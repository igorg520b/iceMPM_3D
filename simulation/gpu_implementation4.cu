#include "gpu_implementation4.h"
#include "model_3d.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#include <spdlog/spdlog.h>

#include "helper_math.cuh"

__device__ int gpu_error_indicator;
__constant__ icy::SimParams3D gprms;


void GPU_Implementation4::initialize()
{
    if(initialized) return;
    cudaError_t err;
    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) throw std::runtime_error("GPU_Implementation3::initialize() cuda error");
    if(deviceCount == 0) throw std::runtime_error("No avaialble CUDA devices");
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    spdlog::info("Compute capability {}.{}",deviceProp.major, deviceProp.minor);
    cudaEventCreate(&eventCycleStart);
    cudaEventCreate(&eventCycleStop);
    err = cudaStreamCreate(&streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Implementation3::initialize() cudaEventCreate");
    initialized = true;
    spdlog::info("GPU_Implementation4::initialize() done");
}

void GPU_Implementation4::cuda_update_constants()
{
    cudaError_t err;
    err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess) throw std::runtime_error("cuda_update_constants()");
    err = cudaMemcpyToSymbol(gprms, &model->prms, sizeof(icy::SimParams3D));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");
    spdlog::info("CUDA constants copied to device");
}

void GPU_Implementation4::cuda_allocate_arrays(size_t nGridNodes, size_t nPoints)
{
    if(!initialized) initialize();
    cudaError_t err;

    // device memory for grid
    cudaFree(model->prms.grid_array);
    cudaFree(model->prms.pts_array);
    cudaFree(model->prms.indenter_force_accumulator);
    cudaFreeHost(tmp_transfer_buffer);
    cudaFreeHost(host_side_indenter_force_accumulator);

    err = cudaMallocPitch (&model->prms.grid_array, &model->prms.nGridPitch, sizeof(real)*nGridNodes, icy::SimParams3D::nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    model->prms.nGridPitch /= sizeof(real); // assume that this divides without a remainder

    // device memory for points
    err = cudaMallocPitch (&model->prms.pts_array, &model->prms.nPtsPitch, sizeof(real)*nPoints, icy::SimParams3D::nPtsArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    model->prms.nPtsPitch /= sizeof(real);

    err = cudaMalloc(&model->prms.indenter_force_accumulator, sizeof(real)*model->prms.indenter_array_size);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    // pinned host memory
    err = cudaMallocHost(&tmp_transfer_buffer, sizeof(real)*model->prms.nPtsPitch*icy::SimParams3D::nPtsArrays);
    if(err!=cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    err = cudaMallocHost(&host_side_indenter_force_accumulator, sizeof(real)*model->prms.indenter_array_size);
    if(err!=cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    double MemGrid = (double)model->prms.nGridPitch*sizeof(real)*icy::SimParams3D::nGridArrays/(1024*1024);
    double MemPoints = (double)model->prms.nPtsPitch*sizeof(real)*icy::SimParams3D::nPtsArrays/(1024*1024);
    double MemTotal = MemGrid + MemPoints;
    spdlog::info("memory use: grid {:03.2f} Mb; points {:03.2f} Mb ; total {:03.2f} Mb", MemGrid, MemPoints, MemTotal);
    error_code = 0;
    spdlog::info("cuda_allocate_arrays done");
}

void GPU_Implementation4::transfer_ponts_to_device()
{
    int pitch = model->prms.nPtsPitch;
    // transfer point data to device
    cudaError_t err = cudaMemcpy(model->prms.pts_array, tmp_transfer_buffer,
                     pitch*sizeof(real)*icy::SimParams3D::nPtsArrays, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    memset(host_side_indenter_force_accumulator, 0, sizeof(real)*model->prms.indenter_array_size);
    spdlog::info("GPU_Implementation4::transfer_ponts_to_device() done");
}

void GPU_Implementation4::cuda_transfer_from_device()
{
    spdlog::info("GPU_Implementation4::cuda_transfer_from_device()");
    cudaError_t err = cudaMemcpyAsync(tmp_transfer_buffer, model->prms.pts_array,
                          model->prms.nPtsPitch*sizeof(real)*icy::SimParams3D::nPtsArrays,
                          cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    err = cudaMemcpyAsync(host_side_indenter_force_accumulator, model->prms.indenter_force_accumulator,
                          sizeof(real)*model->prms.indenter_array_size,
                          cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    err = cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(int), 0, cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    void* userData = reinterpret_cast<void*>(this);
    cudaStreamAddCallback(streamCompute, GPU_Implementation4::callback_transfer_from_device_completion, userData, 0);
}

void CUDART_CB GPU_Implementation4::callback_transfer_from_device_completion(cudaStream_t stream, cudaError_t status, void *userData)
{
    // simulation data was copied to host memory -> proceed with processing of this data
    GPU_Implementation4 *gpu = reinterpret_cast<GPU_Implementation4*>(userData);
    gpu->transfer_ponts_to_host_finalize();
    if(gpu->transfer_completion_callback) gpu->transfer_completion_callback();
}

void GPU_Implementation4::transfer_ponts_to_host_finalize()
{
    // operations that take place right after the data is transferred to host
    Vector3r total;
    total.setZero();

    for(int i=0; i<model->prms.indenter_array_size; i++)
    {
        host_side_indenter_force_accumulator[i] /= model->prms.UpdateEveryNthStep;
        total[i%3] += host_side_indenter_force_accumulator[i];
    }
    this->indenter_force_history.push_back(total);
}

void GPU_Implementation4::cuda_reset_grid()
{
    cudaError_t err = cudaMemsetAsync(model->prms.grid_array, 0,
                                      model->prms.nGridPitch*icy::SimParams3D::nGridArrays*sizeof(real), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}

void GPU_Implementation4::cuda_reset_indenter_force_accumulator()
{
    cudaError_t err = cudaMemsetAsync(model->prms.indenter_force_accumulator, 0,
                                      sizeof(real)*model->prms.indenter_array_size, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}

void GPU_Implementation4::cuda_p2g()
{
    const int nPoints = model->prms.nPts;
    int tpb = model->prms.tpb_P2G;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>();

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        const char* errorDescription = cudaGetErrorString(err);
        spdlog::critical("p2g error {}, {}", err, errorDescription);
        throw std::runtime_error("cuda_p2g");
    }
}

void GPU_Implementation4::cuda_update_nodes(real indenter_x, real indenter_y)
{
    const int nGridNodes = model->prms.GridTotal;
    int tpb = model->prms.tpb_Upd;
    int blocksPerGrid = (nGridNodes + tpb - 1) / tpb;
    kernel_update_nodes<<<blocksPerGrid, tpb, 0, streamCompute>>>(indenter_x, indenter_y);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        const char* errorDescription = cudaGetErrorString(err);
        spdlog::critical("cuda_update_nodes cuda error {}, {}", err, errorDescription);
        throw std::runtime_error("cuda_update_nodes");
    }
}

void GPU_Implementation4::cuda_g2p()
{
    const int nPoints = model->prms.nPts;
    int tpb = model->prms.tpb_G2P;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    kernel_g2p<<<blocksPerGrid, tpb, 0, streamCompute>>>();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        const char* errorDescription = cudaGetErrorString(err);
        spdlog::critical("g2p error {}, {}", err, errorDescription);
        throw std::runtime_error("cuda_g2p");
    }
}


// ============================== functions ====================================

__forceinline__ __device__ Matrix3r dev(Matrix3r A)
{
    return A - A.trace()/3*Matrix3r::Identity();
}

__forceinline__ __device__ void svd3x3(const Matrix3r &A, Matrix3r &_U, Matrix3r &_S, Matrix3r &_V)
{
    double U[9] = {};
    double S[3] = {};
    double V[9] = {};
    svd(A(0,0), A(0,1), A(0,2), A(1,0), A(1,1), A(1,2), A(2,0), A(2,1), A(2,2),
              U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8],
              S[0], S[1], S[2],
              V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
    _U << U[0], U[3], U[6],
          U[1], U[4], U[7],
          U[2], U[5], U[8];
    _S << S[0], 0, 0,
        0, S[1], 0,
        0, 0, S[2];
    _V << V[0], V[3], V[6],
          V[1], V[4], V[7],
          V[2], V[5], V[8];
}


__forceinline__ __device__ Matrix3r KirchhoffStress_Wolper(const Matrix3r &F)
{
    const real &kappa = gprms.kappa;
    const real &mu = gprms.mu;
    const real &dim = 3;

    // Kirchhoff stress as per Wolper (2019)
    real Je = F.determinant();
    Matrix3r b = F*F.transpose();
    Matrix3r PFt = mu*pow(Je, -2./dim)*dev(b) + kappa*(Je*Je-1.)*Matrix3r::Identity();
    return PFt;
}

__forceinline__ __device__ void Wolper_Drucker_Prager(icy::Point3D &p)
{
    const Matrix3r &gradV = p.Bp;
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &dt = gprms.InitialTimeStep;
    const real &tan_phi = gprms.DP_tan_phi;
    const real &DP_threshold_p = gprms.DP_threshold_p;
    constexpr real d = 3;

    Matrix3r FeTr = (Matrix3r::Identity() + dt*gradV) * p.Fe;
    Matrix3r U, V, Sigma;
    svd3x3(FeTr, U, Sigma, V);

    real Je_tr = Sigma(0,0)*Sigma(1,1)*Sigma(2,2);
    real Je_tr_sq = Je_tr*Je_tr;
    Matrix3r SigmaSquared = Sigma*Sigma;
    Matrix3r s_hat_tr = mu * rcbrt(Je_tr_sq) * dev(SigmaSquared);  //  pow(Je_tr, -2./d)
    real p_trial = -(kappa/2.)*(Je_tr_sq - 1);

    if(p_trial < -DP_threshold_p || p.Jp_inv < 1)
    {
        p.q = 1;
        // tear in tension or compress until original state
        real p_new = -DP_threshold_p;
        real Je_new = sqrt(-2.*p_new/kappa + 1.);
        Matrix3r Sigma_new = Matrix3r::Identity() * cbrt(Je_new); //  Matrix3r::Identity()*pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        p.Jp_inv *= Je_new/Je_tr;
    }
    else
    {
        constexpr real coeff1 = 1.2247448713915890; // sqrt((6-d)/2.);
        real q_tr = coeff1*s_hat_tr.norm();
        real q_n_1 = (p_trial+DP_threshold_p)*tan_phi;
        q_n_1 = min(gprms.IceShearStrength, q_n_1);

        if(q_tr < q_n_1)
        {
            // elastic regime
            p.Fe = FeTr;
            p.q = 4;
        }
        else
        {
            // project onto YS
            real s_hat_n_1_norm = q_n_1/coeff1;
            Matrix3r B_hat_E_new = (s_hat_n_1_norm*cbrt(Je_tr_sq)/mu)*s_hat_tr.normalized() + Matrix3r::Identity()*(SigmaSquared.trace()/d);

            Eigen::Array<real,3,1> Sigma_new = B_hat_E_new.diagonal().array().sqrt();
            p.Fe = U * Sigma_new.matrix().asDiagonal() * V.transpose();
            p.q = 3;
        }
    }
}

__forceinline__ __device__ void GetQPP0ForGrain(const int grain, real &p0, real &beta, real &mSq)
{
    // introduce parameter variability depending on the grain number
    real var1 = 1.0 + gprms.GrainVariability*0.05*(-10 + grain%21);
    real var2 = 1.0 + gprms.GrainVariability*0.05*(-10 + (grain+7)%21);
    real var3 = 1.0 + gprms.GrainVariability*0.05*(-10 + (grain+14)%21);

    p0 = gprms.IceCompressiveStrength * var1;
    real p = gprms.IceTensileStrength * var2;
    real q = gprms.IceShearStrength * var3;

    beta = p / p0;
    real NACC_M = (2*q*sqrt(1+2*beta))/(p0*(1+beta));
    mSq = NACC_M*NACC_M;
}



__forceinline__ __device__ void CheckIfPointIsInsideFailureSurface(icy::Point3D &p)
{
    const Matrix3r &gradV = p.Bp;
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &dt = gprms.InitialTimeStep;

    real beta, M_sq, p0;
    GetQPP0ForGrain(p.grain, p0, beta, M_sq);

    constexpr real d = 3; // dimension

    Matrix3r FeTr = (Matrix3r::Identity() + dt*gradV) * p.Fe;
    p.Fe = FeTr;
    Matrix3r U, V, Sigma;
    svd3x3(FeTr, U, Sigma, V);

    real Je_tr = Sigma(0,0)*Sigma(1,1)*Sigma(2,2);
    real Je_tr_sq = Je_tr*Je_tr;
    real p_trial = -(kappa/2.) * (Je_tr*Je_tr - 1.);

    Matrix3r SigmaSquared = Sigma*Sigma;
    Matrix3r s_hat_tr = mu * rcbrt(Je_tr_sq) * dev(SigmaSquared);
    real y = (1.+2.*beta)*(3.-d/2.)*s_hat_tr.squaredNorm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(y > 0) p.q = 3;
}

// ==============================  kernels  ====================================

__global__ void kernel_p2g()
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.nPts;
    if(pt_idx >= nPoints) return;

    const real &dt = gprms.InitialTimeStep;
    const real &vol = gprms.ParticleVolume;
    const real &h = gprms.cellsize;
    const real &h_inv = gprms.cellsize_inv;
    const real &Dinv = gprms.Dp_inv;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const int &gridZ = gprms.GridZ;
    const real &particle_mass = gprms.ParticleMass;
    const int &nGridPitch = gprms.nGridPitch;
    const int &pitch = gprms.nPtsPitch;

    // pull point data from SOA
    const real *buffer = gprms.pts_array;
    Vector3r pos, velocity;
    Matrix3r Bp, Fe;

    for(int i=0; i<3; i++)
    {
        pos[i] = buffer[pt_idx + pitch*(icy::SimParams3D::posx+i)];
        velocity[i] = buffer[pt_idx + pitch*(icy::SimParams3D::velx+i)];
        for(int j=0; j<3; j++)
        {
            Fe(i,j) = buffer[pt_idx + pitch*(icy::SimParams3D::Fe00 + i*3 + j)];
            Bp(i,j) = buffer[pt_idx + pitch*(icy::SimParams3D::Bp00 + i*3 + j)];
        }
    }

//    char* ptr_intact = (char*)(&buffer[pitch*icy::SimParams3D::idx_intact]);
//    char q = ptr_intact[pt_idx];
//    real Jp_inv = buffer[pt_idx + pitch*icy::SimParams3D::idx_Jp_inv];

    Matrix3r PFt = KirchhoffStress_Wolper(Fe);
    Matrix3r subterm2 = particle_mass*Bp - (dt*vol*Dinv)*PFt;

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(pos[0]*h_inv - offset);
    const int j0 = (int)(pos[1]*h_inv - offset);
    const int k0 = (int)(pos[2]*h_inv - offset);

    Vector3r base_coord(i0,j0,k0);
    Vector3r f = pos*h_inv - base_coord;

    // pre-compute the weight function at given offsets
    Array3r arr_v0 = 1.5 - f.array();
    Array3r arr_v1 = f.array() - 1.0;
    Array3r arr_v2 = f.array() - 0.5;
    Array3r ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    // distribute point values to the grid
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            for (int k=0; k<3;k++)
            {
                real Wip = ww[i][0]*ww[j][1]*ww[k][2];
                Vector3r dpos((i-f[0])*h, (j-f[1])*h, (k-f[2])*h);
                Vector3r incV = Wip*(velocity*particle_mass + subterm2*dpos);
                real incM = Wip*particle_mass;

                int idx_gridnode = (i+i0) + (j+j0)*gridX + (k+k0)*gridX*gridY;
                if((i+i0) < 0 || (j+j0) < 0 || (i+i0) >=gridX || (j+j0)>=gridY || (k+k0) < 0 || (k+k0)>=gridZ)
                    gpu_error_indicator = 1;

                // Udpate mass, velocity and force
                atomicAdd(&gprms.grid_array[0*nGridPitch + idx_gridnode], incM);
                for(int idx=0;idx<3;idx++)
                    atomicAdd(&gprms.grid_array[(1+idx)*nGridPitch + idx_gridnode], incV[idx]);
            }
}

__global__ void kernel_update_nodes(real indenter_x, real indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nGridNodes = gprms.GridTotal;
    if(idx >= nGridNodes) return;

    real mass = gprms.grid_array[idx];
    if(mass == 0) return;

    const int &pitch = gprms.nGridPitch;
    const real &gravity = gprms.Gravity;
    const real &indRsq = gprms.IndRSq;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const int &gridZ = gprms.GridZ;
    const real &dt = gprms.InitialTimeStep;
    const real &ind_velocity = gprms.IndVelocity;
    const real &cellsize = gprms.cellsize;

    const Vector3r vco(ind_velocity,0,0);  // velocity of the collision object (indenter)
    const Vector2r_ indCenter(indenter_x, indenter_y);

    // velocity of the node, adjusted for gravity and speed-limited
    Vector3r velocity(gprms.grid_array[pitch+idx], gprms.grid_array[2*pitch+idx], gprms.grid_array[3*pitch+idx]);
    velocity /= mass;
    velocity[1] -= dt*gravity;
    const real vmax = 0.5*cellsize/dt;
    if(velocity.norm() > vmax) velocity = velocity.normalized()*vmax;

    // x-y-z index of the grid node
    int idx_x = idx % gridX;
    int idx_y = (idx / gridX) % gridY;
    int idx_z = idx / (gridX*gridY);

    if(gprms.SetupType == 0)
    {
        // cylindrical indenter for RHITA-type setup
        Vector2r_ gnpos(idx_x*cellsize, idx_y*cellsize);    // position of the grid node
        Vector2r_ n = gnpos - indCenter;    // vector pointing at the node from indenter's center
        if(n.squaredNorm() < indRsq)
        {
            // grid node is inside the indenter
            Vector3r vrel = velocity - vco;
            Vector3r n3d(n[0],n[1],0);
            n3d.normalize();
            real vn = vrel.dot(n3d);   // normal component of velocity
            if(vn < 0)
            {
                Vector3r vt = vrel - n3d*vn;   // tangential portion of relative velocity
                Vector3r prev_velocity = velocity;
                velocity = vco + vt;// + ice_friction_coeff*vn*vt.normalized();

                // record force on the indenter
                Vector3r force = (prev_velocity-velocity)*mass/dt;
                double angle = atan2(n[0],n[1]);
                angle += icy::SimParams3D::pi;
                angle *= gprms.n_indenter_subdivisions_angular/(2*icy::SimParams3D::pi);
                int index_angle = min(max((int)angle, 0), gprms.n_indenter_subdivisions_angular-1);
                int index_z = min(max(idx_z,0),gridZ-1);
                int index = index_z + index_angle*gridZ;
                for(int i=0;i<3;i++) atomicAdd(&gprms.indenter_force_accumulator[i+3*index], force[i]);
            }
        }
    }
    else if(gprms.SetupType == 1)
    {
        // flat indenter for vertical indentation
        real gnpos_y = idx_y*cellsize;
        if(gnpos_y > gprms.indenter_y && velocity[1]>-gprms.IndVelocity)
        {
            Vector3r prev_velocity = velocity;
            velocity[1] = -gprms.IndVelocity;
            Vector3r force = (prev_velocity-velocity)*mass/dt;
            int index_x = min(max(idx_x,0),gridX-1);
            int index_z = min(max(idx_z,0),gridZ-1);
            int index = index_z + index_x*gridZ;
            for(int i=0;i<3;i++) atomicAdd(&gprms.indenter_force_accumulator[i+3*index], force[i]);
        }

    }

    // attached bottom layer and walls
    if(idx_x <= 3 && velocity[0]<0) velocity[0] = 0;
    else if(idx_x >= gridX-4 && velocity[0]>0) velocity[0] = 0;

    if(idx_y <= 3) velocity.setZero();
    else if(idx_y >= gridY-4 && velocity[1]>0) velocity[1] = 0;

    if(idx_z <= 3 && velocity[2]<0) velocity[2] = 0;
    else if(idx_z >= gridZ-4 && velocity[2]>0) velocity[2] = 0;
/*
    // hold lower half of the block at boudaries
    real &hinv = gprms.cellsize_inv;
    int blocksXmin = gprms.BlockOffsetX+5;
    int blocksXmax = gprms.BlockOffsetX+5 + (int)(gprms.IceBlockDimX*hinv);

    int blocksYmid = 2 + (int)(gprms.IceBlockDimY*hinv/2);

    int blocksZmin = gprms.GridZ/2 - (int)(gprms.IceBlockDimZ*gprms.cellsize_inv/2);
    int blocksZmax = gprms.GridZ/2 + (int)(gprms.IceBlockDimZ*gprms.cellsize_inv/2);

    if(idx_y <= blocksYmid && (
                                (idx_x >= (blocksXmin-1) && idx_x < (blocksXmin+2)) ||
                                (idx_x > (blocksXmax-2) && idx_x <= (blocksXmax+1)) ||
                                (idx_z >= (blocksZmin-1) && idx_z < (blocksZmin+2)) ||
                                (idx_z > (blocksZmax-2) && idx_z <= (blocksZmax+1))
                                )) velocity.setZero();
*/
    // write the updated grid velocity back to memory
    for(int i=0;i<3;i++) gprms.grid_array[(1+i)*pitch + idx] = velocity[i];
}

__global__ void kernel_g2p()
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.nPts;
    if(pt_idx >= nPoints) return;

    const int &pitch_pts = gprms.nPtsPitch;
    const int &pitch_grid = gprms.nGridPitch;
    const real &h_inv = gprms.cellsize_inv;
    const real &dt = gprms.InitialTimeStep;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;

    icy::Point3D p;
    p.velocity.setZero();
    p.Bp.setZero();

    real *buffer = gprms.pts_array;

    for(int i=0; i<3; i++)
    {
        p.pos[i] = buffer[pt_idx + pitch_pts*(icy::SimParams3D::posx+i)];
        for(int j=0; j<3; j++) p.Fe(i,j) = buffer[pt_idx + pitch_pts*(icy::SimParams3D::Fe00 + i*3 + j)];
    }

    char* ptr_intact = (char*)(&buffer[pitch_pts*icy::SimParams3D::idx_intact]);
    p.q = ptr_intact[pt_idx];
    short* ptr_grain = (short*)(&ptr_intact[pitch_pts]);
    p.grain = ptr_grain[pt_idx];
    p.Jp_inv = buffer[pt_idx + pitch_pts*icy::SimParams3D::idx_Jp_inv];

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(p.pos[0]*h_inv - offset);
    const int j0 = (int)(p.pos[1]*h_inv - offset);
    const int k0 = (int)(p.pos[2]*h_inv - offset);

    Vector3r base_coord(i0,j0,k0);
    Vector3r f = p.pos*h_inv - base_coord;

    Array3r arr_v0 = 1.5-f.array();
    Array3r arr_v1 = f.array() - 1.0;
    Array3r arr_v2 = f.array() - 0.5;
    Array3r ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            for (int k=0; k<3;k++)
        {
            real weight = ww[i][0]*ww[j][1]*ww[k][2];

            int idx_gridnode = (i+i0) + (j+j0)*gridX + (k+k0)*gridX*gridY;
            Vector3r node_velocity;
            for(int idx=0;idx<3;idx++) node_velocity[idx] = gprms.grid_array[(idx+1)*pitch_grid + idx_gridnode];
            p.velocity += weight * node_velocity;
            Vector3r dpos(i-f[0], j-f[1],k-f[2]);  // note the absence of multiplicaiton by h
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection
    p.pos += dt * p.velocity;

    if(p.q == 0) CheckIfPointIsInsideFailureSurface(p);
    else Wolper_Drucker_Prager(p);

    // distribute the values of p back into GPU memory: pos, velocity, BP, Fe, Jp_inv, q
    ptr_intact[pt_idx] = p.q;
    buffer[pt_idx + pitch_pts*icy::SimParams3D::idx_Jp_inv] = p.Jp_inv;

    for(int i=0; i<3; i++)
    {
        buffer[pt_idx + pitch_pts*(icy::SimParams3D::posx+i)] = p.pos[i];
        buffer[pt_idx + pitch_pts*(icy::SimParams3D::velx+i)] = p.velocity[i];
        for(int j=0; j<3; j++)
        {
            buffer[pt_idx + pitch_pts*(icy::SimParams3D::Fe00 + i*3 + j)] = p.Fe(i,j);
            buffer[pt_idx + pitch_pts*(icy::SimParams3D::Bp00 + i*3 + j)] = p.Bp(i,j);
        }
    }
}

//===========================================================================


__global__ void kernel_hello()
{
    printf("hello from CUDA\n");

    Matrix3r M, U, S, V, _U, _S, _V;
    M << 1.1, -4.1, 22, 0.1, 0.2, -5, 5, 5.1, 0.11;
    _U << -0.976143, 0.0184756, 0.216339, 0.21543, -0.0418997, 0.97562, 0.0270897, 0.998951, 0.0369199;
    _S << 22.9524, 0., 0., 0., 7.12328, 0., 0., 0., 0.732977;
    _V << -0.039942, 0.703452, 0.70962, 0.182265, 0.7034, -0.687028, -0.982438, 0.101898, -0.15631;

    svd3x3(M,U,S,V);
    printf("result of svd\nU:\n");
    for(int i=0;i<3;i++) { for(int j=0;j<3;j++) printf("%f ;",U(i,j)); printf("\n"); }
    printf("\nS:\n");
    for(int i=0;i<3;i++) { for(int j=0;j<3;j++) printf("%f ;",S(i,j)); printf("\n"); }
    printf("\nV:\n");
    for(int i=0;i<3;i++) { for(int j=0;j<3;j++) printf("%f ;",V(i,j)); printf("\n"); }

    Matrix3r USVT = U*S*V.transpose();
    printf("\nUSVT:\n");
    for(int i=0;i<3;i++) { for(int j=0;j<3;j++) printf("%f ;",USVT(i,j)); printf("\n"); }
    printf("\nM:\n");
    for(int i=0;i<3;i++) { for(int j=0;j<3;j++) printf("%f ;",M(i,j)); printf("\n"); }
}


void GPU_Implementation4::test()
{
    cudaError_t err;
    kernel_hello<<<1,1,0,streamCompute>>>();
    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        std::cout << "cuda test error " << err << '\n';
        throw std::runtime_error("cuda test");
    }
    else
    {
        std::cout << "hello kernel executed successfully\n";
    }
    cudaDeviceSynchronize();
}

void GPU_Implementation4::synchronize()
{
    if(!initialized) return;
    cudaDeviceSynchronize();
}

