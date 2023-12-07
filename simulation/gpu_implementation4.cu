#include "gpu_implementation4.h"
#include "model_3d.h".h"

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
}

void GPU_Implementation4::cuda_update_constants()
{
    cudaError_t err;
    err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess) throw std::runtime_error("cuda_update_constants()");
    err = cudaMemcpyToSymbol(gprms, prms, sizeof(icy::SimParams3D));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");
    spdlog::info("CUDA constants copied to device");
}

void GPU_Implementation4::cuda_allocate_arrays(size_t nGridNodes, size_t nPoints)
{
    if(!initialized) initialize();
    cudaError_t err;

    // device memory for grid
    cudaFree(prms->grid_array);
    cudaFree(prms->pts_array);
    cudaFree(prms->indenter_force_accumulator);
    cudaFreeHost(tmp_transfer_buffer);
    cudaFreeHost(host_side_indenter_force_accumulator);

    err = cudaMallocPitch (&prms->grid_array, &prms->nGridPitch, sizeof(real)*nGridNodes, icy::SimParams3D::nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    prms->nGridPitch /= sizeof(real); // assume that this divides without a remainder

    // device memory for points
    err = cudaMallocPitch (&prms->pts_array, &prms->nPtsPitch, sizeof(real)*nPoints, icy::SimParams3D::nPtsArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    prms->nPtsPitch /= sizeof(real);

    err = cudaMalloc(&prms->indenter_force_accumulator, sizeof(real)*icy::SimParams3D::indenter_array_size);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    // pinned host memory
    err = cudaMallocHost(&tmp_transfer_buffer, sizeof(real)*prms->nPtsPitch*icy::SimParams3D::nPtsArrays);
    if(err!=cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    err = cudaMallocHost(&host_side_indenter_force_accumulator, sizeof(real)*icy::SimParams3D::indenter_array_size);
    if(err!=cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    double MemGrid = (double)prms->nGridPitch*sizeof(real)*icy::SimParams::nGridArrays/(1024*1024);
    double MemPoints = (double)prms->nPtsPitch*sizeof(real)*icy::SimParams::nPtsArrays/(1024*1024);
    double MemTotal = MemAllocGrid + MemAllocPoints;
    spdlog::info("memory use: grid {:03.2f} Mb; points {:03.2f} Mb ; total {:03.2f} Mb", MemGrid, MemPoints, MemTotal);
    error_code = 0;
    spdlog::info("cuda_allocate_arrays done");
}

void GPU_Implementation4::transfer_ponts_to_device()
{
    const std::vector<icy::Point3D> &points = model->points;
    int pitch = model->prms.nPtsPitch;
    for(int idx=0;idx<model->prms.nPts;idx++) points[idx].TransferToBuffer(tmp_transfer_buffer, pitch, idx);

    // transfer point data to device
    cudaError_t err = cudaMemcpy(model->prms.pts_array, tmp_transfer_buffer,
                     pitch*sizeof(real)*icy::SimParams3D::nPtsArrays, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");
}

void GPU_Implementation4::cuda_transfer_from_device()
{
    cudaError_t err = cudaMemcpyAsync(tmp_transfer_buffer, model->prms.pts_array,
                          model->prms.nPtsPitch*sizeof(real)*icy::SimParams3D::nPtsArrays,
                          cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    err = cudaMemcpyAsync(host_side_indenter_force_accumulator, model->prms.indenter_force_accumulator,
                          sizeof(real)*icy::SimParams3D::indenter_array_size,
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
    gpu->model->hostside_data_update_mutex.lock();
    gpu->transfer_ponts_to_host_finalize(points);
    gpu->model->hostside_data_update_mutex.unlock();
    if(gpu->transfer_completion_callback) gpu->transfer_completion_callback();
}

void GPU_Implementation4::transfer_ponts_to_host_finalize()
{
    const std::vector<icy::Point3D> &points = model->points;
    int pitch = model->prms.nPtsPitch;
    for(int idx=0;idx<model->prms.nPts;idx++) points[idx].PullFromBuffer(tmp_transfer_buffer, pitch, idx);

    // add up indenter forces
    Vector3r indenter_force;
    indenter_force.setZero();
    for(int i=0; i<icy::SimParams3D::indenter_array_size/3; i++)
        for(int j=0;j<3;j++)
            indenter_force[j] += host_side_indenter_force_accumulator[j+i*3];
    indenter_force /= model->prms.UpdateEveryNthStep;
    model->indenter_force_history.push_back(indenter_force);
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
                                      sizeof(real)*icy::SimParams3D::indenter_array_size, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}

void GPU_Implementation4::cuda_p2g()
{
    const int nPoints = model->prms.nPts;
    int tpb = model->prms.tpb_P2G;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) throw std::runtime_error("cuda_p2g");
}

void GPU_Implementation4::cuda_update_nodes(real indenter_x, real indenter_y)
{
    const int nGridNodes = model->prms.GridTotal;
    int tpb = model->prms.tpb_Upd;
    int blocksPerGrid = (nGridNodes + tpb - 1) / tpb;
    kernel_update_nodes<<<blocksPerGrid, tpb, 0, streamCompute>>>(indenter_x, indenter_y);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) throw std::runtime_error("cuda_update_nodes");
}

void GPU_Implementation4::cuda_g2p()
{
    const int nPoints = model->prms.nPts;
    int tpb = model->prms.tpb_G2P;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    kernel_g2p<<<blocksPerGrid, tpb, 0, streamCompute>>>();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) throw std::runtime_error("cuda_g2p");
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
    svd(A(0,0), A(1,0), A(2,0), A(0,1), A(1,1), A(2,1), A(0,2), A(1,2), A(2,2),     //F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8],
              U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8],
              S[0], S[1], S[2],
              V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
    U << U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8];
    S << S[0], 0, 0,
        0, S[1], 0,
        0, 0, S[2];
    V << V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8];
}


__forceinline__ __device__ Matrix3r KirchhoffStress_Wolper(const Matrix3r &F)
{
    const real &kappa = gprms.kappa;
    const real &mu = gprms.mu;
    const real &dim = icy::SimParams3D::dim;

    // Kirchhoff stress as per Wolper (2019)
    real Je = F.determinant();
    Matrix3r b = F*F.transpose();
    Matrix3r PFt = mu*pow(Je, -2./dim)*dev(b) + kappa*(Je*Je-1.)*Matrix3r::Identity();
    return PFt;
}

__forceinline__ __device__ void Wolper_Drucker_Prager(icy::Point &p)
{
    const Matrix3r &gradV = p.Bp;
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &dt = gprms.InitialTimeStep;
    const real &tan_phi = gprms.DP_tan_phi;
    constexpr real d = 3;

    Matrix3r FeTr = (Matrix3r::Identity() + dt*gradV) * p.Fe;
    Matrix3r U, V, Sigma;
    svd3x3(FeTr, U, Sigma, V);

    real Je_tr = Sigma(0,0)*Sigma(1,1)*Sigma(2,2);
    real Je_tr_sq = Je_tr*Je_tr;
    Matrix3r SigmaSquared = Sigma*Sigma;
    Matrix3r s_hat_tr = mu * rcbrt(Je_tr_sq) * dev(SigmaSquared);  //  pow(Je_tr, -2./d)
    real p_trial = -(kappa/2.)*(Je_tr_sq - 1);

    if(p_trial < 0 || p.Jp_inv < 1)
    {
        p.q = 1;
//        if(p_trial < 1)  p.q = 1;
//        else if(p.Jp_inv < 1) p.q = 2;

        // tear in tension or compress until original state
        real p_new = 0;
        real Je_new = sqrt(-2.*p_new/kappa + 1.);
        Matrix3r Sigma_new = Matrix3r::Identity() * cbrt(Je_new); //  Matrix3r::Identity()*pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        p.Jp_inv *= Je_new/Je_tr;
    }
    else
    {
        constexpr real coeff1 = 1.2247448713915890; // sqrt((6-d)/2.);
        real q_tr = coeff1*s_hat_tr.norm();
        real q_n_1 = p_trial*tan_phi;
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
            Matrix3r Sigma_new;
            Sigma_new << sqrt(B_hat_E_new(0,0)), 0, 0,
                0, sqrt(B_hat_E_new(1,1)), 0,
                0, 0, sqrt(B_hat_E_new(2,2));
            p.Fe = U*Sigma_new*V.transpose();
            p.q = 3;
        }
    }
}


__forceinline__ __device__ void NACCUpdateDeformationGradient_trimmed(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    constexpr real d = 2; // dimensions
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &beta = gprms.NACC_beta;
    const real &dt = gprms.InitialTimeStep;

    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;
    p.Fe = FeTr;
    Matrix2r U, V, Sigma;
    svd2x2(FeTr, U, Sigma, V);

    real Je_tr = Sigma(0,0)*Sigma(1,1);    // this is for 2D
    real p_trial = -(kappa/2.) * (Je_tr*Je_tr - 1.);

    const real &p0 = gprms.IceCompressiveStrength;

    Matrix2r SigmaSquared = Sigma*Sigma;
    Matrix2r s_hat_tr = mu/Je_tr * dev(SigmaSquared); //mu * pow(Je_tr, -2. / (real)d)* dev(SigmaSquared);
    const real &M_sq = gprms.NACC_Msq;
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
    const real &particle_mass = gprms.ParticleMass;
    const int &nGridPitch = gprms.nGridPitch;
    const int &nPtsPitch = gprms.nPtsPitch;

    // pull point data from SOA
    const real *data = gprms.pts_array;
    Vector2r pos(data[pt_idx + nPtsPitch*icy::SimParams::posx], data[pt_idx + nPtsPitch*icy::SimParams::posy]);
    Vector2r velocity(data[pt_idx + nPtsPitch*icy::SimParams::velx], data[pt_idx + nPtsPitch*icy::SimParams::vely]);
    Matrix2r Bp, Fe;
    Bp << data[pt_idx + nPtsPitch*icy::SimParams::Bp00], data[pt_idx + nPtsPitch*icy::SimParams::Bp01],
        data[pt_idx + nPtsPitch*icy::SimParams::Bp10], data[pt_idx + nPtsPitch*icy::SimParams::Bp11];
    Fe << data[pt_idx + nPtsPitch*icy::SimParams::Fe00], data[pt_idx + nPtsPitch*icy::SimParams::Fe01],
        data[pt_idx + nPtsPitch*icy::SimParams::Fe10], data[pt_idx + nPtsPitch*icy::SimParams::Fe11];
    // real Jp_inv =        data[pt_idx + nPtsPitch*icy::SimParams::idx_Jp];
    // real zeta =          data[pt_idx + nPtsPitch*icy::SimParams::idx_zeta];


    Matrix2r PFt = KirchhoffStress_Wolper(Fe);
    Matrix2r subterm2 = particle_mass*Bp - (dt*vol*Dinv)*PFt;

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(pos[0]*h_inv - offset);
    const int j0 = (int)(pos[1]*h_inv - offset);

    Vector2r base_coord(i0,j0);
    Vector2r fx = pos*h_inv - base_coord;

    real v0[2] {1.5-fx[0], 1.5-fx[1]};
    real v1[2] {fx[0]-1.,  fx[1]-1.};
    real v2[2] {fx[0]-.5,  fx[1]-.5};

    real w[3][2] = {{.5*v0[0]*v0[0],  .5*v0[1]*v0[1]},
                    {.75-v1[0]*v1[0], .75-v1[1]*v1[1]},
                    {.5*v2[0]*v2[0],  .5*v2[1]*v2[1]}};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            real Wip = w[i][0]*w[j][1];
            Vector2r dpos((i-fx[0])*h, (j-fx[1])*h);
            Vector2r incV = Wip*(velocity*particle_mass + subterm2*dpos);
            real incM = Wip*particle_mass;

            int idx_gridnode = (i+i0) + (j+j0)*gridX;
            if((i+i0) < 0 || (j+j0) < 0 || (i+i0) >=gridX || (j+j0)>=gridY) gpu_error_indicator = 1;

            // Udpate mass, velocity and force
            atomicAdd(&gprms.grid_array[0*nGridPitch + idx_gridnode], incM);
            atomicAdd(&gprms.grid_array[1*nGridPitch + idx_gridnode], incV[0]);
            atomicAdd(&gprms.grid_array[2*nGridPitch + idx_gridnode], incV[1]);
        }
}

__global__ void v2_kernel_update_nodes(real indenter_x, real indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nGridNodes = gprms.GridX*gprms.GridY;
    if(idx >= nGridNodes) return;

    real mass = gprms.grid_array[idx];
    if(mass == 0) return;

    const int &nGridPitch = gprms.nGridPitch;
    Vector2r velocity(gprms.grid_array[1*nGridPitch + idx], gprms.grid_array[2*nGridPitch + idx]);
    const real &gravity = gprms.Gravity;
    const real &indRsq = gprms.IndRSq;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const real &dt = gprms.InitialTimeStep;
    const real &ind_velocity = gprms.IndVelocity;
    const real &cellsize = gprms.cellsize;
    const real &ice_friction_coeff = gprms.IceFrictionCoefficient;

    const Vector2r vco(ind_velocity,0);  // velocity of the collision object (indenter)
    const Vector2r indCenter(indenter_x, indenter_y);

    velocity /= mass;
    velocity[1] -= dt*gravity;
    real vmax = 0.5*cellsize/dt;
    if(velocity.norm() > vmax) velocity = velocity/velocity.norm()*vmax;

    int idx_x = idx % gridX;
    int idx_y = idx / gridX;

    // indenter
    Vector2r gnpos(idx_x*cellsize, idx_y*cellsize);
    Vector2r n = gnpos - indCenter;
    if(n.squaredNorm() < indRsq)
    {
        // grid node is inside the indenter
        Vector2r vrel = velocity - vco;
        n.normalize();
        real vn = vrel.dot(n);   // normal component of the velocity
        if(vn < 0)
        {
            Vector2r vt = vrel - n*vn;   // tangential portion of relative velocity
            Vector2r prev_velocity = velocity;
            velocity = vco + vt + ice_friction_coeff*vn*vt.normalized();

            // force on the indenter
            Vector2r force = (prev_velocity-velocity)*mass/dt;
            double angle = atan2(n[0],n[1]);
            angle += icy::SimParams::pi;
            angle *= icy::SimParams::n_indenter_subdivisions/ (2*icy::SimParams::pi);
            int index = (int)angle;
            index = max(index, 0);
            index = min(index, icy::SimParams::n_indenter_subdivisions-1);
            atomicAdd(&gprms.indenter_force_accumulator[0+2*index], force[0]);
            atomicAdd(&gprms.indenter_force_accumulator[1+2*index], force[1]);
        }
    }

    // attached bottom layer
    if(idx_y <= 3) velocity.setZero();
    else if(idx_y >= gridY-4 && velocity[1]>0) velocity[1] = 0;
    if(idx_x <= 3 && velocity.x()<0) velocity[0] = 0;
    else if(idx_x >= gridX-5) velocity[0] = 0;
    if(gprms.HoldBlockOnTheRight==1)
    {
        int blocksGridX = gprms.BlockLength*gprms.cellsize_inv+5-2;
        if(idx_x >= blocksGridX) velocity.setZero();
    }
    else if(gprms.HoldBlockOnTheRight==2)
    {
        int blocksGridX = gprms.BlockLength*gprms.cellsize_inv+5-2;
        int blocksGridY = gprms.BlockHeight/2*gprms.cellsize_inv+2;
        if(idx_x >= blocksGridX && idx_x <= blocksGridX + 2 && idx_y < blocksGridY) velocity.setZero();
        if(idx_x <= 7 && idx_x > 4 && idx_y < blocksGridY) velocity.setZero();
    }
    // write the updated grid velocity back to memory
    gprms.grid_array[1*nGridPitch + idx] = velocity[0];
    gprms.grid_array[2*nGridPitch + idx] = velocity[1];
}

__global__ void v2_kernel_g2p()
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.nPts;
    if(pt_idx >= nPoints) return;

    const int &nPtsPitched = gprms.nPtsPitch;
    const int &nGridPitched = gprms.nGridPitch;
    const real &h_inv = gprms.cellsize_inv;
    const real &dt = gprms.InitialTimeStep;
    const int &gridX = gprms.GridX;

    icy::Point p;
    p.pos[0] =      gprms.pts_array[pt_idx + nPtsPitched*icy::SimParams::posx];
    p.pos[1] =      gprms.pts_array[pt_idx + nPtsPitched*icy::SimParams::posy];
    p.Fe(0,0) =     gprms.pts_array[pt_idx + nPtsPitched*icy::SimParams::Fe00];
    p.Fe(0,1) =     gprms.pts_array[pt_idx + nPtsPitched*icy::SimParams::Fe01];
    p.Fe(1,0) =     gprms.pts_array[pt_idx + nPtsPitched*icy::SimParams::Fe10];
    p.Fe(1,1) =     gprms.pts_array[pt_idx + nPtsPitched*icy::SimParams::Fe11];
    p.Jp_inv =      gprms.pts_array[pt_idx + nPtsPitched*icy::SimParams::idx_Jp];
//    p.zeta =        gprms.pts_array[pt_idx + nPtsPitched*icy::SimParams::idx_zeta];
    char* pq_ptr = (char*)&gprms.pts_array[nPtsPitched*icy::SimParams::idx_case];
    p.q =           pq_ptr[pt_idx];

    p.velocity.setZero();
    p.Bp.setZero();

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(p.pos[0]*h_inv - offset);
    const int j0 = (int)(p.pos[1]*h_inv - offset);

    Vector2r base_coord(i0,j0);
    Vector2r fx = p.pos*h_inv - base_coord;

    real v0[2] {1.5-fx[0], 1.5-fx[1]};
    real v1[2] {fx[0]-1.,  fx[1]-1.};
    real v2[2] {fx[0]-.5,  fx[1]-.5};

    real w[3][2] = {{.5*v0[0]*v0[0],  .5*v0[1]*v0[1]},
                    {.75-v1[0]*v1[0], .75-v1[1]*v1[1]},
                    {.5*v2[0]*v2[0],  .5*v2[1]*v2[1]}};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            Vector2r dpos = Vector2r(i, j) - fx;
            real weight = w[i][0]*w[j][1];

            int idx_gridnode = i+i0 + (j+j0)*gridX;
            Vector2r node_velocity;
            node_velocity[0] = gprms.grid_array[1*nGridPitched + idx_gridnode];
            node_velocity[1] = gprms.grid_array[2*nGridPitched + idx_gridnode];
            p.velocity += weight * node_velocity;
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection
    p.pos += dt * p.velocity;

    if(p.q == 0) NACCUpdateDeformationGradient_trimmed(p);
    else Wolper_Drucker_Prager(p);

    gprms.pts_array[icy::SimParams::posx*nPtsPitched + pt_idx] = p.pos[0];
    gprms.pts_array[icy::SimParams::posy*nPtsPitched + pt_idx] = p.pos[1];
    gprms.pts_array[icy::SimParams::velx*nPtsPitched + pt_idx] = p.velocity[0];
    gprms.pts_array[icy::SimParams::vely*nPtsPitched + pt_idx] = p.velocity[1];
    gprms.pts_array[icy::SimParams::Bp00*nPtsPitched + pt_idx] = p.Bp(0,0);
    gprms.pts_array[icy::SimParams::Bp01*nPtsPitched + pt_idx] = p.Bp(0,1);
    gprms.pts_array[icy::SimParams::Bp10*nPtsPitched + pt_idx] = p.Bp(1,0);
    gprms.pts_array[icy::SimParams::Bp11*nPtsPitched + pt_idx] = p.Bp(1,1);
    gprms.pts_array[icy::SimParams::Fe00*nPtsPitched + pt_idx] = p.Fe(0,0);
    gprms.pts_array[icy::SimParams::Fe01*nPtsPitched + pt_idx] = p.Fe(0,1);
    gprms.pts_array[icy::SimParams::Fe10*nPtsPitched + pt_idx] = p.Fe(1,0);
    gprms.pts_array[icy::SimParams::Fe11*nPtsPitched + pt_idx] = p.Fe(1,1);

    gprms.pts_array[icy::SimParams::idx_Jp*nPtsPitched + pt_idx] = p.Jp_inv;
//    gprms.pts_array[icy::SimParams::idx_zeta*nPtsPitched + pt_idx] = p.zeta;

    // visualized variables
//    gprms.pts_array[icy::SimParams::idx_p*nPtsPitched + pt_idx] = p.visualize_p;
//    gprms.pts_array[icy::SimParams::idx_p0*nPtsPitched + pt_idx] = p.visualize_p0;
//    gprms.pts_array[icy::SimParams::idx_q*nPtsPitched + pt_idx] = p.visualize_q;
//    gprms.pts_array[icy::SimParams::idx_psi*nPtsPitched + pt_idx] = p.visualize_psi;
//    gprms.pts_array[icy::SimParams::idx_case*nPtsPitched + pt_idx] = p.q;
//    gprms.pts_array[icy::SimParams::idx_q_limit*nPtsPitched + pt_idx] = p.visualize_q_limit;

    pq_ptr[pt_idx] = p.q;
}

//===========================================================================







// clamp x to range [a, b]
__device__ double clamp(double x, double a, double b)
{
    return max(a, min(b, x));
}


//===========================================================================

//===========================================================================

__device__ void svd(const real a[4], real u[4], real sigma[2], real v[4])
{
    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(a, gu, sigma, gv);
    gu.template fill<2, real>(u);
    gv.template fill<2, real>(v);
}

__device__ void svd2x2(const Matrix2r &mA, Matrix2r &mU, Matrix2r &mS, Matrix2r &mV)
{
    real U[4], V[4], S[2];
    real a[4] = {mA(0,0), mA(0,1), mA(1,0), mA(1,1)};
    svd(a, U, S, V);
    mU << U[0],U[1],U[2],U[3];
    mS << S[0],0,0,S[1];
    mV << V[0],V[1],V[2],V[3];
}


__device__ Matrix2r polar_decomp_R(const Matrix2r &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    real th = atan2(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Matrix2r result;
    result << cosf(th), -sinf(th), sinf(th), cosf(th);
    return result;
}

__global__ void kernel_hello()
{
    printf("hello from CUDA\n");
}


void GPU_Implementation3::test()
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

void GPU_Implementation3::synchronize()
{
    if(!initialized) return;
    cudaDeviceSynchronize();
}

