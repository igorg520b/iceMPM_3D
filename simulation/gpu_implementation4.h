#ifndef GPU_IMPLEMENTATION3_H
#define GPU_IMPLEMENTATION3_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "parameters_sim_3d.h"
#include "point_3d.h"

#include <functional>

__global__ void kernel_p2g();
__global__ void kernel_g2p();
__global__ void kernel_update_nodes(real indenter_x, real indenter_y);

__device__ void Wolper_Drucker_Prager(icy::Point3D &p);
__device__ void NACCUpdateDeformationGradient_trimmed(icy::Point3D &p);
__device__ Matrix3r KirchhoffStress_Wolper(const Matrix3r &F);

//__device__ Matrix2r polar_decomp_R(const Matrix2r &val);
//__device__ void svd(const real a[4], real u[4], real sigma[2], real v[4]);
//__device__ void svd2x2(const Matrix2r &mA, Matrix2r &mU, Matrix2r &mS, Matrix2r &mV);
//__device__ Matrix2r dev(Matrix2r A);


namespace icy { class Model3D; }

// GPU implementation of 3D MPM (this class is used by Model3D)
class GPU_Implementation4
{
public:
    icy::Model3D *model;
    int error_code;
    std::function<void()> transfer_completion_callback;

    void initialize();
    void test();
    void synchronize(); // call before terminating the main thread
    void cuda_update_constants();
    void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
    void cuda_reset_grid();
    void transfer_ponts_to_device();
    void transfer_ponts_to_host_finalize();
    void cuda_p2g();
    void cuda_g2p();
    void cuda_update_nodes(real indenter_x, real indenter_y);
    void cuda_reset_indenter_force_accumulator();

    void cuda_transfer_from_device();

    cudaEvent_t eventCycleStart, eventCycleStop;

    real *tmp_transfer_buffer = nullptr; // buffer in page-locked memory for transferring the data between device and host
    real *host_side_indenter_force_accumulator = nullptr;

private:

    cudaStream_t streamCompute;
    bool initialized = false;

    // callback is invoked when several cycles are complete and data is transferred to host
    static void CUDART_CB callback_transfer_from_device_completion(cudaStream_t stream, cudaError_t status, void *userData);
};

#endif // GPU_IMPLEMENTATION0_H
