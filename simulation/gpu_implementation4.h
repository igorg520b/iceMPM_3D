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

__forceinline__ __device__ void Wolper_Drucker_Prager(icy::Point3D &p);
__forceinline__ __device__ void CheckIfPointIsInsideFailureSurface(icy::Point3D &p);
__forceinline__ __device__ Matrix3r KirchhoffStress_Wolper(const Matrix3r &F);
__forceinline__ __device__ Matrix3r dev(Matrix3r A);

__forceinline__ __device__ void svd3x3(const Matrix3r &A, Matrix3r &U, Matrix3r &S, Matrix3r &V);
__global__ void kernel_hello();


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
