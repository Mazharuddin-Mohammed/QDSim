#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cooperative_groups.h>

namespace QDSim {
namespace GPU {

// Constants for quantum simulations
constexpr double HBAR = 1.054571817e-34;  // Reduced Planck constant (J⋅s)
constexpr double ELECTRON_MASS = 9.1093837015e-31;  // Electron mass (kg)
constexpr double ELECTRON_CHARGE = 1.602176634e-19;  // Elementary charge (C)
constexpr double HBAR_SQ_OVER_2M = (HBAR * HBAR) / (2.0 * ELECTRON_MASS);

// Fused kernel: Hamiltonian assembly + eigenvalue preparation
__global__ void fusedHamiltonianAssembly(
    const double* __restrict__ nodes,
    const int* __restrict__ elements,
    const double* __restrict__ potential,
    const double* __restrict__ effective_mass,
    cuDoubleComplex* __restrict__ hamiltonian_matrix,
    cuDoubleComplex* __restrict__ mass_matrix,
    int num_elements,
    int nodes_per_element,
    int total_nodes);

// Optimized matrix-vector multiplication for sparse matrices
__global__ void fusedSpMV(
    const cuDoubleComplex* __restrict__ matrix_values,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const cuDoubleComplex* __restrict__ input_vector,
    cuDoubleComplex* __restrict__ output_vector,
    int num_rows);

// Fused kernel for electron density computation
__global__ void fusedElectronDensity(
    const cuDoubleComplex* __restrict__ eigenvectors,
    const double* __restrict__ eigenvalues,
    const double* __restrict__ fermi_weights,
    double* __restrict__ electron_density,
    int num_nodes,
    int num_states,
    double temperature,
    double fermi_level);

// Memory coalescing optimization for element assembly
__global__ void coalescedElementAssembly(
    const double* __restrict__ nodes_x,
    const double* __restrict__ nodes_y,
    const int* __restrict__ element_connectivity,
    const double* __restrict__ material_properties,
    cuDoubleComplex* __restrict__ local_matrices,
    int num_elements,
    int nodes_per_element);

// Asynchronous Poisson solver kernel
__global__ void fusedPoissonSolver(
    const double* __restrict__ charge_density,
    const double* __restrict__ permittivity,
    const double* __restrict__ boundary_values,
    const int* __restrict__ boundary_flags,
    double* __restrict__ potential,
    double* __restrict__ electric_field_x,
    double* __restrict__ electric_field_y,
    int nx, int ny,
    double dx, double dy);

// Device functions for shape function computations
__device__ inline void computeShapeFunctionDerivatives(
    const double* element_nodes,
    int i, int j,
    double& dNi_dx, double& dNi_dy,
    double& dNj_dx, double& dNj_dy);

__device__ inline double computeShapeFunctionProduct(
    const double* element_nodes,
    int i, int j);

__device__ inline double computeJacobian(
    const double* element_nodes,
    double xi, double eta);

// Optimized reduction operations
template<typename T>
__device__ inline T warpReduce(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ inline T blockReduce(T val) {
    static __shared__ T shared[32]; // One per warp
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduce(val);
    
    if (lane == 0) shared[wid] = val;
    
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    
    if (wid == 0) val = warpReduce(val);
    
    return val;
}

// Memory coalescing utilities
struct CoalescedAccess {
    static constexpr int WARP_SIZE = 32;
    static constexpr int CACHE_LINE_SIZE = 128; // bytes
    
    template<typename T>
    __device__ static void coalescedLoad(T* shared_mem, const T* global_mem, 
                                        int num_elements, int thread_id) {
        for (int i = thread_id; i < num_elements; i += blockDim.x) {
            shared_mem[i] = global_mem[i];
        }
        __syncthreads();
    }
    
    template<typename T>
    __device__ static void coalescedStore(T* global_mem, const T* shared_mem,
                                         int num_elements, int thread_id) {
        __syncthreads();
        for (int i = thread_id; i < num_elements; i += blockDim.x) {
            global_mem[i] = shared_mem[i];
        }
    }
};

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Launch configuration utilities
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    cudaStream_t stream;
    
    static LaunchConfig optimal1D(int num_elements, int max_threads_per_block = 256) {
        LaunchConfig config;
        config.block.x = std::min(num_elements, max_threads_per_block);
        config.grid.x = (num_elements + config.block.x - 1) / config.block.x;
        config.shared_mem = 0;
        config.stream = 0;
        return config;
    }
    
    static LaunchConfig optimal2D(int nx, int ny, int max_threads_per_block = 256) {
        LaunchConfig config;
        int block_size = static_cast<int>(sqrt(max_threads_per_block));
        config.block.x = std::min(nx, block_size);
        config.block.y = std::min(ny, block_size);
        config.grid.x = (nx + config.block.x - 1) / config.block.x;
        config.grid.y = (ny + config.block.y - 1) / config.block.y;
        config.shared_mem = 0;
        config.stream = 0;
        return config;
    }
};

} // namespace GPU
} // namespace QDSim
