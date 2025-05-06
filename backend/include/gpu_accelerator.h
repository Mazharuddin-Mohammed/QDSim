#pragma once
/**
 * @file gpu_accelerator.h
 * @brief Defines the GPUAccelerator class for GPU-accelerated computations.
 *
 * This file contains the declaration of the GPUAccelerator class, which provides
 * GPU-accelerated implementations of performance-critical operations in the
 * QDSim codebase, including matrix assembly and eigenvalue solving.
 *
 * The implementation uses CUDA for NVIDIA GPUs and falls back to CPU
 * implementations when CUDA is not available.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

// Conditional compilation for CUDA support
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include "gpu_memory_pool.h"
#endif

/**
 * @class GPUAccelerator
 * @brief Provides GPU-accelerated implementations of performance-critical operations.
 *
 * The GPUAccelerator class provides GPU-accelerated implementations of
 * performance-critical operations in the QDSim codebase, including matrix
 * assembly and eigenvalue solving. It uses CUDA for NVIDIA GPUs and falls
 * back to CPU implementations when CUDA is not available.
 */
class GPUAccelerator {
public:
    /**
     * @brief Constructs a new GPUAccelerator object.
     *
     * @param use_gpu Whether to use GPU acceleration if available
     * @param device_id The ID of the GPU device to use
     *
     * @throws std::runtime_error If GPU initialization fails
     */
    GPUAccelerator(bool use_gpu = true, int device_id = 0);

    /**
     * @brief Destructor for the GPUAccelerator object.
     *
     * Cleans up GPU resources.
     */
    ~GPUAccelerator();

    /**
     * @brief Checks if GPU acceleration is available and enabled.
     *
     * @return True if GPU acceleration is available and enabled, false otherwise
     */
    bool is_gpu_enabled() const;

    /**
     * @brief Gets information about the GPU device.
     *
     * @return A string containing information about the GPU device
     */
    std::string get_device_info() const;

    /**
     * @brief Assembles the Hamiltonian and mass matrices on the GPU.
     *
     * This method assembles the Hamiltonian and mass matrices using the finite
     * element method on the GPU. It is significantly faster than the CPU
     * implementation for large meshes.
     *
     * @param mesh The mesh to use for the simulation
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param order The order of the finite elements (1 for P1, 2 for P2, 3 for P3)
     * @param H The Hamiltonian matrix (output)
     * @param M The mass matrix (output)
     *
     * @throws std::runtime_error If the assembly fails
     */
    void assemble_matrices(const Mesh& mesh,
                          double (*m_star)(double, double),
                          double (*V)(double, double),
                          int order,
                          Eigen::SparseMatrix<std::complex<double>>& H,
                          Eigen::SparseMatrix<std::complex<double>>& M);

    /**
     * @brief Solves the generalized eigenvalue problem on the GPU.
     *
     * This method solves the generalized eigenvalue problem H·ψ = E·M·ψ using
     * GPU-accelerated algorithms. It computes the lowest eigenvalues and
     * corresponding eigenvectors.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The computed eigenvalues (output)
     * @param eigenvectors The computed eigenvectors (output)
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve_eigen(const Eigen::SparseMatrix<std::complex<double>>& H,
                    const Eigen::SparseMatrix<std::complex<double>>& M,
                    int num_eigenvalues,
                    std::vector<std::complex<double>>& eigenvalues,
                    std::vector<Eigen::VectorXd>& eigenvectors);

    /**
     * @brief Interpolates a field at arbitrary points using GPU acceleration.
     *
     * This method interpolates a field (like a potential or wavefunction) at
     * arbitrary points in the mesh using GPU acceleration.
     *
     * @param mesh The mesh containing the field
     * @param field The field values at mesh nodes
     * @param points The points at which to interpolate the field
     * @param values The interpolated field values (output)
     *
     * @throws std::runtime_error If the interpolation fails
     */
    void interpolate_field(const Mesh& mesh,
                          const Eigen::VectorXd& field,
                          const std::vector<Eigen::Vector2d>& points,
                          Eigen::VectorXd& values);

    /**
     * @brief Optimized assembly for higher-order elements on the GPU.
     *
     * This method provides specialized optimizations for higher-order elements (P2, P3)
     * which have more complex shape functions and require more quadrature points.
     *
     * @param mesh The mesh to use for the simulation
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param order The order of the finite elements (2 for P2, 3 for P3)
     * @param H The Hamiltonian matrix (output)
     * @param M The mass matrix (output)
     *
     * @throws std::runtime_error If the assembly fails
     */
    void assemble_higher_order_matrices(const Mesh& mesh,
                                      double (*m_star)(double, double),
                                      double (*V)(double, double),
                                      int order,
                                      Eigen::SparseMatrix<std::complex<double>>& H,
                                      Eigen::SparseMatrix<std::complex<double>>& M);

    /**
     * @brief Solves the generalized eigenvalue problem using cuSOLVER.
     *
     * @param H_dense The dense Hamiltonian matrix
     * @param M_dense The dense mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The computed eigenvalues (output)
     * @param eigenvectors The computed eigenvectors (output)
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve_eigen_cusolver(const Eigen::MatrixXcd& H_dense,
                            const Eigen::MatrixXcd& M_dense,
                            int num_eigenvalues,
                            std::vector<std::complex<double>>& eigenvalues,
                            std::vector<Eigen::VectorXd>& eigenvectors);

    /**
     * @brief Optimized eigensolver for large sparse matrices from higher-order elements.
     *
     * This method uses a specialized implementation for large sparse matrices that
     * typically arise from higher-order finite element discretizations. It employs
     * iterative methods that are more suitable for large problems than direct solvers.
     *
     * @param H The sparse Hamiltonian matrix
     * @param M The sparse mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The computed eigenvalues (output)
     * @param eigenvectors The computed eigenvectors (output)
     * @param tolerance The convergence tolerance
     * @param max_iterations The maximum number of iterations
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve_eigen_sparse(const Eigen::SparseMatrix<std::complex<double>>& H,
                          const Eigen::SparseMatrix<std::complex<double>>& M,
                          int num_eigenvalues,
                          std::vector<std::complex<double>>& eigenvalues,
                          std::vector<Eigen::VectorXd>& eigenvectors,
                          double tolerance = 1e-10,
                          int max_iterations = 1000);

    /**
     * @brief Batched eigensolver for multiple small eigenproblems.
     *
     * This method solves multiple small eigenproblems in parallel using batched
     * operations on the GPU. This is useful for parameter sweeps or when solving
     * the same problem with different parameters.
     *
     * @param H_batch Vector of Hamiltonian matrices
     * @param M_batch Vector of mass matrices
     * @param num_eigenvalues The number of eigenvalues to compute for each problem
     * @param eigenvalues_batch Vector of eigenvalue vectors (output)
     * @param eigenvectors_batch Vector of eigenvector vectors (output)
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve_eigen_batched(const std::vector<Eigen::MatrixXcd>& H_batch,
                           const std::vector<Eigen::MatrixXcd>& M_batch,
                           int num_eigenvalues,
                           std::vector<std::vector<std::complex<double>>>& eigenvalues_batch,
                           std::vector<std::vector<Eigen::VectorXd>>& eigenvectors_batch);

private:
    bool use_gpu_;                 ///< Whether to use GPU acceleration
    int device_id_;                ///< The ID of the GPU device to use
    bool gpu_initialized_;         ///< Whether the GPU has been initialized
    std::string device_info_;      ///< Information about the GPU device

#ifdef USE_CUDA
    // CUDA handles
    cudaDeviceProp device_prop_;   ///< Properties of the GPU device
    cusolverDnHandle_t cusolver_handle_; ///< cuSOLVER handle
    cublasHandle_t cublas_handle_; ///< cuBLAS handle
    cusparseHandle_t cusparse_handle_; ///< cuSPARSE handle

    // Persistent GPU memory buffers
    struct GPUBuffer {
        void* ptr;
        size_t size;
        std::string tag;
    };

    // Cache for mesh data to avoid repeated transfers
    struct MeshCache {
        double* d_nodes;
        int* d_elements;
        size_t num_nodes;
        size_t num_elements;
        size_t nodes_per_elem;
        bool valid;
    };

    std::unordered_map<std::string, GPUBuffer> gpu_buffers_; ///< Persistent GPU memory buffers
    MeshCache mesh_cache_; ///< Cache for mesh data

    // Stream for asynchronous operations
    cudaStream_t stream_; ///< CUDA stream for asynchronous operations

    /**
     * @brief Gets a GPU buffer of the specified size.
     *
     * This method gets a GPU buffer of the specified size. If a buffer with the
     * specified tag already exists and is large enough, it is returned. Otherwise,
     * a new buffer is allocated.
     *
     * @param size The size of the buffer in bytes
     * @param tag A tag to identify the buffer
     * @return A pointer to the buffer
     */
    void* get_gpu_buffer(size_t size, const std::string& tag);

    /**
     * @brief Releases a GPU buffer.
     *
     * This method releases a GPU buffer. The buffer is not actually freed, but is
     * marked as available for reuse.
     *
     * @param tag The tag of the buffer to release
     */
    void release_gpu_buffer(const std::string& tag);

    /**
     * @brief Caches mesh data on the GPU.
     *
     * This method caches mesh data on the GPU to avoid repeated transfers.
     *
     * @param mesh The mesh to cache
     * @param order The order of the finite elements
     * @return True if the mesh was cached successfully, false otherwise
     */
    bool cache_mesh(const Mesh& mesh, int order);

    /**
     * @brief Initializes the GPU device and CUDA libraries.
     *
     * @throws std::runtime_error If GPU initialization fails
     */
    void initialize_gpu();

    /**
     * @brief Cleans up GPU resources.
     */
    void cleanup_gpu();

    /**
     * @brief Assembles an element matrix on the GPU.
     *
     * @param element_idx The index of the element
     * @param nodes The mesh nodes
     * @param elements The mesh elements
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param order The order of the finite elements
     * @param H_e The element Hamiltonian matrix (output)
     * @param M_e The element mass matrix (output)
     */
    void assemble_element_matrix_gpu(int element_idx,
                                   const double* nodes,
                                   const int* elements,
                                   double (*m_star)(double, double),
                                   double (*V)(double, double),
                                   int order,
                                   std::complex<double>* H_e,
                                   std::complex<double>* M_e);

    /**
     * @brief Solve the generalized eigenvalue problem on GPU.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The eigenvalues (output)
     * @param eigenvectors The eigenvectors (output)
     */
    void solve_eigenvalue_problem_gpu(
        const Eigen::SparseMatrix<std::complex<double>>& H,
        const Eigen::SparseMatrix<std::complex<double>>& M,
        int num_eigenvalues,
        std::vector<double>& eigenvalues,
        std::vector<Eigen::VectorXcd>& eigenvectors);

    /**
     * @brief Convert Eigen sparse matrix to CSR format.
     *
     * @param mat The Eigen sparse matrix
     * @param rowPtr The CSR row pointer array (output)
     * @param colInd The CSR column index array (output)
     * @param values The CSR value array (output)
     */
    void eigen_sparse_to_csr(
        const Eigen::SparseMatrix<std::complex<double>>& mat,
        std::vector<int>& rowPtr,
        std::vector<int>& colInd,
        std::vector<std::complex<double>>& values);


#endif
};
