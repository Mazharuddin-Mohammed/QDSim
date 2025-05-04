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

// Conditional compilation for CUDA support
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse.h>
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
#endif
};
