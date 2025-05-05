/**
 * @file advanced_gpu_accelerator.h
 * @brief Advanced GPU acceleration for QDSim.
 *
 * This file contains the declaration of the AdvancedGPUAccelerator class,
 * which provides GPU acceleration for QDSim using CUDA.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <functional>
#include <vector>
#include <memory>

#include "mesh.h"

/**
 * @class AdvancedGPUAccelerator
 * @brief Advanced GPU acceleration for QDSim.
 *
 * This class provides GPU acceleration for QDSim using CUDA. It includes
 * optimized implementations for matrix assembly, eigenvalue solving, and
 * interpolation for higher-order elements.
 */
class AdvancedGPUAccelerator {
public:
    /**
     * @brief Constructor.
     *
     * @param use_gpu Whether to use GPU acceleration
     */
    AdvancedGPUAccelerator(bool use_gpu = true);

    /**
     * @brief Destructor.
     */
    ~AdvancedGPUAccelerator();

    /**
     * @brief Check if GPU acceleration is enabled.
     *
     * @return True if GPU acceleration is enabled, false otherwise
     */
    bool is_gpu_enabled() const;

    /**
     * @brief Initialize the GPU accelerator.
     *
     * @return True if initialization was successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Assemble matrices for the Schrödinger equation.
     *
     * @param mesh The mesh
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param order The order of the finite elements
     * @param H The Hamiltonian matrix (output)
     * @param M The mass matrix (output)
     */
    void assemble_matrices(
        const Mesh& mesh,
        std::function<double(double, double)> m_star,
        std::function<double(double, double)> V,
        int order,
        Eigen::SparseMatrix<std::complex<double>>& H,
        Eigen::SparseMatrix<std::complex<double>>& M);

    /**
     * @brief Assemble matrices for higher-order elements.
     *
     * @param mesh The mesh
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param order The order of the finite elements
     * @param H The Hamiltonian matrix (output)
     * @param M The mass matrix (output)
     */
    void assemble_higher_order_matrices(
        const Mesh& mesh,
        std::function<double(double, double)> m_star,
        std::function<double(double, double)> V,
        int order,
        Eigen::SparseMatrix<std::complex<double>>& H,
        Eigen::SparseMatrix<std::complex<double>>& M);

    /**
     * @brief Solve the generalized eigenvalue problem.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The eigenvalues (output)
     * @param eigenvectors The eigenvectors (output)
     */
    void solve_eigen(
        const Eigen::SparseMatrix<std::complex<double>>& H,
        const Eigen::SparseMatrix<std::complex<double>>& M,
        int num_eigenvalues,
        std::vector<double>& eigenvalues,
        std::vector<Eigen::VectorXcd>& eigenvectors);

    /**
     * @brief Interpolate a field at given points.
     *
     * @param mesh The mesh
     * @param field The field values at mesh nodes
     * @param points The points at which to interpolate
     * @param values The interpolated values (output)
     */
    void interpolate_field(
        const Mesh& mesh,
        const Eigen::VectorXd& field,
        const std::vector<std::pair<double, double>>& points,
        Eigen::VectorXd& values);

private:
    bool use_gpu_;
    bool gpu_initialized_;
    
    // CUDA device properties
    int device_count_;
    int current_device_;
    int compute_capability_major_;
    int compute_capability_minor_;
    size_t total_memory_;
    size_t free_memory_;
    
    // CUDA handles
    void* cusolver_handle_;
    void* cusparse_handle_;
    void* cublas_handle_;
    
    // Implementation details
    void assemble_element_matrix_gpu(
        int element_idx,
        const double* nodes,
        const int* elements,
        double (*m_star)(double, double),
        double (*V)(double, double),
        int order,
        std::complex<double>* H_e,
        std::complex<double>* M_e);
        
    void assemble_element_matrices_batched_gpu(
        int batch_start,
        int batch_size,
        const double* nodes,
        const int* elements,
        double (*m_star)(double, double),
        double (*V)(double, double),
        int order,
        std::complex<double>* H_e,
        std::complex<double>* M_e);
        
    void solve_eigen_cusparse(
        const Eigen::SparseMatrix<std::complex<double>>& H,
        const Eigen::SparseMatrix<std::complex<double>>& M,
        int num_eigenvalues,
        std::vector<double>& eigenvalues,
        std::vector<Eigen::VectorXcd>& eigenvectors);
        
    void eigen_sparse_to_csr(
        const Eigen::SparseMatrix<std::complex<double>>& mat,
        std::vector<int>& rowPtr,
        std::vector<int>& colInd,
        std::vector<std::complex<double>>& values);
};
