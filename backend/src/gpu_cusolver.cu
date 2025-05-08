/**
 * @file gpu_cusolver.cu
 * @brief CUDA implementation of eigenvalue solver using cuSOLVER.
 *
 * This file contains the CUDA implementation of the eigenvalue solver using
 * cuSOLVER for the Schrödinger equation.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <complex>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "gpu_accelerator.h"

/**
 * @brief Solves the generalized eigenvalue problem Ax = λBx using cuSOLVER.
 *
 * This function solves the generalized eigenvalue problem Ax = λBx using
 * cuSOLVER. It computes the lowest `num_eigenvalues` eigenvalues and
 * corresponding eigenvectors.
 *
 * @param A The A matrix
 * @param B The B matrix
 * @param num_eigenvalues The number of eigenvalues to compute
 * @param eigenvalues The eigenvalues (output)
 * @param eigenvectors The eigenvectors (output)
 * @return cudaError_t The CUDA error code
 */
cudaError_t solve_generalized_eigenvalue_problem_cusolver(
    const Eigen::MatrixXcd& A,
    const Eigen::MatrixXcd& B,
    int num_eigenvalues,
    std::vector<double>& eigenvalues,
    std::vector<Eigen::VectorXd>& eigenvectors) {
    
    // Get matrix dimensions
    int n = A.rows();
    
    // Create cuSOLVER handle
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t status = cusolverDnCreate(&cusolverH);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return cudaErrorInvalidValue;
    }
    
    // Allocate device memory for matrices
    cuDoubleComplex* d_A = NULL;
    cuDoubleComplex* d_B = NULL;
    double* d_W = NULL;
    
    cudaError_t cuda_status = cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex) * n * n);
    if (cuda_status != cudaSuccess) {
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    cuda_status = cudaMalloc((void**)&d_B, sizeof(cuDoubleComplex) * n * n);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    cuda_status = cudaMalloc((void**)&d_W, sizeof(double) * n);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    // Copy matrices to device
    cuda_status = cudaMemcpy(d_A, A.data(), sizeof(cuDoubleComplex) * n * n, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    cuda_status = cudaMemcpy(d_B, B.data(), sizeof(cuDoubleComplex) * n * n, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    // Workspace size query
    int lwork = 0;
    status = cusolverDnZhegvd_bufferSize(
        cusolverH,
        CUSOLVER_EIG_TYPE_1,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,
        n,
        d_B,
        n,
        d_W,
        &lwork
    );
    
    if (status != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cudaErrorInvalidValue;
    }
    
    // Allocate workspace
    cuDoubleComplex* d_work = NULL;
    cuda_status = cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex) * lwork);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    // Allocate device memory for info
    int* d_info = NULL;
    cuda_status = cudaMalloc((void**)&d_info, sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_work);
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    // Solve generalized eigenvalue problem
    status = cusolverDnZhegvd(
        cusolverH,
        CUSOLVER_EIG_TYPE_1,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,
        n,
        d_B,
        n,
        d_W,
        d_work,
        lwork,
        d_info
    );
    
    if (status != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(d_info);
        cudaFree(d_work);
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cudaErrorInvalidValue;
    }
    
    // Check for errors
    int info = 0;
    cuda_status = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess || info != 0) {
        cudaFree(d_info);
        cudaFree(d_work);
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cudaErrorInvalidValue;
    }
    
    // Copy eigenvalues to host
    std::vector<double> all_eigenvalues(n);
    cuda_status = cudaMemcpy(all_eigenvalues.data(), d_W, sizeof(double) * n, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_info);
        cudaFree(d_work);
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    // Copy eigenvectors to host
    Eigen::MatrixXcd all_eigenvectors(n, n);
    cuda_status = cudaMemcpy(all_eigenvectors.data(), d_A, sizeof(cuDoubleComplex) * n * n, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_info);
        cudaFree(d_work);
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }
    
    // Clean up
    cudaFree(d_info);
    cudaFree(d_work);
    cudaFree(d_W);
    cudaFree(d_B);
    cudaFree(d_A);
    cusolverDnDestroy(cusolverH);
    
    // Sort eigenvalues and eigenvectors
    std::vector<std::pair<double, int>> sorted_eigenvalues(n);
    for (int i = 0; i < n; ++i) {
        sorted_eigenvalues[i] = std::make_pair(all_eigenvalues[i], i);
    }
    std::sort(sorted_eigenvalues.begin(), sorted_eigenvalues.end());
    
    // Extract the lowest num_eigenvalues eigenvalues and eigenvectors
    eigenvalues.resize(num_eigenvalues);
    eigenvectors.resize(num_eigenvalues);
    
    for (int i = 0; i < num_eigenvalues; ++i) {
        eigenvalues[i] = sorted_eigenvalues[i].first;
        
        // Extract eigenvector
        Eigen::VectorXcd complex_eigenvector = all_eigenvectors.col(sorted_eigenvalues[i].second);
        
        // Convert to real
        eigenvectors[i] = complex_eigenvector.cwiseAbs();
    }
    
    return cudaSuccess;
}

/**
 * @brief Implementation of the solve_eigen_cusolver method for the GPUAccelerator class.
 *
 * This function is the implementation of the solve_eigen_cusolver method for the
 * GPUAccelerator class. It solves the generalized eigenvalue problem Ax = λBx
 * using cuSOLVER.
 *
 * @param A The A matrix
 * @param B The B matrix
 * @param num_eigenvalues The number of eigenvalues to compute
 * @param eigenvalues The eigenvalues (output)
 * @param eigenvectors The eigenvectors (output)
 */
void GPUAccelerator::solve_eigen_cusolver(
    const Eigen::MatrixXcd& A,
    const Eigen::MatrixXcd& B,
    int num_eigenvalues,
    std::vector<std::complex<double>>& eigenvalues,
    std::vector<Eigen::VectorXd>& eigenvectors) {
    
    // Solve the generalized eigenvalue problem
    std::vector<double> real_eigenvalues;
    cudaError_t cuda_status = solve_generalized_eigenvalue_problem_cusolver(
        A, B, num_eigenvalues, real_eigenvalues, eigenvectors
    );
    
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to solve generalized eigenvalue problem using cuSOLVER: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Convert eigenvalues to complex
    eigenvalues.resize(num_eigenvalues);
    for (int i = 0; i < num_eigenvalues; ++i) {
        eigenvalues[i] = std::complex<double>(real_eigenvalues[i], 0.0);
    }
}
