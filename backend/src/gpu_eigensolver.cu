/**
 * @file gpu_eigensolver.cu
 * @brief CUDA implementation of eigenvalue solver for Schrödinger equation.
 *
 * This file contains the CUDA implementation of the eigenvalue solver for the
 * Schrödinger equation using cuSOLVER and cuSPARSE libraries.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <thrust/complex.h>
#include <complex>
#include <vector>
#include <iostream>

// Constants
const double TOLERANCE = 1e-10;
const int MAX_ITERATIONS = 1000;

/**
 * @brief Converts a sparse matrix from CSR format to dense format.
 *
 * @param handle The cuSPARSE handle
 * @param rows The number of rows in the matrix
 * @param cols The number of columns in the matrix
 * @param nnz The number of non-zero elements in the matrix
 * @param csrRowPtr The CSR row pointer array
 * @param csrColInd The CSR column index array
 * @param csrVal The CSR value array
 * @param dense The dense matrix (output)
 * @return cudaError_t The CUDA error code
 */
cudaError_t csr2dense(
    cusparseHandle_t handle,
    int rows,
    int cols,
    int nnz,
    const int* csrRowPtr,
    const int* csrColInd,
    const thrust::complex<double>* csrVal,
    thrust::complex<double>* dense) {

    // Create matrix descriptors
    cusparseMatDescr_t descr = 0;
    cusparseStatus_t status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cudaErrorInvalidValue;
    }

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // Convert CSR to dense
    // Use cusparseXcsr2dense for CUDA 11.0+
#if CUDART_VERSION >= 11000
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Create sparse matrix A in CSR format
    status = cusparseCreateCsr(&matA, rows, cols, nnz,
                              (void*)csrRowPtr, (void*)csrColInd, (void*)csrVal,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cudaErrorInvalidValue;
    }

    // Create dense matrix B
    status = cusparseCreateDnMat(&matB, rows, cols, rows, dense, CUDA_C_64F, CUSPARSE_ORDER_COL);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroySpMat(matA);
        return cudaErrorInvalidValue;
    }

    // Get buffer size
    status = cusparseSparseToDense_bufferSize(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyDnMat(matB);
        cusparseDestroySpMat(matA);
        return cudaErrorInvalidValue;
    }

    // Allocate buffer
    cudaError_t cuda_status = cudaMalloc(&dBuffer, bufferSize);
    if (cuda_status != cudaSuccess) {
        cusparseDestroyDnMat(matB);
        cusparseDestroySpMat(matA);
        return cuda_status;
    }

    // Execute sparse to dense conversion
    status = cusparseSparseToDense(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, dBuffer);

    // Clean up
    cudaFree(dBuffer);
    cusparseDestroyDnMat(matB);
    cusparseDestroySpMat(matA);
#else
    // For older CUDA versions, use deprecated function
    status = cusparseZcsr2dense(
        handle,
        rows,
        cols,
        descr,
        reinterpret_cast<const cuDoubleComplex*>(csrVal),
        csrRowPtr,
        csrColInd,
        reinterpret_cast<cuDoubleComplex*>(dense),
        rows
    );
#endif

    cusparseDestroyMatDescr(descr);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

/**
 * @brief Solves the generalized eigenvalue problem Ax = λBx using cuSOLVER.
 *
 * @param n The size of the matrices
 * @param A The A matrix in dense format
 * @param B The B matrix in dense format
 * @param eigenvalues The eigenvalues (output)
 * @param eigenvectors The eigenvectors (output)
 * @param num_eigenvalues The number of eigenvalues to compute
 * @return cudaError_t The CUDA error code
 */
cudaError_t solve_generalized_eigenvalue_problem(
    int n,
    thrust::complex<double>* A,
    thrust::complex<double>* B,
    double* eigenvalues,
    thrust::complex<double>* eigenvectors,
    int num_eigenvalues) {

    // Create cuSOLVER handle
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t status = cusolverDnCreate(&cusolverH);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return cudaErrorInvalidValue;
    }

    // Workspace size query
    int lwork = 0;
    status = cusolverDnZhegvd_bufferSize(
        cusolverH,
        CUSOLVER_EIG_TYPE_1,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        n,
        reinterpret_cast<cuDoubleComplex*>(A),
        n,
        reinterpret_cast<cuDoubleComplex*>(B),
        n,
        eigenvalues,
        &lwork
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        cusolverDnDestroy(cusolverH);
        return cudaErrorInvalidValue;
    }

    // Allocate workspace
    cuDoubleComplex* work = NULL;
    cudaError_t cuda_status = cudaMalloc((void**)&work, sizeof(cuDoubleComplex) * lwork);
    if (cuda_status != cudaSuccess) {
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }

    // Allocate device memory for info
    int* devInfo = NULL;
    cuda_status = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(work);
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
        reinterpret_cast<cuDoubleComplex*>(A),
        n,
        reinterpret_cast<cuDoubleComplex*>(B),
        n,
        eigenvalues,
        work,
        lwork,
        devInfo
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(devInfo);
        cudaFree(work);
        cusolverDnDestroy(cusolverH);
        return cudaErrorInvalidValue;
    }

    // Check for errors
    int info = 0;
    cuda_status = cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess || info != 0) {
        cudaFree(devInfo);
        cudaFree(work);
        cusolverDnDestroy(cusolverH);
        return cudaErrorInvalidValue;
    }

    // Copy eigenvectors
    cuda_status = cudaMemcpy(eigenvectors, A, sizeof(thrust::complex<double>) * n * n, cudaMemcpyDeviceToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(devInfo);
        cudaFree(work);
        cusolverDnDestroy(cusolverH);
        return cuda_status;
    }

    // Clean up
    cudaFree(devInfo);
    cudaFree(work);
    cusolverDnDestroy(cusolverH);

    return cudaSuccess;
}

/**
 * @brief Solves the generalized eigenvalue problem Ax = λBx for sparse matrices.
 *
 * @param n The size of the matrices
 * @param nnz_A The number of non-zero elements in A
 * @param nnz_B The number of non-zero elements in B
 * @param csrRowPtrA The CSR row pointer array for A
 * @param csrColIndA The CSR column index array for A
 * @param csrValA The CSR value array for A
 * @param csrRowPtrB The CSR row pointer array for B
 * @param csrColIndB The CSR column index array for B
 * @param csrValB The CSR value array for B
 * @param eigenvalues The eigenvalues (output)
 * @param eigenvectors The eigenvectors (output)
 * @param num_eigenvalues The number of eigenvalues to compute
 * @return cudaError_t The CUDA error code
 */
extern "C" cudaError_t solve_sparse_generalized_eigenvalue_problem(
    int n,
    int nnz_A,
    int nnz_B,
    const int* csrRowPtrA,
    const int* csrColIndA,
    const std::complex<double>* csrValA,
    const int* csrRowPtrB,
    const int* csrColIndB,
    const std::complex<double>* csrValB,
    double* eigenvalues,
    std::complex<double>* eigenvectors,
    int num_eigenvalues) {

    // Create cuSPARSE handle
    cusparseHandle_t cusparseH = NULL;
    cusparseStatus_t status = cusparseCreate(&cusparseH);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return cudaErrorInvalidValue;
    }

    // Allocate device memory for CSR matrices
    int* d_csrRowPtrA = NULL;
    int* d_csrColIndA = NULL;
    thrust::complex<double>* d_csrValA = NULL;

    int* d_csrRowPtrB = NULL;
    int* d_csrColIndB = NULL;
    thrust::complex<double>* d_csrValB = NULL;

    cudaError_t cuda_status = cudaMalloc((void**)&d_csrRowPtrA, sizeof(int) * (n + 1));
    if (cuda_status != cudaSuccess) {
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&d_csrColIndA, sizeof(int) * nnz_A);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&d_csrValA, sizeof(thrust::complex<double>) * nnz_A);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&d_csrRowPtrB, sizeof(int) * (n + 1));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&d_csrColIndB, sizeof(int) * nnz_B);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&d_csrValB, sizeof(thrust::complex<double>) * nnz_B);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    // Copy CSR matrices to device
    cuda_status = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnz_A, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMemcpy(d_csrValA, csrValA, sizeof(thrust::complex<double>) * nnz_A, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMemcpy(d_csrRowPtrB, csrRowPtrB, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMemcpy(d_csrColIndB, csrColIndB, sizeof(int) * nnz_B, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMemcpy(d_csrValB, csrValB, sizeof(thrust::complex<double>) * nnz_B, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    // Allocate device memory for dense matrices
    thrust::complex<double>* d_A = NULL;
    thrust::complex<double>* d_B = NULL;

    cuda_status = cudaMalloc((void**)&d_A, sizeof(thrust::complex<double>) * n * n);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&d_B, sizeof(thrust::complex<double>) * n * n);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    // Convert CSR to dense
    cuda_status = csr2dense(cusparseH, n, n, nnz_A, d_csrRowPtrA, d_csrColIndA, d_csrValA, d_A);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = csr2dense(cusparseH, n, n, nnz_B, d_csrRowPtrB, d_csrColIndB, d_csrValB, d_B);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    // Allocate device memory for eigenvalues and eigenvectors
    double* d_eigenvalues = NULL;
    thrust::complex<double>* d_eigenvectors = NULL;

    cuda_status = cudaMalloc((void**)&d_eigenvalues, sizeof(double) * n);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&d_eigenvectors, sizeof(thrust::complex<double>) * n * n);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_eigenvalues);
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    // Solve generalized eigenvalue problem
    cuda_status = solve_generalized_eigenvalue_problem(n, d_A, d_B, d_eigenvalues, d_eigenvectors, num_eigenvalues);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_eigenvectors);
        cudaFree(d_eigenvalues);
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    // Copy results back to host
    cuda_status = cudaMemcpy(eigenvalues, d_eigenvalues, sizeof(double) * num_eigenvalues, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_eigenvectors);
        cudaFree(d_eigenvalues);
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    cuda_status = cudaMemcpy(eigenvectors, d_eigenvectors, sizeof(thrust::complex<double>) * n * num_eigenvalues, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_eigenvectors);
        cudaFree(d_eigenvalues);
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_csrValB);
        cudaFree(d_csrColIndB);
        cudaFree(d_csrRowPtrB);
        cudaFree(d_csrValA);
        cudaFree(d_csrColIndA);
        cudaFree(d_csrRowPtrA);
        cusparseDestroy(cusparseH);
        return cuda_status;
    }

    // Clean up
    cudaFree(d_eigenvectors);
    cudaFree(d_eigenvalues);
    cudaFree(d_B);
    cudaFree(d_A);
    cudaFree(d_csrValB);
    cudaFree(d_csrColIndB);
    cudaFree(d_csrRowPtrB);
    cudaFree(d_csrValA);
    cudaFree(d_csrColIndA);
    cudaFree(d_csrRowPtrA);
    cusparseDestroy(cusparseH);

    return cudaSuccess;
}
