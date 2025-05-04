/**
 * @file gpu_accelerator.cpp
 * @brief Implementation of the GPUAccelerator class for GPU-accelerated computations.
 *
 * This file contains the implementation of the GPUAccelerator class, which provides
 * GPU-accelerated implementations of performance-critical operations in the
 * QDSim codebase, including matrix assembly and eigenvalue solving.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "gpu_accelerator.h"
#include <iostream>
#include <stdexcept>
#include <sstream>

// Constructor
GPUAccelerator::GPUAccelerator(bool use_gpu, int device_id)
    : use_gpu_(use_gpu), device_id_(device_id), gpu_initialized_(false) {
    
    // Initialize GPU if requested
    if (use_gpu_) {
#ifdef USE_CUDA
        try {
            initialize_gpu();
            gpu_initialized_ = true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to initialize GPU: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation." << std::endl;
            use_gpu_ = false;
        }
#else
        std::cerr << "Warning: CUDA support not compiled. Falling back to CPU implementation." << std::endl;
        use_gpu_ = false;
#endif
    }
}

// Destructor
GPUAccelerator::~GPUAccelerator() {
#ifdef USE_CUDA
    if (gpu_initialized_) {
        cleanup_gpu();
    }
#endif
}

// Check if GPU acceleration is enabled
bool GPUAccelerator::is_gpu_enabled() const {
    return use_gpu_ && gpu_initialized_;
}

// Get GPU device information
std::string GPUAccelerator::get_device_info() const {
    return device_info_;
}

#ifdef USE_CUDA
// Initialize GPU
void GPUAccelerator::initialize_gpu() {
    // Get number of devices
    int device_count;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device count: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Check if requested device is available
    if (device_id_ >= device_count) {
        throw std::runtime_error("Requested GPU device " + std::to_string(device_id_) + 
                                " not available. Only " + std::to_string(device_count) + 
                                " devices found.");
    }
    
    // Set device
    cuda_status = cudaSetDevice(device_id_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Get device properties
    cuda_status = cudaGetDeviceProperties(&device_prop_, device_id_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device properties: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Initialize cuSOLVER
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle_);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuSOLVER handle");
    }
    
    // Initialize cuBLAS
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cusolverDnDestroy(cusolver_handle_);
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
    
    // Initialize cuSPARSE
    cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle_);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cublasDestroy(cublas_handle_);
        cusolverDnDestroy(cusolver_handle_);
        throw std::runtime_error("Failed to create cuSPARSE handle");
    }
    
    // Build device info string
    std::stringstream ss;
    ss << "GPU Device: " << device_prop_.name << "\n";
    ss << "Compute Capability: " << device_prop_.major << "." << device_prop_.minor << "\n";
    ss << "Total Global Memory: " << device_prop_.totalGlobalMem / (1024 * 1024) << " MB\n";
    ss << "Multiprocessors: " << device_prop_.multiProcessorCount << "\n";
    ss << "Max Threads per Block: " << device_prop_.maxThreadsPerBlock << "\n";
    ss << "Max Threads per Multiprocessor: " << device_prop_.maxThreadsPerMultiProcessor << "\n";
    ss << "Warp Size: " << device_prop_.warpSize;
    device_info_ = ss.str();
}

// Cleanup GPU resources
void GPUAccelerator::cleanup_gpu() {
    // Destroy cuSPARSE handle
    cusparseDestroy(cusparse_handle_);
    
    // Destroy cuBLAS handle
    cublasDestroy(cublas_handle_);
    
    // Destroy cuSOLVER handle
    cusolverDnDestroy(cusolver_handle_);
    
    // Reset device
    cudaDeviceReset();
}
#endif

// Assemble matrices on GPU
void GPUAccelerator::assemble_matrices(const Mesh& mesh,
                                     double (*m_star)(double, double),
                                     double (*V)(double, double),
                                     int order,
                                     Eigen::SparseMatrix<std::complex<double>>& H,
                                     Eigen::SparseMatrix<std::complex<double>>& M) {
    // Check if GPU acceleration is enabled
    if (!is_gpu_enabled()) {
        throw std::runtime_error("GPU acceleration not available for matrix assembly");
    }
    
#ifdef USE_CUDA
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    int num_nodes = nodes.size();
    int num_elements = elements.size();
    
    // Resize matrices
    H.resize(num_nodes, num_nodes);
    M.resize(num_nodes, num_nodes);
    
    // Create triplet lists for sparse matrix assembly
    std::vector<Eigen::Triplet<std::complex<double>>> H_triplets;
    std::vector<Eigen::Triplet<std::complex<double>>> M_triplets;
    
    // Reserve space for triplets (estimate: 9 entries per element for linear elements)
    int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;
    int entries_per_elem = nodes_per_elem * nodes_per_elem;
    H_triplets.reserve(num_elements * entries_per_elem);
    M_triplets.reserve(num_elements * entries_per_elem);
    
    // Copy mesh data to GPU
    double* d_nodes = nullptr;
    int* d_elements = nullptr;
    
    // Allocate memory on GPU
    cudaError_t cuda_status = cudaMalloc(&d_nodes, 2 * num_nodes * sizeof(double));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for nodes: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    cuda_status = cudaMalloc(&d_elements, 3 * num_elements * sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to allocate GPU memory for elements: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Copy data to GPU
    // Flatten nodes array
    std::vector<double> nodes_flat(2 * num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes_flat[2 * i] = nodes[i][0];
        nodes_flat[2 * i + 1] = nodes[i][1];
    }
    
    cuda_status = cudaMemcpy(d_nodes, nodes_flat.data(), 2 * num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to copy nodes to GPU: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Flatten elements array
    std::vector<int> elements_flat(3 * num_elements);
    for (int i = 0; i < num_elements; ++i) {
        elements_flat[3 * i] = elements[i][0];
        elements_flat[3 * i + 1] = elements[i][1];
        elements_flat[3 * i + 2] = elements[i][2];
    }
    
    cuda_status = cudaMemcpy(d_elements, elements_flat.data(), 3 * num_elements * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to copy elements to GPU: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Allocate memory for element matrices
    std::complex<double>* H_e = new std::complex<double>[nodes_per_elem * nodes_per_elem];
    std::complex<double>* M_e = new std::complex<double>[nodes_per_elem * nodes_per_elem];
    
    // Process elements
    for (int e = 0; e < num_elements; ++e) {
        // Assemble element matrices on GPU
        assemble_element_matrix_gpu(e, d_nodes, d_elements, m_star, V, order, H_e, M_e);
        
        // Add to triplet lists
        for (int i = 0; i < nodes_per_elem; ++i) {
            for (int j = 0; j < nodes_per_elem; ++j) {
                int global_i = elements[e][i];
                int global_j = elements[e][j];
                H_triplets.emplace_back(global_i, global_j, H_e[i * nodes_per_elem + j]);
                M_triplets.emplace_back(global_i, global_j, M_e[i * nodes_per_elem + j]);
            }
        }
    }
    
    // Clean up
    delete[] H_e;
    delete[] M_e;
    cudaFree(d_elements);
    cudaFree(d_nodes);
    
    // Set matrices from triplets
    H.setFromTriplets(H_triplets.begin(), H_triplets.end());
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());
    
    // Compress matrices
    H.makeCompressed();
    M.makeCompressed();
#endif
}

// Solve eigenvalue problem on GPU
void GPUAccelerator::solve_eigen(const Eigen::SparseMatrix<std::complex<double>>& H,
                               const Eigen::SparseMatrix<std::complex<double>>& M,
                               int num_eigenvalues,
                               std::vector<std::complex<double>>& eigenvalues,
                               std::vector<Eigen::VectorXd>& eigenvectors) {
    // Check if GPU acceleration is enabled
    if (!is_gpu_enabled()) {
        throw std::runtime_error("GPU acceleration not available for eigenvalue solving");
    }
    
#ifdef USE_CUDA
    // Convert sparse matrices to dense
    Eigen::MatrixXcd H_dense = Eigen::MatrixXcd(H);
    Eigen::MatrixXcd M_dense = Eigen::MatrixXcd(M);
    
    // Make sure matrices are Hermitian
    H_dense = (H_dense + H_dense.adjoint()) / 2.0;
    M_dense = (M_dense + M_dense.adjoint()) / 2.0;
    
    // Solve using cuSOLVER
    solve_eigen_cusolver(H_dense, M_dense, num_eigenvalues, eigenvalues, eigenvectors);
#endif
}

// Interpolate field on GPU
void GPUAccelerator::interpolate_field(const Mesh& mesh,
                                     const Eigen::VectorXd& field,
                                     const std::vector<Eigen::Vector2d>& points,
                                     Eigen::VectorXd& values) {
    // Check if GPU acceleration is enabled
    if (!is_gpu_enabled()) {
        throw std::runtime_error("GPU acceleration not available for field interpolation");
    }
    
#ifdef USE_CUDA
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    int num_nodes = nodes.size();
    int num_elements = elements.size();
    int num_points = points.size();
    
    // Resize output vector
    values.resize(num_points);
    
    // Copy mesh data to GPU
    double* d_nodes = nullptr;
    int* d_elements = nullptr;
    double* d_field = nullptr;
    double* d_points = nullptr;
    double* d_values = nullptr;
    
    // Allocate memory on GPU
    cudaError_t cuda_status = cudaMalloc(&d_nodes, 2 * num_nodes * sizeof(double));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for nodes: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    cuda_status = cudaMalloc(&d_elements, 3 * num_elements * sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to allocate GPU memory for elements: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    cuda_status = cudaMalloc(&d_field, num_nodes * sizeof(double));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to allocate GPU memory for field: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    cuda_status = cudaMalloc(&d_points, 2 * num_points * sizeof(double));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to allocate GPU memory for points: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    cuda_status = cudaMalloc(&d_values, num_points * sizeof(double));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to allocate GPU memory for values: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Copy data to GPU
    // Flatten nodes array
    std::vector<double> nodes_flat(2 * num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes_flat[2 * i] = nodes[i][0];
        nodes_flat[2 * i + 1] = nodes[i][1];
    }
    
    cuda_status = cudaMemcpy(d_nodes, nodes_flat.data(), 2 * num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to copy nodes to GPU: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Flatten elements array
    std::vector<int> elements_flat(3 * num_elements);
    for (int i = 0; i < num_elements; ++i) {
        elements_flat[3 * i] = elements[i][0];
        elements_flat[3 * i + 1] = elements[i][1];
        elements_flat[3 * i + 2] = elements[i][2];
    }
    
    cuda_status = cudaMemcpy(d_elements, elements_flat.data(), 3 * num_elements * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to copy elements to GPU: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Copy field data
    cuda_status = cudaMemcpy(d_field, field.data(), num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to copy field to GPU: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Flatten points array
    std::vector<double> points_flat(2 * num_points);
    for (int i = 0; i < num_points; ++i) {
        points_flat[2 * i] = points[i][0];
        points_flat[2 * i + 1] = points[i][1];
    }
    
    cuda_status = cudaMemcpy(d_points, points_flat.data(), 2 * num_points * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to copy points to GPU: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // TODO: Implement GPU kernel for field interpolation
    
    // Copy results back to CPU
    cuda_status = cudaMemcpy(values.data(), d_values, num_points * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to copy values from GPU: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    
    // Clean up
    cudaFree(d_values);
    cudaFree(d_points);
    cudaFree(d_field);
    cudaFree(d_elements);
    cudaFree(d_nodes);
#endif
}
