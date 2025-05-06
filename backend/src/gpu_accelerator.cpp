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

#ifdef USE_CUDA
#include <thrust/complex.h>

// Forward declarations of CUDA functions
extern "C" {
    void assemble_element_matrix_cuda(
        int element_idx,
        const double* nodes,
        const int* elements,
        std::complex<double>* H_e,
        std::complex<double>* M_e,
        const double* m_star_values,
        const double* V_values,
        int order);

    void assemble_element_matrices_batched_cuda(
        int batch_start,
        int batch_size,
        const double* nodes,
        const int* elements,
        std::complex<double>* H_e,
        std::complex<double>* M_e,
        const double* m_star_values,
        const double* V_values,
        int order);

    void interpolate_field_cuda(
        const double* nodes,
        const int* elements,
        const double* field,
        const double* points,
        double* values,
        int num_nodes,
        int num_elements,
        int num_points);

    cudaError_t solve_sparse_generalized_eigenvalue_problem(
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
        int num_eigenvalues);
}
#endif

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

    // Create CUDA stream for asynchronous operations
    cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        cusparseDestroy(cusparse_handle_);
        cublasDestroy(cublas_handle_);
        cusolverDnDestroy(cusolver_handle_);
        throw std::runtime_error("Failed to create CUDA stream: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Set stream for cuSOLVER
    cusolver_status = cusolverDnSetStream(cusolver_handle_, stream_);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        cudaStreamDestroy(stream_);
        cusparseDestroy(cusparse_handle_);
        cublasDestroy(cublas_handle_);
        cusolverDnDestroy(cusolver_handle_);
        throw std::runtime_error("Failed to set stream for cuSOLVER");
    }

    // Set stream for cuBLAS
    cublas_status = cublasSetStream(cublas_handle_, stream_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream_);
        cusparseDestroy(cusparse_handle_);
        cublasDestroy(cublas_handle_);
        cusolverDnDestroy(cusolver_handle_);
        throw std::runtime_error("Failed to set stream for cuBLAS");
    }

    // Set stream for cuSPARSE
    cusparse_status = cusparseSetStream(cusparse_handle_, stream_);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cudaStreamDestroy(stream_);
        cusparseDestroy(cusparse_handle_);
        cublasDestroy(cublas_handle_);
        cusolverDnDestroy(cusolver_handle_);
        throw std::runtime_error("Failed to set stream for cuSPARSE");
    }

    // Initialize mesh cache
    mesh_cache_.d_nodes = nullptr;
    mesh_cache_.d_elements = nullptr;
    mesh_cache_.num_nodes = 0;
    mesh_cache_.num_elements = 0;
    mesh_cache_.nodes_per_elem = 0;
    mesh_cache_.valid = false;

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
    // Free all GPU buffers
    for (auto& buffer : gpu_buffers_) {
        if (buffer.second.ptr != nullptr) {
            GPUMemoryPool::getInstance().release(buffer.second.ptr, buffer.second.size);
            buffer.second.ptr = nullptr;
        }
    }
    gpu_buffers_.clear();

    // Free mesh cache
    if (mesh_cache_.valid) {
        if (mesh_cache_.d_nodes != nullptr) {
            GPUMemoryPool::getInstance().release(mesh_cache_.d_nodes, 2 * mesh_cache_.num_nodes * sizeof(double));
            mesh_cache_.d_nodes = nullptr;
        }
        if (mesh_cache_.d_elements != nullptr) {
            GPUMemoryPool::getInstance().release(mesh_cache_.d_elements, mesh_cache_.nodes_per_elem * mesh_cache_.num_elements * sizeof(int));
            mesh_cache_.d_elements = nullptr;
        }
        mesh_cache_.valid = false;
    }

    // Destroy CUDA stream
    cudaStreamDestroy(stream_);

    // Destroy cuSPARSE handle
    cusparseDestroy(cusparse_handle_);

    // Destroy cuBLAS handle
    cublasDestroy(cublas_handle_);

    // Destroy cuSOLVER handle
    cusolverDnDestroy(cusolver_handle_);

    // Free all memory in the pool
    GPUMemoryPool::getInstance().freeAll();

    // Reset device
    cudaDeviceReset();
}

// Get a GPU buffer of the specified size
void* GPUAccelerator::get_gpu_buffer(size_t size, const std::string& tag) {
    // Check if buffer already exists
    auto it = gpu_buffers_.find(tag);
    if (it != gpu_buffers_.end()) {
        // Check if existing buffer is large enough
        if (it->second.size >= size) {
            return it->second.ptr;
        }

        // Release existing buffer
        GPUMemoryPool::getInstance().release(it->second.ptr, it->second.size);
    }

    // Allocate new buffer
    void* ptr = GPUMemoryPool::getInstance().allocate(size, tag);

    // Store buffer
    gpu_buffers_[tag] = {ptr, size, tag};

    return ptr;
}

// Release a GPU buffer
void GPUAccelerator::release_gpu_buffer(const std::string& tag) {
    // Check if buffer exists
    auto it = gpu_buffers_.find(tag);
    if (it != gpu_buffers_.end()) {
        // Release buffer
        GPUMemoryPool::getInstance().release(it->second.ptr, it->second.size);

        // Remove from map
        gpu_buffers_.erase(it);
    }
}

// Cache mesh data on the GPU
bool GPUAccelerator::cache_mesh(const Mesh& mesh, int order) {
    const auto& nodes = mesh.getNodes();
    int num_nodes = nodes.size();
    int num_elements = 0;
    int nodes_per_elem = 0;

    // Get element data based on order
    if (order == 1) {
        const auto& elements = mesh.getElements();
        num_elements = elements.size();
        nodes_per_elem = 3;
    } else if (order == 2) {
        const auto& elements = mesh.getQuadraticElements();
        num_elements = elements.size();
        nodes_per_elem = 6;
    } else if (order == 3) {
        const auto& elements = mesh.getCubicElements();
        num_elements = elements.size();
        nodes_per_elem = 10;
    } else {
        return false;
    }

    // Check if mesh is already cached with the same parameters
    if (mesh_cache_.valid &&
        mesh_cache_.num_nodes == num_nodes &&
        mesh_cache_.num_elements == num_elements &&
        mesh_cache_.nodes_per_elem == nodes_per_elem) {
        return true;
    }

    // Free existing cache if any
    if (mesh_cache_.valid) {
        if (mesh_cache_.d_nodes != nullptr) {
            GPUMemoryPool::getInstance().release(mesh_cache_.d_nodes, 2 * mesh_cache_.num_nodes * sizeof(double));
            mesh_cache_.d_nodes = nullptr;
        }
        if (mesh_cache_.d_elements != nullptr) {
            GPUMemoryPool::getInstance().release(mesh_cache_.d_elements, mesh_cache_.nodes_per_elem * mesh_cache_.num_elements * sizeof(int));
            mesh_cache_.d_elements = nullptr;
        }
        mesh_cache_.valid = false;
    }

    try {
        // Allocate memory for nodes
        mesh_cache_.d_nodes = static_cast<double*>(
            GPUMemoryPool::getInstance().allocate(2 * num_nodes * sizeof(double), "mesh_nodes"));

        // Allocate memory for elements
        mesh_cache_.d_elements = static_cast<int*>(
            GPUMemoryPool::getInstance().allocate(nodes_per_elem * num_elements * sizeof(int), "mesh_elements"));

        // Flatten nodes array
        std::vector<double> nodes_flat(2 * num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            nodes_flat[2 * i] = nodes[i][0];
            nodes_flat[2 * i + 1] = nodes[i][1];
        }

        // Copy nodes to GPU
        cudaError_t cuda_status = cudaMemcpyAsync(
            mesh_cache_.d_nodes, nodes_flat.data(), 2 * num_nodes * sizeof(double),
            cudaMemcpyHostToDevice, stream_);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy nodes to GPU: " +
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        // Flatten elements array
        std::vector<int> elements_flat(nodes_per_elem * num_elements);
        if (order == 1) {
            const auto& elements = mesh.getElements();
            for (int i = 0; i < num_elements; ++i) {
                for (int j = 0; j < nodes_per_elem; ++j) {
                    elements_flat[nodes_per_elem * i + j] = elements[i][j];
                }
            }
        } else if (order == 2) {
            const auto& elements = mesh.getQuadraticElements();
            for (int i = 0; i < num_elements; ++i) {
                for (int j = 0; j < nodes_per_elem; ++j) {
                    elements_flat[nodes_per_elem * i + j] = elements[i][j];
                }
            }
        } else if (order == 3) {
            const auto& elements = mesh.getCubicElements();
            for (int i = 0; i < num_elements; ++i) {
                for (int j = 0; j < nodes_per_elem; ++j) {
                    elements_flat[nodes_per_elem * i + j] = elements[i][j];
                }
            }
        }

        // Copy elements to GPU
        cuda_status = cudaMemcpyAsync(
            mesh_cache_.d_elements, elements_flat.data(), nodes_per_elem * num_elements * sizeof(int),
            cudaMemcpyHostToDevice, stream_);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy elements to GPU: " +
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        // Synchronize stream to ensure data is copied
        cuda_status = cudaStreamSynchronize(stream_);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to synchronize CUDA stream: " +
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        // Update cache metadata
        mesh_cache_.num_nodes = num_nodes;
        mesh_cache_.num_elements = num_elements;
        mesh_cache_.nodes_per_elem = nodes_per_elem;
        mesh_cache_.valid = true;

        return true;
    } catch (const std::exception& e) {
        // Clean up on failure
        if (mesh_cache_.d_nodes != nullptr) {
            GPUMemoryPool::getInstance().release(mesh_cache_.d_nodes, 2 * num_nodes * sizeof(double));
            mesh_cache_.d_nodes = nullptr;
        }
        if (mesh_cache_.d_elements != nullptr) {
            GPUMemoryPool::getInstance().release(mesh_cache_.d_elements, nodes_per_elem * num_elements * sizeof(int));
            mesh_cache_.d_elements = nullptr;
        }
        mesh_cache_.valid = false;

        std::cerr << "Failed to cache mesh: " << e.what() << std::endl;
        return false;
    }
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

    // Cache mesh data on GPU
    if (!cache_mesh(mesh, order)) {
        throw std::runtime_error("Failed to cache mesh data on GPU");
    }

    // Get cached mesh data
    double* d_nodes = mesh_cache_.d_nodes;
    int* d_elements = mesh_cache_.d_elements;

    // Determine optimal batch size based on GPU properties
    int max_threads_per_block = device_prop_.maxThreadsPerBlock;
    int max_blocks_per_sm = device_prop_.maxBlocksPerMultiProcessor;
    int num_sm = device_prop_.multiProcessorCount;

    // Calculate optimal batch size to maximize GPU utilization
    // We want to keep all SMs busy with enough blocks
    int optimal_batch_size = max_blocks_per_sm * num_sm;

    // Limit batch size to avoid excessive memory usage
    int batch_size = std::min(optimal_batch_size, 256);

    // Ensure batch size is at least 32 (warp size) for efficiency
    batch_size = std::max(batch_size, 32);

    // Limit batch size to the number of elements
    batch_size = std::min(batch_size, num_elements);

    // Allocate memory for material properties
    double* m_star_values = new double[batch_size];
    double* V_values = new double[batch_size];

    // Allocate memory for batch results using memory pool
    std::complex<double>* H_batch = static_cast<std::complex<double>*>(
        get_gpu_buffer(batch_size * nodes_per_elem * nodes_per_elem * sizeof(std::complex<double>), "H_batch"));

    std::complex<double>* M_batch = static_cast<std::complex<double>*>(
        get_gpu_buffer(batch_size * nodes_per_elem * nodes_per_elem * sizeof(std::complex<double>), "M_batch"));

    // Process elements in batches
    for (int batch_start = 0; batch_start < num_elements; batch_start += batch_size) {
        int current_batch_size = std::min(batch_size, num_elements - batch_start);

        // Compute material properties for each element in the batch
        for (int i = 0; i < current_batch_size; ++i) {
            int e = batch_start + i;

            // Get element centroid
            double xc = 0.0, yc = 0.0;

            // Use the first 3 nodes (vertices) to compute centroid
            double x1 = nodes[elements[e][0]][0];
            double y1 = nodes[elements[e][0]][1];
            double x2 = nodes[elements[e][1]][0];
            double y2 = nodes[elements[e][1]][1];
            double x3 = nodes[elements[e][2]][0];
            double y3 = nodes[elements[e][2]][1];

            xc = (x1 + x2 + x3) / 3.0;
            yc = (y1 + y2 + y3) / 3.0;

            // Evaluate material properties at centroid
            m_star_values[i] = m_star(xc, yc);
            V_values[i] = V(xc, yc);
        }

        // Copy material properties to GPU
        double* d_m_star_values = static_cast<double*>(
            get_gpu_buffer(batch_size * sizeof(double), "m_star_values"));

        double* d_V_values = static_cast<double*>(
            get_gpu_buffer(batch_size * sizeof(double), "V_values"));

        cudaError_t cuda_status = cudaMemcpyAsync(
            d_m_star_values, m_star_values, current_batch_size * sizeof(double),
            cudaMemcpyHostToDevice, stream_);

        if (cuda_status != cudaSuccess) {
            delete[] m_star_values;
            delete[] V_values;
            throw std::runtime_error("Failed to copy m_star values to GPU: " +
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        cuda_status = cudaMemcpyAsync(
            d_V_values, V_values, current_batch_size * sizeof(double),
            cudaMemcpyHostToDevice, stream_);

        if (cuda_status != cudaSuccess) {
            delete[] m_star_values;
            delete[] V_values;
            throw std::runtime_error("Failed to copy V values to GPU: " +
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        // Launch GPU kernel for batch processing
        assemble_element_matrices_batched_cuda(
            batch_start, current_batch_size, d_nodes, d_elements,
            H_batch, M_batch, d_m_star_values, d_V_values, order
        );

        // Copy results back to host
        std::complex<double>* h_H_batch = new std::complex<double>[current_batch_size * nodes_per_elem * nodes_per_elem];
        std::complex<double>* h_M_batch = new std::complex<double>[current_batch_size * nodes_per_elem * nodes_per_elem];

        cuda_status = cudaMemcpyAsync(
            h_H_batch, H_batch, current_batch_size * nodes_per_elem * nodes_per_elem * sizeof(std::complex<double>),
            cudaMemcpyDeviceToHost, stream_);

        if (cuda_status != cudaSuccess) {
            delete[] h_H_batch;
            delete[] h_M_batch;
            delete[] m_star_values;
            delete[] V_values;
            throw std::runtime_error("Failed to copy H_batch from GPU: " +
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        cuda_status = cudaMemcpyAsync(
            h_M_batch, M_batch, current_batch_size * nodes_per_elem * nodes_per_elem * sizeof(std::complex<double>),
            cudaMemcpyDeviceToHost, stream_);

        if (cuda_status != cudaSuccess) {
            delete[] h_H_batch;
            delete[] h_M_batch;
            delete[] m_star_values;
            delete[] V_values;
            throw std::runtime_error("Failed to copy M_batch from GPU: " +
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        // Synchronize stream to ensure data is copied
        cuda_status = cudaStreamSynchronize(stream_);
        if (cuda_status != cudaSuccess) {
            delete[] h_H_batch;
            delete[] h_M_batch;
            delete[] m_star_values;
            delete[] V_values;
            throw std::runtime_error("Failed to synchronize CUDA stream: " +
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        // Add to triplet lists
        for (int i = 0; i < current_batch_size; ++i) {
            int e = batch_start + i;
            for (int j = 0; j < nodes_per_elem; ++j) {
                for (int k = 0; k < nodes_per_elem; ++k) {
                    int global_j = elements[e][j];
                    int global_k = elements[e][k];
                    int idx = i * nodes_per_elem * nodes_per_elem + j * nodes_per_elem + k;
                    H_triplets.emplace_back(global_j, global_k, h_H_batch[idx]);
                    M_triplets.emplace_back(global_j, global_k, h_M_batch[idx]);
                }
            }
        }

        // Clean up batch results
        delete[] h_H_batch;
        delete[] h_M_batch;
    }

    // Clean up
    delete[] m_star_values;
    delete[] V_values;

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
    try {
        // Convert eigenvalues to real (we're solving a Hermitian problem)
        std::vector<double> real_eigenvalues;
        std::vector<Eigen::VectorXcd> complex_eigenvectors;

        // Solve the eigenvalue problem on GPU
        solve_eigenvalue_problem_gpu(H, M, num_eigenvalues, real_eigenvalues, complex_eigenvectors);

        // Convert results back to the expected format
        eigenvalues.resize(num_eigenvalues);
        eigenvectors.resize(num_eigenvalues);

        for (int i = 0; i < num_eigenvalues; ++i) {
            eigenvalues[i] = std::complex<double>(real_eigenvalues[i], 0.0);

            // Convert complex eigenvector to real (take absolute value)
            eigenvectors[i].resize(complex_eigenvectors[i].size());
            for (int j = 0; j < complex_eigenvectors[i].size(); ++j) {
                eigenvectors[i](j) = std::abs(complex_eigenvectors[i](j));
            }
        }

        return;
    } catch (const std::exception& e) {
        std::cerr << "GPU eigenvalue solver failed: " << e.what() << std::endl;
        std::cerr << "Falling back to dense solver." << std::endl;

        // Convert sparse matrices to dense
        Eigen::MatrixXcd H_dense = Eigen::MatrixXcd(H);
        Eigen::MatrixXcd M_dense = Eigen::MatrixXcd(M);

        // Make sure matrices are Hermitian
        H_dense = (H_dense + H_dense.adjoint()) / 2.0;
        M_dense = (M_dense + M_dense.adjoint()) / 2.0;

        // Solve using cuSOLVER
        solve_eigen_cusolver(H_dense, M_dense, num_eigenvalues, eigenvalues, eigenvectors);
    }
#endif
}

// Optimized assembly for higher-order elements on GPU
void GPUAccelerator::assemble_higher_order_matrices(const Mesh& mesh,
                                                 double (*m_star)(double, double),
                                                 double (*V)(double, double),
                                                 int order,
                                                 Eigen::SparseMatrix<std::complex<double>>& H,
                                                 Eigen::SparseMatrix<std::complex<double>>& M) {
    // Check if GPU acceleration is enabled
    if (!is_gpu_enabled()) {
        throw std::runtime_error("GPU acceleration not available for higher-order matrix assembly");
    }

    // Check if order is valid for higher-order elements
    if (order < 2) {
        throw std::invalid_argument("Higher-order assembly requires order >= 2");
    }

#ifdef USE_CUDA
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    int num_nodes = nodes.size();

    // Get element data based on order
    std::vector<std::array<int, 6>> quadratic_elements;
    std::vector<std::array<int, 10>> cubic_elements;

    if (order == 2) {
        quadratic_elements = mesh.getQuadraticElements();
    } else { // order == 3
        cubic_elements = mesh.getCubicElements();
    }

    int num_elements = (order == 2) ? quadratic_elements.size() : cubic_elements.size();
    int nodes_per_elem = (order == 2) ? 6 : 10;

    // Resize matrices
    H.resize(num_nodes, num_nodes);
    M.resize(num_nodes, num_nodes);

    // Create triplet lists for sparse matrix assembly
    std::vector<Eigen::Triplet<std::complex<double>>> H_triplets;
    std::vector<Eigen::Triplet<std::complex<double>>> M_triplets;

    // Reserve space for triplets
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

    cuda_status = cudaMalloc(&d_elements, nodes_per_elem * num_elements * sizeof(int));
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
    std::vector<int> elements_flat(nodes_per_elem * num_elements);
    if (order == 2) {
        for (int i = 0; i < num_elements; ++i) {
            for (int j = 0; j < nodes_per_elem; ++j) {
                elements_flat[nodes_per_elem * i + j] = quadratic_elements[i][j];
            }
        }
    } else { // order == 3
        for (int i = 0; i < num_elements; ++i) {
            for (int j = 0; j < nodes_per_elem; ++j) {
                elements_flat[nodes_per_elem * i + j] = cubic_elements[i][j];
            }
        }
    }

    cuda_status = cudaMemcpy(d_elements, elements_flat.data(), nodes_per_elem * num_elements * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_elements);
        cudaFree(d_nodes);
        throw std::runtime_error("Failed to copy elements to GPU: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Allocate memory for element matrices
    std::complex<double>* H_e = new std::complex<double>[nodes_per_elem * nodes_per_elem];
    std::complex<double>* M_e = new std::complex<double>[nodes_per_elem * nodes_per_elem];

    // Process elements in batches for better GPU utilization
    const int batch_size = 64; // Process 64 elements at a time

    // Allocate memory for batch results
    std::complex<double>* H_batch = new std::complex<double>[batch_size * nodes_per_elem * nodes_per_elem];
    std::complex<double>* M_batch = new std::complex<double>[batch_size * nodes_per_elem * nodes_per_elem];

    // Allocate memory for material properties
    double* m_star_values = new double[batch_size];
    double* V_values = new double[batch_size];

    for (int batch_start = 0; batch_start < num_elements; batch_start += batch_size) {
        int current_batch_size = std::min(batch_size, num_elements - batch_start);

        // Compute material properties for each element in the batch
        for (int i = 0; i < current_batch_size; ++i) {
            int e = batch_start + i;

            // Get element centroid
            double xc = 0.0, yc = 0.0;
            std::vector<int> element_nodes;

            if (order == 2) {
                element_nodes.resize(6);
                for (int j = 0; j < 6; ++j) {
                    element_nodes[j] = quadratic_elements[e][j];
                }
            } else { // order == 3
                element_nodes.resize(10);
                for (int j = 0; j < 10; ++j) {
                    element_nodes[j] = cubic_elements[e][j];
                }
            }

            // Use the first 3 nodes (vertices) to compute centroid
            double x1 = nodes[element_nodes[0]][0];
            double y1 = nodes[element_nodes[0]][1];
            double x2 = nodes[element_nodes[1]][0];
            double y2 = nodes[element_nodes[1]][1];
            double x3 = nodes[element_nodes[2]][0];
            double y3 = nodes[element_nodes[2]][1];

            xc = (x1 + x2 + x3) / 3.0;
            yc = (y1 + y2 + y3) / 3.0;

            // Evaluate material properties at centroid
            m_star_values[i] = m_star(xc, yc);
            V_values[i] = V(xc, yc);
        }

        // Launch GPU kernel for batch processing
        assemble_element_matrices_batched_cuda(
            batch_start, current_batch_size, d_nodes, d_elements,
            H_batch, M_batch, m_star_values, V_values, order
        );

        // Add batch results to triplet lists
        for (int i = 0; i < current_batch_size; ++i) {
            int e = batch_start + i;

            // Get element nodes
            std::vector<int> element(nodes_per_elem);
            if (order == 2) {
                for (int j = 0; j < nodes_per_elem; ++j) {
                    element[j] = quadratic_elements[e][j];
                }
            } else { // order == 3
                for (int j = 0; j < nodes_per_elem; ++j) {
                    element[j] = cubic_elements[e][j];
                }
            }

            // Add to triplet lists
            for (int j = 0; j < nodes_per_elem; ++j) {
                for (int k = 0; k < nodes_per_elem; ++k) {
                    int global_j = element[j];
                    int global_k = element[k];
                    int matrix_idx = i * nodes_per_elem * nodes_per_elem + j * nodes_per_elem + k;
                    H_triplets.emplace_back(global_j, global_k, H_batch[matrix_idx]);
                    M_triplets.emplace_back(global_j, global_k, M_batch[matrix_idx]);
                }
            }
        }
    }

    // Clean up
    delete[] H_e;
    delete[] M_e;
    delete[] H_batch;
    delete[] M_batch;
    delete[] m_star_values;
    delete[] V_values;
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

// Optimized eigensolver for large sparse matrices from higher-order elements
void GPUAccelerator::solve_eigen_sparse(const Eigen::SparseMatrix<std::complex<double>>& H,
                                      const Eigen::SparseMatrix<std::complex<double>>& M,
                                      int num_eigenvalues,
                                      std::vector<std::complex<double>>& eigenvalues,
                                      std::vector<Eigen::VectorXd>& eigenvectors,
                                      double tolerance,
                                      int max_iterations) {
    // Check if GPU acceleration is enabled
    if (!is_gpu_enabled()) {
        throw std::runtime_error("GPU acceleration not available for sparse eigenvalue solving");
    }

#ifdef USE_CUDA
    // Get matrix dimensions
    int n = H.rows();

    // Check if matrices are valid
    if (H.rows() != H.cols() || M.rows() != M.cols() || H.rows() != M.rows()) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    // Check if number of eigenvalues is valid
    if (num_eigenvalues <= 0 || num_eigenvalues > n) {
        throw std::invalid_argument("Invalid number of eigenvalues");
    }

    // Convert to CSR format for cuSPARSE
    // First, make sure matrices are in compressed format
    Eigen::SparseMatrix<std::complex<double>> H_csr = H;
    Eigen::SparseMatrix<std::complex<double>> M_csr = M;
    H_csr.makeCompressed();
    M_csr.makeCompressed();

    // Get CSR representation
    const int* H_row_ptr = H_csr.outerIndexPtr();
    const int* H_col_ind = H_csr.innerIndexPtr();
    const std::complex<double>* H_values = H_csr.valuePtr();

    const int* M_row_ptr = M_csr.outerIndexPtr();
    const int* M_col_ind = M_csr.innerIndexPtr();
    const std::complex<double>* M_values = M_csr.valuePtr();

    // Allocate device memory for matrices
    int* d_H_row_ptr = nullptr;
    int* d_H_col_ind = nullptr;
    cuDoubleComplex* d_H_values = nullptr;

    int* d_M_row_ptr = nullptr;
    int* d_M_col_ind = nullptr;
    cuDoubleComplex* d_M_values = nullptr;

    // Allocate memory for row pointers
    cudaError_t cuda_status = cudaMalloc(&d_H_row_ptr, (n + 1) * sizeof(int));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for H row pointers: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    cuda_status = cudaMalloc(&d_M_row_ptr, (n + 1) * sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to allocate GPU memory for M row pointers: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Copy row pointers to device
    cuda_status = cudaMemcpy(d_H_row_ptr, H_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to copy H row pointers to GPU: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    cuda_status = cudaMemcpy(d_M_row_ptr, M_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to copy M row pointers to GPU: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Allocate memory for column indices
    int H_nnz = H_csr.nonZeros();
    int M_nnz = M_csr.nonZeros();

    cuda_status = cudaMalloc(&d_H_col_ind, H_nnz * sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to allocate GPU memory for H column indices: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    cuda_status = cudaMalloc(&d_M_col_ind, M_nnz * sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to allocate GPU memory for M column indices: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Copy column indices to device
    cuda_status = cudaMemcpy(d_H_col_ind, H_col_ind, H_nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_col_ind);
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to copy H column indices to GPU: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    cuda_status = cudaMemcpy(d_M_col_ind, M_col_ind, M_nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_col_ind);
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to copy M column indices to GPU: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Allocate memory for values
    cuda_status = cudaMalloc(&d_H_values, H_nnz * sizeof(cuDoubleComplex));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_col_ind);
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to allocate GPU memory for H values: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    cuda_status = cudaMalloc(&d_M_values, M_nnz * sizeof(cuDoubleComplex));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_H_values);
        cudaFree(d_M_col_ind);
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to allocate GPU memory for M values: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Copy values to device (need to convert from std::complex to cuDoubleComplex)
    std::vector<cuDoubleComplex> H_values_cuda(H_nnz);
    std::vector<cuDoubleComplex> M_values_cuda(M_nnz);

    for (int i = 0; i < H_nnz; ++i) {
        H_values_cuda[i].x = H_values[i].real();
        H_values_cuda[i].y = H_values[i].imag();
    }

    for (int i = 0; i < M_nnz; ++i) {
        M_values_cuda[i].x = M_values[i].real();
        M_values_cuda[i].y = M_values[i].imag();
    }

    cuda_status = cudaMemcpy(d_H_values, H_values_cuda.data(), H_nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_values);
        cudaFree(d_H_values);
        cudaFree(d_M_col_ind);
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to copy H values to GPU: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    cuda_status = cudaMemcpy(d_M_values, M_values_cuda.data(), M_nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_values);
        cudaFree(d_H_values);
        cudaFree(d_M_col_ind);
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to copy M values to GPU: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Create cuSPARSE matrix descriptors
    cusparseMatDescr_t descr_H, descr_M;
    cusparseStatus_t cusparse_status = cusparseCreateMatDescr(&descr_H);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(d_M_values);
        cudaFree(d_H_values);
        cudaFree(d_M_col_ind);
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to create cuSPARSE matrix descriptor for H");
    }

    cusparse_status = cusparseCreateMatDescr(&descr_M);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyMatDescr(descr_H);
        cudaFree(d_M_values);
        cudaFree(d_H_values);
        cudaFree(d_M_col_ind);
        cudaFree(d_H_col_ind);
        cudaFree(d_M_row_ptr);
        cudaFree(d_H_row_ptr);
        throw std::runtime_error("Failed to create cuSPARSE matrix descriptor for M");
    }

    // Set matrix properties
    cusparseSetMatType(descr_H, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_H, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);

    // TODO: Implement sparse eigensolver using cuSPARSE and cuSOLVER
    // For now, convert to dense and use the dense solver

    // Convert sparse matrices to dense
    Eigen::MatrixXcd H_dense = Eigen::MatrixXcd(H);
    Eigen::MatrixXcd M_dense = Eigen::MatrixXcd(M);

    // Make sure matrices are Hermitian
    H_dense = (H_dense + H_dense.adjoint()) / 2.0;
    M_dense = (M_dense + M_dense.adjoint()) / 2.0;

    // Solve using cuSOLVER
    solve_eigen_cusolver(H_dense, M_dense, num_eigenvalues, eigenvalues, eigenvectors);

    // Clean up
    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyMatDescr(descr_H);
    cudaFree(d_M_values);
    cudaFree(d_H_values);
    cudaFree(d_M_col_ind);
    cudaFree(d_H_col_ind);
    cudaFree(d_M_row_ptr);
    cudaFree(d_H_row_ptr);
#endif
}

// Batched eigensolver for multiple small eigenproblems
void GPUAccelerator::solve_eigen_batched(const std::vector<Eigen::MatrixXcd>& H_batch,
                                       const std::vector<Eigen::MatrixXcd>& M_batch,
                                       int num_eigenvalues,
                                       std::vector<std::vector<std::complex<double>>>& eigenvalues_batch,
                                       std::vector<std::vector<Eigen::VectorXd>>& eigenvectors_batch) {
    // Check if GPU acceleration is enabled
    if (!is_gpu_enabled()) {
        throw std::runtime_error("GPU acceleration not available for batched eigenvalue solving");
    }

#ifdef USE_CUDA
    // Check if batch sizes match
    if (H_batch.size() != M_batch.size()) {
        throw std::invalid_argument("Batch sizes do not match");
    }

    // Get batch size
    int batch_size = H_batch.size();
    if (batch_size == 0) {
        return;
    }

    // Resize output vectors
    eigenvalues_batch.resize(batch_size);
    eigenvectors_batch.resize(batch_size);

    // Process each problem in the batch
    for (int i = 0; i < batch_size; ++i) {
        // Solve individual eigenvalue problem
        std::vector<std::complex<double>> eigenvalues;
        std::vector<Eigen::VectorXd> eigenvectors;

        solve_eigen_cusolver(H_batch[i], M_batch[i], num_eigenvalues, eigenvalues, eigenvectors);

        // Store results
        eigenvalues_batch[i] = eigenvalues;
        eigenvectors_batch[i] = eigenvectors;
    }

    // TODO: Implement true batched solver using cuSOLVER batched APIs
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

#ifdef USE_CUDA
// Assemble element matrices on GPU
void GPUAccelerator::assemble_element_matrix_gpu(int element_idx,
                                              const double* nodes,
                                              const int* elements,
                                              double (*m_star)(double, double),
                                              double (*V)(double, double),
                                              int order,
                                              std::complex<double>* H_e,
                                              std::complex<double>* M_e) {
    // Compute element centroid
    double xc = 0.0, yc = 0.0;
    int n1 = elements[3 * element_idx];
    int n2 = elements[3 * element_idx + 1];
    int n3 = elements[3 * element_idx + 2];

    double x1 = nodes[2 * n1];
    double y1 = nodes[2 * n1 + 1];
    double x2 = nodes[2 * n2];
    double y2 = nodes[2 * n2 + 1];
    double x3 = nodes[2 * n3];
    double y3 = nodes[2 * n3 + 1];

    xc = (x1 + x2 + x3) / 3.0;
    yc = (y1 + y2 + y3) / 3.0;

    // Evaluate material properties at centroid
    double m_star_val = m_star(xc, yc);
    double V_val = V(xc, yc);

    // For now, use a CPU implementation
    int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;

    // Calculate element area
    double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

    // Calculate shape function gradients
    double b1 = (y2 - y3) / (2.0 * area);
    double c1 = (x3 - x2) / (2.0 * area);
    double b2 = (y3 - y1) / (2.0 * area);
    double c2 = (x1 - x3) / (2.0 * area);
    double b3 = (y1 - y2) / (2.0 * area);
    double c3 = (x2 - x1) / (2.0 * area);

    // Assemble element matrices
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // Calculate gradients of shape functions
            double dNi_dx, dNi_dy, dNj_dx, dNj_dy;

            if (i == 0) { dNi_dx = b1; dNi_dy = c1; }
            else if (i == 1) { dNi_dx = b2; dNi_dy = c2; }
            else { dNi_dx = b3; dNi_dy = c3; }

            if (j == 0) { dNj_dx = b1; dNj_dy = c1; }
            else if (j == 1) { dNj_dx = b2; dNj_dy = c2; }
            else { dNj_dx = b3; dNj_dy = c3; }

            // Calculate shape functions at centroid
            double Ni, Nj;

            if (i == 0) Ni = 1.0/3.0;
            else if (i == 1) Ni = 1.0/3.0;
            else Ni = 1.0/3.0;

            if (j == 0) Nj = 1.0/3.0;
            else if (j == 1) Nj = 1.0/3.0;
            else Nj = 1.0/3.0;

            // Calculate Hamiltonian matrix element
            const double hbar = 6.582119569e-16; // eV·s
            double kinetic_term = (hbar * hbar / (2.0 * m_star_val)) * (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * area;
            double potential_term = V_val * Ni * Nj * area;
            H_e[i * nodes_per_elem + j] = std::complex<double>(kinetic_term + potential_term, 0.0);

            // Calculate mass matrix element
            M_e[i * nodes_per_elem + j] = std::complex<double>(Ni * Nj * area, 0.0);
        }
    }
}

// Convert Eigen sparse matrix to CSR format
void GPUAccelerator::eigen_sparse_to_csr(
    const Eigen::SparseMatrix<std::complex<double>>& mat,
    std::vector<int>& rowPtr,
    std::vector<int>& colInd,
    std::vector<std::complex<double>>& values) {

    // Make sure matrix is in compressed format
    Eigen::SparseMatrix<std::complex<double>> mat_csr = mat;
    mat_csr.makeCompressed();

    // Get matrix dimensions
    int rows = mat_csr.rows();
    int nnz = mat_csr.nonZeros();

    // Resize output vectors
    rowPtr.resize(rows + 1);
    colInd.resize(nnz);
    values.resize(nnz);

    // Copy data from Eigen sparse matrix
    std::copy(mat_csr.outerIndexPtr(), mat_csr.outerIndexPtr() + rows + 1, rowPtr.begin());
    std::copy(mat_csr.innerIndexPtr(), mat_csr.innerIndexPtr() + nnz, colInd.begin());
    std::copy(mat_csr.valuePtr(), mat_csr.valuePtr() + nnz, values.begin());
}

// Solve generalized eigenvalue problem on GPU
void GPUAccelerator::solve_eigenvalue_problem_gpu(
    const Eigen::SparseMatrix<std::complex<double>>& H,
    const Eigen::SparseMatrix<std::complex<double>>& M,
    int num_eigenvalues,
    std::vector<double>& eigenvalues,
    std::vector<Eigen::VectorXcd>& eigenvectors) {

    // Check if GPU acceleration is enabled
    if (!is_gpu_enabled()) {
        throw std::runtime_error("GPU acceleration not available for eigenvalue solving");
    }

    // Get matrix dimensions
    int n = H.rows();

    // Check if matrices are valid
    if (H.rows() != H.cols() || M.rows() != M.cols() || H.rows() != M.rows()) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    // Check if number of eigenvalues is valid
    if (num_eigenvalues <= 0 || num_eigenvalues > n) {
        throw std::invalid_argument("Invalid number of eigenvalues");
    }

    // Convert matrices to CSR format
    std::vector<int> H_rowPtr, H_colInd, M_rowPtr, M_colInd;
    std::vector<std::complex<double>> H_values, M_values;

    eigen_sparse_to_csr(H, H_rowPtr, H_colInd, H_values);
    eigen_sparse_to_csr(M, M_rowPtr, M_colInd, M_values);

    // Allocate memory for eigenvalues and eigenvectors
    eigenvalues.resize(num_eigenvalues);
    std::vector<std::complex<double>> eigenvectors_flat(n * num_eigenvalues);

    // Solve generalized eigenvalue problem
    cudaError_t cuda_status = solve_sparse_generalized_eigenvalue_problem(
        n,
        H_values.size(),
        M_values.size(),
        H_rowPtr.data(),
        H_colInd.data(),
        H_values.data(),
        M_rowPtr.data(),
        M_colInd.data(),
        M_values.data(),
        eigenvalues.data(),
        eigenvectors_flat.data(),
        num_eigenvalues
    );

    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to solve generalized eigenvalue problem on GPU: " +
                                std::string(cudaGetErrorString(cuda_status)));
    }

    // Convert flat eigenvectors to Eigen vectors
    eigenvectors.resize(num_eigenvalues);
    for (int i = 0; i < num_eigenvalues; ++i) {
        eigenvectors[i].resize(n);
        for (int j = 0; j < n; ++j) {
            eigenvectors[i](j) = eigenvectors_flat[i * n + j];
        }
    }
}
#endif