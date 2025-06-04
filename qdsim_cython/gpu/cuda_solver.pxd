# distutils: language = c++
# cython: language_level = 3

"""
Cython declaration file for CUDA GPU acceleration

High-performance GPU-accelerated quantum mechanical calculations
using CUDA for massive parallel processing.
"""

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as bint
from ..eigen cimport VectorXd, VectorXcd, MatrixXcd, SparseMatrixXcd
from ..core.mesh cimport Mesh

# CUDA device management
cdef extern from "cuda_device_manager.h":
    cdef cppclass CudaDeviceManager:
        CudaDeviceManager() except +
        
        # Device queries
        int get_device_count() except +
        bint is_cuda_available() except +
        string get_device_name(int device_id) except +
        size_t get_device_memory(int device_id) except +
        size_t get_free_memory(int device_id) except +
        
        # Device selection and management
        void set_device(int device_id) except +
        int get_current_device() except +
        void synchronize_device() except +
        void reset_device() except +
        
        # Memory management
        void* allocate_device_memory(size_t size) except +
        void free_device_memory(void* ptr) except +
        void copy_to_device(void* dst, const void* src, size_t size) except +
        void copy_from_device(void* dst, const void* src, size_t size) except +
        
        # Performance monitoring
        double get_memory_bandwidth() except +
        int get_compute_capability_major() except +
        int get_compute_capability_minor() except +
        int get_multiprocessor_count() except +
        int get_max_threads_per_block() except +

# CUDA-accelerated Schrödinger solver
cdef extern from "cuda_schrodinger_solver.h":
    cdef cppclass CudaSchrodingerSolver:
        CudaSchrodingerSolver(const Mesh& mesh, int device_id) except +
        
        # Core solving methods
        void solve_gpu(int num_eigenvalues) except +
        void solve_gpu_iterative(int num_eigenvalues, double tolerance) except +
        void solve_gpu_batched(const vector[int]& eigenvalue_counts) except +
        
        # Hybrid CPU-GPU solving
        void solve_hybrid(int num_eigenvalues, double cpu_fraction) except +
        
        # Results access
        VectorXd get_eigenvalues_gpu() except +
        vector[VectorXd] get_eigenvectors_gpu() except +
        
        # GPU-specific configuration
        void set_block_size(int block_size) except +
        void set_grid_size(int grid_size) except +
        void enable_gpu_memory_pool(bint enable) except +
        void set_gpu_memory_limit(size_t limit_bytes) except +
        
        # Performance optimization
        void enable_tensor_cores(bint enable) except +
        void enable_mixed_precision(bint enable) except +
        void set_cuda_streams(int num_streams) except +
        
        # Memory management
        void preload_data_to_gpu() except +
        void clear_gpu_memory() except +
        size_t get_gpu_memory_usage() except +
        
        # Performance monitoring
        double get_gpu_solve_time() except +
        double get_memory_transfer_time() except +
        double get_total_gpu_time() except +
        string get_gpu_performance_report() except +

# CUDA-accelerated matrix operations
cdef extern from "cuda_matrix_ops.h":
    cdef cppclass CudaMatrixOperations:
        CudaMatrixOperations(int device_id) except +
        
        # Basic matrix operations
        void matrix_vector_multiply_gpu(const SparseMatrixXcd& A, 
                                      const VectorXcd& x, 
                                      VectorXcd& y) except +
        
        void matrix_matrix_multiply_gpu(const MatrixXcd& A, 
                                      const MatrixXcd& B, 
                                      MatrixXcd& C) except +
        
        # Eigenvalue solvers
        void solve_eigenvalues_gpu(const SparseMatrixXcd& A, 
                                 int num_eigenvalues,
                                 VectorXd& eigenvalues,
                                 vector[VectorXcd]& eigenvectors) except +
        
        void solve_generalized_eigenvalues_gpu(const SparseMatrixXcd& A,
                                             const SparseMatrixXcd& B,
                                             int num_eigenvalues,
                                             VectorXd& eigenvalues,
                                             vector[VectorXcd]& eigenvectors) except +
        
        # Linear system solvers
        void solve_linear_system_gpu(const SparseMatrixXcd& A,
                                   const VectorXcd& b,
                                   VectorXcd& x) except +
        
        void solve_linear_system_iterative_gpu(const SparseMatrixXcd& A,
                                             const VectorXcd& b,
                                             VectorXcd& x,
                                             double tolerance,
                                             int max_iterations) except +
        
        # Matrix factorizations
        void lu_factorization_gpu(const MatrixXcd& A,
                                MatrixXcd& L,
                                MatrixXcd& U) except +
        
        void cholesky_factorization_gpu(const MatrixXcd& A,
                                      MatrixXcd& L) except +
        
        # Performance optimization
        void enable_cublas_tensor_ops(bint enable) except +
        void enable_cusolver_optimizations(bint enable) except +
        void set_cublas_math_mode(int mode) except +

# CUDA memory management
cdef extern from "cuda_memory_manager.h":
    cdef cppclass CudaMemoryManager:
        CudaMemoryManager(int device_id) except +
        
        # Unified memory management
        void* allocate_unified_memory(size_t size) except +
        void free_unified_memory(void* ptr) except +
        void prefetch_to_gpu(void* ptr, size_t size) except +
        void prefetch_to_cpu(void* ptr, size_t size) except +
        
        # Memory pools
        void create_memory_pool(size_t pool_size) except +
        void destroy_memory_pool() except +
        void* allocate_from_pool(size_t size) except +
        void free_to_pool(void* ptr) except +
        
        # Memory optimization
        void enable_memory_compression(bint enable) except +
        void set_memory_growth_policy(int policy) except +
        void optimize_memory_layout() except +
        
        # Memory monitoring
        size_t get_allocated_memory() except +
        size_t get_peak_memory_usage() except +
        double get_memory_fragmentation() except +
        string get_memory_report() except +

# CUDA kernel management
cdef extern from "cuda_kernels.h":
    cdef cppclass CudaKernelManager:
        CudaKernelManager(int device_id) except +
        
        # Kernel compilation and caching
        void compile_kernels() except +
        void cache_kernels(bint enable) except +
        void clear_kernel_cache() except +
        
        # Kernel execution configuration
        void set_default_block_size(int block_size) except +
        void set_default_grid_size(int grid_size) except +
        void enable_dynamic_parallelism(bint enable) except +
        
        # Specialized quantum kernels
        void launch_hamiltonian_assembly_kernel(const Mesh& mesh,
                                               void* mass_data,
                                               void* potential_data,
                                               SparseMatrixXcd& hamiltonian) except +
        
        void launch_eigenvalue_kernel(const SparseMatrixXcd& hamiltonian,
                                    int num_eigenvalues,
                                    VectorXd& eigenvalues,
                                    vector[VectorXcd]& eigenvectors) except +
        
        void launch_wavefunction_normalization_kernel(vector[VectorXcd]& eigenvectors) except +
        
        # Performance optimization kernels
        void launch_matrix_compression_kernel(SparseMatrixXcd& matrix) except +
        void launch_memory_coalescing_kernel(void* data, size_t size) except +
        
        # Kernel performance monitoring
        double get_kernel_execution_time(const string& kernel_name) except +
        string get_kernel_performance_report() except +

# Multi-GPU support
cdef extern from "multi_gpu_manager.h":
    cdef cppclass MultiGpuManager:
        MultiGpuManager() except +
        
        # Multi-GPU initialization
        void initialize_multi_gpu(const vector[int]& device_ids) except +
        void finalize_multi_gpu() except +
        
        # Load balancing
        void distribute_workload(const vector[double]& workload_weights) except +
        void enable_dynamic_load_balancing(bint enable) except +
        
        # Communication
        void enable_peer_to_peer_access() except +
        void synchronize_all_devices() except +
        void broadcast_data(void* data, size_t size, int source_device) except +
        void reduce_data(void* data, size_t size, int reduction_op) except +
        
        # Multi-GPU solving
        void solve_distributed(const SparseMatrixXcd& hamiltonian,
                             int num_eigenvalues,
                             VectorXd& eigenvalues,
                             vector[VectorXcd]& eigenvectors) except +
        
        # Performance monitoring
        vector[double] get_device_utilizations() except +
        vector[size_t] get_device_memory_usage() except +
        string get_multi_gpu_performance_report() except +

# Error handling for CUDA operations
cdef extern from "cuda_errors.h":
    cdef cppclass CudaError:
        CudaError(const string& message) except +
        const char* what() except +
    
    cdef cppclass CudaMemoryError:
        CudaMemoryError(const string& message) except +
        const char* what() except +
    
    cdef cppclass CudaKernelError:
        CudaKernelError(const string& message) except +
        const char* what() except +
    
    cdef cppclass CudaDeviceError:
        CudaDeviceError(const string& message) except +
        const char* what() except +

# CUDA profiling and debugging
cdef extern from "cuda_profiler.h":
    cdef cppclass CudaProfiler:
        CudaProfiler() except +
        
        # Profiling control
        void start_profiling() except +
        void stop_profiling() except +
        void reset_profiling() except +
        
        # Performance metrics
        double get_kernel_time(const string& kernel_name) except +
        double get_memory_transfer_time() except +
        double get_total_gpu_time() except +
        double get_gpu_utilization() except +
        
        # Memory profiling
        size_t get_peak_memory_usage() except +
        double get_memory_bandwidth_utilization() except +
        
        # Profiling reports
        string generate_performance_report() except +
        void export_profiling_data(const string& filename) except +
