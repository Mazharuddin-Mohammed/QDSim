#include "unified_parallel_manager.h"
#include "unified_memory_manager.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>

#ifdef __linux__
#include <numa.h>
#include <sched.h>
#endif

namespace QDSim {

thread_local UnifiedParallelManager::ParallelContext 
    UnifiedParallelManager::current_context_;

bool UnifiedParallelManager::initialize(const ParallelConfig& config) {
    if (initialized_) return true;
    
    config_ = config;
    
    try {
        // 1. Initialize MPI with thread support
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            std::cerr << "Warning: MPI implementation doesn't support MPI_THREAD_MULTIPLE" << std::endl;
        }
        
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        
        // 2. Initialize NUMA topology
        if (config_.enable_numa_binding) {
            initializeNUMABinding();
        }
        
        // 3. Initialize CUDA devices
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        
        if (cuda_err == cudaSuccess && device_count > 0) {
            // Distribute devices among MPI ranks
            int devices_per_rank = std::max(1, device_count / mpi_size_);
            int start_device = mpi_rank_ * devices_per_rank;
            int end_device = std::min(start_device + devices_per_rank, device_count);
            
            for (int i = start_device; i < end_device; ++i) {
                available_devices_.push_back(i);
            }
            
            if (!available_devices_.empty()) {
                current_device_ = available_devices_[0];
                cudaSetDevice(current_device_);
                
                // Enable GPU Direct if available
                if (config_.enable_gpu_direct) {
                    enableGPUDirect();
                }
            }
        } else {
            std::cerr << "Warning: No CUDA devices available or CUDA not installed" << std::endl;
        }
        
        // 4. Initialize OpenMP
        omp_set_num_threads(config_.omp_threads_per_rank);
        omp_set_nested(1); // Enable nested parallelism
        
        // 5. Initialize unified memory manager
        UnifiedMemoryManager::getInstance().initialize(config_);
        
        initialized_ = true;
        
        if (mpi_rank_ == 0) {
            std::cout << "UnifiedParallelManager initialized successfully:" << std::endl;
            std::cout << "  MPI ranks: " << mpi_size_ << std::endl;
            std::cout << "  OpenMP threads per rank: " << config_.omp_threads_per_rank << std::endl;
            std::cout << "  CUDA devices available: " << available_devices_.size() << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize UnifiedParallelManager: " << e.what() << std::endl;
        return false;
    }
}

void UnifiedParallelManager::finalize() {
    if (!initialized_) return;
    
    try {
        // Synchronize all processes before cleanup
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Cleanup CUDA resources
        if (!available_devices_.empty()) {
            cudaDeviceSynchronize();
            cudaDeviceReset();
        }
        
        // Cleanup unified memory manager
        UnifiedMemoryManager::getInstance().cleanup();
        
        // Finalize MPI
        MPI_Finalize();
        
        initialized_ = false;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during UnifiedParallelManager finalization: " << e.what() << std::endl;
    }
}

UnifiedParallelManager::ParallelContext UnifiedParallelManager::getCurrentContext() const {
    return current_context_;
}

void UnifiedParallelManager::barrier() const {
    if (initialized_) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void UnifiedParallelManager::gpuSynchronize() const {
    if (!available_devices_.empty()) {
        cudaDeviceSynchronize();
    }
}

void UnifiedParallelManager::initializeNUMABinding() {
#ifdef __linux__
    if (numa_available() != -1) {
        int num_numa_nodes = numa_max_node() + 1;
        int threads_per_node = std::max(1, config_.omp_threads_per_rank / num_numa_nodes);
        
        thread_to_numa_mapping_.resize(config_.omp_threads_per_rank);
        
        for (int t = 0; t < config_.omp_threads_per_rank; ++t) {
            int numa_node = std::min(t / threads_per_node, num_numa_nodes - 1);
            thread_to_numa_mapping_[t] = numa_node;
        }
        
        if (mpi_rank_ == 0) {
            std::cout << "NUMA binding enabled with " << num_numa_nodes << " nodes" << std::endl;
        }
    }
#endif
}

void UnifiedParallelManager::enableGPUDirect() {
#ifdef CUDA_AWARE_MPI
    try {
        // Enable GPU Direct RDMA for MPI communications
        cudaDeviceSetAttribute(cudaDevAttrGPUDirectRDMAWritesOrdering, 1, current_device_);
        cudaDeviceSetAttribute(cudaDevAttrGPUDirectRDMAFlushWritesOptions, 1, current_device_);
        
        if (mpi_rank_ == 0) {
            std::cout << "GPU Direct enabled for device " << current_device_ << std::endl;
        }
    } catch (...) {
        std::cerr << "Warning: Failed to enable GPU Direct" << std::endl;
    }
#endif
}

int UnifiedParallelManager::getCurrentNUMANode() const {
#ifdef __linux__
    if (!thread_to_numa_mapping_.empty()) {
        int thread_id = omp_get_thread_num();
        if (thread_id < thread_to_numa_mapping_.size()) {
            return thread_to_numa_mapping_[thread_id];
        }
    }
    return numa_node_of_cpu(sched_getcpu());
#else
    return 0;
#endif
}

} // namespace QDSim
