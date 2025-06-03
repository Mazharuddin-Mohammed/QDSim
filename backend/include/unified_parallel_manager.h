#pragma once

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace QDSim {

class UnifiedParallelManager {
public:
    struct ParallelConfig {
        int mpi_ranks = 1;
        int omp_threads_per_rank = 1;
        int cuda_devices_per_rank = 1;
        bool enable_gpu_direct = true;
        bool enable_numa_binding = true;
        size_t gpu_memory_pool_size = 0; // 0 = auto-detect
    };

    // Singleton pattern for global access
    static UnifiedParallelManager& getInstance() {
        static UnifiedParallelManager instance;
        return instance;
    }

    // Initialize the parallel environment
    bool initialize(const ParallelConfig& config);
    
    // Shutdown and cleanup
    void finalize();
    
    // Get current parallel context
    struct ParallelContext {
        int mpi_rank;
        int mpi_size;
        int omp_thread_id;
        int omp_num_threads;
        int cuda_device_id;
        int numa_node;
    };
    
    ParallelContext getCurrentContext() const;
    
    // Work distribution and synchronization
    template<typename WorkItem>
    void distributeWork(const std::vector<WorkItem>& work_items,
                       std::function<void(const WorkItem&, const ParallelContext&)> processor);
    
    void barrier() const;
    void gpuSynchronize() const;
    
    // Resource management
    bool isInitialized() const { return initialized_; }
    int getMPIRank() const { return mpi_rank_; }
    int getMPISize() const { return mpi_size_; }
    int getCurrentDevice() const { return current_device_; }
    
private:
    UnifiedParallelManager() = default;
    ~UnifiedParallelManager() { finalize(); }
    
    // Disable copy and move
    UnifiedParallelManager(const UnifiedParallelManager&) = delete;
    UnifiedParallelManager& operator=(const UnifiedParallelManager&) = delete;
    
    ParallelConfig config_;
    bool initialized_ = false;
    
    // MPI state
    int mpi_rank_ = 0;
    int mpi_size_ = 1;
    
    // OpenMP state
    std::vector<int> thread_to_numa_mapping_;
    
    // CUDA state
    std::vector<int> available_devices_;
    int current_device_ = 0;
    
    // Thread-local storage for context
    thread_local static ParallelContext current_context_;
    
    // Helper methods
    void initializeNUMABinding();
    void enableGPUDirect();
    int getCurrentNUMANode() const;
};

// Template implementation
template<typename WorkItem>
void UnifiedParallelManager::distributeWork(
    const std::vector<WorkItem>& work_items,
    std::function<void(const WorkItem&, const ParallelContext&)> processor) {
    
    if (!initialized_) {
        throw std::runtime_error("UnifiedParallelManager not initialized");
    }
    
    // Calculate work distribution
    size_t total_work = work_items.size();
    size_t work_per_rank = total_work / mpi_size_;
    size_t start_idx = mpi_rank_ * work_per_rank;
    size_t end_idx = (mpi_rank_ == mpi_size_ - 1) ? total_work : start_idx + work_per_rank;
    
    // Process work items in parallel using OpenMP
    #pragma omp parallel
    {
        // Update thread-local context
        current_context_.mpi_rank = mpi_rank_;
        current_context_.mpi_size = mpi_size_;
        current_context_.omp_thread_id = omp_get_thread_num();
        current_context_.omp_num_threads = omp_get_num_threads();
        current_context_.cuda_device_id = current_device_;
        current_context_.numa_node = getCurrentNUMANode();
        
        // Set CUDA device for this thread
        if (!available_devices_.empty()) {
            cudaSetDevice(current_device_);
        }
        
        #pragma omp for schedule(dynamic)
        for (size_t i = start_idx; i < end_idx; ++i) {
            processor(work_items[i], current_context_);
        }
    }
    
    // MPI barrier to synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace QDSim
