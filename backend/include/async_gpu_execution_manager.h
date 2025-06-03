#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "fused_gpu_kernels.cuh"

namespace QDSim {

class AsyncGPUExecutionManager {
public:
    static constexpr int NUM_STREAMS = 8;
    static constexpr int NUM_EVENTS = 16;
    
    AsyncGPUExecutionManager();
    ~AsyncGPUExecutionManager();
    
    // Disable copy and move
    AsyncGPUExecutionManager(const AsyncGPUExecutionManager&) = delete;
    AsyncGPUExecutionManager& operator=(const AsyncGPUExecutionManager&) = delete;
    
    // Kernel launch interface
    template<typename KernelFunc, typename... Args>
    void launchKernelAsync(int stream_id, dim3 grid, dim3 block, 
                          size_t shared_mem, KernelFunc kernel, Args... args);
    
    template<typename KernelFunc, typename... Args>
    void launchKernelAsync(int stream_id, const GPU::LaunchConfig& config,
                          KernelFunc kernel, Args... args);
    
    // Memory operations
    void memcpyAsync(void* dst, const void* src, size_t size, 
                    cudaMemcpyKind kind, int stream_id);
    
    void memsetAsync(void* ptr, int value, size_t size, int stream_id);
    
    // Synchronization
    void synchronizeStream(int stream_id);
    void synchronizeAll();
    void synchronizeEvent(int event_id);
    
    // Event management
    int recordEvent(int stream_id);
    void waitForEvent(int event_id, int stream_id);
    bool isEventComplete(int event_id);
    
    // Pipeline execution with overlapped computation and communication
    struct ComputeTask {
        std::function<void(cudaStream_t)> task;
        int priority = 0;
        std::string name;
    };
    
    struct CommTask {
        std::function<void(cudaStream_t)> task;
        int dependency_event = -1;
        std::string name;
    };
    
    void executePipeline(const std::vector<ComputeTask>& compute_tasks,
                        const std::vector<CommTask>& comm_tasks);
    
    // Performance monitoring
    struct StreamStats {
        int stream_id;
        size_t kernels_launched = 0;
        size_t bytes_transferred = 0;
        double total_compute_time = 0.0;
        double total_transfer_time = 0.0;
        double utilization_percent = 0.0;
    };
    
    std::vector<StreamStats> getStreamStats() const;
    void resetStats();
    void printStats() const;
    
    // Resource management
    cudaStream_t getStream(int stream_id) const;
    cudaEvent_t getEvent(int event_id) const;
    int getOptimalStreamCount() const;
    
    // Error handling
    void checkErrors() const;
    std::string getLastError() const;
    
private:
    cudaStream_t streams_[NUM_STREAMS];
    cudaEvent_t events_[NUM_EVENTS];
    
    // Performance tracking
    mutable std::mutex stats_mutex_;
    std::vector<StreamStats> stream_stats_;
    std::atomic<int> next_event_id_{0};
    
    // Error tracking
    mutable std::mutex error_mutex_;
    std::string last_error_;
    
    // Helper methods
    void initializeStreams();
    void initializeEvents();
    void cleanup();
    int getNextEventId();
    void updateStreamStats(int stream_id, size_t bytes, double time, bool is_compute);
    
    // Timing utilities
    double getElapsedTime(cudaEvent_t start, cudaEvent_t end) const;
    cudaEvent_t createTimingEvent();
};

// Template implementations
template<typename KernelFunc, typename... Args>
void AsyncGPUExecutionManager::launchKernelAsync(int stream_id, dim3 grid, dim3 block, 
                                                size_t shared_mem, KernelFunc kernel, Args... args) {
    
    stream_id = stream_id % NUM_STREAMS;
    
    try {
        // Record start time
        cudaEvent_t start_event = createTimingEvent();
        cudaEventRecord(start_event, streams_[stream_id]);
        
        // Launch kernel asynchronously
        kernel<<<grid, block, shared_mem, streams_[stream_id]>>>(args...);
        
        // Record end time
        cudaEvent_t end_event = createTimingEvent();
        cudaEventRecord(end_event, streams_[stream_id]);
        
        // Update statistics (asynchronously)
        std::thread([this, stream_id, start_event, end_event]() {
            cudaEventSynchronize(end_event);
            double elapsed = getElapsedTime(start_event, end_event);
            updateStreamStats(stream_id, 0, elapsed, true);
            
            cudaEventDestroy(start_event);
            cudaEventDestroy(end_event);
        }).detach();
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = "Kernel launch failed: " + std::string(e.what());
        throw;
    }
}

template<typename KernelFunc, typename... Args>
void AsyncGPUExecutionManager::launchKernelAsync(int stream_id, const GPU::LaunchConfig& config,
                                                KernelFunc kernel, Args... args) {
    launchKernelAsync(stream_id, config.grid, config.block, config.shared_mem, kernel, args...);
}

// Specialized execution patterns for quantum simulations
class QuantumSimulationExecutor {
public:
    QuantumSimulationExecutor(AsyncGPUExecutionManager& gpu_manager) 
        : gpu_manager_(gpu_manager) {}
    
    // Execute Hamiltonian assembly with optimal streaming
    void executeHamiltonianAssembly(
        const double* nodes, const int* elements,
        const double* potential, const double* effective_mass,
        cuDoubleComplex* H_matrix, cuDoubleComplex* M_matrix,
        int num_elements, int nodes_per_element, int total_nodes);
    
    // Execute eigenvalue computation with pipeline optimization
    void executeEigenvalueSolver(
        const cuDoubleComplex* H_matrix, const cuDoubleComplex* M_matrix,
        double* eigenvalues, cuDoubleComplex* eigenvectors,
        int matrix_size, int num_eigenvalues);
    
    // Execute self-consistent field iteration
    void executeSCFIteration(
        const cuDoubleComplex* eigenvectors, const double* eigenvalues,
        double* electron_density, double* potential,
        int num_nodes, int num_states, double temperature, double fermi_level);
    
private:
    AsyncGPUExecutionManager& gpu_manager_;
    
    // Optimal launch configurations for different kernel types
    GPU::LaunchConfig getHamiltonianConfig(int num_elements) const;
    GPU::LaunchConfig getEigensolverConfig(int matrix_size) const;
    GPU::LaunchConfig getDensityConfig(int num_nodes) const;
};

} // namespace QDSim
