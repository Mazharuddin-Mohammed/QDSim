#include "async_gpu_execution_manager.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <chrono>

namespace QDSim {

AsyncGPUExecutionManager::AsyncGPUExecutionManager() {
    try {
        initializeStreams();
        initializeEvents();
        
        // Initialize stream statistics
        stream_stats_.resize(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; ++i) {
            stream_stats_[i].stream_id = i;
        }
        
        std::cout << "AsyncGPUExecutionManager initialized with " << NUM_STREAMS 
                  << " streams and " << NUM_EVENTS << " events" << std::endl;
                  
    } catch (const std::exception& e) {
        cleanup();
        throw std::runtime_error("Failed to initialize AsyncGPUExecutionManager: " + std::string(e.what()));
    }
}

AsyncGPUExecutionManager::~AsyncGPUExecutionManager() {
    cleanup();
}

void AsyncGPUExecutionManager::initializeStreams() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaError_t err = cudaStreamCreate(&streams_[i]);
        if (err != cudaSuccess) {
            // Cleanup already created streams
            for (int j = 0; j < i; ++j) {
                cudaStreamDestroy(streams_[j]);
            }
            throw std::runtime_error("Failed to create CUDA stream " + std::to_string(i) + 
                                   ": " + cudaGetErrorString(err));
        }
    }
}

void AsyncGPUExecutionManager::initializeEvents() {
    for (int i = 0; i < NUM_EVENTS; ++i) {
        cudaError_t err = cudaEventCreate(&events_[i]);
        if (err != cudaSuccess) {
            // Cleanup already created events
            for (int j = 0; j < i; ++j) {
                cudaEventDestroy(events_[j]);
            }
            throw std::runtime_error("Failed to create CUDA event " + std::to_string(i) + 
                                   ": " + cudaGetErrorString(err));
        }
    }
}

void AsyncGPUExecutionManager::cleanup() {
    try {
        // Synchronize all streams before cleanup
        synchronizeAll();
        
        // Destroy streams
        for (int i = 0; i < NUM_STREAMS; ++i) {
            if (streams_[i]) {
                cudaStreamDestroy(streams_[i]);
                streams_[i] = nullptr;
            }
        }
        
        // Destroy events
        for (int i = 0; i < NUM_EVENTS; ++i) {
            if (events_[i]) {
                cudaEventDestroy(events_[i]);
                events_[i] = nullptr;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during AsyncGPUExecutionManager cleanup: " << e.what() << std::endl;
    }
}

void AsyncGPUExecutionManager::memcpyAsync(void* dst, const void* src, size_t size, 
                                          cudaMemcpyKind kind, int stream_id) {
    stream_id = stream_id % NUM_STREAMS;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cudaError_t err = cudaMemcpyAsync(dst, src, size, kind, streams_[stream_id]);
    if (err != cudaSuccess) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = "Async memcpy failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(last_error_);
    }
    
    // Update statistics asynchronously
    std::thread([this, stream_id, size, start_time]() {
        cudaStreamSynchronize(streams_[stream_id]);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double elapsed_ms = duration.count() / 1000.0;
        
        updateStreamStats(stream_id, size, elapsed_ms, false);
    }).detach();
}

void AsyncGPUExecutionManager::memsetAsync(void* ptr, int value, size_t size, int stream_id) {
    stream_id = stream_id % NUM_STREAMS;
    
    cudaError_t err = cudaMemsetAsync(ptr, value, size, streams_[stream_id]);
    if (err != cudaSuccess) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = "Async memset failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(last_error_);
    }
}

void AsyncGPUExecutionManager::synchronizeStream(int stream_id) {
    stream_id = stream_id % NUM_STREAMS;
    
    cudaError_t err = cudaStreamSynchronize(streams_[stream_id]);
    if (err != cudaSuccess) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = "Stream synchronization failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(last_error_);
    }
}

void AsyncGPUExecutionManager::synchronizeAll() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        synchronizeStream(i);
    }
}

int AsyncGPUExecutionManager::recordEvent(int stream_id) {
    stream_id = stream_id % NUM_STREAMS;
    int event_id = getNextEventId();
    
    cudaError_t err = cudaEventRecord(events_[event_id], streams_[stream_id]);
    if (err != cudaSuccess) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = "Event recording failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(last_error_);
    }
    
    return event_id;
}

void AsyncGPUExecutionManager::waitForEvent(int event_id, int stream_id) {
    stream_id = stream_id % NUM_STREAMS;
    event_id = event_id % NUM_EVENTS;
    
    cudaError_t err = cudaStreamWaitEvent(streams_[stream_id], events_[event_id], 0);
    if (err != cudaSuccess) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = "Stream wait for event failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(last_error_);
    }
}

bool AsyncGPUExecutionManager::isEventComplete(int event_id) {
    event_id = event_id % NUM_EVENTS;
    
    cudaError_t err = cudaEventQuery(events_[event_id]);
    return (err == cudaSuccess);
}

void AsyncGPUExecutionManager::executePipeline(const std::vector<ComputeTask>& compute_tasks,
                                              const std::vector<CommTask>& comm_tasks) {
    
    int num_compute = compute_tasks.size();
    int num_comm = comm_tasks.size();
    
    std::vector<int> compute_events(num_compute, -1);
    
    // Launch compute tasks
    for (int i = 0; i < num_compute; ++i) {
        int compute_stream = i % NUM_STREAMS;
        
        // Execute compute task
        compute_tasks[i].task(streams_[compute_stream]);
        
        // Record completion event
        compute_events[i] = recordEvent(compute_stream);
    }
    
    // Launch communication tasks with dependencies
    for (int i = 0; i < num_comm; ++i) {
        int comm_stream = (i + NUM_STREAMS/2) % NUM_STREAMS;
        
        // Wait for dependency if specified
        if (comm_tasks[i].dependency_event >= 0 && 
            comm_tasks[i].dependency_event < compute_events.size()) {
            waitForEvent(compute_events[comm_tasks[i].dependency_event], comm_stream);
        }
        
        // Execute communication task
        comm_tasks[i].task(streams_[comm_stream]);
    }
    
    // Synchronize all streams
    synchronizeAll();
}

cudaStream_t AsyncGPUExecutionManager::getStream(int stream_id) const {
    return streams_[stream_id % NUM_STREAMS];
}

cudaEvent_t AsyncGPUExecutionManager::getEvent(int event_id) const {
    return events_[event_id % NUM_EVENTS];
}

int AsyncGPUExecutionManager::getOptimalStreamCount() const {
    // Query device properties to determine optimal stream count
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Use number of SMs as a heuristic for optimal stream count
    return std::min(NUM_STREAMS, prop.multiProcessorCount);
}

void AsyncGPUExecutionManager::checkErrors() const {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error detected: " + std::string(cudaGetErrorString(err)));
    }
}

std::string AsyncGPUExecutionManager::getLastError() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

std::vector<AsyncGPUExecutionManager::StreamStats> AsyncGPUExecutionManager::getStreamStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stream_stats_;
}

void AsyncGPUExecutionManager::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    for (auto& stats : stream_stats_) {
        stats.kernels_launched = 0;
        stats.bytes_transferred = 0;
        stats.total_compute_time = 0.0;
        stats.total_transfer_time = 0.0;
        stats.utilization_percent = 0.0;
    }
}

void AsyncGPUExecutionManager::printStats() const {
    auto stats = getStreamStats();
    
    std::cout << "=== AsyncGPUExecutionManager Statistics ===" << std::endl;
    std::cout << "Stream | Kernels | Transfers (MB) | Compute (ms) | Transfer (ms) | Utilization (%)" << std::endl;
    std::cout << "-------|---------|----------------|--------------|---------------|----------------" << std::endl;
    
    for (const auto& stat : stats) {
        std::cout << std::setw(6) << stat.stream_id << " | "
                  << std::setw(7) << stat.kernels_launched << " | "
                  << std::setw(14) << std::fixed << std::setprecision(2) 
                  << (stat.bytes_transferred / (1024.0 * 1024.0)) << " | "
                  << std::setw(12) << stat.total_compute_time << " | "
                  << std::setw(13) << stat.total_transfer_time << " | "
                  << std::setw(14) << stat.utilization_percent << std::endl;
    }
}

int AsyncGPUExecutionManager::getNextEventId() {
    return next_event_id_.fetch_add(1) % NUM_EVENTS;
}

void AsyncGPUExecutionManager::updateStreamStats(int stream_id, size_t bytes, 
                                                double time, bool is_compute) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (stream_id >= 0 && stream_id < NUM_STREAMS) {
        auto& stats = stream_stats_[stream_id];
        
        if (is_compute) {
            stats.kernels_launched++;
            stats.total_compute_time += time;
        } else {
            stats.bytes_transferred += bytes;
            stats.total_transfer_time += time;
        }
        
        // Calculate utilization (simplified)
        double total_time = stats.total_compute_time + stats.total_transfer_time;
        if (total_time > 0) {
            stats.utilization_percent = (stats.total_compute_time / total_time) * 100.0;
        }
    }
}

cudaEvent_t AsyncGPUExecutionManager::createTimingEvent() {
    cudaEvent_t event;
    cudaEventCreate(&event);
    return event;
}

double AsyncGPUExecutionManager::getElapsedTime(cudaEvent_t start, cudaEvent_t end) const {
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    return static_cast<double>(elapsed_ms);
}

} // namespace QDSim
