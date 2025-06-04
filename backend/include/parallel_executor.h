#pragma once

/**
 * @file parallel_executor.h
 * @brief Unified Parallel Execution Framework for QDSim
 * 
 * Hybrid MPI+OpenMP+CUDA architecture with integrated GPU memory management,
 * RAII-based thread-safe design, and performance optimizations including
 * kernel fusion and async execution.
 */

#include <memory>
#include <vector>
#include <future>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>

#ifdef QDSIM_USE_MPI
#include <mpi.h>
#endif

#ifdef QDSIM_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

#include "unified_memory.h"

namespace QDSim {
namespace Parallel {

/**
 * @brief Execution backends available
 */
enum class ExecutionBackend {
    CPU_SEQUENTIAL,     ///< Single-threaded CPU
    CPU_OPENMP,         ///< Multi-threaded CPU with OpenMP
    GPU_CUDA,           ///< NVIDIA GPU with CUDA
    HYBRID_CPU_GPU,     ///< Hybrid CPU+GPU execution
    DISTRIBUTED_MPI     ///< Distributed MPI execution
};

/**
 * @brief Task priority levels
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief Execution context for tasks
 */
struct ExecutionContext {
    ExecutionBackend backend;
    int device_id;
    int thread_count;
    size_t memory_limit;
    TaskPriority priority;
    std::chrono::milliseconds timeout;
    
    ExecutionContext(ExecutionBackend b = ExecutionBackend::CPU_OPENMP)
        : backend(b), device_id(0), thread_count(0), memory_limit(0),
          priority(TaskPriority::NORMAL), timeout(std::chrono::milliseconds(0)) {}
};

/**
 * @brief Base class for parallel tasks
 */
class ParallelTask {
public:
    virtual ~ParallelTask() = default;
    
    /**
     * @brief Execute the task
     */
    virtual void execute(const ExecutionContext& context) = 0;
    
    /**
     * @brief Get task name for debugging
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Estimate computational cost
     */
    virtual double estimated_cost() const { return 1.0; }
    
    /**
     * @brief Check if task can run on given backend
     */
    virtual bool supports_backend(ExecutionBackend backend) const = 0;
    
    /**
     * @brief Get required memory
     */
    virtual size_t required_memory() const { return 0; }
};

/**
 * @brief Thread-safe task queue with priority scheduling
 */
class TaskQueue {
private:
    struct QueuedTask {
        std::unique_ptr<ParallelTask> task;
        TaskPriority priority;
        std::chrono::steady_clock::time_point submit_time;
        std::promise<void> completion_promise;
        
        bool operator<(const QueuedTask& other) const {
            if (priority != other.priority) {
                return priority < other.priority; // Higher priority first
            }
            return submit_time > other.submit_time; // Earlier submission first
        }
    };
    
    std::priority_queue<QueuedTask> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_;
    
public:
    TaskQueue() : shutdown_(false) {}
    ~TaskQueue() { shutdown(); }
    
    /**
     * @brief Submit task for execution
     */
    std::future<void> submit(std::unique_ptr<ParallelTask> task, 
                            TaskPriority priority = TaskPriority::NORMAL);
    
    /**
     * @brief Get next task (blocking)
     */
    std::unique_ptr<ParallelTask> get_next_task();
    
    /**
     * @brief Check if queue is empty
     */
    bool empty() const;
    
    /**
     * @brief Get queue size
     */
    size_t size() const;
    
    /**
     * @brief Shutdown queue
     */
    void shutdown();
};

/**
 * @brief Worker thread for executing tasks
 */
class WorkerThread {
private:
    std::thread thread_;
    TaskQueue& queue_;
    ExecutionContext context_;
    std::atomic<bool> running_;
    std::atomic<size_t> tasks_completed_;
    
public:
    WorkerThread(TaskQueue& queue, ExecutionContext context);
    ~WorkerThread();
    
    /**
     * @brief Start worker thread
     */
    void start();
    
    /**
     * @brief Stop worker thread
     */
    void stop();
    
    /**
     * @brief Get statistics
     */
    size_t tasks_completed() const { return tasks_completed_.load(); }
    bool is_running() const { return running_.load(); }

private:
    void worker_loop();
};

/**
 * @brief Main parallel executor with hybrid architecture
 */
class ParallelExecutor {
private:
    // Thread pools for different backends
    std::vector<std::unique_ptr<WorkerThread>> cpu_workers_;
    std::vector<std::unique_ptr<WorkerThread>> gpu_workers_;
    
    // Task queues
    std::unique_ptr<TaskQueue> cpu_queue_;
    std::unique_ptr<TaskQueue> gpu_queue_;
    
    // MPI configuration
    int mpi_rank_;
    int mpi_size_;
    bool mpi_initialized_;
    
    // CUDA configuration
    std::vector<int> cuda_devices_;
    std::vector<cudaStream_t> cuda_streams_;
    std::vector<cublasHandle_t> cublas_handles_;
    
    // Performance monitoring
    std::atomic<size_t> total_tasks_executed_;
    std::atomic<double> total_execution_time_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Configuration
    size_t max_cpu_threads_;
    size_t max_gpu_threads_;
    bool auto_load_balance_;
    
    std::mutex executor_mutex_;
    
public:
    /**
     * @brief Constructor with configuration
     */
    explicit ParallelExecutor(size_t cpu_threads = 0, size_t gpu_threads = 1);
    
    /**
     * @brief Destructor with RAII cleanup
     */
    ~ParallelExecutor();
    
    // Non-copyable but movable
    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;
    ParallelExecutor(ParallelExecutor&&) = default;
    ParallelExecutor& operator=(ParallelExecutor&&) = default;
    
    /**
     * @brief Initialize executor
     */
    void initialize();
    
    /**
     * @brief Shutdown executor
     */
    void shutdown();
    
    /**
     * @brief Submit task for execution
     */
    std::future<void> submit(std::unique_ptr<ParallelTask> task,
                            ExecutionBackend preferred_backend = ExecutionBackend::CPU_OPENMP,
                            TaskPriority priority = TaskPriority::NORMAL);
    
    /**
     * @brief Submit multiple tasks
     */
    std::vector<std::future<void>> submit_batch(
        std::vector<std::unique_ptr<ParallelTask>> tasks,
        ExecutionBackend preferred_backend = ExecutionBackend::CPU_OPENMP);
    
    /**
     * @brief Wait for all tasks to complete
     */
    void wait_all();
    
    /**
     * @brief Execute task synchronously
     */
    void execute_sync(std::unique_ptr<ParallelTask> task,
                     ExecutionBackend backend = ExecutionBackend::CPU_OPENMP);
    
    /**
     * @brief Configuration
     */
    void set_cpu_threads(size_t count);
    void set_gpu_threads(size_t count);
    void enable_auto_load_balance(bool enable = true);
    void set_cuda_devices(const std::vector<int>& devices);
    
    /**
     * @brief Performance monitoring
     */
    size_t total_tasks_executed() const { return total_tasks_executed_.load(); }
    double average_execution_time() const;
    double throughput() const; // tasks per second
    void print_performance_report() const;
    
    /**
     * @brief MPI support
     */
    bool is_mpi_initialized() const { return mpi_initialized_; }
    int mpi_rank() const { return mpi_rank_; }
    int mpi_size() const { return mpi_size_; }
    
    /**
     * @brief CUDA support
     */
    bool is_cuda_available() const;
    size_t cuda_device_count() const { return cuda_devices_.size(); }
    void synchronize_cuda() const;

private:
    void initialize_mpi();
    void initialize_cuda();
    void cleanup_mpi();
    void cleanup_cuda();
    
    ExecutionBackend select_optimal_backend(const ParallelTask& task) const;
    void balance_load();
};

/**
 * @brief Quantum-specific parallel tasks
 */
namespace QuantumTasks {

/**
 * @brief Matrix-vector multiplication task
 */
class MatrixVectorTask : public ParallelTask {
private:
    Memory::UnifiedMemoryBlock<std::complex<double>>& matrix_;
    Memory::UnifiedMemoryBlock<std::complex<double>>& vector_;
    Memory::UnifiedMemoryBlock<std::complex<double>>& result_;
    size_t rows_, cols_;
    
public:
    MatrixVectorTask(Memory::UnifiedMemoryBlock<std::complex<double>>& matrix,
                    Memory::UnifiedMemoryBlock<std::complex<double>>& vector,
                    Memory::UnifiedMemoryBlock<std::complex<double>>& result,
                    size_t rows, size_t cols);
    
    void execute(const ExecutionContext& context) override;
    std::string name() const override { return "MatrixVector"; }
    bool supports_backend(ExecutionBackend backend) const override;
    size_t required_memory() const override;
};

/**
 * @brief FFT computation task
 */
class FFTTask : public ParallelTask {
private:
    Memory::UnifiedMemoryBlock<std::complex<double>>& data_;
    size_t size_;
    bool forward_;
    
public:
    FFTTask(Memory::UnifiedMemoryBlock<std::complex<double>>& data,
           size_t size, bool forward = true);
    
    void execute(const ExecutionContext& context) override;
    std::string name() const override { return "FFT"; }
    bool supports_backend(ExecutionBackend backend) const override;
    double estimated_cost() const override;
};

/**
 * @brief Eigenvalue computation task
 */
class EigenvalueTask : public ParallelTask {
private:
    Memory::UnifiedMemoryBlock<std::complex<double>>& matrix_;
    Memory::UnifiedMemoryBlock<std::complex<double>>& eigenvalues_;
    Memory::UnifiedMemoryBlock<std::complex<double>>& eigenvectors_;
    size_t size_;
    
public:
    EigenvalueTask(Memory::UnifiedMemoryBlock<std::complex<double>>& matrix,
                  Memory::UnifiedMemoryBlock<std::complex<double>>& eigenvalues,
                  Memory::UnifiedMemoryBlock<std::complex<double>>& eigenvectors,
                  size_t size);
    
    void execute(const ExecutionContext& context) override;
    std::string name() const override { return "Eigenvalue"; }
    bool supports_backend(ExecutionBackend backend) const override;
    double estimated_cost() const override;
    size_t required_memory() const override;
};

} // namespace QuantumTasks

/**
 * @brief Performance profiler for parallel execution
 */
class ParallelProfiler {
public:
    struct ProfileData {
        size_t cpu_tasks;
        size_t gpu_tasks;
        double cpu_time;
        double gpu_time;
        double load_balance_efficiency;
        size_t memory_transfers;
        double transfer_time;
    };
    
    static void start_profiling();
    static void stop_profiling();
    static ProfileData get_profile_data();
    static void print_profile_report();
};

} // namespace Parallel
} // namespace QDSim

// Convenience macros
#define QDSIM_PARALLEL_SCOPE() QDSim::Parallel::ParallelExecutor executor
#define QDSIM_SUBMIT_TASK(executor, task) executor.submit(std::make_unique<decltype(task)>(task))
#define QDSIM_EXECUTE_SYNC(executor, task) executor.execute_sync(std::make_unique<decltype(task)>(task))
