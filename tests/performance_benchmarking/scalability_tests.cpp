#include "../../backend/include/unified_parallel_manager.h"
#include "../../backend/include/unified_memory_manager.h"
#include "../../backend/include/async_gpu_execution_manager.h"
#include "../../backend/include/fused_gpu_kernels.cuh"
#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace QDSim;

class ScalabilityTest : public ::testing::Test {
protected:
    struct BenchmarkResult {
        int num_threads;
        int num_processes;
        int num_gpu_devices;
        size_t problem_size;
        double execution_time_ms;
        double throughput_ops_per_sec;
        double parallel_efficiency;
        double speedup_factor;
        double memory_bandwidth_gb_s;
        std::string test_name;
    };
    
    std::vector<BenchmarkResult> results_;
    double serial_baseline_time_ = 0.0;
    
    void SetUp() override {
        // Initialize with minimal configuration for baseline
        UnifiedParallelManager::ParallelConfig config;
        config.mpi_ranks = 1;
        config.omp_threads_per_rank = 1;
        config.cuda_devices_per_rank = 0;
        
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.initialize(config);
    }
    
    void TearDown() override {
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.finalize();
        
        // Generate performance report
        generatePerformanceReport();
    }
    
    double measureExecutionTime(std::function<void()> operation) {
        auto start = std::chrono::high_resolution_clock::now();
        operation();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
    
    void generatePerformanceReport() {
        std::ofstream report("scalability_performance_report.csv");
        report << "Test,Threads,Processes,GPUs,ProblemSize,TimeMS,ThroughputOPS,Efficiency%,Speedup,BandwidthGBs\n";
        
        for (const auto& result : results_) {
            report << result.test_name << ","
                   << result.num_threads << ","
                   << result.num_processes << ","
                   << result.num_gpu_devices << ","
                   << result.problem_size << ","
                   << std::fixed << std::setprecision(3) << result.execution_time_ms << ","
                   << std::scientific << std::setprecision(2) << result.throughput_ops_per_sec << ","
                   << std::fixed << std::setprecision(1) << result.parallel_efficiency << ","
                   << std::setprecision(2) << result.speedup_factor << ","
                   << std::setprecision(1) << result.memory_bandwidth_gb_s << "\n";
        }
        
        std::cout << "\nPerformance report saved to: scalability_performance_report.csv" << std::endl;
    }
};

// Test CPU thread scalability
TEST_F(ScalabilityTest, CPUThreadScalability) {
    const size_t problem_size = 10000000; // 10M elements
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32};
    
    // Create test data
    std::vector<double> input_data(problem_size);
    std::vector<double> output_data(problem_size);
    std::iota(input_data.begin(), input_data.end(), 1.0);
    
    for (int num_threads : thread_counts) {
        // Reconfigure parallel manager
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.finalize();
        
        UnifiedParallelManager::ParallelConfig config;
        config.mpi_ranks = 1;
        config.omp_threads_per_rank = num_threads;
        config.cuda_devices_per_rank = 0;
        parallel_mgr.initialize(config);
        
        // Benchmark parallel computation
        auto computation = [&]() {
            #pragma omp parallel for
            for (size_t i = 0; i < problem_size; ++i) {
                // Simulate computational work
                output_data[i] = std::sin(input_data[i]) * std::cos(input_data[i]) + 
                                std::sqrt(std::abs(input_data[i]));
            }
        };
        
        double execution_time = measureExecutionTime(computation);
        
        // Calculate metrics
        double throughput = problem_size / (execution_time / 1000.0); // ops per second
        double speedup = (num_threads == 1) ? 1.0 : serial_baseline_time_ / execution_time;
        double efficiency = (speedup / num_threads) * 100.0;
        
        if (num_threads == 1) {
            serial_baseline_time_ = execution_time;
            speedup = 1.0;
            efficiency = 100.0;
        }
        
        BenchmarkResult result;
        result.test_name = "CPU_Thread_Scalability";
        result.num_threads = num_threads;
        result.num_processes = 1;
        result.num_gpu_devices = 0;
        result.problem_size = problem_size;
        result.execution_time_ms = execution_time;
        result.throughput_ops_per_sec = throughput;
        result.parallel_efficiency = efficiency;
        result.speedup_factor = speedup;
        result.memory_bandwidth_gb_s = (problem_size * sizeof(double) * 2) / (execution_time / 1000.0) / 1e9;
        
        results_.push_back(result);
        
        std::cout << "Threads: " << num_threads 
                  << ", Time: " << std::fixed << std::setprecision(2) << execution_time << " ms"
                  << ", Speedup: " << speedup << "x"
                  << ", Efficiency: " << efficiency << "%" << std::endl;
        
        // Verify correctness
        for (size_t i = 0; i < std::min(size_t(100), problem_size); ++i) {
            double expected = std::sin(input_data[i]) * std::cos(input_data[i]) + 
                             std::sqrt(std::abs(input_data[i]));
            EXPECT_NEAR(output_data[i], expected, 1e-10) << "Computation error at index " << i;
        }
    }
    
    // Check that we achieve reasonable speedup
    auto best_result = *std::max_element(results_.begin(), results_.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.speedup_factor < b.speedup_factor;
        });
    
    EXPECT_GT(best_result.speedup_factor, 2.0) << "Poor thread scalability detected";
}

// Test memory bandwidth scalability
TEST_F(ScalabilityTest, MemoryBandwidthScalability) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    std::vector<size_t> data_sizes = {
        1024 * 1024,      // 1 MB
        16 * 1024 * 1024, // 16 MB
        64 * 1024 * 1024, // 64 MB
        256 * 1024 * 1024 // 256 MB
    };
    
    for (size_t data_size : data_sizes) {
        // Allocate memory blocks
        auto src_block = memory_mgr.allocate(data_size, "bandwidth_src",
                                           UnifiedMemoryManager::MemoryType::HOST_PINNED);
        auto dst_block = memory_mgr.allocate(data_size, "bandwidth_dst",
                                           UnifiedMemoryManager::MemoryType::HOST_PINNED);
        
        ASSERT_NE(src_block, nullptr);
        ASSERT_NE(dst_block, nullptr);
        
        // Initialize source data
        std::memset(src_block->ptr, 0xAA, data_size);
        
        // Benchmark memory copy
        const int num_iterations = 100;
        auto copy_operation = [&]() {
            for (int i = 0; i < num_iterations; ++i) {
                std::memcpy(dst_block->ptr, src_block->ptr, data_size);
            }
        };
        
        double execution_time = measureExecutionTime(copy_operation);
        
        // Calculate bandwidth
        double total_bytes = data_size * num_iterations * 2; // Read + Write
        double bandwidth_gb_s = total_bytes / (execution_time / 1000.0) / 1e9;
        
        BenchmarkResult result;
        result.test_name = "Memory_Bandwidth";
        result.num_threads = 1;
        result.num_processes = 1;
        result.num_gpu_devices = 0;
        result.problem_size = data_size;
        result.execution_time_ms = execution_time;
        result.throughput_ops_per_sec = num_iterations / (execution_time / 1000.0);
        result.parallel_efficiency = 100.0;
        result.speedup_factor = 1.0;
        result.memory_bandwidth_gb_s = bandwidth_gb_s;
        
        results_.push_back(result);
        
        std::cout << "Data size: " << (data_size / (1024 * 1024)) << " MB"
                  << ", Bandwidth: " << std::fixed << std::setprecision(1) << bandwidth_gb_s << " GB/s"
                  << std::endl;
        
        // Verify data integrity
        EXPECT_EQ(std::memcmp(src_block->ptr, dst_block->ptr, data_size), 0)
            << "Memory copy verification failed for size " << data_size;
    }
}

// Test GPU scalability (if available)
TEST_F(ScalabilityTest, GPUScalability) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA not available, skipping GPU scalability test";
    }
    
    AsyncGPUExecutionManager gpu_manager;
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    std::vector<int> stream_counts = {1, 2, 4, 8};
    const size_t problem_size = 1024 * 1024; // 1M elements
    const size_t data_size = problem_size * sizeof(float);
    
    for (int num_streams : stream_counts) {
        // Allocate GPU memory
        auto gpu_data = memory_mgr.allocate(data_size, "gpu_scalability",
                                          UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        ASSERT_NE(gpu_data, nullptr);
        
        // Benchmark GPU operations
        auto gpu_operation = [&]() {
            for (int s = 0; s < num_streams; ++s) {
                // Launch memory operations on different streams
                gpu_manager.memsetAsync(gpu_data->ptr, s % 256, data_size / num_streams, s);
            }
            gpu_manager.synchronizeAll();
        };
        
        double execution_time = measureExecutionTime(gpu_operation);
        
        // Calculate metrics
        double throughput = (data_size * num_streams) / (execution_time / 1000.0);
        double bandwidth_gb_s = throughput / 1e9;
        
        BenchmarkResult result;
        result.test_name = "GPU_Stream_Scalability";
        result.num_threads = 1;
        result.num_processes = 1;
        result.num_gpu_devices = 1;
        result.problem_size = problem_size;
        result.execution_time_ms = execution_time;
        result.throughput_ops_per_sec = throughput;
        result.parallel_efficiency = (num_streams == 1) ? 100.0 : 
                                    (results_[0].execution_time_ms / execution_time / num_streams) * 100.0;
        result.speedup_factor = (num_streams == 1) ? 1.0 : 
                               results_[0].execution_time_ms / execution_time;
        result.memory_bandwidth_gb_s = bandwidth_gb_s;
        
        results_.push_back(result);
        
        std::cout << "GPU Streams: " << num_streams
                  << ", Time: " << std::fixed << std::setprecision(2) << execution_time << " ms"
                  << ", Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    }
}

// Test work distribution scalability
TEST_F(ScalabilityTest, WorkDistributionScalability) {
    auto& parallel_mgr = UnifiedParallelManager::getInstance();
    
    struct WorkItem {
        int id;
        std::vector<double> data;
        double result;
        
        WorkItem(int i, size_t size) : id(i), data(size, i * 0.1), result(0.0) {}
    };
    
    std::vector<size_t> work_sizes = {1000, 10000, 100000, 1000000};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    for (size_t work_size : work_sizes) {
        for (int num_threads : thread_counts) {
            // Reconfigure parallel manager
            parallel_mgr.finalize();
            
            UnifiedParallelManager::ParallelConfig config;
            config.mpi_ranks = 1;
            config.omp_threads_per_rank = num_threads;
            config.cuda_devices_per_rank = 0;
            parallel_mgr.initialize(config);
            
            // Create work items
            std::vector<WorkItem> work_items;
            for (size_t i = 0; i < work_size; ++i) {
                work_items.emplace_back(i, 100); // 100 elements per work item
            }
            
            // Benchmark work distribution
            auto work_operation = [&]() {
                parallel_mgr.distributeWork<WorkItem>(work_items,
                    [](const WorkItem& item, const UnifiedParallelManager::ParallelContext& ctx) {
                        // Simulate computational work
                        double sum = 0.0;
                        for (double val : item.data) {
                            sum += std::sin(val) * std::cos(val);
                        }
                        const_cast<WorkItem&>(item).result = sum;
                    });
            };
            
            double execution_time = measureExecutionTime(work_operation);
            
            // Calculate metrics
            double throughput = work_size / (execution_time / 1000.0);
            double speedup = 1.0;
            double efficiency = 100.0;
            
            // Find baseline for this work size
            auto baseline_it = std::find_if(results_.begin(), results_.end(),
                [work_size](const BenchmarkResult& r) {
                    return r.test_name == "Work_Distribution" && 
                           r.problem_size == work_size && 
                           r.num_threads == 1;
                });
            
            if (baseline_it != results_.end()) {
                speedup = baseline_it->execution_time_ms / execution_time;
                efficiency = (speedup / num_threads) * 100.0;
            }
            
            BenchmarkResult result;
            result.test_name = "Work_Distribution";
            result.num_threads = num_threads;
            result.num_processes = 1;
            result.num_gpu_devices = 0;
            result.problem_size = work_size;
            result.execution_time_ms = execution_time;
            result.throughput_ops_per_sec = throughput;
            result.parallel_efficiency = efficiency;
            result.speedup_factor = speedup;
            result.memory_bandwidth_gb_s = 0.0; // Not applicable
            
            results_.push_back(result);
            
            std::cout << "Work items: " << work_size 
                      << ", Threads: " << num_threads
                      << ", Time: " << std::fixed << std::setprecision(2) << execution_time << " ms"
                      << ", Efficiency: " << efficiency << "%" << std::endl;
            
            // Verify all work items were processed
            for (const auto& item : work_items) {
                EXPECT_NE(item.result, 0.0) << "Work item " << item.id << " was not processed";
            }
        }
    }
}

// Test strong scaling (fixed problem size, increasing resources)
TEST_F(ScalabilityTest, StrongScaling) {
    const size_t fixed_problem_size = 10000000; // 10M operations
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    std::vector<double> data(fixed_problem_size);
    std::iota(data.begin(), data.end(), 1.0);
    
    double baseline_time = 0.0;
    
    for (int num_threads : thread_counts) {
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.finalize();
        
        UnifiedParallelManager::ParallelConfig config;
        config.mpi_ranks = 1;
        config.omp_threads_per_rank = num_threads;
        config.cuda_devices_per_rank = 0;
        parallel_mgr.initialize(config);
        
        auto computation = [&]() {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (size_t i = 0; i < fixed_problem_size; ++i) {
                sum += std::sin(data[i]) * std::cos(data[i]);
            }
            // Prevent optimization
            volatile double result = sum;
            (void)result;
        };
        
        double execution_time = measureExecutionTime(computation);
        
        if (num_threads == 1) {
            baseline_time = execution_time;
        }
        
        double speedup = baseline_time / execution_time;
        double efficiency = (speedup / num_threads) * 100.0;
        
        BenchmarkResult result;
        result.test_name = "Strong_Scaling";
        result.num_threads = num_threads;
        result.num_processes = 1;
        result.num_gpu_devices = 0;
        result.problem_size = fixed_problem_size;
        result.execution_time_ms = execution_time;
        result.throughput_ops_per_sec = fixed_problem_size / (execution_time / 1000.0);
        result.parallel_efficiency = efficiency;
        result.speedup_factor = speedup;
        result.memory_bandwidth_gb_s = 0.0;
        
        results_.push_back(result);
        
        std::cout << "Strong scaling - Threads: " << num_threads
                  << ", Speedup: " << std::fixed << std::setprecision(2) << speedup << "x"
                  << ", Efficiency: " << efficiency << "%" << std::endl;
    }
    
    // Check for reasonable strong scaling
    auto best_efficiency = std::max_element(results_.end() - thread_counts.size(), results_.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.parallel_efficiency < b.parallel_efficiency;
        });
    
    EXPECT_GT(best_efficiency->parallel_efficiency, 50.0) << "Poor strong scaling efficiency";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running Scalability Performance Tests..." << std::endl;
    std::cout << "Results will be saved to: scalability_performance_report.csv" << std::endl;
    
    return RUN_ALL_TESTS();
}

// Test weak scaling (increasing problem size with resources)
TEST_F(ScalabilityTest, WeakScaling) {
    const size_t base_problem_size = 1000000; // 1M operations per thread
    std::vector<int> thread_counts = {1, 2, 4, 8};

    double baseline_time = 0.0;

    for (int num_threads : thread_counts) {
        size_t problem_size = base_problem_size * num_threads;

        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.finalize();

        UnifiedParallelManager::ParallelConfig config;
        config.mpi_ranks = 1;
        config.omp_threads_per_rank = num_threads;
        config.cuda_devices_per_rank = 0;
        parallel_mgr.initialize(config);

        std::vector<double> data(problem_size);
        std::iota(data.begin(), data.end(), 1.0);

        auto computation = [&]() {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (size_t i = 0; i < problem_size; ++i) {
                sum += std::sin(data[i]) * std::cos(data[i]);
            }
            volatile double result = sum;
            (void)result;
        };

        double execution_time = measureExecutionTime(computation);

        if (num_threads == 1) {
            baseline_time = execution_time;
        }

        double efficiency = (baseline_time / execution_time) * 100.0;

        BenchmarkResult result;
        result.test_name = "Weak_Scaling";
        result.num_threads = num_threads;
        result.num_processes = 1;
        result.num_gpu_devices = 0;
        result.problem_size = problem_size;
        result.execution_time_ms = execution_time;
        result.throughput_ops_per_sec = problem_size / (execution_time / 1000.0);
        result.parallel_efficiency = efficiency;
        result.speedup_factor = baseline_time / execution_time;
        result.memory_bandwidth_gb_s = 0.0;

        results_.push_back(result);

        std::cout << "Weak scaling - Threads: " << num_threads
                  << ", Problem size: " << problem_size
                  << ", Time: " << std::fixed << std::setprecision(2) << execution_time << " ms"
                  << ", Efficiency: " << efficiency << "%" << std::endl;
    }

    // Check for good weak scaling (time should remain roughly constant)
    auto worst_efficiency = std::min_element(results_.end() - thread_counts.size(), results_.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.parallel_efficiency < b.parallel_efficiency;
        });

    EXPECT_GT(worst_efficiency->parallel_efficiency, 70.0) << "Poor weak scaling efficiency";
}
