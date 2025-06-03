#include "../backend/include/unified_parallel_manager.h"
#include "../backend/include/unified_memory_manager.h"
#include "../backend/include/async_gpu_execution_manager.h"
#include "../backend/include/thread_safe_resource_manager.h"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <random>
#include <atomic>

using namespace QDSim;

class StressTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize with maximum available resources
        UnifiedParallelManager::ParallelConfig config;
        config.mpi_ranks = 1;
        config.omp_threads_per_rank = std::thread::hardware_concurrency();
        config.cuda_devices_per_rank = 1;
        config.enable_gpu_direct = false;
        config.enable_numa_binding = true;
        config.gpu_memory_pool_size = 1024 * 1024 * 1024; // 1 GB
        
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        bool success = parallel_mgr.initialize(config);
        ASSERT_TRUE(success) << "Failed to initialize for stress testing";
        
        std::cout << "Stress testing with " << config.omp_threads_per_rank << " threads" << std::endl;
    }
    
    void TearDown() override {
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.finalize();
    }
};

// Long-running memory stress test
TEST_F(StressTest, LongRunningMemoryStress) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    const int test_duration_minutes = 5;
    const int max_concurrent_blocks = 1000;
    const size_t min_block_size = 1024;
    const size_t max_block_size = 16 * 1024 * 1024; // 16 MB
    
    std::cout << "Running memory stress test for " << test_duration_minutes << " minutes" << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::minutes(test_duration_minutes);
    
    std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> active_blocks;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> size_dist(min_block_size, max_block_size);
    std::uniform_int_distribution<int> action_dist(0, 2); // 0=allocate, 1=free, 2=copy
    
    std::atomic<long long> total_allocations{0};
    std::atomic<long long> total_deallocations{0};
    std::atomic<long long> total_copies{0};
    std::atomic<long long> allocation_failures{0};
    
    while (std::chrono::steady_clock::now() < end_time) {
        int action = action_dist(gen);
        
        if (action == 0 || active_blocks.size() < 10) { // Allocate
            if (active_blocks.size() < max_concurrent_blocks) {
                size_t size = size_dist(gen);
                
                try {
                    auto block = memory_mgr.allocate(size, 
                        "stress_" + std::to_string(total_allocations.load()),
                        UnifiedMemoryManager::MemoryType::HOST_PINNED);
                    
                    if (block) {
                        // Initialize memory with pattern
                        std::memset(block->ptr, 0xAA, size);
                        active_blocks.push_back(block);
                        total_allocations.fetch_add(1);
                    } else {
                        allocation_failures.fetch_add(1);
                    }
                } catch (const std::exception& e) {
                    allocation_failures.fetch_add(1);
                }
            }
            
        } else if (action == 1 && !active_blocks.empty()) { // Free
            std::uniform_int_distribution<size_t> block_dist(0, active_blocks.size() - 1);
            size_t idx = block_dist(gen);
            
            active_blocks.erase(active_blocks.begin() + idx);
            total_deallocations.fetch_add(1);
            
        } else if (action == 2 && active_blocks.size() >= 2) { // Copy
            std::uniform_int_distribution<size_t> block_dist(0, active_blocks.size() - 1);
            size_t src_idx = block_dist(gen);
            size_t dst_idx = block_dist(gen);
            
            if (src_idx != dst_idx) {
                auto& src_block = active_blocks[src_idx];
                auto& dst_block = active_blocks[dst_idx];
                
                size_t copy_size = std::min(src_block->size, dst_block->size);
                
                try {
                    memory_mgr.copyAsync(dst_block, src_block, copy_size);
                    total_copies.fetch_add(1);
                } catch (const std::exception& e) {
                    // Copy failure is acceptable under stress
                }
            }
        }
        
        // Print progress every 30 seconds
        static auto last_print = start_time;
        auto now = std::chrono::steady_clock::now();
        if (now - last_print > std::chrono::seconds(30)) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            std::cout << "Stress test progress: " << elapsed.count() << "s, "
                      << "Allocations: " << total_allocations.load() << ", "
                      << "Active blocks: " << active_blocks.size() << ", "
                      << "Failures: " << allocation_failures.load() << std::endl;
            last_print = now;
        }
    }
    
    // Final statistics
    std::cout << "\nMemory stress test completed:" << std::endl;
    std::cout << "  Total allocations: " << total_allocations.load() << std::endl;
    std::cout << "  Total deallocations: " << total_deallocations.load() << std::endl;
    std::cout << "  Total copies: " << total_copies.load() << std::endl;
    std::cout << "  Allocation failures: " << allocation_failures.load() << std::endl;
    std::cout << "  Final active blocks: " << active_blocks.size() << std::endl;
    
    // Verify system is still functional
    auto test_block = memory_mgr.allocate(1024, "final_test",
                                        UnifiedMemoryManager::MemoryType::HOST_PINNED);
    EXPECT_NE(test_block, nullptr) << "System not functional after stress test";
    
    // Check memory statistics
    auto stats = memory_mgr.getStats();
    std::cout << "  Memory manager stats:" << std::endl;
    std::cout << "    Active blocks: " << stats.active_blocks << std::endl;
    std::cout << "    Total allocated: " << (stats.total_allocated / (1024 * 1024)) << " MB" << std::endl;
    
    // Failure rate should be reasonable
    double failure_rate = static_cast<double>(allocation_failures.load()) / 
                         static_cast<double>(total_allocations.load() + allocation_failures.load());
    EXPECT_LT(failure_rate, 0.1) << "Excessive allocation failure rate: " << (failure_rate * 100) << "%";
}

// High-concurrency thread stress test
TEST_F(StressTest, HighConcurrencyThreadStress) {
    auto& parallel_mgr = UnifiedParallelManager::getInstance();
    
    const int num_work_batches = 100;
    const int work_items_per_batch = 10000;
    const int test_duration_minutes = 3;
    
    std::cout << "Running high-concurrency thread stress test for " << test_duration_minutes << " minutes" << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::minutes(test_duration_minutes);
    
    std::atomic<long long> total_work_items_processed{0};
    std::atomic<long long> total_batches_completed{0};
    std::atomic<long long> processing_errors{0};
    
    int batch_count = 0;
    
    while (std::chrono::steady_clock::now() < end_time) {
        // Create work items
        std::vector<int> work_items(work_items_per_batch);
        std::iota(work_items.begin(), work_items.end(), batch_count * work_items_per_batch);
        
        try {
            parallel_mgr.distributeWork<int>(work_items,
                [&total_work_items_processed, &processing_errors](
                    const int& item, const UnifiedParallelManager::ParallelContext& ctx) {
                    
                    try {
                        // Simulate computational work with varying complexity
                        double result = 0.0;
                        int complexity = (item % 100) + 10;
                        
                        for (int i = 0; i < complexity; ++i) {
                            result += std::sin(static_cast<double>(item + i)) * 
                                     std::cos(static_cast<double>(item - i));
                        }
                        
                        // Simulate memory access
                        volatile double temp = result;
                        (void)temp;
                        
                        total_work_items_processed.fetch_add(1);
                        
                    } catch (const std::exception& e) {
                        processing_errors.fetch_add(1);
                    }
                });
            
            total_batches_completed.fetch_add(1);
            
        } catch (const std::exception& e) {
            processing_errors.fetch_add(1);
        }
        
        batch_count++;
        
        // Print progress
        if (batch_count % 10 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time);
            std::cout << "Thread stress progress: " << elapsed.count() << "s, "
                      << "Batches: " << total_batches_completed.load() << ", "
                      << "Work items: " << total_work_items_processed.load() << ", "
                      << "Errors: " << processing_errors.load() << std::endl;
        }
    }
    
    std::cout << "\nThread stress test completed:" << std::endl;
    std::cout << "  Total batches: " << total_batches_completed.load() << std::endl;
    std::cout << "  Total work items: " << total_work_items_processed.load() << std::endl;
    std::cout << "  Processing errors: " << processing_errors.load() << std::endl;
    
    // Error rate should be minimal
    double error_rate = static_cast<double>(processing_errors.load()) / 
                       static_cast<double>(total_work_items_processed.load());
    EXPECT_LT(error_rate, 0.001) << "Excessive processing error rate: " << (error_rate * 100) << "%";
    
    // Should have processed a reasonable amount of work
    EXPECT_GT(total_work_items_processed.load(), 100000) << "Insufficient work processed";
}

// GPU stress test (if available)
TEST_F(StressTest, GPUStress) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA not available, skipping GPU stress test";
    }
    
    AsyncGPUExecutionManager gpu_manager;
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    const int test_duration_minutes = 3;
    const int max_concurrent_operations = 100;
    const size_t operation_size = 1024 * 1024; // 1 MB per operation
    
    std::cout << "Running GPU stress test for " << test_duration_minutes << " minutes" << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::minutes(test_duration_minutes);
    
    std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> gpu_blocks;
    std::atomic<long long> total_gpu_operations{0};
    std::atomic<long long> gpu_errors{0};
    
    // Pre-allocate GPU memory blocks
    for (int i = 0; i < max_concurrent_operations; ++i) {
        try {
            auto block = memory_mgr.allocate(operation_size, 
                "gpu_stress_" + std::to_string(i),
                UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
            
            if (block) {
                gpu_blocks.push_back(block);
            }
        } catch (const std::exception& e) {
            // Some allocation failures are expected under stress
        }
    }
    
    std::cout << "Allocated " << gpu_blocks.size() << " GPU memory blocks" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> block_dist(0, gpu_blocks.size() - 1);
    std::uniform_int_distribution<int> stream_dist(0, 7); // 8 streams
    std::uniform_int_distribution<int> value_dist(0, 255);
    
    while (std::chrono::steady_clock::now() < end_time && !gpu_blocks.empty()) {
        try {
            // Random GPU operation
            int block_idx = block_dist(gen);
            int stream_id = stream_dist(gen);
            int value = value_dist(gen);
            
            gpu_manager.memsetAsync(gpu_blocks[block_idx]->ptr, value, 
                                   operation_size, stream_id);
            
            total_gpu_operations.fetch_add(1);
            
            // Occasionally synchronize to prevent queue overflow
            if (total_gpu_operations.load() % 1000 == 0) {
                gpu_manager.synchronizeAll();
            }
            
        } catch (const std::exception& e) {
            gpu_errors.fetch_add(1);
        }
        
        // Print progress
        static auto last_print = start_time;
        auto now = std::chrono::steady_clock::now();
        if (now - last_print > std::chrono::seconds(30)) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            std::cout << "GPU stress progress: " << elapsed.count() << "s, "
                      << "Operations: " << total_gpu_operations.load() << ", "
                      << "Errors: " << gpu_errors.load() << std::endl;
            last_print = now;
        }
    }
    
    // Final synchronization
    gpu_manager.synchronizeAll();
    
    std::cout << "\nGPU stress test completed:" << std::endl;
    std::cout << "  Total GPU operations: " << total_gpu_operations.load() << std::endl;
    std::cout << "  GPU errors: " << gpu_errors.load() << std::endl;
    
    // Verify GPU is still functional
    if (!gpu_blocks.empty()) {
        cudaError_t err = cudaMemset(gpu_blocks[0]->ptr, 0xFF, 1024);
        EXPECT_EQ(err, cudaSuccess) << "GPU not functional after stress test";
    }
    
    // Error rate should be minimal
    if (total_gpu_operations.load() > 0) {
        double error_rate = static_cast<double>(gpu_errors.load()) / 
                           static_cast<double>(total_gpu_operations.load());
        EXPECT_LT(error_rate, 0.01) << "Excessive GPU error rate: " << (error_rate * 100) << "%";
    }
}

// Combined system stress test
TEST_F(StressTest, CombinedSystemStress) {
    const int test_duration_minutes = 2;
    
    std::cout << "Running combined system stress test for " << test_duration_minutes << " minutes" << std::endl;
    std::cout << "This test exercises all components simultaneously" << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::minutes(test_duration_minutes);
    
    std::atomic<bool> stop_flag{false};
    std::atomic<long long> total_operations{0};
    std::atomic<long long> total_errors{0};
    
    // Launch multiple stress threads
    std::vector<std::thread> stress_threads;
    
    // Memory stress thread
    stress_threads.emplace_back([&]() {
        auto& memory_mgr = UnifiedMemoryManager::getInstance();
        std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> blocks;
        std::random_device rd;
        std::mt19937 gen(rd());
        
        while (!stop_flag.load()) {
            try {
                if (blocks.size() < 100) {
                    auto block = memory_mgr.allocate(1024 * 1024, "combined_stress",
                                                   UnifiedMemoryManager::MemoryType::HOST_PINNED);
                    if (block) blocks.push_back(block);
                } else {
                    blocks.clear();
                }
                total_operations.fetch_add(1);
            } catch (...) {
                total_errors.fetch_add(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    // Parallel work stress thread
    stress_threads.emplace_back([&]() {
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        
        while (!stop_flag.load()) {
            try {
                std::vector<int> work(1000);
                std::iota(work.begin(), work.end(), 0);
                
                parallel_mgr.distributeWork<int>(work,
                    [](const int& item, const UnifiedParallelManager::ParallelContext& ctx) {
                        volatile double result = std::sin(item) * std::cos(item);
                        (void)result;
                    });
                
                total_operations.fetch_add(1);
            } catch (...) {
                total_errors.fetch_add(1);
            }
        }
    });
    
    // GPU stress thread (if available)
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        stress_threads.emplace_back([&]() {
            AsyncGPUExecutionManager gpu_manager;
            auto& memory_mgr = UnifiedMemoryManager::getInstance();
            
            auto gpu_block = memory_mgr.allocate(1024 * 1024, "gpu_combined_stress",
                                               UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
            
            while (!stop_flag.load() && gpu_block) {
                try {
                    gpu_manager.memsetAsync(gpu_block->ptr, rand() % 256, 1024 * 1024, 0);
                    total_operations.fetch_add(1);
                } catch (...) {
                    total_errors.fetch_add(1);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    
    // Wait for test duration
    std::this_thread::sleep_until(end_time);
    stop_flag.store(true);
    
    // Wait for all threads to complete
    for (auto& thread : stress_threads) {
        thread.join();
    }
    
    std::cout << "\nCombined system stress test completed:" << std::endl;
    std::cout << "  Total operations: " << total_operations.load() << std::endl;
    std::cout << "  Total errors: " << total_errors.load() << std::endl;
    
    // System should still be functional
    EXPECT_GT(total_operations.load(), 1000) << "Insufficient operations completed";
    
    double error_rate = static_cast<double>(total_errors.load()) / 
                       static_cast<double>(total_operations.load());
    EXPECT_LT(error_rate, 0.05) << "Excessive error rate under combined stress: " << (error_rate * 100) << "%";
    
    std::cout << "✓ System remained stable under combined stress" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running QDSim Stress Tests..." << std::endl;
    std::cout << "These tests will run for extended periods and stress all system components" << std::endl;
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    
    return RUN_ALL_TESTS();
}
