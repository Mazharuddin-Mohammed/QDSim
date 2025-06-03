#include "../../backend/include/unified_parallel_manager.h"
#include "../../backend/include/unified_memory_manager.h"
#include "../../backend/include/thread_safe_resource_manager.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>

using namespace QDSim;

class ThreadSanitizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize parallel manager for testing
        UnifiedParallelManager::ParallelConfig config;
        config.mpi_ranks = 1;
        config.omp_threads_per_rank = 8;
        config.cuda_devices_per_rank = 1;
        config.enable_gpu_direct = false;
        config.enable_numa_binding = false;
        
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.initialize(config);
    }
    
    void TearDown() override {
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.finalize();
    }
};

// Test for race conditions in memory allocation
TEST_F(ThreadSanitizerTest, MemoryAllocationRaceConditions) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    const int num_threads = 16;
    const int allocations_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> successful_allocations{0};
    std::atomic<int> failed_allocations{0};
    
    // Stress test memory allocation from multiple threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&memory_mgr, &successful_allocations, &failed_allocations, 
                             allocations_per_thread, t]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> size_dist(1024, 1024 * 1024); // 1KB to 1MB
            
            for (int i = 0; i < allocations_per_thread; ++i) {
                try {
                    size_t size = size_dist(gen);
                    auto block = memory_mgr.allocate(size, 
                        "thread_" + std::to_string(t) + "_alloc_" + std::to_string(i),
                        UnifiedMemoryManager::MemoryType::HOST_PINNED);
                    
                    ASSERT_NE(block, nullptr);
                    ASSERT_NE(block->ptr, nullptr);
                    ASSERT_GE(block->size, size);
                    
                    // Write to memory to ensure it's valid
                    std::memset(block->ptr, 0xAA, std::min(size, size_t(4096)));
                    
                    successful_allocations.fetch_add(1);
                    
                    // Random delay to increase chance of race conditions
                    if (i % 10 == 0) {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                    
                } catch (const std::exception& e) {
                    failed_allocations.fetch_add(1);
                    FAIL() << "Memory allocation failed: " << e.what();
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successful_allocations.load(), num_threads * allocations_per_thread);
    EXPECT_EQ(failed_allocations.load(), 0);
    
    // Verify memory manager statistics are consistent
    auto stats = memory_mgr.getStats();
    EXPECT_GT(stats.total_allocated, 0);
    EXPECT_GT(stats.active_blocks, 0);
}

// Test for deadlocks in resource manager
TEST_F(ThreadSanitizerTest, ResourceManagerDeadlockDetection) {
    struct TestResource {
        int id;
        std::mutex mutex;
        bool in_use = false;
        
        TestResource() : id(rand() % 10000) {}
    };
    
    auto factory = []() { return std::make_shared<TestResource>(); };
    auto deleter = [](std::shared_ptr<TestResource> res) { 
        std::lock_guard<std::mutex> lock(res->mutex);
        res->in_use = false;
    };
    
    ThreadSafeResourceManager<TestResource> resource_mgr(factory, deleter, 4);
    
    const int num_threads = 20;
    std::vector<std::thread> threads;
    std::atomic<int> successful_acquisitions{0};
    std::atomic<int> deadlock_timeouts{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&resource_mgr, &successful_acquisitions, &deadlock_timeouts, t]() {
            try {
                // Use timeout to detect potential deadlocks
                auto start_time = std::chrono::steady_clock::now();
                auto guard = resource_mgr.acquire();
                auto acquire_time = std::chrono::steady_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    acquire_time - start_time);
                
                // If acquisition takes more than 5 seconds, consider it a potential deadlock
                if (duration.count() > 5000) {
                    deadlock_timeouts.fetch_add(1);
                    FAIL() << "Potential deadlock detected in thread " << t 
                           << " (waited " << duration.count() << " ms)";
                }
                
                ASSERT_TRUE(guard.valid());
                
                // Lock the resource and simulate work
                {
                    std::lock_guard<std::mutex> lock(guard->mutex);
                    guard->in_use = true;
                    
                    // Simulate work with the resource
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    
                    guard->in_use = false;
                }
                
                successful_acquisitions.fetch_add(1);
                
            } catch (const std::exception& e) {
                FAIL() << "Resource acquisition failed in thread " << t << ": " << e.what();
            }
        });
    }
    
    // Wait for all threads with timeout
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successful_acquisitions.load(), num_threads);
    EXPECT_EQ(deadlock_timeouts.load(), 0);
}

// Test for data races in parallel work distribution
TEST_F(ThreadSanitizerTest, ParallelWorkDistributionRaces) {
    auto& parallel_mgr = UnifiedParallelManager::getInstance();
    
    struct WorkItem {
        int id;
        std::atomic<int> process_count{0};
        mutable std::mutex mutex;
        std::vector<int> data;
        
        WorkItem(int i) : id(i), data(1000, i) {}
    };
    
    const int num_work_items = 1000;
    std::vector<WorkItem> work_items;
    for (int i = 0; i < num_work_items; ++i) {
        work_items.emplace_back(i);
    }
    
    std::atomic<int> total_processed{0};
    std::atomic<int> race_conditions_detected{0};
    
    parallel_mgr.distributeWork<WorkItem>(work_items,
        [&total_processed, &race_conditions_detected](
            const WorkItem& item, const UnifiedParallelManager::ParallelContext& ctx) {
            
            // Check for race conditions by ensuring each item is processed exactly once
            int expected = 0;
            if (!item.process_count.compare_exchange_strong(expected, 1)) {
                race_conditions_detected.fetch_add(1);
                return;
            }
            
            // Simulate work with potential for race conditions
            {
                std::lock_guard<std::mutex> lock(item.mutex);
                
                // Verify data integrity
                for (size_t i = 0; i < item.data.size(); ++i) {
                    if (item.data[i] != item.id) {
                        race_conditions_detected.fetch_add(1);
                        return;
                    }
                }
                
                // Modify data in a thread-safe manner
                std::fill(item.data.begin(), item.data.end(), item.id + 1000);
            }
            
            total_processed.fetch_add(1);
        });
    
    EXPECT_EQ(total_processed.load(), num_work_items);
    EXPECT_EQ(race_conditions_detected.load(), 0);
    
    // Verify all items were processed exactly once
    for (const auto& item : work_items) {
        EXPECT_EQ(item.process_count.load(), 1);
        
        // Verify data was modified correctly
        for (int val : item.data) {
            EXPECT_EQ(val, item.id + 1000);
        }
    }
}

// Test for GPU memory race conditions
TEST_F(ThreadSanitizerTest, GPUMemoryRaceConditions) {
    // Skip if CUDA is not available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA not available, skipping GPU memory race condition test";
    }
    
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    const int num_threads = 8;
    const int gpu_allocations_per_thread = 50;
    std::vector<std::thread> threads;
    std::atomic<int> successful_gpu_allocations{0};
    std::atomic<int> gpu_memory_errors{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&memory_mgr, &successful_gpu_allocations, &gpu_memory_errors,
                             gpu_allocations_per_thread, t]() {
            try {
                for (int i = 0; i < gpu_allocations_per_thread; ++i) {
                    size_t size = (1024 + i) * sizeof(float);
                    
                    auto gpu_block = memory_mgr.allocate(size,
                        "gpu_thread_" + std::to_string(t) + "_" + std::to_string(i),
                        UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
                    
                    ASSERT_NE(gpu_block, nullptr);
                    ASSERT_NE(gpu_block->ptr, nullptr);
                    
                    // Test GPU memory operations
                    cudaError_t err = cudaMemset(gpu_block->ptr, 0, size);
                    if (err != cudaSuccess) {
                        gpu_memory_errors.fetch_add(1);
                        continue;
                    }
                    
                    successful_gpu_allocations.fetch_add(1);
                }
            } catch (const std::exception& e) {
                gpu_memory_errors.fetch_add(1);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successful_gpu_allocations.load(), num_threads * gpu_allocations_per_thread);
    EXPECT_EQ(gpu_memory_errors.load(), 0);
}

// Test for lock-free queue correctness under high contention
TEST_F(ThreadSanitizerTest, LockFreeQueueCorrectness) {
    LockFreeQueue<int> queue;
    
    const int num_producers = 8;
    const int num_consumers = 8;
    const int items_per_producer = 1000;
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    std::atomic<int> items_produced{0};
    std::atomic<int> items_consumed{0};
    std::atomic<bool> production_done{false};
    
    // Start producers
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&queue, &items_produced, items_per_producer, p]() {
            for (int i = 0; i < items_per_producer; ++i) {
                int value = p * items_per_producer + i;
                queue.enqueue(value);
                items_produced.fetch_add(1);
                
                // Add some randomness to increase contention
                if (i % 100 == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });
    }
    
    // Start consumers
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&queue, &items_consumed, &production_done]() {
            int value;
            while (!production_done.load() || !queue.empty()) {
                if (queue.dequeue(value)) {
                    items_consumed.fetch_add(1);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Wait for producers to finish
    for (auto& producer : producers) {
        producer.join();
    }
    production_done.store(true);
    
    // Wait for consumers to finish
    for (auto& consumer : consumers) {
        consumer.join();
    }
    
    EXPECT_EQ(items_produced.load(), num_producers * items_per_producer);
    EXPECT_EQ(items_consumed.load(), num_producers * items_per_producer);
    EXPECT_TRUE(queue.empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Enable ThreadSanitizer detection
    std::cout << "Running ThreadSanitizer tests..." << std::endl;
    std::cout << "Compile with: -fsanitize=thread -g -O1" << std::endl;
    std::cout << "Run with: TSAN_OPTIONS=\"halt_on_error=1:abort_on_error=1\" ./thread_sanitizer_tests" << std::endl;
    
    return RUN_ALL_TESTS();
}
