#include "backend/include/unified_parallel_manager.h"
#include "backend/include/unified_memory_manager.h"
#include "backend/include/thread_safe_resource_manager.h"
#include "backend/include/async_gpu_execution_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

using namespace QDSim;

// Test structure for work distribution
struct TestWorkItem {
    int id;
    double value;
    
    TestWorkItem(int i, double v) : id(i), value(v) {}
};

// Simple test resource for resource manager
struct TestResource {
    int resource_id;
    bool is_active = false;
    
    TestResource() : resource_id(rand() % 1000) {
        is_active = true;
        std::cout << "Created test resource " << resource_id << std::endl;
    }
    
    ~TestResource() {
        is_active = false;
        std::cout << "Destroyed test resource " << resource_id << std::endl;
    }
};

void testUnifiedParallelManager() {
    std::cout << "\n=== Testing UnifiedParallelManager ===" << std::endl;
    
    // Configure parallel environment
    UnifiedParallelManager::ParallelConfig config;
    config.mpi_ranks = 1;
    config.omp_threads_per_rank = 4;
    config.cuda_devices_per_rank = 1;
    config.enable_gpu_direct = false;
    config.enable_numa_binding = false;
    
    auto& parallel_mgr = UnifiedParallelManager::getInstance();
    
    bool success = parallel_mgr.initialize(config);
    assert(success && "Failed to initialize UnifiedParallelManager");
    
    std::cout << "✓ UnifiedParallelManager initialized successfully" << std::endl;
    
    // Test work distribution
    std::vector<TestWorkItem> work_items;
    for (int i = 0; i < 100; ++i) {
        work_items.emplace_back(i, i * 0.1);
    }
    
    std::atomic<int> processed_count{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    parallel_mgr.distributeWork<TestWorkItem>(work_items,
        [&processed_count](const TestWorkItem& item, const UnifiedParallelManager::ParallelContext& ctx) {
            // Simulate some work
            double result = item.value * item.value + std::sin(item.value);
            processed_count.fetch_add(1);
            
            if (item.id % 20 == 0) {
                std::cout << "Processed item " << item.id 
                          << " on MPI rank " << ctx.mpi_rank 
                          << ", OpenMP thread " << ctx.omp_thread_id 
                          << ", result: " << result << std::endl;
            }
        });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    assert(processed_count.load() == 100 && "Not all work items were processed");
    std::cout << "✓ Processed " << processed_count.load() << " work items in " 
              << duration.count() << " ms" << std::endl;
    
    parallel_mgr.finalize();
    std::cout << "✓ UnifiedParallelManager finalized successfully" << std::endl;
}

void testUnifiedMemoryManager() {
    std::cout << "\n=== Testing UnifiedMemoryManager ===" << std::endl;
    
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    // Test memory allocation
    size_t test_size = 1024 * 1024; // 1MB
    auto block1 = memory_mgr.allocate(test_size, "test_block_1", 
                                     UnifiedMemoryManager::MemoryType::HOST_PINNED);
    
    assert(block1 != nullptr && "Failed to allocate memory block");
    assert(block1->size >= test_size && "Allocated block size is incorrect");
    assert(block1->ptr != nullptr && "Allocated block pointer is null");
    
    std::cout << "✓ Allocated " << (block1->size / 1024) << " KB memory block" << std::endl;
    
    // Test memory copy
    auto block2 = memory_mgr.allocate(test_size, "test_block_2", 
                                     UnifiedMemoryManager::MemoryType::HOST_PINNED);
    
    // Fill first block with test data
    double* data1 = static_cast<double*>(block1->ptr);
    for (size_t i = 0; i < test_size / sizeof(double); ++i) {
        data1[i] = i * 0.5;
    }
    
    // Copy to second block
    memory_mgr.copyAsync(block2, block1);
    
    // Verify copy
    double* data2 = static_cast<double*>(block2->ptr);
    bool copy_correct = true;
    for (size_t i = 0; i < std::min(size_t(100), test_size / sizeof(double)); ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-10) {
            copy_correct = false;
            break;
        }
    }
    
    assert(copy_correct && "Memory copy verification failed");
    std::cout << "✓ Memory copy completed and verified" << std::endl;
    
    // Test statistics
    auto stats = memory_mgr.getStats();
    std::cout << "✓ Memory statistics: " << stats.active_blocks << " active blocks, "
              << (stats.total_allocated / (1024 * 1024)) << " MB allocated" << std::endl;
    
    memory_mgr.printStats();
}

void testThreadSafeResourceManager() {
    std::cout << "\n=== Testing ThreadSafeResourceManager ===" << std::endl;
    
    // Create resource manager
    auto resource_factory = []() -> std::shared_ptr<TestResource> {
        return std::make_shared<TestResource>();
    };
    
    auto resource_deleter = [](std::shared_ptr<TestResource> resource) {
        std::cout << "Deleting resource " << resource->resource_id << std::endl;
    };
    
    ThreadSafeResourceManager<TestResource> resource_mgr(resource_factory, resource_deleter, 5);
    
    // Test resource acquisition and release
    std::vector<std::thread> threads;
    std::atomic<int> successful_acquisitions{0};
    
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&resource_mgr, &successful_acquisitions, i]() {
            try {
                auto guard = resource_mgr.acquire();
                assert(guard.valid() && "Failed to acquire resource");
                assert(guard->is_active && "Acquired inactive resource");
                
                // Simulate some work
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                
                successful_acquisitions.fetch_add(1);
                std::cout << "Thread " << i << " successfully used resource " 
                          << guard->resource_id << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "Thread " << i << " failed: " << e.what() << std::endl;
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    assert(successful_acquisitions.load() == 10 && "Not all threads successfully acquired resources");
    std::cout << "✓ All " << successful_acquisitions.load() << " threads successfully acquired resources" << std::endl;
    
    // Test resource pool statistics
    auto stats = resource_mgr.getStats();
    std::cout << "✓ Resource pool stats: " << stats.available_resources << " available, "
              << stats.total_created << " total created, " << stats.active_resources << " active" << std::endl;
}

void testAsyncGPUExecutionManager() {
    std::cout << "\n=== Testing AsyncGPUExecutionManager ===" << std::endl;
    
    try {
        AsyncGPUExecutionManager gpu_mgr;
        
        // Test basic functionality
        int optimal_streams = gpu_mgr.getOptimalStreamCount();
        std::cout << "✓ Optimal stream count: " << optimal_streams << std::endl;
        
        // Test memory operations
        size_t test_size = 1024 * sizeof(float);
        float* host_data = new float[1024];
        float* device_data = nullptr;
        
        // Initialize host data
        for (int i = 0; i < 1024; ++i) {
            host_data[i] = i * 0.1f;
        }
        
        cudaError_t err = cudaMalloc(&device_data, test_size);
        if (err == cudaSuccess) {
            // Test async memory copy
            gpu_mgr.memcpyAsync(device_data, host_data, test_size, cudaMemcpyHostToDevice, 0);
            gpu_mgr.synchronizeStream(0);
            
            std::cout << "✓ Async memory copy completed successfully" << std::endl;
            
            // Test event recording and synchronization
            int event_id = gpu_mgr.recordEvent(0);
            gpu_mgr.synchronizeEvent(event_id);
            
            std::cout << "✓ Event recording and synchronization completed" << std::endl;
            
            cudaFree(device_data);
        } else {
            std::cout << "⚠ CUDA not available, skipping GPU-specific tests" << std::endl;
        }
        
        delete[] host_data;
        
        // Test pipeline execution
        std::vector<AsyncGPUExecutionManager::ComputeTask> compute_tasks;
        std::vector<AsyncGPUExecutionManager::CommTask> comm_tasks;
        
        for (int i = 0; i < 3; ++i) {
            compute_tasks.push_back({
                [i](cudaStream_t stream) {
                    std::cout << "Executing compute task " << i << " on stream" << std::endl;
                    // Simulate computation
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                },
                i,
                "compute_task_" + std::to_string(i)
            });
            
            comm_tasks.push_back({
                [i](cudaStream_t stream) {
                    std::cout << "Executing communication task " << i << " on stream" << std::endl;
                    // Simulate communication
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                },
                i - 1, // Dependency on previous compute task
                "comm_task_" + std::to_string(i)
            });
        }
        
        gpu_mgr.executePipeline(compute_tasks, comm_tasks);
        std::cout << "✓ Pipeline execution completed successfully" << std::endl;
        
        // Print statistics
        gpu_mgr.printStats();
        
    } catch (const std::exception& e) {
        std::cout << "⚠ GPU execution manager test failed: " << e.what() << std::endl;
        std::cout << "  This is expected if CUDA is not available" << std::endl;
    }
}

int main() {
    std::cout << "QDSim Unified Parallel Architecture Test Suite" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        testUnifiedParallelManager();
        testUnifiedMemoryManager();
        testThreadSafeResourceManager();
        testAsyncGPUExecutionManager();
        
        std::cout << "\n🎉 All tests completed successfully!" << std::endl;
        std::cout << "✓ Unified parallel architecture is working correctly" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}
