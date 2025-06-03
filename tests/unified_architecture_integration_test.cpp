#include "../backend/include/unified_parallel_manager.h"
#include "../backend/include/unified_memory_manager.h"
#include "../backend/include/async_gpu_execution_manager.h"
#include "../backend/include/thread_safe_resource_manager.h"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <random>

using namespace QDSim;

class UnifiedArchitectureIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the unified parallel architecture
        UnifiedParallelManager::ParallelConfig config;
        config.mpi_ranks = 1;
        config.omp_threads_per_rank = 4;
        config.cuda_devices_per_rank = 1;
        config.enable_gpu_direct = false;
        config.enable_numa_binding = true;
        config.gpu_memory_pool_size = 512 * 1024 * 1024; // 512 MB
        
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        bool success = parallel_mgr.initialize(config);
        ASSERT_TRUE(success) << "Failed to initialize unified parallel manager";
        
        std::cout << "Unified parallel architecture initialized successfully" << std::endl;
    }
    
    void TearDown() override {
        auto& parallel_mgr = UnifiedParallelManager::getInstance();
        parallel_mgr.finalize();
    }
};

// Test complete workflow integration
TEST_F(UnifiedArchitectureIntegrationTest, CompleteWorkflowIntegration) {
    auto& parallel_mgr = UnifiedParallelManager::getInstance();
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    // Simulate a quantum dot simulation workflow
    const size_t num_nodes = 10000;
    const size_t matrix_size = num_nodes * num_nodes;
    
    std::cout << "Testing complete workflow with " << num_nodes << " nodes" << std::endl;
    
    // 1. Allocate unified memory for matrices
    auto H_matrix_block = memory_mgr.allocate(matrix_size * sizeof(std::complex<double>),
                                            "hamiltonian_matrix",
                                            UnifiedMemoryManager::MemoryType::UNIFIED_MANAGED);
    
    auto M_matrix_block = memory_mgr.allocate(matrix_size * sizeof(std::complex<double>),
                                            "mass_matrix", 
                                            UnifiedMemoryManager::MemoryType::UNIFIED_MANAGED);
    
    auto eigenvalues_block = memory_mgr.allocate(num_nodes * sizeof(double),
                                               "eigenvalues",
                                               UnifiedMemoryManager::MemoryType::UNIFIED_MANAGED);
    
    ASSERT_NE(H_matrix_block, nullptr);
    ASSERT_NE(M_matrix_block, nullptr);
    ASSERT_NE(eigenvalues_block, nullptr);
    
    auto* H_matrix = static_cast<std::complex<double>*>(H_matrix_block->ptr);
    auto* M_matrix = static_cast<std::complex<double>*>(M_matrix_block->ptr);
    auto* eigenvalues = static_cast<double*>(eigenvalues_block->ptr);
    
    // 2. Parallel matrix assembly simulation
    struct MatrixAssemblyTask {
        size_t start_row, end_row;
        size_t matrix_size;
        
        MatrixAssemblyTask(size_t start, size_t end, size_t size) 
            : start_row(start), end_row(end), matrix_size(size) {}
    };
    
    std::vector<MatrixAssemblyTask> assembly_tasks;
    const size_t rows_per_task = 100;
    for (size_t row = 0; row < num_nodes; row += rows_per_task) {
        size_t end_row = std::min(row + rows_per_task, num_nodes);
        assembly_tasks.emplace_back(row, end_row, num_nodes);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    parallel_mgr.distributeWork<MatrixAssemblyTask>(assembly_tasks,
        [H_matrix, M_matrix](const MatrixAssemblyTask& task, 
                            const UnifiedParallelManager::ParallelContext& ctx) {
            
            // Simulate matrix assembly work
            std::random_device rd;
            std::mt19937 gen(rd() + ctx.omp_thread_id);
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            
            for (size_t i = task.start_row; i < task.end_row; ++i) {
                for (size_t j = 0; j < task.matrix_size; ++j) {
                    size_t idx = i * task.matrix_size + j;
                    
                    // Hamiltonian matrix (Hermitian)
                    if (i == j) {
                        H_matrix[idx] = std::complex<double>(dis(gen), 0.0);
                    } else if (i < j) {
                        double real_part = dis(gen) * 0.1;
                        double imag_part = dis(gen) * 0.1;
                        H_matrix[idx] = std::complex<double>(real_part, imag_part);
                        H_matrix[j * task.matrix_size + i] = std::conj(H_matrix[idx]);
                    }
                    
                    // Mass matrix (identity for simplicity)
                    M_matrix[idx] = (i == j) ? std::complex<double>(1.0, 0.0) : 
                                              std::complex<double>(0.0, 0.0);
                }
            }
        });
    
    auto assembly_time = std::chrono::high_resolution_clock::now();
    auto assembly_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        assembly_time - start_time);
    
    std::cout << "Matrix assembly completed in " << assembly_duration.count() << " ms" << std::endl;
    
    // 3. Simulate eigenvalue computation (simplified)
    for (size_t i = 0; i < num_nodes; ++i) {
        eigenvalues[i] = static_cast<double>(i) * 0.1; // Dummy eigenvalues
    }
    
    // 4. Test memory operations across different types
    auto host_buffer = memory_mgr.allocate(num_nodes * sizeof(double),
                                         "host_buffer",
                                         UnifiedMemoryManager::MemoryType::HOST_PINNED);
    ASSERT_NE(host_buffer, nullptr);
    
    // Copy eigenvalues to host buffer
    memory_mgr.copyAsync(host_buffer, eigenvalues_block);
    
    // Verify data integrity
    auto* host_eigenvalues = static_cast<double*>(host_buffer->ptr);
    for (size_t i = 0; i < std::min(num_nodes, size_t(100)); ++i) {
        EXPECT_NEAR(host_eigenvalues[i], eigenvalues[i], 1e-10)
            << "Data integrity check failed at index " << i;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Complete workflow finished in " << total_duration.count() << " ms" << std::endl;
    
    // 5. Verify memory statistics
    auto stats = memory_mgr.getStats();
    EXPECT_GT(stats.active_blocks, 0);
    EXPECT_GT(stats.total_allocated, 0);
    
    std::cout << "Memory statistics:" << std::endl;
    std::cout << "  Active blocks: " << stats.active_blocks << std::endl;
    std::cout << "  Total allocated: " << (stats.total_allocated / (1024 * 1024)) << " MB" << std::endl;
}

// Test GPU integration (if available)
TEST_F(UnifiedArchitectureIntegrationTest, GPUIntegration) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA not available, skipping GPU integration test";
    }
    
    AsyncGPUExecutionManager gpu_manager;
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    const size_t data_size = 1024 * 1024; // 1M elements
    const int num_operations = 10;
    
    std::cout << "Testing GPU integration with " << data_size << " elements" << std::endl;
    
    // Allocate GPU memory
    std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> gpu_blocks;
    for (int i = 0; i < num_operations; ++i) {
        auto block = memory_mgr.allocate(data_size * sizeof(float),
                                       "gpu_block_" + std::to_string(i),
                                       UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        ASSERT_NE(block, nullptr);
        gpu_blocks.push_back(block);
    }
    
    // Launch asynchronous GPU operations
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        int stream_id = i % 4; // Use 4 streams
        gpu_manager.memsetAsync(gpu_blocks[i]->ptr, i + 1, 
                               data_size * sizeof(float), stream_id);
    }
    
    gpu_manager.synchronizeAll();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "GPU operations completed in " << duration.count() << " ms" << std::endl;
    
    // Verify GPU operations
    for (int i = 0; i < num_operations; ++i) {
        std::vector<float> readback(data_size);
        cudaError_t err = cudaMemcpy(readback.data(), gpu_blocks[i]->ptr,
                                    data_size * sizeof(float), cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess);
        
        // Check a few values
        for (int j = 0; j < 10; ++j) {
            uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(&readback[j]);
            for (int b = 0; b < 4; ++b) {
                EXPECT_EQ(byte_ptr[b], i + 1) 
                    << "GPU operation verification failed for block " << i;
            }
        }
    }
    
    // Test GPU statistics
    auto gpu_stats = gpu_manager.getStreamStats();
    EXPECT_GT(gpu_stats.size(), 0);
    
    std::cout << "GPU stream statistics:" << std::endl;
    for (const auto& stat : gpu_stats) {
        std::cout << "  Stream " << stat.stream_id 
                  << ": " << stat.bytes_transferred << " bytes transferred" << std::endl;
    }
}

// Test resource manager integration
TEST_F(UnifiedArchitectureIntegrationTest, ResourceManagerIntegration) {
    // Test resource manager with simulated CUDA contexts
    struct MockCUDAContext {
        int device_id;
        bool is_active;
        
        MockCUDAContext() : device_id(0), is_active(true) {}
        ~MockCUDAContext() { is_active = false; }
    };
    
    auto factory = []() { return std::make_shared<MockCUDAContext>(); };
    auto deleter = [](std::shared_ptr<MockCUDAContext> ctx) { 
        ctx->is_active = false; 
    };
    
    ThreadSafeResourceManager<MockCUDAContext> resource_mgr(factory, deleter, 4);
    
    const int num_threads = 8;
    const int acquisitions_per_thread = 50;
    std::vector<std::thread> threads;
    std::atomic<int> successful_acquisitions{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&resource_mgr, &successful_acquisitions, 
                             acquisitions_per_thread, t]() {
            for (int i = 0; i < acquisitions_per_thread; ++i) {
                try {
                    auto guard = resource_mgr.acquire();
                    ASSERT_TRUE(guard.valid());
                    ASSERT_TRUE(guard->is_active);
                    
                    // Simulate work
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    
                    successful_acquisitions.fetch_add(1);
                    
                } catch (const std::exception& e) {
                    FAIL() << "Resource acquisition failed in thread " << t << ": " << e.what();
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_EQ(successful_acquisitions.load(), num_threads * acquisitions_per_thread);
    
    std::cout << "Resource manager test completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Successful acquisitions: " << successful_acquisitions.load() << std::endl;
    
    // Check resource pool statistics
    auto stats = resource_mgr.getStats();
    std::cout << "Resource pool statistics:" << std::endl;
    std::cout << "  Available resources: " << stats.available_resources << std::endl;
    std::cout << "  Total created: " << stats.total_created << std::endl;
    std::cout << "  Active resources: " << stats.active_resources << std::endl;
}

// Test error handling and recovery
TEST_F(UnifiedArchitectureIntegrationTest, ErrorHandlingAndRecovery) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    std::cout << "Testing error handling and recovery mechanisms" << std::endl;
    
    // Test memory allocation failure recovery
    std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> blocks;
    
    try {
        // Try to allocate increasingly large blocks until failure
        for (size_t size = 1024 * 1024; size <= 16ULL * 1024 * 1024 * 1024; size *= 2) {
            auto block = memory_mgr.allocate(size, "stress_test_" + std::to_string(size),
                                           UnifiedMemoryManager::MemoryType::HOST_PINNED);
            
            if (block) {
                blocks.push_back(block);
                std::cout << "Allocated " << (size / (1024 * 1024)) << " MB" << std::endl;
            } else {
                std::cout << "Allocation failed at " << (size / (1024 * 1024)) << " MB" << std::endl;
                break;
            }
            
            // Stop if we've allocated a reasonable amount
            if (blocks.size() > 10) break;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Expected allocation failure: " << e.what() << std::endl;
    }
    
    // Verify system can recover
    blocks.clear(); // Free all memory
    
    // Try a small allocation to verify recovery
    auto recovery_block = memory_mgr.allocate(1024, "recovery_test",
                                            UnifiedMemoryManager::MemoryType::HOST_PINNED);
    EXPECT_NE(recovery_block, nullptr) << "System failed to recover from allocation failure";
    
    std::cout << "✓ System recovered successfully from allocation stress" << std::endl;
    
    // Test parallel manager error handling
    auto& parallel_mgr = UnifiedParallelManager::getInstance();
    
    // Test with invalid work distribution
    std::vector<int> empty_work;
    
    try {
        parallel_mgr.distributeWork<int>(empty_work,
            [](const int& item, const UnifiedParallelManager::ParallelContext& ctx) {
                // This should not be called
                FAIL() << "Work processor called with empty work list";
            });
        
        std::cout << "✓ Empty work distribution handled gracefully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Work distribution error (expected): " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running Unified Architecture Integration Tests..." << std::endl;
    std::cout << "Testing complete system integration and workflows" << std::endl;
    
    return RUN_ALL_TESTS();
}
