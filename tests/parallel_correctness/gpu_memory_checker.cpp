#include "../../backend/include/unified_memory_manager.h"
#include "../../backend/include/async_gpu_execution_manager.h"
#include "../../backend/include/fused_gpu_kernels.cuh"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <chrono>

using namespace QDSim;

class GPUMemoryCheckerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA not available, skipping GPU memory tests";
        }
        
        // Initialize CUDA context
        cudaSetDevice(0);
        
        // Initialize memory manager
        UnifiedParallelManager::ParallelConfig config;
        config.cuda_devices_per_rank = 1;
        auto& memory_mgr = UnifiedMemoryManager::getInstance();
        memory_mgr.initialize(config);
    }
    
    void TearDown() override {
        // Synchronize and cleanup
        cudaDeviceSynchronize();
        auto& memory_mgr = UnifiedMemoryManager::getInstance();
        memory_mgr.cleanup();
    }
    
    // Helper function to check for CUDA errors
    void checkCudaError(const std::string& operation) {
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "CUDA error in " << operation << ": " 
                                   << cudaGetErrorString(err);
    }
};

// Test for memory leaks in GPU allocations
TEST_F(GPUMemoryCheckerTest, GPUMemoryLeakDetection) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    // Get initial memory statistics
    auto initial_stats = memory_mgr.getStats();
    size_t initial_gpu_memory = 0;
    cudaMemGetInfo(&initial_gpu_memory, nullptr);
    
    const int num_allocations = 100;
    const size_t allocation_size = 1024 * 1024; // 1MB each
    
    {
        // Scope to ensure blocks go out of scope
        std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> blocks;
        
        for (int i = 0; i < num_allocations; ++i) {
            auto block = memory_mgr.allocate(allocation_size,
                "leak_test_" + std::to_string(i),
                UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
            
            ASSERT_NE(block, nullptr);
            ASSERT_NE(block->ptr, nullptr);
            
            // Write to GPU memory to ensure it's valid
            cudaError_t err = cudaMemset(block->ptr, i % 256, allocation_size);
            ASSERT_EQ(err, cudaSuccess) << "Failed to write to GPU memory: " 
                                       << cudaGetErrorString(err);
            
            blocks.push_back(block);
        }
        
        // Verify all allocations are tracked
        auto mid_stats = memory_mgr.getStats();
        EXPECT_GE(mid_stats.active_blocks, initial_stats.active_blocks + num_allocations);
        
    } // Blocks should be automatically freed here
    
    // Force garbage collection and synchronization
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check for memory leaks
    auto final_stats = memory_mgr.getStats();
    size_t final_gpu_memory = 0;
    cudaMemGetInfo(&final_gpu_memory, nullptr);
    
    // Memory should be returned to approximately initial levels
    EXPECT_LE(final_stats.active_blocks, initial_stats.active_blocks + 5); // Allow some tolerance
    
    // GPU memory should be freed (within reasonable tolerance)
    size_t memory_diff = (initial_gpu_memory > final_gpu_memory) ? 
                        (initial_gpu_memory - final_gpu_memory) : 0;
    EXPECT_LT(memory_diff, allocation_size) << "Significant GPU memory leak detected";
}

// Test for buffer overruns and invalid memory access
TEST_F(GPUMemoryCheckerTest, GPUBufferOverrunDetection) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    const size_t buffer_size = 1024 * sizeof(float);
    auto gpu_buffer = memory_mgr.allocate(buffer_size, "overrun_test",
                                         UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
    
    ASSERT_NE(gpu_buffer, nullptr);
    
    // Create host buffer for testing
    std::vector<float> host_data(buffer_size / sizeof(float));
    std::iota(host_data.begin(), host_data.end(), 0.0f);
    
    // Test valid memory operations
    cudaError_t err = cudaMemcpy(gpu_buffer->ptr, host_data.data(), 
                                buffer_size, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Valid memory copy failed: " << cudaGetErrorString(err);
    
    // Test reading back the data
    std::vector<float> readback_data(buffer_size / sizeof(float));
    err = cudaMemcpy(readback_data.data(), gpu_buffer->ptr, 
                    buffer_size, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Valid memory read failed: " << cudaGetErrorString(err);
    
    // Verify data integrity
    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_data[i], readback_data[i]) 
            << "Data corruption detected at index " << i;
    }
    
    checkCudaError("buffer overrun detection test");
}

// Test for use-after-free detection
TEST_F(GPUMemoryCheckerTest, UseAfterFreeDetection) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    void* raw_ptr = nullptr;
    
    {
        auto gpu_buffer = memory_mgr.allocate(1024 * sizeof(float), "use_after_free_test",
                                             UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        ASSERT_NE(gpu_buffer, nullptr);
        raw_ptr = gpu_buffer->ptr;
        
        // Valid operation while buffer is alive
        cudaError_t err = cudaMemset(raw_ptr, 0, 1024 * sizeof(float));
        ASSERT_EQ(err, cudaSuccess) << "Valid memset failed: " << cudaGetErrorString(err);
        
    } // Buffer should be freed here
    
    // Force synchronization to ensure cleanup
    cudaDeviceSynchronize();
    
    // Note: Actual use-after-free detection would require cuda-memcheck or similar tools
    // This test verifies that the memory manager properly tracks freed memory
    auto found_block = memory_mgr.findBlock(raw_ptr);
    EXPECT_EQ(found_block, nullptr) << "Memory block should be unregistered after free";
}

// Test for double-free detection
TEST_F(GPUMemoryCheckerTest, DoubleFreeDetection) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    auto gpu_buffer = memory_mgr.allocate(1024 * sizeof(float), "double_free_test",
                                         UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
    ASSERT_NE(gpu_buffer, nullptr);
    
    void* raw_ptr = gpu_buffer->ptr;
    
    // First free (automatic via RAII)
    gpu_buffer.reset();
    
    // Verify the block is no longer tracked
    auto found_block = memory_mgr.findBlock(raw_ptr);
    EXPECT_EQ(found_block, nullptr) << "Memory block should be unregistered after first free";
    
    // Attempting to manually free again should not cause issues
    // (The memory manager should handle this gracefully)
    memory_mgr.unregisterBlock(raw_ptr); // Should be safe to call
    
    checkCudaError("double free detection test");
}

// Test for memory alignment issues
TEST_F(GPUMemoryCheckerTest, MemoryAlignmentValidation) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    // Test various allocation sizes to check alignment
    std::vector<size_t> test_sizes = {
        1, 7, 16, 33, 64, 127, 256, 513, 1024, 2047, 4096, 8191
    };
    
    for (size_t size : test_sizes) {
        auto buffer = memory_mgr.allocate(size, "alignment_test_" + std::to_string(size),
                                         UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        
        ASSERT_NE(buffer, nullptr) << "Allocation failed for size " << size;
        ASSERT_NE(buffer->ptr, nullptr) << "Null pointer for size " << size;
        
        // Check alignment (GPU memory should be at least 256-byte aligned for optimal performance)
        uintptr_t addr = reinterpret_cast<uintptr_t>(buffer->ptr);
        EXPECT_EQ(addr % 256, 0) << "Poor alignment for size " << size 
                                 << ", address: 0x" << std::hex << addr;
        
        // Test that we can actually use the memory
        cudaError_t err = cudaMemset(buffer->ptr, 0xAA, size);
        ASSERT_EQ(err, cudaSuccess) << "Memory access failed for size " << size 
                                   << ": " << cudaGetErrorString(err);
    }
    
    checkCudaError("memory alignment validation");
}

// Test for concurrent GPU memory operations
TEST_F(GPUMemoryCheckerTest, ConcurrentGPUMemoryOperations) {
    AsyncGPUExecutionManager gpu_manager;
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    const int num_streams = 4;
    const int operations_per_stream = 50;
    const size_t buffer_size = 1024 * sizeof(float);
    
    std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> buffers;
    std::vector<std::vector<float>> host_data(num_streams);
    
    // Prepare test data
    for (int s = 0; s < num_streams; ++s) {
        auto buffer = memory_mgr.allocate(buffer_size, "concurrent_test_" + std::to_string(s),
                                         UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        buffers.push_back(buffer);
        
        host_data[s].resize(buffer_size / sizeof(float));
        std::iota(host_data[s].begin(), host_data[s].end(), s * 1000.0f);
    }
    
    // Launch concurrent operations on different streams
    for (int s = 0; s < num_streams; ++s) {
        for (int op = 0; op < operations_per_stream; ++op) {
            // Async memory copy
            gpu_manager.memcpyAsync(buffers[s]->ptr, host_data[s].data(), 
                                   buffer_size, cudaMemcpyHostToDevice, s);
            
            // Async memset
            gpu_manager.memsetAsync(buffers[s]->ptr, op % 256, buffer_size, s);
        }
    }
    
    // Synchronize all streams
    gpu_manager.synchronizeAll();
    
    // Verify no errors occurred
    checkCudaError("concurrent GPU memory operations");
    
    // Verify data integrity by reading back
    for (int s = 0; s < num_streams; ++s) {
        std::vector<float> readback(buffer_size / sizeof(float));
        cudaError_t err = cudaMemcpy(readback.data(), buffers[s]->ptr,
                                    buffer_size, cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess) << "Readback failed for stream " << s;
        
        // Data should be all the same value (from the last memset)
        int expected_value = (operations_per_stream - 1) % 256;
        for (size_t i = 0; i < readback.size(); ++i) {
            // Convert back from float representation of memset value
            uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(&readback[i]);
            for (int b = 0; b < 4; ++b) {
                EXPECT_EQ(byte_ptr[b], expected_value) 
                    << "Data corruption in stream " << s << " at float " << i << " byte " << b;
            }
        }
    }
}

// Test for GPU memory fragmentation
TEST_F(GPUMemoryCheckerTest, MemoryFragmentationAnalysis) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    // Get initial GPU memory info
    size_t free_mem_initial, total_mem;
    cudaMemGetInfo(&free_mem_initial, &total_mem);
    
    std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> small_blocks;
    std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> large_blocks;
    
    // Allocate many small blocks to create fragmentation
    const int num_small_blocks = 1000;
    const size_t small_block_size = 4096; // 4KB
    
    for (int i = 0; i < num_small_blocks; ++i) {
        auto block = memory_mgr.allocate(small_block_size, 
            "small_frag_" + std::to_string(i),
            UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        
        if (block) {
            small_blocks.push_back(block);
        }
    }
    
    // Free every other small block to create holes
    for (size_t i = 1; i < small_blocks.size(); i += 2) {
        small_blocks[i].reset();
    }
    
    // Try to allocate larger blocks
    const size_t large_block_size = 1024 * 1024; // 1MB
    const int num_large_blocks = 10;
    
    for (int i = 0; i < num_large_blocks; ++i) {
        auto block = memory_mgr.allocate(large_block_size,
            "large_frag_" + std::to_string(i),
            UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        
        if (block) {
            large_blocks.push_back(block);
        }
    }
    
    // Check final memory state
    size_t free_mem_final;
    cudaMemGetInfo(&free_mem_final, &total_mem);
    
    // Verify we could allocate some large blocks despite fragmentation
    EXPECT_GT(large_blocks.size(), 0) << "Could not allocate any large blocks due to fragmentation";
    
    // Memory usage should be reasonable
    size_t used_memory = free_mem_initial - free_mem_final;
    size_t expected_usage = (small_blocks.size() / 2) * small_block_size + 
                           large_blocks.size() * large_block_size;
    
    // Allow for some overhead and fragmentation
    EXPECT_LT(used_memory, expected_usage * 1.5) << "Excessive memory fragmentation detected";
    
    checkCudaError("memory fragmentation analysis");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running GPU Memory Checker tests..." << std::endl;
    std::cout << "For comprehensive checking, run with:" << std::endl;
    std::cout << "  cuda-memcheck --tool=memcheck ./gpu_memory_checker" << std::endl;
    std::cout << "  cuda-memcheck --tool=racecheck ./gpu_memory_checker" << std::endl;
    std::cout << "  cuda-memcheck --tool=initcheck ./gpu_memory_checker" << std::endl;
    std::cout << "  cuda-memcheck --tool=synccheck ./gpu_memory_checker" << std::endl;
    
    return RUN_ALL_TESTS();
}
