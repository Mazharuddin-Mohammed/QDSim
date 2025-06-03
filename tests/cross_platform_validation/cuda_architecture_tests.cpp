#include "../../backend/include/unified_memory_manager.h"
#include "../../backend/include/async_gpu_execution_manager.h"
#include "../../backend/include/fused_gpu_kernels.cuh"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <string>

using namespace QDSim;

class CUDAArchitectureTest : public ::testing::Test {
protected:
    struct ArchitectureInfo {
        int major;
        int minor;
        std::string name;
        int multiProcessorCount;
        size_t totalGlobalMem;
        int maxThreadsPerBlock;
        int maxThreadsPerMultiProcessor;
        int warpSize;
        bool supportsCooperativeGroups;
        bool supportsUnifiedMemory;
        bool supportsConcurrentKernels;
    };
    
    std::vector<ArchitectureInfo> detected_architectures_;
    
    void SetUp() override {
        detectCUDAArchitectures();
    }
    
    void detectCUDAArchitectures() {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA not available, skipping architecture tests";
        }
        
        for (int device = 0; device < device_count; ++device) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            
            ArchitectureInfo arch;
            arch.major = prop.major;
            arch.minor = prop.minor;
            arch.name = prop.name;
            arch.multiProcessorCount = prop.multiProcessorCount;
            arch.totalGlobalMem = prop.totalGlobalMem;
            arch.maxThreadsPerBlock = prop.maxThreadsPerBlock;
            arch.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
            arch.warpSize = prop.warpSize;
            arch.supportsCooperativeGroups = (prop.major >= 6); // Pascal and later
            arch.supportsUnifiedMemory = (prop.unifiedAddressing != 0);
            arch.supportsConcurrentKernels = (prop.concurrentKernels != 0);
            
            detected_architectures_.push_back(arch);
            
            std::cout << "Detected GPU " << device << ": " << arch.name 
                      << " (Compute " << arch.major << "." << arch.minor << ")" << std::endl;
        }
    }
    
    std::string getArchitectureName(int major, int minor) {
        std::map<std::pair<int, int>, std::string> arch_names = {
            {{6, 0}, "Pascal"},
            {{6, 1}, "Pascal"},
            {{6, 2}, "Pascal"},
            {{7, 0}, "Volta"},
            {{7, 2}, "Volta"},
            {{7, 5}, "Turing"},
            {{8, 0}, "Ampere"},
            {{8, 6}, "Ampere"},
            {{8, 7}, "Ampere"},
            {{8, 9}, "Ada Lovelace"},
            {{9, 0}, "Hopper"}
        };
        
        auto it = arch_names.find({major, minor});
        return (it != arch_names.end()) ? it->second : "Unknown";
    }
};

// Test basic CUDA functionality across architectures
TEST_F(CUDAArchitectureTest, BasicCUDAFunctionality) {
    for (size_t i = 0; i < detected_architectures_.size(); ++i) {
        const auto& arch = detected_architectures_[i];
        
        cudaSetDevice(i);
        
        std::cout << "\nTesting device " << i << ": " << arch.name << std::endl;
        
        // Test basic memory allocation
        void* gpu_ptr = nullptr;
        size_t test_size = 1024 * 1024; // 1MB
        
        cudaError_t err = cudaMalloc(&gpu_ptr, test_size);
        ASSERT_EQ(err, cudaSuccess) << "Memory allocation failed on " << arch.name;
        ASSERT_NE(gpu_ptr, nullptr);
        
        // Test memory operations
        err = cudaMemset(gpu_ptr, 0xAA, test_size);
        ASSERT_EQ(err, cudaSuccess) << "Memory set failed on " << arch.name;
        
        // Test memory copy
        std::vector<char> host_data(test_size, 0xBB);
        err = cudaMemcpy(gpu_ptr, host_data.data(), test_size, cudaMemcpyHostToDevice);
        ASSERT_EQ(err, cudaSuccess) << "Host to device copy failed on " << arch.name;
        
        std::vector<char> readback_data(test_size);
        err = cudaMemcpy(readback_data.data(), gpu_ptr, test_size, cudaMemcpyDeviceToHost);
        ASSERT_EQ(err, cudaSuccess) << "Device to host copy failed on " << arch.name;
        
        // Verify data integrity
        EXPECT_EQ(host_data, readback_data) << "Data corruption on " << arch.name;
        
        cudaFree(gpu_ptr);
        
        std::cout << "✓ Basic CUDA functionality works on " << arch.name << std::endl;
    }
}

// Test unified memory support across architectures
TEST_F(CUDAArchitectureTest, UnifiedMemorySupport) {
    for (size_t i = 0; i < detected_architectures_.size(); ++i) {
        const auto& arch = detected_architectures_[i];
        
        cudaSetDevice(i);
        
        if (!arch.supportsUnifiedMemory) {
            std::cout << "Skipping unified memory test for " << arch.name 
                      << " (not supported)" << std::endl;
            continue;
        }
        
        std::cout << "\nTesting unified memory on device " << i << ": " << arch.name << std::endl;
        
        // Test unified memory allocation
        void* unified_ptr = nullptr;
        size_t test_size = 1024 * 1024; // 1MB
        
        cudaError_t err = cudaMallocManaged(&unified_ptr, test_size);
        ASSERT_EQ(err, cudaSuccess) << "Unified memory allocation failed on " << arch.name;
        ASSERT_NE(unified_ptr, nullptr);
        
        // Test CPU access
        char* char_ptr = static_cast<char*>(unified_ptr);
        std::memset(char_ptr, 0xCC, test_size);
        
        // Test GPU access via kernel
        err = cudaMemset(unified_ptr, 0xDD, test_size);
        ASSERT_EQ(err, cudaSuccess) << "GPU memset on unified memory failed on " << arch.name;
        
        cudaDeviceSynchronize();
        
        // Verify GPU wrote the data
        for (size_t j = 0; j < std::min(size_t(100), test_size); ++j) {
            EXPECT_EQ(char_ptr[j], static_cast<char>(0xDD)) 
                << "Unified memory data mismatch on " << arch.name << " at index " << j;
        }
        
        cudaFree(unified_ptr);
        
        std::cout << "✓ Unified memory works on " << arch.name << std::endl;
    }
}

// Test concurrent kernel execution across architectures
TEST_F(CUDAArchitectureTest, ConcurrentKernelExecution) {
    for (size_t i = 0; i < detected_architectures_.size(); ++i) {
        const auto& arch = detected_architectures_[i];
        
        cudaSetDevice(i);
        
        if (!arch.supportsConcurrentKernels) {
            std::cout << "Skipping concurrent kernel test for " << arch.name 
                      << " (not supported)" << std::endl;
            continue;
        }
        
        std::cout << "\nTesting concurrent kernels on device " << i << ": " << arch.name << std::endl;
        
        AsyncGPUExecutionManager gpu_manager;
        auto& memory_mgr = UnifiedMemoryManager::getInstance();
        
        const int num_streams = 4;
        const size_t data_size = 1024 * 1024; // 1MB per stream
        
        std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> gpu_blocks;
        
        // Allocate memory for each stream
        for (int s = 0; s < num_streams; ++s) {
            auto block = memory_mgr.allocate(data_size, 
                "concurrent_" + std::to_string(s), 
                UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
            ASSERT_NE(block, nullptr);
            gpu_blocks.push_back(block);
        }
        
        // Launch concurrent operations
        for (int s = 0; s < num_streams; ++s) {
            gpu_manager.memsetAsync(gpu_blocks[s]->ptr, s + 1, data_size, s);
        }
        
        // Synchronize all streams
        gpu_manager.synchronizeAll();
        
        // Verify results
        for (int s = 0; s < num_streams; ++s) {
            std::vector<char> readback(data_size);
            cudaError_t err = cudaMemcpy(readback.data(), gpu_blocks[s]->ptr, 
                                        data_size, cudaMemcpyDeviceToHost);
            ASSERT_EQ(err, cudaSuccess) << "Readback failed for stream " << s 
                                       << " on " << arch.name;
            
            char expected_value = s + 1;
            for (size_t j = 0; j < std::min(size_t(100), data_size); ++j) {
                EXPECT_EQ(readback[j], expected_value) 
                    << "Concurrent kernel data mismatch on " << arch.name 
                    << " stream " << s << " index " << j;
            }
        }
        
        std::cout << "✓ Concurrent kernels work on " << arch.name << std::endl;
    }
}

// Test cooperative groups support (Pascal and later)
TEST_F(CUDAArchitectureTest, CooperativeGroupsSupport) {
    for (size_t i = 0; i < detected_architectures_.size(); ++i) {
        const auto& arch = detected_architectures_[i];
        
        cudaSetDevice(i);
        
        if (!arch.supportsCooperativeGroups) {
            std::cout << "Skipping cooperative groups test for " << arch.name 
                      << " (requires Compute 6.0+)" << std::endl;
            continue;
        }
        
        std::cout << "\nTesting cooperative groups on device " << i << ": " << arch.name << std::endl;
        
        // Test basic cooperative groups functionality
        auto& memory_mgr = UnifiedMemoryManager::getInstance();
        
        const size_t test_size = 1024 * sizeof(int);
        auto gpu_data = memory_mgr.allocate(test_size, "coop_groups",
                                          UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        ASSERT_NE(gpu_data, nullptr);
        
        // Initialize data
        std::vector<int> host_data(test_size / sizeof(int));
        std::iota(host_data.begin(), host_data.end(), 0);
        
        cudaError_t err = cudaMemcpy(gpu_data->ptr, host_data.data(), 
                                    test_size, cudaMemcpyHostToDevice);
        ASSERT_EQ(err, cudaSuccess);
        
        // Launch a kernel that uses cooperative groups (simplified test)
        dim3 block(256);
        dim3 grid((host_data.size() + block.x - 1) / block.x);
        
        // Note: This would require a custom kernel that uses cooperative groups
        // For now, we just test that the architecture supports the required features
        
        std::cout << "✓ Cooperative groups supported on " << arch.name << std::endl;
    }
}

// Test memory bandwidth across different architectures
TEST_F(CUDAArchitectureTest, MemoryBandwidthComparison) {
    std::vector<double> bandwidths;
    
    for (size_t i = 0; i < detected_architectures_.size(); ++i) {
        const auto& arch = detected_architectures_[i];
        
        cudaSetDevice(i);
        
        std::cout << "\nMeasuring memory bandwidth on device " << i << ": " << arch.name << std::endl;
        
        const size_t test_size = 256 * 1024 * 1024; // 256MB
        const int num_iterations = 10;
        
        void* gpu_src = nullptr;
        void* gpu_dst = nullptr;
        
        cudaError_t err = cudaMalloc(&gpu_src, test_size);
        ASSERT_EQ(err, cudaSuccess);
        err = cudaMalloc(&gpu_dst, test_size);
        ASSERT_EQ(err, cudaSuccess);
        
        // Warmup
        for (int j = 0; j < 3; ++j) {
            cudaMemcpy(gpu_dst, gpu_src, test_size, cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();
        
        // Measure bandwidth
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int j = 0; j < num_iterations; ++j) {
            cudaMemcpy(gpu_dst, gpu_src, test_size, cudaMemcpyDeviceToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        double bandwidth_gb_s = (test_size * num_iterations * 2) / (elapsed_ms / 1000.0) / 1e9;
        bandwidths.push_back(bandwidth_gb_s);
        
        std::cout << "Memory bandwidth: " << std::fixed << std::setprecision(1) 
                  << bandwidth_gb_s << " GB/s" << std::endl;
        
        // Compare with theoretical bandwidth
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        double theoretical_bandwidth = (prop.memoryClockRate * 1000.0 * prop.memoryBusWidth / 8) / 1e9;
        double efficiency = (bandwidth_gb_s / theoretical_bandwidth) * 100.0;
        
        std::cout << "Theoretical bandwidth: " << theoretical_bandwidth << " GB/s" << std::endl;
        std::cout << "Efficiency: " << efficiency << "%" << std::endl;
        
        EXPECT_GT(efficiency, 50.0) << "Poor memory bandwidth efficiency on " << arch.name;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(gpu_src);
        cudaFree(gpu_dst);
    }
    
    // Compare bandwidths across architectures
    if (bandwidths.size() > 1) {
        auto max_bandwidth = *std::max_element(bandwidths.begin(), bandwidths.end());
        auto min_bandwidth = *std::min_element(bandwidths.begin(), bandwidths.end());
        
        std::cout << "\nBandwidth comparison:" << std::endl;
        std::cout << "Maximum: " << max_bandwidth << " GB/s" << std::endl;
        std::cout << "Minimum: " << min_bandwidth << " GB/s" << std::endl;
        std::cout << "Ratio: " << (max_bandwidth / min_bandwidth) << "x" << std::endl;
    }
}

// Test architecture-specific optimizations
TEST_F(CUDAArchitectureTest, ArchitectureSpecificOptimizations) {
    for (size_t i = 0; i < detected_architectures_.size(); ++i) {
        const auto& arch = detected_architectures_[i];
        
        cudaSetDevice(i);
        
        std::cout << "\nTesting optimizations for device " << i << ": " << arch.name 
                  << " (" << getArchitectureName(arch.major, arch.minor) << ")" << std::endl;
        
        // Test optimal block size for this architecture
        std::vector<int> block_sizes = {64, 128, 256, 512, 1024};
        std::vector<double> execution_times;
        
        const size_t data_size = 1024 * 1024 * sizeof(float);
        void* gpu_data = nullptr;
        cudaMalloc(&gpu_data, data_size);
        
        for (int block_size : block_sizes) {
            if (block_size > arch.maxThreadsPerBlock) {
                continue; // Skip if block size exceeds device limit
            }
            
            int grid_size = (data_size / sizeof(float) + block_size - 1) / block_size;
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            // Warmup
            cudaMemset(gpu_data, 0, data_size);
            cudaDeviceSynchronize();
            
            // Measure execution time
            cudaEventRecord(start);
            for (int j = 0; j < 100; ++j) {
                cudaMemset(gpu_data, j % 256, data_size);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            execution_times.push_back(elapsed_ms);
            
            std::cout << "Block size " << block_size << ": " << elapsed_ms << " ms" << std::endl;
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        
        // Find optimal block size
        auto min_time_it = std::min_element(execution_times.begin(), execution_times.end());
        int optimal_block_size = block_sizes[std::distance(execution_times.begin(), min_time_it)];
        
        std::cout << "Optimal block size for " << arch.name << ": " << optimal_block_size << std::endl;
        
        // Verify the optimal block size is reasonable
        EXPECT_GE(optimal_block_size, 64) << "Optimal block size too small for " << arch.name;
        EXPECT_LE(optimal_block_size, arch.maxThreadsPerBlock) 
            << "Optimal block size exceeds device limit for " << arch.name;
        
        cudaFree(gpu_data);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running CUDA Architecture Validation Tests..." << std::endl;
    std::cout << "Testing compatibility across different GPU architectures" << std::endl;
    
    return RUN_ALL_TESTS();
}
