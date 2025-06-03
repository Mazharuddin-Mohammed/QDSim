#include "../../backend/include/unified_memory_manager.h"
#include "../../backend/include/async_gpu_execution_manager.h"
#include "../../backend/include/fused_gpu_kernels.cuh"
#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace QDSim;

class BandwidthUtilizationTest : public ::testing::Test {
protected:
    struct BandwidthResult {
        std::string test_name;
        size_t data_size_mb;
        double achieved_bandwidth_gb_s;
        double theoretical_bandwidth_gb_s;
        double utilization_percent;
        double latency_us;
        std::string memory_type;
        int num_streams;
    };
    
    std::vector<BandwidthResult> results_;
    double peak_memory_bandwidth_ = 0.0;
    double peak_gpu_bandwidth_ = 0.0;
    
    void SetUp() override {
        // Initialize memory manager
        UnifiedParallelManager::ParallelConfig config;
        config.cuda_devices_per_rank = 1;
        auto& memory_mgr = UnifiedMemoryManager::getInstance();
        memory_mgr.initialize(config);
        
        // Measure theoretical peak bandwidths
        measureTheoreticalBandwidths();
    }
    
    void TearDown() override {
        generateBandwidthReport();
    }
    
    void measureTheoreticalBandwidths() {
        // Measure CPU memory bandwidth
        const size_t test_size = 256 * 1024 * 1024; // 256 MB
        std::vector<char> src(test_size), dst(test_size);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            std::memcpy(dst.data(), src.data(), test_size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        peak_memory_bandwidth_ = (test_size * 10 * 2) / (duration.count() / 1e6) / 1e9; // GB/s
        
        // Measure GPU bandwidth (if available)
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            peak_gpu_bandwidth_ = (prop.memoryClockRate * 1000.0 * prop.memoryBusWidth / 8) / 1e9;
        }
        
        std::cout << "Theoretical CPU bandwidth: " << peak_memory_bandwidth_ << " GB/s" << std::endl;
        std::cout << "Theoretical GPU bandwidth: " << peak_gpu_bandwidth_ << " GB/s" << std::endl;
    }
    
    double measureBandwidth(std::function<void()> operation, size_t bytes_transferred) {
        const int warmup_iterations = 3;
        const int measurement_iterations = 10;
        
        // Warmup
        for (int i = 0; i < warmup_iterations; ++i) {
            operation();
        }
        
        // Measure
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < measurement_iterations; ++i) {
            operation();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_seconds = duration.count() / 1e6;
        
        return (bytes_transferred * measurement_iterations) / time_seconds / 1e9; // GB/s
    }
    
    void generateBandwidthReport() {
        std::ofstream report("bandwidth_utilization_report.csv");
        report << "Test,DataSizeMB,AchievedGBs,TheoreticalGBs,UtilizationPercent,LatencyUs,MemoryType,Streams\n";
        
        for (const auto& result : results_) {
            report << result.test_name << ","
                   << result.data_size_mb << ","
                   << std::fixed << std::setprecision(2) << result.achieved_bandwidth_gb_s << ","
                   << result.theoretical_bandwidth_gb_s << ","
                   << std::setprecision(1) << result.utilization_percent << ","
                   << std::setprecision(0) << result.latency_us << ","
                   << result.memory_type << ","
                   << result.num_streams << "\n";
        }
        
        std::cout << "\nBandwidth utilization report saved to: bandwidth_utilization_report.csv" << std::endl;
    }
};

// Test CPU memory bandwidth utilization
TEST_F(BandwidthUtilizationTest, CPUMemoryBandwidth) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    std::vector<size_t> data_sizes = {
        1 * 1024 * 1024,    // 1 MB
        4 * 1024 * 1024,    // 4 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        256 * 1024 * 1024   // 256 MB
    };
    
    for (size_t data_size : data_sizes) {
        auto src_block = memory_mgr.allocate(data_size, "cpu_bw_src",
                                           UnifiedMemoryManager::MemoryType::HOST_PINNED);
        auto dst_block = memory_mgr.allocate(data_size, "cpu_bw_dst",
                                           UnifiedMemoryManager::MemoryType::HOST_PINNED);
        
        ASSERT_NE(src_block, nullptr);
        ASSERT_NE(dst_block, nullptr);
        
        // Initialize source data
        std::memset(src_block->ptr, 0xAA, data_size);
        
        auto copy_operation = [&]() {
            std::memcpy(dst_block->ptr, src_block->ptr, data_size);
        };
        
        double bandwidth = measureBandwidth(copy_operation, data_size * 2); // Read + Write
        double utilization = (bandwidth / peak_memory_bandwidth_) * 100.0;
        
        BandwidthResult result;
        result.test_name = "CPU_Memory_Copy";
        result.data_size_mb = data_size / (1024 * 1024);
        result.achieved_bandwidth_gb_s = bandwidth;
        result.theoretical_bandwidth_gb_s = peak_memory_bandwidth_;
        result.utilization_percent = utilization;
        result.latency_us = 0.0; // Not measured for bulk transfers
        result.memory_type = "HOST_PINNED";
        result.num_streams = 1;
        
        results_.push_back(result);
        
        std::cout << "CPU Memory - Size: " << result.data_size_mb << " MB"
                  << ", Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
                  << ", Utilization: " << utilization << "%" << std::endl;
        
        // Verify data integrity
        EXPECT_EQ(std::memcmp(src_block->ptr, dst_block->ptr, data_size), 0)
            << "Data corruption detected for size " << data_size;
    }
}

// Test GPU memory bandwidth utilization
TEST_F(BandwidthUtilizationTest, GPUMemoryBandwidth) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA not available, skipping GPU bandwidth test";
    }
    
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    std::vector<size_t> data_sizes = {
        1 * 1024 * 1024,    // 1 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        256 * 1024 * 1024   // 256 MB
    };
    
    for (size_t data_size : data_sizes) {
        auto gpu_src = memory_mgr.allocate(data_size, "gpu_bw_src",
                                          UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        auto gpu_dst = memory_mgr.allocate(data_size, "gpu_bw_dst",
                                          UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        
        ASSERT_NE(gpu_src, nullptr);
        ASSERT_NE(gpu_dst, nullptr);
        
        // Initialize source data
        cudaMemset(gpu_src->ptr, 0xAA, data_size);
        
        auto copy_operation = [&]() {
            cudaMemcpy(gpu_dst->ptr, gpu_src->ptr, data_size, cudaMemcpyDeviceToDevice);
        };
        
        double bandwidth = measureBandwidth(copy_operation, data_size * 2);
        double utilization = (bandwidth / peak_gpu_bandwidth_) * 100.0;
        
        BandwidthResult result;
        result.test_name = "GPU_Device_Copy";
        result.data_size_mb = data_size / (1024 * 1024);
        result.achieved_bandwidth_gb_s = bandwidth;
        result.theoretical_bandwidth_gb_s = peak_gpu_bandwidth_;
        result.utilization_percent = utilization;
        result.latency_us = 0.0;
        result.memory_type = "DEVICE_ONLY";
        result.num_streams = 1;
        
        results_.push_back(result);
        
        std::cout << "GPU Memory - Size: " << result.data_size_mb << " MB"
                  << ", Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
                  << ", Utilization: " << utilization << "%" << std::endl;
    }
}

// Test host-to-device transfer bandwidth
TEST_F(BandwidthUtilizationTest, HostToDeviceBandwidth) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA not available, skipping host-to-device bandwidth test";
    }
    
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    std::vector<size_t> data_sizes = {
        1 * 1024 * 1024,    // 1 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        256 * 1024 * 1024   // 256 MB
    };
    
    for (size_t data_size : data_sizes) {
        auto host_block = memory_mgr.allocate(data_size, "h2d_host",
                                            UnifiedMemoryManager::MemoryType::HOST_PINNED);
        auto device_block = memory_mgr.allocate(data_size, "h2d_device",
                                              UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        
        ASSERT_NE(host_block, nullptr);
        ASSERT_NE(device_block, nullptr);
        
        // Initialize host data
        std::memset(host_block->ptr, 0xBB, data_size);
        
        auto transfer_operation = [&]() {
            cudaMemcpy(device_block->ptr, host_block->ptr, data_size, cudaMemcpyHostToDevice);
        };
        
        double bandwidth = measureBandwidth(transfer_operation, data_size);
        
        // For PCIe transfers, theoretical bandwidth is lower
        double pcie_bandwidth = 16.0; // PCIe 4.0 x16 theoretical: ~16 GB/s
        double utilization = (bandwidth / pcie_bandwidth) * 100.0;
        
        BandwidthResult result;
        result.test_name = "Host_To_Device";
        result.data_size_mb = data_size / (1024 * 1024);
        result.achieved_bandwidth_gb_s = bandwidth;
        result.theoretical_bandwidth_gb_s = pcie_bandwidth;
        result.utilization_percent = utilization;
        result.latency_us = 0.0;
        result.memory_type = "HOST_TO_DEVICE";
        result.num_streams = 1;
        
        results_.push_back(result);
        
        std::cout << "Host->Device - Size: " << result.data_size_mb << " MB"
                  << ", Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
                  << ", PCIe Utilization: " << utilization << "%" << std::endl;
    }
}

// Test async transfer bandwidth with multiple streams
TEST_F(BandwidthUtilizationTest, AsyncTransferBandwidth) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA not available, skipping async transfer test";
    }
    
    AsyncGPUExecutionManager gpu_manager;
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    std::vector<int> stream_counts = {1, 2, 4, 8};
    const size_t total_data_size = 64 * 1024 * 1024; // 64 MB total
    
    for (int num_streams : stream_counts) {
        size_t data_per_stream = total_data_size / num_streams;
        
        std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> host_blocks;
        std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> device_blocks;
        
        // Allocate memory for each stream
        for (int s = 0; s < num_streams; ++s) {
            auto host_block = memory_mgr.allocate(data_per_stream, 
                "async_host_" + std::to_string(s), UnifiedMemoryManager::MemoryType::HOST_PINNED);
            auto device_block = memory_mgr.allocate(data_per_stream,
                "async_device_" + std::to_string(s), UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
            
            host_blocks.push_back(host_block);
            device_blocks.push_back(device_block);
            
            // Initialize data
            std::memset(host_block->ptr, s + 1, data_per_stream);
        }
        
        auto async_transfer = [&]() {
            // Launch transfers on all streams
            for (int s = 0; s < num_streams; ++s) {
                gpu_manager.memcpyAsync(device_blocks[s]->ptr, host_blocks[s]->ptr,
                                       data_per_stream, cudaMemcpyHostToDevice, s);
            }
            gpu_manager.synchronizeAll();
        };
        
        double bandwidth = measureBandwidth(async_transfer, total_data_size);
        
        double pcie_bandwidth = 16.0; // PCIe theoretical
        double utilization = (bandwidth / pcie_bandwidth) * 100.0;
        
        BandwidthResult result;
        result.test_name = "Async_Multi_Stream";
        result.data_size_mb = total_data_size / (1024 * 1024);
        result.achieved_bandwidth_gb_s = bandwidth;
        result.theoretical_bandwidth_gb_s = pcie_bandwidth;
        result.utilization_percent = utilization;
        result.latency_us = 0.0;
        result.memory_type = "HOST_TO_DEVICE_ASYNC";
        result.num_streams = num_streams;
        
        results_.push_back(result);
        
        std::cout << "Async " << num_streams << " streams - Total: " << result.data_size_mb << " MB"
                  << ", Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
                  << ", Utilization: " << utilization << "%" << std::endl;
    }
}

// Test memory latency
TEST_F(BandwidthUtilizationTest, MemoryLatency) {
    auto& memory_mgr = UnifiedMemoryManager::getInstance();
    
    const size_t small_size = 4096; // 4KB for latency measurement
    const int num_iterations = 10000;
    
    // CPU memory latency
    auto host_block = memory_mgr.allocate(small_size, "latency_host",
                                        UnifiedMemoryManager::MemoryType::HOST_PINNED);
    ASSERT_NE(host_block, nullptr);
    
    volatile char* ptr = static_cast<volatile char*>(host_block->ptr);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        volatile char value = ptr[i % small_size];
        (void)value; // Prevent optimization
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double cpu_latency_ns = static_cast<double>(duration.count()) / num_iterations;
    
    BandwidthResult cpu_result;
    cpu_result.test_name = "CPU_Memory_Latency";
    cpu_result.data_size_mb = 0;
    cpu_result.achieved_bandwidth_gb_s = 0.0;
    cpu_result.theoretical_bandwidth_gb_s = 0.0;
    cpu_result.utilization_percent = 0.0;
    cpu_result.latency_us = cpu_latency_ns / 1000.0;
    cpu_result.memory_type = "HOST_PINNED";
    cpu_result.num_streams = 1;
    
    results_.push_back(cpu_result);
    
    std::cout << "CPU Memory Latency: " << std::fixed << std::setprecision(1) 
              << cpu_latency_ns << " ns" << std::endl;
    
    // GPU memory latency (if available)
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        auto gpu_block = memory_mgr.allocate(small_size, "latency_gpu",
                                           UnifiedMemoryManager::MemoryType::DEVICE_ONLY);
        ASSERT_NE(gpu_block, nullptr);
        
        // Measure GPU kernel launch latency
        cudaEvent_t start_event, end_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
        
        cudaEventRecord(start_event);
        for (int i = 0; i < 100; ++i) {
            cudaMemset(gpu_block->ptr, i % 256, 1); // Minimal operation
        }
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        
        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, start_event, end_event);
        double gpu_latency_us = (gpu_time_ms * 1000.0) / 100.0;
        
        BandwidthResult gpu_result;
        gpu_result.test_name = "GPU_Kernel_Latency";
        gpu_result.data_size_mb = 0;
        gpu_result.achieved_bandwidth_gb_s = 0.0;
        gpu_result.theoretical_bandwidth_gb_s = 0.0;
        gpu_result.utilization_percent = 0.0;
        gpu_result.latency_us = gpu_latency_us;
        gpu_result.memory_type = "DEVICE_ONLY";
        gpu_result.num_streams = 1;
        
        results_.push_back(gpu_result);
        
        std::cout << "GPU Kernel Launch Latency: " << std::fixed << std::setprecision(1)
                  << gpu_latency_us << " μs" << std::endl;
        
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running Bandwidth Utilization Tests..." << std::endl;
    std::cout << "Results will be saved to: bandwidth_utilization_report.csv" << std::endl;
    
    return RUN_ALL_TESTS();
}
