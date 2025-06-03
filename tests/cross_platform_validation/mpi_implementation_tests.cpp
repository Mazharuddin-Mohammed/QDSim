#include "../../backend/include/unified_parallel_manager.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <string>
#include <map>
#include <chrono>

using namespace QDSim;

class MPIImplementationTest : public ::testing::Test {
protected:
    struct MPIInfo {
        std::string implementation_name;
        std::string version;
        int thread_support_level;
        bool supports_cuda_aware;
        bool supports_gpu_direct;
        int max_tag;
        int max_processor_name_len;
    };
    
    MPIInfo mpi_info_;
    int rank_;
    int size_;
    
    void SetUp() override {
        // Initialize MPI if not already done
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        
        detectMPIImplementation();
        
        if (rank_ == 0) {
            std::cout << "MPI Implementation: " << mpi_info_.implementation_name << std::endl;
            std::cout << "Version: " << mpi_info_.version << std::endl;
            std::cout << "Processes: " << size_ << std::endl;
            std::cout << "Thread support: " << getThreadSupportName(mpi_info_.thread_support_level) << std::endl;
        }
    }
    
    void TearDown() override {
        MPI_Finalize();
    }
    
    void detectMPIImplementation() {
        // Get MPI version
        int version, subversion;
        MPI_Get_version(&version, &subversion);
        mpi_info_.version = std::to_string(version) + "." + std::to_string(subversion);
        
        // Get thread support level
        int provided;
        MPI_Query_thread(&provided);
        mpi_info_.thread_support_level = provided;
        
        // Try to detect implementation name
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        
        // Check for implementation-specific features
        detectImplementationSpecificFeatures();
        
        // Get MPI limits
        int flag;
        void* attr_val;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &attr_val, &flag);
        if (flag) {
            mpi_info_.max_tag = *static_cast<int*>(attr_val);
        }
    }
    
    void detectImplementationSpecificFeatures() {
        // Try to detect OpenMPI
        #ifdef OPEN_MPI
        mpi_info_.implementation_name = "OpenMPI";
        #elif defined(MPICH)
        mpi_info_.implementation_name = "MPICH";
        #elif defined(INTEL_MPI)
        mpi_info_.implementation_name = "Intel MPI";
        #elif defined(CRAY_MPI)
        mpi_info_.implementation_name = "Cray MPI";
        #else
        mpi_info_.implementation_name = "Unknown";
        #endif
        
        // Check for CUDA-aware MPI support
        #ifdef CUDA_AWARE_MPI
        mpi_info_.supports_cuda_aware = true;
        #else
        mpi_info_.supports_cuda_aware = false;
        #endif
        
        // Check for GPU Direct support
        mpi_info_.supports_gpu_direct = false; // Will be tested dynamically
    }
    
    std::string getThreadSupportName(int level) {
        switch (level) {
            case MPI_THREAD_SINGLE: return "MPI_THREAD_SINGLE";
            case MPI_THREAD_FUNNELED: return "MPI_THREAD_FUNNELED";
            case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
            case MPI_THREAD_MULTIPLE: return "MPI_THREAD_MULTIPLE";
            default: return "Unknown";
        }
    }
};

// Test basic MPI functionality
TEST_F(MPIImplementationTest, BasicMPIFunctionality) {
    // Test basic point-to-point communication
    if (size_ >= 2) {
        const int message_size = 1000;
        std::vector<int> send_data(message_size);
        std::vector<int> recv_data(message_size);
        
        // Initialize data
        for (int i = 0; i < message_size; ++i) {
            send_data[i] = rank_ * message_size + i;
        }
        
        if (rank_ == 0) {
            // Send to rank 1
            MPI_Send(send_data.data(), message_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
            
            // Receive from rank 1
            MPI_Recv(recv_data.data(), message_size, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Verify received data
            for (int i = 0; i < message_size; ++i) {
                EXPECT_EQ(recv_data[i], 1 * message_size + i) 
                    << "Point-to-point communication failed at index " << i;
            }
            
            std::cout << "✓ Point-to-point communication works" << std::endl;
            
        } else if (rank_ == 1) {
            // Receive from rank 0
            MPI_Recv(recv_data.data(), message_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Send to rank 0
            MPI_Send(send_data.data(), message_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
            
            // Verify received data
            for (int i = 0; i < message_size; ++i) {
                EXPECT_EQ(recv_data[i], 0 * message_size + i) 
                    << "Point-to-point communication failed at index " << i;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// Test collective operations
TEST_F(MPIImplementationTest, CollectiveOperations) {
    const int data_size = 100;
    std::vector<int> local_data(data_size);
    std::vector<int> global_data;
    
    // Initialize local data
    for (int i = 0; i < data_size; ++i) {
        local_data[i] = rank_ * data_size + i;
    }
    
    // Test Allgather
    global_data.resize(data_size * size_);
    MPI_Allgather(local_data.data(), data_size, MPI_INT,
                  global_data.data(), data_size, MPI_INT, MPI_COMM_WORLD);
    
    // Verify Allgather results
    for (int r = 0; r < size_; ++r) {
        for (int i = 0; i < data_size; ++i) {
            int expected = r * data_size + i;
            int actual = global_data[r * data_size + i];
            EXPECT_EQ(actual, expected) 
                << "Allgather failed for rank " << r << " index " << i;
        }
    }
    
    if (rank_ == 0) {
        std::cout << "✓ Allgather works correctly" << std::endl;
    }
    
    // Test reduction operations
    int local_sum = 0;
    for (int val : local_data) {
        local_sum += val;
    }
    
    int global_sum = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Calculate expected global sum
    int expected_global_sum = 0;
    for (int r = 0; r < size_; ++r) {
        for (int i = 0; i < data_size; ++i) {
            expected_global_sum += r * data_size + i;
        }
    }
    
    EXPECT_EQ(global_sum, expected_global_sum) << "Allreduce sum incorrect";
    
    if (rank_ == 0) {
        std::cout << "✓ Allreduce works correctly" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// Test thread safety with MPI
TEST_F(MPIImplementationTest, ThreadSafety) {
    if (mpi_info_.thread_support_level < MPI_THREAD_MULTIPLE) {
        GTEST_SKIP() << "MPI implementation doesn't support MPI_THREAD_MULTIPLE";
    }
    
    const int num_threads = 4;
    const int messages_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> successful_operations{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, messages_per_thread, &successful_operations]() {
            try {
                for (int i = 0; i < messages_per_thread; ++i) {
                    int data = rank_ * 10000 + t * 1000 + i;
                    int result = 0;
                    
                    // Thread-safe MPI operation
                    MPI_Allreduce(&data, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                    
                    // Verify result is reasonable
                    EXPECT_GT(result, 0) << "Thread-safe MPI operation failed";
                    
                    successful_operations.fetch_add(1);
                }
            } catch (const std::exception& e) {
                FAIL() << "Thread " << t << " failed: " << e.what();
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successful_operations.load(), num_threads * messages_per_thread);
    
    if (rank_ == 0) {
        std::cout << "✓ Thread-safe MPI operations work" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// Test large message handling
TEST_F(MPIImplementationTest, LargeMessageHandling) {
    if (size_ < 2) {
        GTEST_SKIP() << "Need at least 2 MPI processes for large message test";
    }
    
    // Test with different message sizes
    std::vector<size_t> message_sizes = {
        1024,           // 1 KB
        1024 * 1024,    // 1 MB
        16 * 1024 * 1024, // 16 MB
        64 * 1024 * 1024  // 64 MB
    };
    
    for (size_t msg_size : message_sizes) {
        std::vector<char> send_buffer(msg_size);
        std::vector<char> recv_buffer(msg_size);
        
        // Initialize with pattern
        for (size_t i = 0; i < msg_size; ++i) {
            send_buffer[i] = static_cast<char>((rank_ + i) % 256);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (rank_ == 0) {
            MPI_Send(send_buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(recv_buffer.data(), msg_size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank_ == 1) {
            MPI_Recv(recv_buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_buffer.data(), msg_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (rank_ == 0) {
            // Verify data integrity
            bool data_correct = true;
            for (size_t i = 0; i < std::min(msg_size, size_t(1000)); ++i) {
                char expected = static_cast<char>((1 + i) % 256);
                if (recv_buffer[i] != expected) {
                    data_correct = false;
                    break;
                }
            }
            
            EXPECT_TRUE(data_correct) << "Large message data corruption for size " << msg_size;
            
            double bandwidth_mb_s = (msg_size * 2) / (duration.count() / 1e6) / (1024 * 1024);
            std::cout << "Message size: " << (msg_size / (1024 * 1024)) << " MB, "
                      << "Bandwidth: " << std::fixed << std::setprecision(1) << bandwidth_mb_s << " MB/s" << std::endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank_ == 0) {
        std::cout << "✓ Large message handling works" << std::endl;
    }
}

// Test CUDA-aware MPI (if supported)
TEST_F(MPIImplementationTest, CUDAAwareMPI) {
    if (!mpi_info_.supports_cuda_aware) {
        GTEST_SKIP() << "CUDA-aware MPI not supported";
    }
    
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA not available for CUDA-aware MPI test";
    }
    
    if (size_ < 2) {
        GTEST_SKIP() << "Need at least 2 MPI processes for CUDA-aware MPI test";
    }
    
    // Set CUDA device based on rank
    cudaSetDevice(rank_ % device_count);
    
    const size_t data_size = 1024 * 1024; // 1MB
    float* gpu_send_buffer = nullptr;
    float* gpu_recv_buffer = nullptr;
    
    cudaError_t err = cudaMalloc(&gpu_send_buffer, data_size * sizeof(float));
    ASSERT_EQ(err, cudaSuccess) << "GPU memory allocation failed";
    
    err = cudaMalloc(&gpu_recv_buffer, data_size * sizeof(float));
    ASSERT_EQ(err, cudaSuccess) << "GPU memory allocation failed";
    
    // Initialize GPU data
    std::vector<float> host_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        host_data[i] = rank_ * 1000.0f + static_cast<float>(i);
    }
    
    err = cudaMemcpy(gpu_send_buffer, host_data.data(), 
                    data_size * sizeof(float), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);
    
    // Test CUDA-aware MPI communication
    if (rank_ == 0) {
        MPI_Send(gpu_send_buffer, data_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(gpu_recv_buffer, data_size, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank_ == 1) {
        MPI_Recv(gpu_recv_buffer, data_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(gpu_send_buffer, data_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
    
    // Verify data on GPU
    std::vector<float> recv_host_data(data_size);
    err = cudaMemcpy(recv_host_data.data(), gpu_recv_buffer,
                    data_size * sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
    
    if (rank_ == 0) {
        // Verify received data from rank 1
        bool data_correct = true;
        for (size_t i = 0; i < std::min(data_size, size_t(100)); ++i) {
            float expected = 1 * 1000.0f + static_cast<float>(i);
            if (std::abs(recv_host_data[i] - expected) > 1e-6) {
                data_correct = false;
                break;
            }
        }
        EXPECT_TRUE(data_correct) << "CUDA-aware MPI data corruption";
        std::cout << "✓ CUDA-aware MPI works correctly" << std::endl;
    }
    
    cudaFree(gpu_send_buffer);
    cudaFree(gpu_recv_buffer);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// Test MPI performance characteristics
TEST_F(MPIImplementationTest, PerformanceCharacteristics) {
    if (size_ < 2) {
        GTEST_SKIP() << "Need at least 2 MPI processes for performance test";
    }
    
    // Test latency (small messages)
    const int num_iterations = 1000;
    const int small_msg_size = 8; // 8 bytes
    
    if (rank_ == 0) {
        char send_data[small_msg_size] = {0};
        char recv_data[small_msg_size] = {0};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            MPI_Send(send_data, small_msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(recv_data, small_msg_size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double latency_us = duration.count() / (2.0 * num_iterations); // Round-trip time / 2
        std::cout << "MPI Latency: " << std::fixed << std::setprecision(2) << latency_us << " μs" << std::endl;
        
        // Latency should be reasonable (less than 100 μs for most implementations)
        EXPECT_LT(latency_us, 100.0) << "MPI latency too high";
        
    } else if (rank_ == 1) {
        char send_data[small_msg_size] = {0};
        char recv_data[small_msg_size] = {0};
        
        for (int i = 0; i < num_iterations; ++i) {
            MPI_Recv(recv_data, small_msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_data, small_msg_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank_ == 0) {
        std::cout << "✓ MPI performance characteristics measured" << std::endl;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Note: MPI_Init is called in SetUp(), not here
    std::cout << "Running MPI Implementation Validation Tests..." << std::endl;
    
    return RUN_ALL_TESTS();
}
