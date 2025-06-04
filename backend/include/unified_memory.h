#pragma once

/**
 * @file unified_memory.h
 * @brief Unified Memory Architecture for QDSim
 * 
 * Provides unified memory management across CPU/GPU with RAII-based design
 * for quantum device simulations with automatic memory optimization.
 */

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstddef>

namespace QDSim {
namespace Memory {

/**
 * @brief Memory allocation strategies
 */
enum class AllocationStrategy {
    CPU_ONLY,           ///< CPU memory only
    GPU_ONLY,           ///< GPU memory only  
    UNIFIED,            ///< Unified CPU/GPU memory
    ADAPTIVE            ///< Adaptive based on usage patterns
};

/**
 * @brief Memory access patterns for optimization
 */
enum class AccessPattern {
    SEQUENTIAL,         ///< Sequential access
    RANDOM,            ///< Random access
    STREAMING,         ///< Streaming access
    COMPUTE_INTENSIVE  ///< Compute-intensive workloads
};

/**
 * @brief RAII-based unified memory block
 * 
 * Automatically manages memory allocation/deallocation across CPU/GPU
 * with transparent data movement and caching.
 */
template<typename T>
class UnifiedMemoryBlock {
private:
    T* cpu_ptr_;                    ///< CPU memory pointer
    T* gpu_ptr_;                    ///< GPU memory pointer
    size_t size_;                   ///< Number of elements
    size_t capacity_;               ///< Allocated capacity
    AllocationStrategy strategy_;   ///< Memory allocation strategy
    AccessPattern pattern_;         ///< Access pattern hint
    
    mutable std::mutex mutex_;      ///< Thread safety
    mutable bool cpu_valid_;        ///< CPU data is valid
    mutable bool gpu_valid_;        ///< GPU data is valid
    
    std::atomic<size_t> access_count_;  ///< Access counter for optimization
    
public:
    /**
     * @brief Constructor with size and strategy
     */
    explicit UnifiedMemoryBlock(size_t size, 
                               AllocationStrategy strategy = AllocationStrategy::ADAPTIVE,
                               AccessPattern pattern = AccessPattern::SEQUENTIAL);
    
    /**
     * @brief Destructor - RAII cleanup
     */
    ~UnifiedMemoryBlock();
    
    // Non-copyable but movable
    UnifiedMemoryBlock(const UnifiedMemoryBlock&) = delete;
    UnifiedMemoryBlock& operator=(const UnifiedMemoryBlock&) = delete;
    UnifiedMemoryBlock(UnifiedMemoryBlock&& other) noexcept;
    UnifiedMemoryBlock& operator=(UnifiedMemoryBlock&& other) noexcept;
    
    /**
     * @brief Get CPU pointer with automatic synchronization
     */
    T* cpu_data() const;
    
    /**
     * @brief Get GPU pointer with automatic synchronization  
     */
    T* gpu_data() const;
    
    /**
     * @brief Get raw pointer (preferred location)
     */
    T* data() const;
    
    /**
     * @brief Element access with bounds checking
     */
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    
    /**
     * @brief Size and capacity
     */
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    
    /**
     * @brief Resize with reallocation if needed
     */
    void resize(size_t new_size);
    void reserve(size_t new_capacity);
    
    /**
     * @brief Memory synchronization control
     */
    void sync_to_cpu() const;
    void sync_to_gpu() const;
    void invalidate_cpu() const;
    void invalidate_gpu() const;
    
    /**
     * @brief Performance optimization hints
     */
    void set_access_pattern(AccessPattern pattern);
    void prefetch_to_gpu() const;
    void prefetch_to_cpu() const;
    
    /**
     * @brief Memory statistics
     */
    size_t memory_usage() const;
    double cache_hit_ratio() const;
    size_t access_count() const { return access_count_.load(); }

private:
    void allocate_cpu();
    void allocate_gpu();
    void deallocate_cpu();
    void deallocate_gpu();
    void sync_cpu_to_gpu() const;
    void sync_gpu_to_cpu() const;
};

/**
 * @brief Unified memory allocator
 * 
 * Global memory manager with automatic optimization and caching.
 */
class UnifiedAllocator {
private:
    static UnifiedAllocator* instance_;
    static std::mutex instance_mutex_;
    
    std::unordered_map<void*, size_t> allocations_;
    std::mutex allocations_mutex_;
    
    // Memory pools for different sizes
    std::vector<std::unique_ptr<void, void(*)(void*)>> small_pool_;
    std::vector<std::unique_ptr<void, void(*)(void*)>> large_pool_;
    
    // Statistics
    std::atomic<size_t> total_allocated_;
    std::atomic<size_t> peak_usage_;
    std::atomic<size_t> allocation_count_;
    
public:
    /**
     * @brief Singleton access
     */
    static UnifiedAllocator& instance();
    
    /**
     * @brief Allocate unified memory
     */
    template<typename T>
    UnifiedMemoryBlock<T> allocate(size_t count, 
                                  AllocationStrategy strategy = AllocationStrategy::ADAPTIVE);
    
    /**
     * @brief Memory statistics
     */
    size_t total_allocated() const { return total_allocated_.load(); }
    size_t peak_usage() const { return peak_usage_.load(); }
    size_t allocation_count() const { return allocation_count_.load(); }
    
    /**
     * @brief Memory optimization
     */
    void optimize_memory_layout();
    void garbage_collect();
    void set_memory_limit(size_t limit_bytes);
    
    /**
     * @brief Performance monitoring
     */
    void enable_profiling(bool enable = true);
    void print_memory_report() const;

private:
    UnifiedAllocator() = default;
    ~UnifiedAllocator() = default;
    
    void* allocate_raw(size_t bytes, AllocationStrategy strategy);
    void deallocate_raw(void* ptr);
};

/**
 * @brief RAII memory scope guard
 * 
 * Automatically manages memory scope and cleanup for quantum simulations.
 */
class MemoryScope {
private:
    std::vector<std::function<void()>> cleanup_functions_;
    std::string scope_name_;
    
public:
    explicit MemoryScope(const std::string& name);
    ~MemoryScope();
    
    /**
     * @brief Register cleanup function
     */
    void register_cleanup(std::function<void()> cleanup);
    
    /**
     * @brief Create scoped memory block
     */
    template<typename T>
    UnifiedMemoryBlock<T>& create_block(size_t size, 
                                       AllocationStrategy strategy = AllocationStrategy::ADAPTIVE);
};

/**
 * @brief Memory-mapped quantum arrays for large datasets
 */
template<typename T>
class QuantumArray {
private:
    UnifiedMemoryBlock<T> data_;
    std::vector<size_t> dimensions_;
    std::vector<size_t> strides_;
    
public:
    /**
     * @brief Constructor for multi-dimensional arrays
     */
    explicit QuantumArray(const std::vector<size_t>& dimensions,
                         AllocationStrategy strategy = AllocationStrategy::ADAPTIVE);
    
    /**
     * @brief Multi-dimensional access
     */
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;
    
    /**
     * @brief Array properties
     */
    const std::vector<size_t>& dimensions() const { return dimensions_; }
    size_t total_size() const;
    size_t dimension_count() const { return dimensions_.size(); }
    
    /**
     * @brief Memory access
     */
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    
    /**
     * @brief Quantum-specific operations
     */
    void zero_initialize();
    void random_initialize(double mean = 0.0, double stddev = 1.0);
    void copy_from(const QuantumArray<T>& other);
    
    /**
     * @brief Performance optimization
     */
    void optimize_for_fft();
    void optimize_for_matrix_ops();
    void prefetch_for_computation();
};

// Type aliases for common quantum data types
using ComplexArray = QuantumArray<std::complex<double>>;
using RealArray = QuantumArray<double>;
using IntArray = QuantumArray<int>;

/**
 * @brief Memory performance profiler
 */
class MemoryProfiler {
public:
    struct ProfileData {
        size_t allocations;
        size_t deallocations;
        size_t bytes_allocated;
        size_t bytes_deallocated;
        double avg_allocation_time;
        double avg_access_time;
        size_t cache_hits;
        size_t cache_misses;
    };
    
    static void start_profiling();
    static void stop_profiling();
    static ProfileData get_profile_data();
    static void reset_profile_data();
    static void print_profile_report();
};

} // namespace Memory
} // namespace QDSim

// Convenience macros for memory management
#define QDSIM_MEMORY_SCOPE(name) QDSim::Memory::MemoryScope _scope(name)
#define QDSIM_ALLOC(type, size) QDSim::Memory::UnifiedAllocator::instance().allocate<type>(size)
#define QDSIM_ARRAY(type, dims) QDSim::Memory::QuantumArray<type>(dims)
