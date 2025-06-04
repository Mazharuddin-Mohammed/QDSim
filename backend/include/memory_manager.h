#pragma once

/**
 * @file memory_manager.h
 * @brief Advanced Memory Management for QDSim
 * 
 * RAII-based thread-safe memory manager with automatic optimization,
 * garbage collection, and performance monitoring for quantum simulations.
 */

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <condition_variable>

#include "unified_memory.h"

namespace QDSim {
namespace Memory {

/**
 * @brief Memory allocation statistics
 */
struct AllocationStats {
    size_t total_allocations;
    size_t total_deallocations;
    size_t current_allocations;
    size_t peak_allocations;
    size_t bytes_allocated;
    size_t bytes_deallocated;
    size_t current_bytes;
    size_t peak_bytes;
    double average_allocation_size;
    double fragmentation_ratio;
    std::chrono::steady_clock::time_point last_update;
};

/**
 * @brief Memory pool for efficient allocation of same-sized objects
 */
template<typename T>
class MemoryPool {
private:
    struct Block {
        alignas(T) char data[sizeof(T)];
        Block* next;
    };
    
    Block* free_list_;
    std::vector<std::unique_ptr<Block[]>> chunks_;
    size_t chunk_size_;
    size_t objects_per_chunk_;
    
    mutable std::mutex pool_mutex_;
    std::atomic<size_t> allocated_count_;
    std::atomic<size_t> total_capacity_;
    
public:
    /**
     * @brief Constructor with initial chunk size
     */
    explicit MemoryPool(size_t initial_chunk_size = 1024);
    
    /**
     * @brief Destructor
     */
    ~MemoryPool();
    
    /**
     * @brief Allocate object
     */
    T* allocate();
    
    /**
     * @brief Deallocate object
     */
    void deallocate(T* ptr);
    
    /**
     * @brief Construct object in-place
     */
    template<typename... Args>
    T* construct(Args&&... args);
    
    /**
     * @brief Destroy object
     */
    void destroy(T* ptr);
    
    /**
     * @brief Pool statistics
     */
    size_t allocated_count() const { return allocated_count_.load(); }
    size_t total_capacity() const { return total_capacity_.load(); }
    double utilization() const;
    
    /**
     * @brief Pool management
     */
    void shrink_to_fit();
    void reserve(size_t count);

private:
    void allocate_chunk();
    bool is_from_pool(T* ptr) const;
};

/**
 * @brief Smart pointer with automatic memory tracking
 */
template<typename T>
class TrackedPtr {
private:
    T* ptr_;
    std::shared_ptr<void> tracker_;
    
public:
    /**
     * @brief Constructors
     */
    TrackedPtr() : ptr_(nullptr) {}
    explicit TrackedPtr(T* ptr);
    TrackedPtr(const TrackedPtr& other) = default;
    TrackedPtr(TrackedPtr&& other) noexcept = default;
    
    /**
     * @brief Assignment operators
     */
    TrackedPtr& operator=(const TrackedPtr& other) = default;
    TrackedPtr& operator=(TrackedPtr&& other) noexcept = default;
    
    /**
     * @brief Pointer operations
     */
    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    
    /**
     * @brief Reset pointer
     */
    void reset(T* new_ptr = nullptr);
    
    /**
     * @brief Release ownership
     */
    T* release();
    
    /**
     * @brief Check if unique
     */
    bool unique() const;
    
    /**
     * @brief Reference count
     */
    long use_count() const;
};

/**
 * @brief Garbage collector for automatic memory cleanup
 */
class GarbageCollector {
private:
    std::unordered_set<void*> tracked_pointers_;
    std::unordered_map<void*, std::function<void()>> cleanup_functions_;
    
    mutable std::shared_mutex gc_mutex_;
    std::thread gc_thread_;
    std::atomic<bool> running_;
    std::condition_variable gc_cv_;
    std::mutex gc_cv_mutex_;
    
    // GC configuration
    std::chrono::milliseconds gc_interval_;
    size_t gc_threshold_bytes_;
    double gc_threshold_ratio_;
    
    // Statistics
    std::atomic<size_t> collections_performed_;
    std::atomic<size_t> objects_collected_;
    std::atomic<size_t> bytes_freed_;
    
public:
    /**
     * @brief Constructor
     */
    GarbageCollector();
    
    /**
     * @brief Destructor
     */
    ~GarbageCollector();
    
    /**
     * @brief Start garbage collection thread
     */
    void start();
    
    /**
     * @brief Stop garbage collection thread
     */
    void stop();
    
    /**
     * @brief Register object for tracking
     */
    void track_object(void* ptr, std::function<void()> cleanup);
    
    /**
     * @brief Unregister object
     */
    void untrack_object(void* ptr);
    
    /**
     * @brief Force garbage collection
     */
    void collect();
    
    /**
     * @brief Configuration
     */
    void set_collection_interval(std::chrono::milliseconds interval);
    void set_threshold_bytes(size_t bytes);
    void set_threshold_ratio(double ratio);
    
    /**
     * @brief Statistics
     */
    size_t collections_performed() const { return collections_performed_.load(); }
    size_t objects_collected() const { return objects_collected_.load(); }
    size_t bytes_freed() const { return bytes_freed_.load(); }
    size_t tracked_objects() const;

private:
    void gc_loop();
    bool should_collect() const;
    void perform_collection();
};

/**
 * @brief Memory leak detector
 */
class LeakDetector {
private:
    struct AllocationInfo {
        size_t size;
        std::chrono::steady_clock::time_point timestamp;
        std::string location;
        std::thread::id thread_id;
    };
    
    std::unordered_map<void*, AllocationInfo> allocations_;
    mutable std::shared_mutex detector_mutex_;
    
    std::atomic<bool> enabled_;
    std::atomic<size_t> total_allocations_;
    std::atomic<size_t> total_deallocations_;
    
public:
    /**
     * @brief Constructor
     */
    LeakDetector();
    
    /**
     * @brief Enable/disable leak detection
     */
    void enable(bool enabled = true);
    bool is_enabled() const { return enabled_.load(); }
    
    /**
     * @brief Track allocation
     */
    void track_allocation(void* ptr, size_t size, const std::string& location = "");
    
    /**
     * @brief Track deallocation
     */
    void track_deallocation(void* ptr);
    
    /**
     * @brief Check for leaks
     */
    std::vector<std::pair<void*, AllocationInfo>> find_leaks() const;
    
    /**
     * @brief Print leak report
     */
    void print_leak_report() const;
    
    /**
     * @brief Clear tracking data
     */
    void clear();
    
    /**
     * @brief Statistics
     */
    size_t active_allocations() const;
    size_t total_allocations() const { return total_allocations_.load(); }
    size_t total_deallocations() const { return total_deallocations_.load(); }
};

/**
 * @brief Main memory manager with comprehensive features
 */
class MemoryManager {
private:
    static std::unique_ptr<MemoryManager> instance_;
    static std::mutex instance_mutex_;
    
    // Memory pools for different types
    std::unordered_map<std::type_index, std::unique_ptr<void, void(*)(void*)>> pools_;
    mutable std::shared_mutex pools_mutex_;
    
    // Unified allocator
    std::unique_ptr<UnifiedAllocator> unified_allocator_;
    
    // Garbage collector
    std::unique_ptr<GarbageCollector> garbage_collector_;
    
    // Leak detector
    std::unique_ptr<LeakDetector> leak_detector_;
    
    // Statistics
    AllocationStats stats_;
    mutable std::mutex stats_mutex_;
    
    // Configuration
    bool auto_gc_enabled_;
    bool leak_detection_enabled_;
    bool pool_allocation_enabled_;
    size_t memory_limit_;
    
public:
    /**
     * @brief Get singleton instance
     */
    static MemoryManager& instance();
    
    /**
     * @brief Destructor
     */
    ~MemoryManager();
    
    /**
     * @brief Initialize memory manager
     */
    void initialize();
    
    /**
     * @brief Shutdown memory manager
     */
    void shutdown();
    
    /**
     * @brief Allocate memory
     */
    template<typename T>
    TrackedPtr<T> allocate(size_t count = 1);
    
    /**
     * @brief Allocate from pool
     */
    template<typename T>
    TrackedPtr<T> allocate_from_pool();
    
    /**
     * @brief Allocate unified memory
     */
    template<typename T>
    UnifiedMemoryBlock<T> allocate_unified(size_t count, 
                                          AllocationStrategy strategy = AllocationStrategy::ADAPTIVE);
    
    /**
     * @brief Create quantum array
     */
    template<typename T>
    std::unique_ptr<QuantumArray<T>> create_quantum_array(const std::vector<size_t>& dimensions);
    
    /**
     * @brief Memory management
     */
    void garbage_collect();
    void optimize_memory();
    void defragment();
    
    /**
     * @brief Configuration
     */
    void enable_auto_gc(bool enable = true);
    void enable_leak_detection(bool enable = true);
    void enable_pool_allocation(bool enable = true);
    void set_memory_limit(size_t limit_bytes);
    
    /**
     * @brief Statistics and monitoring
     */
    AllocationStats get_stats() const;
    void print_memory_report() const;
    void print_leak_report() const;
    
    /**
     * @brief Memory health check
     */
    bool check_memory_health() const;
    double memory_utilization() const;
    double fragmentation_ratio() const;

private:
    MemoryManager();
    
    template<typename T>
    MemoryPool<T>& get_pool();
    
    void update_stats(size_t bytes_allocated, bool is_allocation);
    void check_memory_limit() const;
};

/**
 * @brief RAII memory scope with automatic cleanup
 */
class ScopedMemoryManager {
private:
    std::vector<std::function<void()>> cleanup_functions_;
    std::string scope_name_;
    std::chrono::steady_clock::time_point start_time_;
    
public:
    /**
     * @brief Constructor
     */
    explicit ScopedMemoryManager(const std::string& name);
    
    /**
     * @brief Destructor with automatic cleanup
     */
    ~ScopedMemoryManager();
    
    /**
     * @brief Register cleanup function
     */
    void register_cleanup(std::function<void()> cleanup);
    
    /**
     * @brief Allocate scoped memory
     */
    template<typename T>
    TrackedPtr<T> allocate(size_t count = 1);
    
    /**
     * @brief Create scoped quantum array
     */
    template<typename T>
    std::unique_ptr<QuantumArray<T>> create_quantum_array(const std::vector<size_t>& dimensions);
};

/**
 * @brief Memory performance monitor
 */
class MemoryMonitor {
public:
    struct PerformanceMetrics {
        double allocation_rate;      // allocations per second
        double deallocation_rate;    // deallocations per second
        double memory_growth_rate;   // bytes per second
        double gc_efficiency;        // percentage of memory freed by GC
        double pool_hit_ratio;       // percentage of allocations from pools
        double fragmentation_trend;  // fragmentation change over time
    };
    
    static void start_monitoring();
    static void stop_monitoring();
    static PerformanceMetrics get_metrics();
    static void print_performance_report();
};

} // namespace Memory
} // namespace QDSim

// Convenience macros for memory management
#define QDSIM_MEMORY_MANAGER() QDSim::Memory::MemoryManager::instance()
#define QDSIM_SCOPED_MEMORY(name) QDSim::Memory::ScopedMemoryManager _scoped_memory(name)
#define QDSIM_ALLOC_TRACKED(type, count) QDSIM_MEMORY_MANAGER().allocate<type>(count)
#define QDSIM_ALLOC_POOL(type) QDSIM_MEMORY_MANAGER().allocate_from_pool<type>()
#define QDSIM_QUANTUM_ARRAY(type, dims) QDSIM_MEMORY_MANAGER().create_quantum_array<type>(dims)
