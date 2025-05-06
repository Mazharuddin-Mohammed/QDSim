#pragma once
/**
 * @file gpu_memory_pool.h
 * @brief Defines the GPUMemoryPool class for efficient GPU memory management.
 *
 * This file contains the declaration of the GPUMemoryPool class, which provides
 * efficient memory management for GPU operations by reusing memory allocations
 * and reducing the overhead of frequent allocations and deallocations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <string>

// Conditional compilation for CUDA support
#ifdef USE_CUDA
#include <cuda_runtime.h>

/**
 * @class GPUMemoryPool
 * @brief Provides efficient memory management for GPU operations.
 *
 * The GPUMemoryPool class provides efficient memory management for GPU operations
 * by reusing memory allocations and reducing the overhead of frequent allocations
 * and deallocations. It maintains a pool of pre-allocated memory blocks that can
 * be reused for subsequent operations.
 */
class GPUMemoryPool {
public:
    /**
     * @brief Gets the singleton instance of the GPUMemoryPool.
     *
     * @return The singleton instance of the GPUMemoryPool
     */
    static GPUMemoryPool& getInstance();

    /**
     * @brief Allocates memory from the pool.
     *
     * This method allocates memory from the pool. If a suitable memory block
     * is available in the pool, it is returned. Otherwise, a new memory block
     * is allocated and returned.
     *
     * @param size The size of the memory block to allocate in bytes
     * @param tag An optional tag to identify the allocation (for debugging)
     * @return A pointer to the allocated memory block
     * @throws std::runtime_error If memory allocation fails
     */
    void* allocate(size_t size, const std::string& tag = "");

    /**
     * @brief Releases memory back to the pool.
     *
     * This method releases memory back to the pool for reuse. The memory block
     * is not actually freed, but is marked as available for reuse.
     *
     * @param ptr The pointer to the memory block to release
     * @param size The size of the memory block in bytes
     */
    void release(void* ptr, size_t size);

    /**
     * @brief Frees all memory in the pool.
     *
     * This method frees all memory in the pool. It should be called when the
     * pool is no longer needed, typically at program termination.
     */
    void freeAll();

    /**
     * @brief Gets statistics about the memory pool.
     *
     * @return A string containing statistics about the memory pool
     */
    std::string getStats() const;

    /**
     * @brief Sets the maximum pool size.
     *
     * @param size The maximum pool size in bytes
     */
    void setMaxPoolSize(size_t size);

    /**
     * @brief Sets the minimum block size.
     *
     * @param size The minimum block size in bytes
     */
    void setMinBlockSize(size_t size);

    /**
     * @brief Sets the maximum block size.
     *
     * @param size The maximum block size in bytes
     */
    void setMaxBlockSize(size_t size);

    /**
     * @brief Sets the growth factor for block sizes.
     *
     * @param factor The growth factor (must be > 1.0)
     */
    void setGrowthFactor(float factor);

    /**
     * @brief Sets the time-to-live for unused blocks.
     *
     * @param ttl The time-to-live in seconds
     */
    void setBlockTTL(std::chrono::seconds ttl);

    /**
     * @brief Prefetches memory blocks of specified sizes.
     *
     * This method prefetches memory blocks of specified sizes to avoid
     * allocation overhead during runtime.
     *
     * @param sizes A vector of block sizes to prefetch
     */
    void prefetch(const std::vector<size_t>& sizes);

    /**
     * @brief Trims the memory pool to free unused memory.
     *
     * This method trims the memory pool by freeing memory blocks that
     * have not been used for a specified time.
     */
    void trim();

    /**
     * @brief Gets the current memory usage.
     *
     * @return The current memory usage in bytes
     */
    size_t getCurrentUsage() const;

    /**
     * @brief Gets the total allocated memory.
     *
     * @return The total allocated memory in bytes
     */
    size_t getTotalAllocated() const;

private:
    /**
     * @brief Constructs a new GPUMemoryPool object.
     *
     * The constructor is private to enforce the singleton pattern.
     */
    GPUMemoryPool();

    /**
     * @brief Destructor for the GPUMemoryPool object.
     *
     * Frees all memory in the pool.
     */
    ~GPUMemoryPool();

    // Prevent copy and assignment
    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

    // Structure to represent a memory block
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        std::string tag;
        std::chrono::steady_clock::time_point last_used;
    };

    // Memory block size categories for faster lookup
    enum class SizeCategory {
        TINY,     // < 1 KB
        SMALL,    // 1 KB - 64 KB
        MEDIUM,   // 64 KB - 1 MB
        LARGE,    // 1 MB - 16 MB
        HUGE,     // 16 MB - 256 MB
        ENORMOUS  // > 256 MB
    };

    // Convert size to category
    SizeCategory getSizeCategory(size_t size) const;

    // Get blocks by size category
    std::vector<MemoryBlock*> getBlocksByCategory(SizeCategory category);

    // Trim the pool to free memory when needed
    void trimPool(size_t required_size);

    // Check if we should grow the pool
    bool shouldGrowPool(size_t required_size) const;

    // Get available GPU memory
    size_t getAvailableGPUMemory() const;

    // Map of memory blocks by size category
    std::unordered_map<SizeCategory, std::vector<MemoryBlock>> memory_blocks_by_category_;

    // List of all memory blocks for iteration
    std::vector<MemoryBlock*> all_memory_blocks_;

    mutable std::mutex mutex_;                ///< Mutex for thread safety

    // Configuration
    size_t max_pool_size_;                    ///< Maximum pool size in bytes
    size_t min_block_size_;                   ///< Minimum block size in bytes
    size_t max_block_size_;                   ///< Maximum block size in bytes
    float growth_factor_;                     ///< Growth factor for block sizes
    std::chrono::seconds block_ttl_;          ///< Time-to-live for unused blocks

    // Statistics
    size_t total_allocated_;                  ///< Total memory allocated
    size_t current_used_;                     ///< Current memory in use
    size_t allocation_count_;                 ///< Number of allocations
    size_t reuse_count_;                      ///< Number of reuses
    size_t trim_count_;                       ///< Number of pool trims
    size_t oom_count_;                        ///< Number of out-of-memory errors
};

#endif // USE_CUDA
