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
    };

    std::vector<MemoryBlock> memory_blocks_;  ///< List of memory blocks
    mutable std::mutex mutex_;                ///< Mutex for thread safety

    // Statistics
    size_t total_allocated_;                  ///< Total memory allocated
    size_t current_used_;                     ///< Current memory in use
    size_t allocation_count_;                 ///< Number of allocations
    size_t reuse_count_;                      ///< Number of reuses
};

#endif // USE_CUDA
