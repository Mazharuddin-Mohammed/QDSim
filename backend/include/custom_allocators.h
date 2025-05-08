#pragma once
/**
 * @file custom_allocators.h
 * @brief Defines custom allocators for specific data structures in QDSim.
 *
 * This file contains declarations of custom allocators for specific data structures
 * to improve memory efficiency in large-scale quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <memory>
#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <string>
#include <mutex>
#include <atomic>
#include <cstddef>
#include <type_traits>
#include <limits>
#include <new>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Allocators {

/**
 * @class ArenaAllocator
 * @brief Memory arena allocator for temporary allocations.
 *
 * The ArenaAllocator class provides a memory arena for temporary allocations.
 * It allocates memory in large chunks and then suballocates from these chunks,
 * reducing the overhead of frequent allocations and deallocations.
 */
class ArenaAllocator {
public:
    /**
     * @brief Gets the singleton instance of the ArenaAllocator.
     *
     * @return The singleton instance
     */
    static ArenaAllocator& getInstance();
    
    /**
     * @brief Allocates memory from the arena.
     *
     * @param size The size of the memory to allocate in bytes
     * @param alignment The alignment of the memory (default: 16)
     * @return Pointer to the allocated memory
     */
    void* allocate(size_t size, size_t alignment = 16);
    
    /**
     * @brief Deallocates memory from the arena.
     *
     * @param ptr The pointer to the memory to deallocate
     * @param size The size of the memory in bytes
     */
    void deallocate(void* ptr, size_t size);
    
    /**
     * @brief Resets the arena.
     *
     * This method resets the arena, making all memory available for allocation again.
     * It does not free the memory, but marks it as unused.
     */
    void reset();
    
    /**
     * @brief Clears the arena.
     *
     * This method clears the arena, freeing all memory.
     */
    void clear();
    
    /**
     * @brief Gets statistics about the arena.
     *
     * @return A string containing statistics about the arena
     */
    std::string getStats() const;
    
    /**
     * @brief Sets the chunk size.
     *
     * @param size The chunk size in bytes
     */
    void setChunkSize(size_t size);
    
    /**
     * @brief Gets the chunk size.
     *
     * @return The chunk size in bytes
     */
    size_t getChunkSize() const;
    
    /**
     * @brief Sets the maximum arena size.
     *
     * @param size The maximum arena size in bytes
     */
    void setMaxArenaSize(size_t size);
    
    /**
     * @brief Gets the maximum arena size.
     *
     * @return The maximum arena size in bytes
     */
    size_t getMaxArenaSize() const;
    
private:
    /**
     * @brief Constructs a new ArenaAllocator object.
     */
    ArenaAllocator();
    
    /**
     * @brief Destructor for the ArenaAllocator object.
     */
    ~ArenaAllocator();
    
    // Prevent copying and assignment
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;
    
    // Structure to represent a memory chunk
    struct Chunk {
        void* memory;       ///< Pointer to the memory
        size_t size;        ///< Size of the memory in bytes
        size_t used;        ///< Used memory in bytes
        Chunk* next;        ///< Next chunk in the list
    };
    
    // Structure to represent an allocation
    struct Allocation {
        Chunk* chunk;       ///< Chunk containing the allocation
        size_t offset;      ///< Offset within the chunk
        size_t size;        ///< Size of the allocation in bytes
    };
    
    Chunk* current_chunk_;                  ///< Current chunk for allocation
    std::unordered_map<void*, Allocation> allocations_; ///< Map of allocations
    size_t chunk_size_;                     ///< Chunk size in bytes
    size_t max_arena_size_;                 ///< Maximum arena size in bytes
    size_t total_allocated_;                ///< Total allocated memory in bytes
    mutable std::mutex mutex_;              ///< Mutex for thread safety
    
    /**
     * @brief Allocates a new chunk.
     *
     * @param min_size The minimum size of the chunk in bytes
     * @return Pointer to the new chunk
     */
    Chunk* allocateChunk(size_t min_size);
    
    /**
     * @brief Frees a chunk.
     *
     * @param chunk The chunk to free
     */
    void freeChunk(Chunk* chunk);
};

/**
 * @class ArenaAllocatorAdapter
 * @brief Adapter for using ArenaAllocator with STL containers.
 *
 * The ArenaAllocatorAdapter class provides an adapter for using ArenaAllocator
 * with STL containers.
 *
 * @tparam T The value type
 */
template <typename T>
class ArenaAllocatorAdapter {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    /**
     * @brief Rebind allocator to another type.
     *
     * @tparam U The other type
     */
    template <typename U>
    struct rebind {
        using other = ArenaAllocatorAdapter<U>;
    };
    
    /**
     * @brief Constructs a new ArenaAllocatorAdapter object.
     */
    ArenaAllocatorAdapter() noexcept = default;
    
    /**
     * @brief Copy constructor.
     *
     * @param other The other allocator
     */
    ArenaAllocatorAdapter(const ArenaAllocatorAdapter& other) noexcept = default;
    
    /**
     * @brief Copy constructor for different type.
     *
     * @tparam U The other type
     * @param other The other allocator
     */
    template <typename U>
    ArenaAllocatorAdapter(const ArenaAllocatorAdapter<U>& other) noexcept {}
    
    /**
     * @brief Destructor.
     */
    ~ArenaAllocatorAdapter() noexcept = default;
    
    /**
     * @brief Allocates memory.
     *
     * @param n The number of elements to allocate
     * @return Pointer to the allocated memory
     */
    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        size_type bytes = n * sizeof(T);
        pointer p = static_cast<pointer>(ArenaAllocator::getInstance().allocate(bytes, alignof(T)));
        
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        
        return p;
    }
    
    /**
     * @brief Deallocates memory.
     *
     * @param p Pointer to the memory
     * @param n The number of elements
     */
    void deallocate(pointer p, size_type n) noexcept {
        ArenaAllocator::getInstance().deallocate(p, n * sizeof(T));
    }
    
    /**
     * @brief Constructs an object.
     *
     * @param p Pointer to the memory
     * @param args Arguments for the constructor
     */
    template <typename... Args>
    void construct(pointer p, Args&&... args) {
        new(p) T(std::forward<Args>(args)...);
    }
    
    /**
     * @brief Destroys an object.
     *
     * @param p Pointer to the object
     */
    void destroy(pointer p) {
        p->~T();
    }
    
    /**
     * @brief Equality operator.
     *
     * @param other The other allocator
     * @return True if the allocators are equal, false otherwise
     */
    bool operator==(const ArenaAllocatorAdapter& other) const noexcept {
        return true;
    }
    
    /**
     * @brief Inequality operator.
     *
     * @param other The other allocator
     * @return True if the allocators are not equal, false otherwise
     */
    bool operator!=(const ArenaAllocatorAdapter& other) const noexcept {
        return false;
    }
};

/**
 * @class PoolAllocator
 * @brief Memory pool allocator for fixed-size allocations.
 *
 * The PoolAllocator class provides a memory pool for fixed-size allocations.
 * It allocates memory in large chunks and then suballocates fixed-size blocks
 * from these chunks, reducing the overhead of frequent allocations and deallocations.
 *
 * @tparam BlockSize The size of each block in bytes
 */
template <size_t BlockSize>
class PoolAllocator {
public:
    /**
     * @brief Gets the singleton instance of the PoolAllocator.
     *
     * @return The singleton instance
     */
    static PoolAllocator& getInstance() {
        static PoolAllocator instance;
        return instance;
    }
    
    /**
     * @brief Allocates a block from the pool.
     *
     * @return Pointer to the allocated block
     */
    void* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if there are free blocks
        if (free_blocks_ == nullptr) {
            // Allocate a new chunk
            allocateChunk();
            
            if (free_blocks_ == nullptr) {
                throw std::bad_alloc();
            }
        }
        
        // Get a free block
        void* block = free_blocks_;
        free_blocks_ = *reinterpret_cast<void**>(free_blocks_);
        
        // Update statistics
        allocated_blocks_++;
        
        return block;
    }
    
    /**
     * @brief Deallocates a block back to the pool.
     *
     * @param block Pointer to the block
     */
    void deallocate(void* block) {
        if (block == nullptr) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Add the block to the free list
        *reinterpret_cast<void**>(block) = free_blocks_;
        free_blocks_ = block;
        
        // Update statistics
        allocated_blocks_--;
    }
    
    /**
     * @brief Gets statistics about the pool.
     *
     * @return A string containing statistics about the pool
     */
    std::string getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ostringstream ss;
        ss << "Pool Allocator (Block Size: " << BlockSize << " bytes) Statistics:" << std::endl;
        ss << "  Chunks: " << chunks_.size() << std::endl;
        ss << "  Blocks per chunk: " << blocks_per_chunk_ << std::endl;
        ss << "  Total blocks: " << chunks_.size() * blocks_per_chunk_ << std::endl;
        ss << "  Allocated blocks: " << allocated_blocks_ << std::endl;
        ss << "  Free blocks: " << (chunks_.size() * blocks_per_chunk_ - allocated_blocks_) << std::endl;
        ss << "  Memory usage: " << (chunks_.size() * chunk_size_ / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        return ss.str();
    }
    
    /**
     * @brief Sets the chunk size.
     *
     * @param size The chunk size in bytes
     */
    void setChunkSize(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Ensure the chunk size is a multiple of the block size
        chunk_size_ = (size + BlockSize - 1) / BlockSize * BlockSize;
        blocks_per_chunk_ = chunk_size_ / BlockSize;
    }
    
    /**
     * @brief Gets the chunk size.
     *
     * @return The chunk size in bytes
     */
    size_t getChunkSize() const {
        return chunk_size_;
    }
    
    /**
     * @brief Sets the maximum pool size.
     *
     * @param size The maximum pool size in bytes
     */
    void setMaxPoolSize(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_pool_size_ = size;
    }
    
    /**
     * @brief Gets the maximum pool size.
     *
     * @return The maximum pool size in bytes
     */
    size_t getMaxPoolSize() const {
        return max_pool_size_;
    }
    
private:
    /**
     * @brief Constructs a new PoolAllocator object.
     */
    PoolAllocator()
        : free_blocks_(nullptr), allocated_blocks_(0) {
        
        // Set default chunk size to 1 MB or 1000 blocks, whichever is larger
        chunk_size_ = std::max(size_t(1024 * 1024), BlockSize * 1000);
        blocks_per_chunk_ = chunk_size_ / BlockSize;
        
        // Set default maximum pool size to 1 GB
        max_pool_size_ = 1024 * 1024 * 1024;
    }
    
    /**
     * @brief Destructor for the PoolAllocator object.
     */
    ~PoolAllocator() {
        // Free all chunks
        for (void* chunk : chunks_) {
            ::operator delete(chunk);
        }
    }
    
    // Prevent copying and assignment
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;
    
    /**
     * @brief Allocates a new chunk.
     */
    void allocateChunk() {
        // Check if we've reached the maximum pool size
        if (chunks_.size() * chunk_size_ >= max_pool_size_) {
            return;
        }
        
        // Allocate a new chunk
        void* chunk = ::operator new(chunk_size_);
        chunks_.push_back(chunk);
        
        // Initialize the free list
        char* blocks = static_cast<char*>(chunk);
        for (size_t i = 0; i < blocks_per_chunk_ - 1; ++i) {
            void* block = blocks + i * BlockSize;
            *reinterpret_cast<void**>(block) = blocks + (i + 1) * BlockSize;
        }
        
        // Set the last block's next pointer to the current free list
        void* last_block = blocks + (blocks_per_chunk_ - 1) * BlockSize;
        *reinterpret_cast<void**>(last_block) = free_blocks_;
        
        // Update the free list
        free_blocks_ = blocks;
    }
    
    void* free_blocks_;                 ///< Pointer to the first free block
    std::vector<void*> chunks_;         ///< Chunks of memory
    size_t chunk_size_;                 ///< Chunk size in bytes
    size_t blocks_per_chunk_;           ///< Number of blocks per chunk
    size_t max_pool_size_;              ///< Maximum pool size in bytes
    std::atomic<size_t> allocated_blocks_; ///< Number of allocated blocks
    mutable std::mutex mutex_;          ///< Mutex for thread safety
};

/**
 * @class PoolAllocatorAdapter
 * @brief Adapter for using PoolAllocator with STL containers.
 *
 * The PoolAllocatorAdapter class provides an adapter for using PoolAllocator
 * with STL containers.
 *
 * @tparam T The value type
 */
template <typename T>
class PoolAllocatorAdapter {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    /**
     * @brief Rebind allocator to another type.
     *
     * @tparam U The other type
     */
    template <typename U>
    struct rebind {
        using other = PoolAllocatorAdapter<U>;
    };
    
    /**
     * @brief Constructs a new PoolAllocatorAdapter object.
     */
    PoolAllocatorAdapter() noexcept = default;
    
    /**
     * @brief Copy constructor.
     *
     * @param other The other allocator
     */
    PoolAllocatorAdapter(const PoolAllocatorAdapter& other) noexcept = default;
    
    /**
     * @brief Copy constructor for different type.
     *
     * @tparam U The other type
     * @param other The other allocator
     */
    template <typename U>
    PoolAllocatorAdapter(const PoolAllocatorAdapter<U>& other) noexcept {}
    
    /**
     * @brief Destructor.
     */
    ~PoolAllocatorAdapter() noexcept = default;
    
    /**
     * @brief Allocates memory.
     *
     * @param n The number of elements to allocate
     * @return Pointer to the allocated memory
     */
    pointer allocate(size_type n) {
        if (n != 1) {
            // Pool allocator only supports allocating one element at a time
            return static_cast<pointer>(::operator new(n * sizeof(T)));
        }
        
        return static_cast<pointer>(PoolAllocator<sizeof(T)>::getInstance().allocate());
    }
    
    /**
     * @brief Deallocates memory.
     *
     * @param p Pointer to the memory
     * @param n The number of elements
     */
    void deallocate(pointer p, size_type n) noexcept {
        if (n != 1) {
            // Pool allocator only supports deallocating one element at a time
            ::operator delete(p);
            return;
        }
        
        PoolAllocator<sizeof(T)>::getInstance().deallocate(p);
    }
    
    /**
     * @brief Constructs an object.
     *
     * @param p Pointer to the memory
     * @param args Arguments for the constructor
     */
    template <typename... Args>
    void construct(pointer p, Args&&... args) {
        new(p) T(std::forward<Args>(args)...);
    }
    
    /**
     * @brief Destroys an object.
     *
     * @param p Pointer to the object
     */
    void destroy(pointer p) {
        p->~T();
    }
    
    /**
     * @brief Equality operator.
     *
     * @param other The other allocator
     * @return True if the allocators are equal, false otherwise
     */
    bool operator==(const PoolAllocatorAdapter& other) const noexcept {
        return true;
    }
    
    /**
     * @brief Inequality operator.
     *
     * @param other The other allocator
     * @return True if the allocators are not equal, false otherwise
     */
    bool operator!=(const PoolAllocatorAdapter& other) const noexcept {
        return false;
    }
};

// Type aliases for STL containers with custom allocators
template <typename T>
using ArenaVector = std::vector<T, ArenaAllocatorAdapter<T>>;

template <typename T>
using ArenaList = std::list<T, ArenaAllocatorAdapter<T>>;

template <typename K, typename V, typename Compare = std::less<K>>
using ArenaMap = std::map<K, V, Compare, ArenaAllocatorAdapter<std::pair<const K, V>>>;

template <typename K, typename V, typename Hash = std::hash<K>, typename Equal = std::equal_to<K>>
using ArenaUnorderedMap = std::unordered_map<K, V, Hash, Equal, ArenaAllocatorAdapter<std::pair<const K, V>>>;

template <typename T>
using PoolVector = std::vector<T, PoolAllocatorAdapter<T>>;

template <typename T>
using PoolList = std::list<T, PoolAllocatorAdapter<T>>;

template <typename K, typename V, typename Compare = std::less<K>>
using PoolMap = std::map<K, V, Compare, PoolAllocatorAdapter<std::pair<const K, V>>>;

template <typename K, typename V, typename Hash = std::hash<K>, typename Equal = std::equal_to<K>>
using PoolUnorderedMap = std::unordered_map<K, V, Hash, Equal, PoolAllocatorAdapter<std::pair<const K, V>>>;

} // namespace Allocators
