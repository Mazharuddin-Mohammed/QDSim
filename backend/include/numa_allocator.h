#pragma once
/**
 * @file numa_allocator.h
 * @brief Defines NUMA-aware memory allocation utilities for QDSim.
 *
 * This file contains declarations of NUMA-aware memory allocation utilities
 * for improving performance in large-scale quantum simulations on NUMA systems.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <thread>
#include <functional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

namespace NUMA {

/**
 * @struct NumaTopology
 * @brief Structure to represent NUMA topology.
 */
struct NumaTopology {
    int num_nodes;                      ///< Number of NUMA nodes
    std::vector<int> node_ids;          ///< NUMA node IDs
    std::vector<size_t> node_memory;    ///< Memory available on each node in bytes
    std::vector<std::vector<int>> node_cpus; ///< CPUs on each node
};

/**
 * @enum NumaPolicy
 * @brief Enumeration of NUMA memory policies.
 */
enum class NumaPolicy {
    DEFAULT,    ///< Default policy (local allocation)
    BIND,       ///< Bind to specific nodes
    INTERLEAVE, ///< Interleave across nodes
    PREFERRED   ///< Prefer specific nodes
};

/**
 * @brief Gets the NUMA topology.
 *
 * @return The NUMA topology
 */
NumaTopology getNumaTopology();

/**
 * @brief Gets the NUMA node for the current thread.
 *
 * @return The NUMA node ID
 */
int getCurrentNumaNode();

/**
 * @brief Sets the NUMA node for the current thread.
 *
 * @param node The NUMA node ID
 * @return True if successful, false otherwise
 */
bool setCurrentNumaNode(int node);

/**
 * @brief Gets the NUMA node for a memory address.
 *
 * @param ptr The memory address
 * @return The NUMA node ID
 */
int getNumaNodeForAddress(const void* ptr);

/**
 * @brief Sets the NUMA policy for a memory range.
 *
 * @param ptr The memory address
 * @param size The size of the memory range in bytes
 * @param policy The NUMA policy
 * @param nodes The NUMA nodes to use (for BIND and PREFERRED policies)
 * @return True if successful, false otherwise
 */
bool setNumaPolicy(void* ptr, size_t size, NumaPolicy policy, const std::vector<int>& nodes = {});

/**
 * @brief Allocates memory on a specific NUMA node.
 *
 * @param size The size of the memory to allocate in bytes
 * @param node The NUMA node ID
 * @return Pointer to the allocated memory
 */
void* allocateOnNode(size_t size, int node);

/**
 * @brief Frees memory allocated with allocateOnNode.
 *
 * @param ptr The memory address
 * @param size The size of the memory in bytes
 */
void freeOnNode(void* ptr, size_t size);

/**
 * @brief Allocates memory interleaved across NUMA nodes.
 *
 * @param size The size of the memory to allocate in bytes
 * @param nodes The NUMA nodes to use (empty for all nodes)
 * @return Pointer to the allocated memory
 */
void* allocateInterleaved(size_t size, const std::vector<int>& nodes = {});

/**
 * @brief Frees memory allocated with allocateInterleaved.
 *
 * @param ptr The memory address
 * @param size The size of the memory in bytes
 */
void freeInterleaved(void* ptr, size_t size);

/**
 * @brief Moves memory to a specific NUMA node.
 *
 * @param ptr The memory address
 * @param size The size of the memory in bytes
 * @param node The NUMA node ID
 * @return True if successful, false otherwise
 */
bool moveToNode(void* ptr, size_t size, int node);

/**
 * @brief Checks if NUMA is available.
 *
 * @return True if NUMA is available, false otherwise
 */
bool isNumaAvailable();

/**
 * @brief Gets a string representation of the NUMA topology.
 *
 * @return A string representation of the NUMA topology
 */
std::string getNumaTopologyString();

/**
 * @class NumaAllocator
 * @brief NUMA-aware allocator for STL containers.
 *
 * The NumaAllocator class provides a NUMA-aware allocator for STL containers
 * that allocates memory on a specific NUMA node or interleaved across nodes.
 *
 * @tparam T The value type
 */
template <typename T>
class NumaAllocator {
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
        using other = NumaAllocator<U>;
    };
    
    /**
     * @brief Constructs a new NumaAllocator object.
     *
     * @param node The NUMA node ID (-1 for interleaved allocation)
     * @param nodes The NUMA nodes to use for interleaved allocation
     */
    NumaAllocator(int node = -1, const std::vector<int>& nodes = {})
        : node_(node), nodes_(nodes) {
        
        // Check if NUMA is available
        if (!isNumaAvailable()) {
            node_ = -2;  // Disable NUMA allocation
        }
    }
    
    /**
     * @brief Copy constructor.
     *
     * @param other The other allocator
     */
    NumaAllocator(const NumaAllocator& other) noexcept
        : node_(other.node_), nodes_(other.nodes_) {}
    
    /**
     * @brief Copy constructor for different type.
     *
     * @tparam U The other type
     * @param other The other allocator
     */
    template <typename U>
    NumaAllocator(const NumaAllocator<U>& other) noexcept
        : node_(other.node_), nodes_(other.nodes_) {}
    
    /**
     * @brief Destructor.
     */
    ~NumaAllocator() noexcept = default;
    
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
        pointer p = nullptr;
        
        if (node_ == -2) {
            // NUMA not available, use standard allocation
            p = static_cast<pointer>(::operator new(bytes));
        } else if (node_ == -1) {
            // Interleaved allocation
            p = static_cast<pointer>(allocateInterleaved(bytes, nodes_));
        } else {
            // Node-specific allocation
            p = static_cast<pointer>(allocateOnNode(bytes, node_));
        }
        
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
        size_type bytes = n * sizeof(T);
        
        if (node_ == -2) {
            // NUMA not available, use standard deallocation
            ::operator delete(p);
        } else if (node_ == -1) {
            // Interleaved allocation
            freeInterleaved(p, bytes);
        } else {
            // Node-specific allocation
            freeOnNode(p, bytes);
        }
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
     * @brief Gets the NUMA node ID.
     *
     * @return The NUMA node ID
     */
    int getNode() const {
        return node_;
    }
    
    /**
     * @brief Gets the NUMA nodes for interleaved allocation.
     *
     * @return The NUMA nodes
     */
    const std::vector<int>& getNodes() const {
        return nodes_;
    }
    
    /**
     * @brief Equality operator.
     *
     * @param other The other allocator
     * @return True if the allocators are equal, false otherwise
     */
    bool operator==(const NumaAllocator& other) const {
        return node_ == other.node_ && nodes_ == other.nodes_;
    }
    
    /**
     * @brief Inequality operator.
     *
     * @param other The other allocator
     * @return True if the allocators are not equal, false otherwise
     */
    bool operator!=(const NumaAllocator& other) const {
        return !(*this == other);
    }
    
private:
    int node_;                  ///< NUMA node ID (-1 for interleaved allocation, -2 for disabled)
    std::vector<int> nodes_;    ///< NUMA nodes for interleaved allocation
    
    // Make all NumaAllocator instantiations friends
    template <typename U>
    friend class NumaAllocator;
};

/**
 * @class NumaMemoryPool
 * @brief NUMA-aware memory pool.
 *
 * The NumaMemoryPool class provides a NUMA-aware memory pool
 * that allocates memory on specific NUMA nodes or interleaved across nodes.
 */
class NumaMemoryPool {
public:
    /**
     * @brief Gets the singleton instance of the NumaMemoryPool.
     *
     * @return The singleton instance
     */
    static NumaMemoryPool& getInstance();
    
    /**
     * @brief Allocates memory.
     *
     * @param size The size of the memory to allocate in bytes
     * @param node The NUMA node ID (-1 for interleaved allocation)
     * @param tag An optional tag for debugging
     * @return Pointer to the allocated memory
     */
    void* allocate(size_t size, int node = -1, const std::string& tag = "");
    
    /**
     * @brief Deallocates memory.
     *
     * @param ptr Pointer to the memory
     * @param size The size of the memory in bytes
     */
    void deallocate(void* ptr, size_t size);
    
    /**
     * @brief Gets the NUMA node for a memory address.
     *
     * @param ptr The memory address
     * @return The NUMA node ID
     */
    int getNodeForAddress(const void* ptr) const;
    
    /**
     * @brief Gets statistics about the memory pool.
     *
     * @return A string containing statistics about the memory pool
     */
    std::string getStats() const;
    
    /**
     * @brief Clears the memory pool.
     */
    void clear();
    
private:
    /**
     * @brief Constructs a new NumaMemoryPool object.
     */
    NumaMemoryPool();
    
    /**
     * @brief Destructor for the NumaMemoryPool object.
     */
    ~NumaMemoryPool();
    
    // Prevent copying and assignment
    NumaMemoryPool(const NumaMemoryPool&) = delete;
    NumaMemoryPool& operator=(const NumaMemoryPool&) = delete;
    
    // Structure to represent a memory block
    struct MemoryBlock {
        void* ptr;          ///< Pointer to the memory
        size_t size;        ///< Size of the memory in bytes
        int node;           ///< NUMA node ID
        std::string tag;    ///< Tag for debugging
    };
    
    std::vector<MemoryBlock> blocks_;   ///< Memory blocks
    mutable std::mutex mutex_;          ///< Mutex for thread safety
    
    // Statistics
    std::vector<size_t> allocated_per_node_;    ///< Memory allocated per node
    std::vector<size_t> deallocated_per_node_;  ///< Memory deallocated per node
    size_t interleaved_allocated_;              ///< Interleaved memory allocated
    size_t interleaved_deallocated_;            ///< Interleaved memory deallocated
};

} // namespace NUMA
