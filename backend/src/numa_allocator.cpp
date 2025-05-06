/**
 * @file numa_allocator.cpp
 * @brief Implementation of NUMA-aware memory allocation utilities for QDSim.
 *
 * This file contains implementations of NUMA-aware memory allocation utilities
 * for improving performance in large-scale quantum simulations on NUMA systems.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "numa_allocator.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace NUMA {

// Check if NUMA is available
bool isNumaAvailable() {
#ifdef __linux__
    // Check if NUMA is available on Linux
    if (numa_available() == -1) {
        return false;
    }
    return true;
#elif defined(_WIN32)
    // Check if NUMA is available on Windows
    ULONG highest_node_number;
    if (GetNumaHighestNodeNumber(&highest_node_number)) {
        return highest_node_number > 0;
    }
    return false;
#else
    // NUMA not supported on other platforms
    return false;
#endif
}

// Get NUMA topology
NumaTopology getNumaTopology() {
    NumaTopology topology;
    
#ifdef __linux__
    if (!isNumaAvailable()) {
        // NUMA not available, return single-node topology
        topology.num_nodes = 1;
        topology.node_ids = {0};
        topology.node_memory = {0};
        topology.node_cpus = {{0}};
        return topology;
    }
    
    // Get number of NUMA nodes
    topology.num_nodes = numa_num_configured_nodes();
    
    // Get node IDs
    topology.node_ids.resize(topology.num_nodes);
    for (int i = 0; i < topology.num_nodes; ++i) {
        topology.node_ids[i] = i;
    }
    
    // Get memory per node
    topology.node_memory.resize(topology.num_nodes);
    for (int i = 0; i < topology.num_nodes; ++i) {
        topology.node_memory[i] = numa_node_size64(i, nullptr);
    }
    
    // Get CPUs per node
    topology.node_cpus.resize(topology.num_nodes);
    int num_cpus = numa_num_configured_cpus();
    for (int i = 0; i < num_cpus; ++i) {
        int node = numa_node_of_cpu(i);
        if (node >= 0 && node < topology.num_nodes) {
            topology.node_cpus[node].push_back(i);
        }
    }
#elif defined(_WIN32)
    if (!isNumaAvailable()) {
        // NUMA not available, return single-node topology
        topology.num_nodes = 1;
        topology.node_ids = {0};
        topology.node_memory = {0};
        topology.node_cpus = {{0}};
        return topology;
    }
    
    // Get number of NUMA nodes
    ULONG highest_node_number;
    if (GetNumaHighestNodeNumber(&highest_node_number)) {
        topology.num_nodes = highest_node_number + 1;
    } else {
        topology.num_nodes = 1;
    }
    
    // Get node IDs
    topology.node_ids.resize(topology.num_nodes);
    for (int i = 0; i < topology.num_nodes; ++i) {
        topology.node_ids[i] = i;
    }
    
    // Get memory per node
    topology.node_memory.resize(topology.num_nodes);
    for (int i = 0; i < topology.num_nodes; ++i) {
        ULONGLONG node_memory = 0;
        if (GetNumaAvailableMemoryNode(i, &node_memory)) {
            topology.node_memory[i] = node_memory;
        }
    }
    
    // Get CPUs per node
    topology.node_cpus.resize(topology.num_nodes);
    DWORD_PTR process_mask, system_mask;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        for (int i = 0; i < 64; ++i) {
            if ((system_mask & (1ULL << i)) != 0) {
                UCHAR node = 0;
                if (GetNumaNodeProcessorMask(i, &node)) {
                    topology.node_cpus[node].push_back(i);
                }
            }
        }
    }
#else
    // NUMA not supported on other platforms
    topology.num_nodes = 1;
    topology.node_ids = {0};
    topology.node_memory = {0};
    topology.node_cpus = {{0}};
#endif
    
    return topology;
}

// Get current NUMA node
int getCurrentNumaNode() {
#ifdef __linux__
    if (!isNumaAvailable()) {
        return 0;
    }
    
    // Get current CPU
    int cpu = sched_getcpu();
    if (cpu < 0) {
        return 0;
    }
    
    // Get NUMA node for CPU
    int node = numa_node_of_cpu(cpu);
    if (node < 0) {
        return 0;
    }
    
    return node;
#elif defined(_WIN32)
    if (!isNumaAvailable()) {
        return 0;
    }
    
    // Get current processor number
    PROCESSOR_NUMBER processor_number;
    GetCurrentProcessorNumberEx(&processor_number);
    
    // Get NUMA node for processor
    USHORT node = 0;
    if (GetNumaProcessorNode(processor_number.Number, &node)) {
        return node;
    }
    
    return 0;
#else
    // NUMA not supported on other platforms
    return 0;
#endif
}

// Set current NUMA node
bool setCurrentNumaNode(int node) {
#ifdef __linux__
    if (!isNumaAvailable()) {
        return false;
    }
    
    // Check if node is valid
    if (node < 0 || node >= numa_num_configured_nodes()) {
        return false;
    }
    
    // Create CPU mask for the node
    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);
    
    // Get CPUs for the node
    struct bitmask* node_mask = numa_allocate_cpumask();
    numa_node_to_cpus(node, node_mask);
    
    // Set CPU mask
    for (int i = 0; i < numa_num_configured_cpus(); ++i) {
        if (numa_bitmask_isbitset(node_mask, i)) {
            CPU_SET(i, &cpu_mask);
        }
    }
    
    numa_free_cpumask(node_mask);
    
    // Set CPU affinity
    if (sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask) != 0) {
        return false;
    }
    
    return true;
#elif defined(_WIN32)
    if (!isNumaAvailable()) {
        return false;
    }
    
    // Check if node is valid
    ULONG highest_node_number;
    if (!GetNumaHighestNodeNumber(&highest_node_number) || node < 0 || node > highest_node_number) {
        return false;
    }
    
    // Get processor mask for the node
    ULONGLONG processor_mask = 0;
    if (!GetNumaNodeProcessorMask(node, &processor_mask)) {
        return false;
    }
    
    // Set processor affinity
    if (!SetProcessAffinityMask(GetCurrentProcess(), processor_mask)) {
        return false;
    }
    
    return true;
#else
    // NUMA not supported on other platforms
    return false;
#endif
}

// Get NUMA node for address
int getNumaNodeForAddress(const void* ptr) {
#ifdef __linux__
    if (!isNumaAvailable() || ptr == nullptr) {
        return 0;
    }
    
    // Get page size
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 0;
    }
    
    // Align address to page boundary
    void* page_addr = (void*)((uintptr_t)ptr & ~(page_size - 1));
    
    // Get NUMA node for page
    int node = -1;
    if (get_mempolicy(&node, nullptr, 0, page_addr, MPOL_F_NODE | MPOL_F_ADDR) != 0) {
        return 0;
    }
    
    return node;
#elif defined(_WIN32)
    if (!isNumaAvailable() || ptr == nullptr) {
        return 0;
    }
    
    // Windows doesn't provide a direct way to get the NUMA node for an address
    // We would need to use VirtualQuery to get the memory region and then check
    // the NUMA node for that region, but this is not reliable
    return 0;
#else
    // NUMA not supported on other platforms
    return 0;
#endif
}

// Set NUMA policy for memory range
bool setNumaPolicy(void* ptr, size_t size, NumaPolicy policy, const std::vector<int>& nodes) {
#ifdef __linux__
    if (!isNumaAvailable() || ptr == nullptr || size == 0) {
        return false;
    }
    
    // Create node mask
    struct bitmask* node_mask = nullptr;
    
    if (policy == NumaPolicy::BIND || policy == NumaPolicy::PREFERRED) {
        // Check if nodes are valid
        if (nodes.empty()) {
            return false;
        }
        
        // Create node mask for specified nodes
        node_mask = numa_allocate_nodemask();
        numa_bitmask_clearall(node_mask);
        
        for (int node : nodes) {
            if (node >= 0 && node < numa_num_configured_nodes()) {
                numa_bitmask_setbit(node_mask, node);
            }
        }
    } else if (policy == NumaPolicy::INTERLEAVE) {
        // Create node mask for all nodes or specified nodes
        node_mask = numa_allocate_nodemask();
        
        if (nodes.empty()) {
            // Use all nodes
            numa_bitmask_setall(node_mask);
        } else {
            // Use specified nodes
            numa_bitmask_clearall(node_mask);
            
            for (int node : nodes) {
                if (node >= 0 && node < numa_num_configured_nodes()) {
                    numa_bitmask_setbit(node_mask, node);
                }
            }
        }
    }
    
    // Set memory policy
    int result = 0;
    
    switch (policy) {
        case NumaPolicy::DEFAULT:
            result = mbind(ptr, size, MPOL_DEFAULT, nullptr, 0, 0);
            break;
            
        case NumaPolicy::BIND:
            result = mbind(ptr, size, MPOL_BIND, node_mask->maskp, node_mask->size, 0);
            break;
            
        case NumaPolicy::INTERLEAVE:
            result = mbind(ptr, size, MPOL_INTERLEAVE, node_mask->maskp, node_mask->size, 0);
            break;
            
        case NumaPolicy::PREFERRED:
            result = mbind(ptr, size, MPOL_PREFERRED, node_mask->maskp, node_mask->size, 0);
            break;
    }
    
    // Free node mask
    if (node_mask != nullptr) {
        numa_free_nodemask(node_mask);
    }
    
    return result == 0;
#elif defined(_WIN32)
    // Windows doesn't provide a direct way to set NUMA policy for a memory range
    return false;
#else
    // NUMA not supported on other platforms
    return false;
#endif
}

// Allocate memory on a specific NUMA node
void* allocateOnNode(size_t size, int node) {
#ifdef __linux__
    if (!isNumaAvailable() || size == 0) {
        return nullptr;
    }
    
    // Check if node is valid
    if (node < 0 || node >= numa_num_configured_nodes()) {
        return nullptr;
    }
    
    // Allocate memory on the specified node
    void* ptr = numa_alloc_onnode(size, node);
    if (ptr == nullptr) {
        return nullptr;
    }
    
    // Touch pages to ensure they are allocated on the specified node
    memset(ptr, 0, size);
    
    return ptr;
#elif defined(_WIN32)
    if (!isNumaAvailable() || size == 0) {
        return nullptr;
    }
    
    // Check if node is valid
    ULONG highest_node_number;
    if (!GetNumaHighestNodeNumber(&highest_node_number) || node < 0 || node > highest_node_number) {
        return nullptr;
    }
    
    // Allocate memory on the specified node
    void* ptr = VirtualAllocExNuma(GetCurrentProcess(), nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, node);
    if (ptr == nullptr) {
        return nullptr;
    }
    
    // Touch pages to ensure they are allocated on the specified node
    memset(ptr, 0, size);
    
    return ptr;
#else
    // NUMA not supported on other platforms
    return ::operator new(size);
#endif
}

// Free memory allocated with allocateOnNode
void freeOnNode(void* ptr, size_t size) {
#ifdef __linux__
    if (!isNumaAvailable() || ptr == nullptr) {
        return;
    }
    
    // Free memory
    numa_free(ptr, size);
#elif defined(_WIN32)
    if (!isNumaAvailable() || ptr == nullptr) {
        return;
    }
    
    // Free memory
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    // NUMA not supported on other platforms
    ::operator delete(ptr);
#endif
}

// Allocate memory interleaved across NUMA nodes
void* allocateInterleaved(size_t size, const std::vector<int>& nodes) {
#ifdef __linux__
    if (!isNumaAvailable() || size == 0) {
        return nullptr;
    }
    
    void* ptr = nullptr;
    
    if (nodes.empty()) {
        // Interleave across all nodes
        ptr = numa_alloc_interleaved(size);
    } else {
        // Interleave across specified nodes
        struct bitmask* node_mask = numa_allocate_nodemask();
        numa_bitmask_clearall(node_mask);
        
        for (int node : nodes) {
            if (node >= 0 && node < numa_num_configured_nodes()) {
                numa_bitmask_setbit(node_mask, node);
            }
        }
        
        ptr = numa_alloc_interleaved_subset(size, node_mask);
        numa_free_nodemask(node_mask);
    }
    
    if (ptr == nullptr) {
        return nullptr;
    }
    
    // Touch pages to ensure they are allocated
    memset(ptr, 0, size);
    
    return ptr;
#elif defined(_WIN32)
    // Windows doesn't provide a direct way to allocate interleaved memory
    // Allocate on the first available node
    if (!isNumaAvailable() || size == 0) {
        return nullptr;
    }
    
    int node = 0;
    if (!nodes.empty()) {
        node = nodes[0];
    }
    
    return allocateOnNode(size, node);
#else
    // NUMA not supported on other platforms
    return ::operator new(size);
#endif
}

// Free memory allocated with allocateInterleaved
void freeInterleaved(void* ptr, size_t size) {
#ifdef __linux__
    if (!isNumaAvailable() || ptr == nullptr) {
        return;
    }
    
    // Free memory
    numa_free(ptr, size);
#elif defined(_WIN32)
    if (!isNumaAvailable() || ptr == nullptr) {
        return;
    }
    
    // Free memory
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    // NUMA not supported on other platforms
    ::operator delete(ptr);
#endif
}

// Move memory to a specific NUMA node
bool moveToNode(void* ptr, size_t size, int node) {
#ifdef __linux__
    if (!isNumaAvailable() || ptr == nullptr || size == 0) {
        return false;
    }
    
    // Check if node is valid
    if (node < 0 || node >= numa_num_configured_nodes()) {
        return false;
    }
    
    // Create node mask for the specified node
    struct bitmask* node_mask = numa_allocate_nodemask();
    numa_bitmask_clearall(node_mask);
    numa_bitmask_setbit(node_mask, node);
    
    // Move memory to the specified node
    int result = mbind(ptr, size, MPOL_BIND, node_mask->maskp, node_mask->size, MPOL_MF_MOVE);
    
    // Free node mask
    numa_free_nodemask(node_mask);
    
    return result == 0;
#elif defined(_WIN32)
    // Windows doesn't provide a direct way to move memory between NUMA nodes
    return false;
#else
    // NUMA not supported on other platforms
    return false;
#endif
}

// Get string representation of NUMA topology
std::string getNumaTopologyString() {
    if (!isNumaAvailable()) {
        return "NUMA not available";
    }
    
    NumaTopology topology = getNumaTopology();
    
    std::ostringstream ss;
    ss << "NUMA Topology:" << std::endl;
    ss << "  Number of nodes: " << topology.num_nodes << std::endl;
    
    for (int i = 0; i < topology.num_nodes; ++i) {
        ss << "  Node " << i << ":" << std::endl;
        ss << "    Memory: " << (topology.node_memory[i] / (1024.0 * 1024.0)) << " MB" << std::endl;
        ss << "    CPUs: ";
        
        for (size_t j = 0; j < topology.node_cpus[i].size(); ++j) {
            ss << topology.node_cpus[i][j];
            if (j < topology.node_cpus[i].size() - 1) {
                ss << ", ";
            }
        }
        
        ss << std::endl;
    }
    
    return ss.str();
}

// NumaMemoryPool implementation
NumaMemoryPool& NumaMemoryPool::getInstance() {
    static NumaMemoryPool instance;
    return instance;
}

NumaMemoryPool::NumaMemoryPool() {
    // Initialize statistics
    NumaTopology topology = getNumaTopology();
    allocated_per_node_.resize(topology.num_nodes, 0);
    deallocated_per_node_.resize(topology.num_nodes, 0);
    interleaved_allocated_ = 0;
    interleaved_deallocated_ = 0;
}

NumaMemoryPool::~NumaMemoryPool() {
    clear();
}

void* NumaMemoryPool::allocate(size_t size, int node, const std::string& tag) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    void* ptr = nullptr;
    
    if (node < 0) {
        // Interleaved allocation
        ptr = allocateInterleaved(size);
        interleaved_allocated_ += size;
    } else {
        // Node-specific allocation
        ptr = allocateOnNode(size, node);
        
        if (node < static_cast<int>(allocated_per_node_.size())) {
            allocated_per_node_[node] += size;
        }
    }
    
    if (ptr == nullptr) {
        return nullptr;
    }
    
    // Add to blocks
    MemoryBlock block;
    block.ptr = ptr;
    block.size = size;
    block.node = node;
    block.tag = tag;
    blocks_.push_back(block);
    
    return ptr;
}

void NumaMemoryPool::deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find the block
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
                          [ptr](const MemoryBlock& block) {
                              return block.ptr == ptr;
                          });
    
    if (it == blocks_.end()) {
        // Block not found, use standard deallocation
        ::operator delete(ptr);
        return;
    }
    
    // Update statistics
    if (it->node < 0) {
        // Interleaved allocation
        interleaved_deallocated_ += it->size;
        freeInterleaved(ptr, it->size);
    } else {
        // Node-specific allocation
        if (it->node < static_cast<int>(deallocated_per_node_.size())) {
            deallocated_per_node_[it->node] += it->size;
        }
        freeOnNode(ptr, it->size);
    }
    
    // Remove from blocks
    blocks_.erase(it);
}

int NumaMemoryPool::getNodeForAddress(const void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find the block
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
                          [ptr](const MemoryBlock& block) {
                              return block.ptr == ptr;
                          });
    
    if (it == blocks_.end()) {
        // Block not found, use NUMA API
        return getNumaNodeForAddress(ptr);
    }
    
    return it->node;
}

std::string NumaMemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream ss;
    ss << "NUMA Memory Pool Statistics:" << std::endl;
    
    // Node-specific statistics
    for (size_t i = 0; i < allocated_per_node_.size(); ++i) {
        ss << "  Node " << i << ":" << std::endl;
        ss << "    Allocated: " << (allocated_per_node_[i] / (1024.0 * 1024.0)) << " MB" << std::endl;
        ss << "    Deallocated: " << (deallocated_per_node_[i] / (1024.0 * 1024.0)) << " MB" << std::endl;
        ss << "    Current: " << ((allocated_per_node_[i] - deallocated_per_node_[i]) / (1024.0 * 1024.0)) << " MB" << std::endl;
    }
    
    // Interleaved statistics
    ss << "  Interleaved:" << std::endl;
    ss << "    Allocated: " << (interleaved_allocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "    Deallocated: " << (interleaved_deallocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "    Current: " << ((interleaved_allocated_ - interleaved_deallocated_) / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // Total statistics
    size_t total_allocated = std::accumulate(allocated_per_node_.begin(), allocated_per_node_.end(), interleaved_allocated_);
    size_t total_deallocated = std::accumulate(deallocated_per_node_.begin(), deallocated_per_node_.end(), interleaved_deallocated_);
    
    ss << "  Total:" << std::endl;
    ss << "    Allocated: " << (total_allocated / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "    Deallocated: " << (total_deallocated / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "    Current: " << ((total_allocated - total_deallocated) / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // Block count
    ss << "  Block count: " << blocks_.size() << std::endl;
    
    return ss.str();
}

void NumaMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all blocks
    for (const auto& block : blocks_) {
        if (block.node < 0) {
            // Interleaved allocation
            freeInterleaved(block.ptr, block.size);
        } else {
            // Node-specific allocation
            freeOnNode(block.ptr, block.size);
        }
    }
    
    // Clear blocks
    blocks_.clear();
    
    // Reset statistics
    std::fill(allocated_per_node_.begin(), allocated_per_node_.end(), 0);
    std::fill(deallocated_per_node_.begin(), deallocated_per_node_.end(), 0);
    interleaved_allocated_ = 0;
    interleaved_deallocated_ = 0;
}

} // namespace NUMA
