#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <string>
#include <vector>

// Forward declaration for Python integration
struct _object;
typedef _object PyObject;

namespace QDSim {

class UnifiedMemoryManager {
public:
    enum class MemoryType {
        HOST_PINNED,
        DEVICE_ONLY,
        UNIFIED_MANAGED,
        HOST_MAPPED
    };
    
    struct MemoryBlock {
        void* ptr = nullptr;
        size_t size = 0;
        MemoryType type = MemoryType::HOST_PINNED;
        int device_id = -1;
        std::atomic<int> ref_count{1};
        std::string tag;
        
        // Python object tracking
        PyObject* python_owner = nullptr;
        bool is_python_managed = false;
        
        // RAII cleanup
        ~MemoryBlock();
    };
    
    static UnifiedMemoryManager& getInstance() {
        static UnifiedMemoryManager instance;
        return instance;
    }
    
    // Forward declaration
    struct ParallelConfig;

    // Initialization and cleanup
    void initialize(const ParallelConfig& config);
    void cleanup();
    
    // Memory allocation with automatic type selection
    std::shared_ptr<MemoryBlock> allocate(size_t size, 
                                         const std::string& tag = "",
                                         MemoryType preferred_type = MemoryType::UNIFIED_MANAGED);
    
    // Python integration
    std::shared_ptr<MemoryBlock> wrapPythonArray(PyObject* array_obj);
    PyObject* createPythonArray(std::shared_ptr<MemoryBlock> block, int ndim, const long* dims);
    
    // Memory operations
    void copyAsync(std::shared_ptr<MemoryBlock> dst, 
                  std::shared_ptr<MemoryBlock> src,
                  cudaStream_t stream = 0);
    
    void prefetchToDevice(std::shared_ptr<MemoryBlock> block, int device_id);
    void prefetchToHost(std::shared_ptr<MemoryBlock> block);
    
    // Memory pool management
    void setPoolSize(size_t size_bytes);
    void trimPool();
    size_t getPoolUsage() const;
    
    // Statistics and debugging
    struct MemoryStats {
        size_t total_allocated = 0;
        size_t peak_usage = 0;
        size_t current_usage = 0;
        size_t pool_size = 0;
        int active_blocks = 0;
        int python_blocks = 0;
    };
    
    MemoryStats getStats() const;
    void printStats() const;
    
    // Thread-safe block management
    void registerBlock(std::shared_ptr<MemoryBlock> block);
    void unregisterBlock(void* ptr);
    std::shared_ptr<MemoryBlock> findBlock(void* ptr);
    
private:
    UnifiedMemoryManager() = default;
    ~UnifiedMemoryManager() { cleanup(); }
    
    // Disable copy and move
    UnifiedMemoryManager(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager& operator=(const UnifiedMemoryManager&) = delete;
    
    // Memory pool implementation
    struct MemoryPool {
        std::vector<std::shared_ptr<MemoryBlock>> free_blocks;
        std::mutex mutex;
        size_t total_size = 0;
        size_t max_size = 0;
    };
    
    std::unordered_map<MemoryType, MemoryPool> memory_pools_;
    std::unordered_map<void*, std::shared_ptr<MemoryBlock>> active_blocks_;
    mutable std::shared_mutex active_blocks_mutex_;
    
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_usage_{0};
    std::atomic<int> active_block_count_{0};
    std::atomic<int> python_block_count_{0};
    
    bool initialized_ = false;
    size_t max_pool_size_ = 0;
    
    // Helper methods
    void* allocateRaw(size_t size, MemoryType type);
    void deallocateRaw(void* ptr, MemoryType type);
    cudaMemcpyKind getMemcpyKind(MemoryType src_type, MemoryType dst_type);
    std::shared_ptr<MemoryBlock> findInPool(size_t size, MemoryType type);
    void returnToPool(std::shared_ptr<MemoryBlock> block);
};

// RAII wrapper for automatic memory management
template<typename T>
class ManagedArray {
public:
    ManagedArray(size_t count, const std::string& tag = "", 
                UnifiedMemoryManager::MemoryType type = UnifiedMemoryManager::MemoryType::UNIFIED_MANAGED)
        : count_(count) {
        block_ = UnifiedMemoryManager::getInstance().allocate(
            count * sizeof(T), tag, type);
    }
    
    T* data() const { return static_cast<T*>(block_->ptr); }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    
    T& operator[](size_t index) { return data()[index]; }
    const T& operator[](size_t index) const { return data()[index]; }
    
    // Iterator support
    T* begin() { return data(); }
    T* end() { return data() + count_; }
    const T* begin() const { return data(); }
    const T* end() const { return data() + count_; }
    
    std::shared_ptr<UnifiedMemoryManager::MemoryBlock> getBlock() const { return block_; }
    
private:
    std::shared_ptr<UnifiedMemoryManager::MemoryBlock> block_;
    size_t count_;
};

} // namespace QDSim
