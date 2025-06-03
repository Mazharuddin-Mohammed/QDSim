#include "unified_memory_manager.h"
#include "unified_parallel_manager.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>

// Python integration (optional)
#ifdef PYTHON_INTEGRATION
#include <Python.h>
#include <numpy/arrayobject.h>
#endif

namespace QDSim {

// MemoryBlock destructor implementation
UnifiedMemoryManager::MemoryBlock::~MemoryBlock() {
    if (ptr && !is_python_managed) {
        // Cleanup based on memory type
        switch (type) {
            case MemoryType::HOST_PINNED:
            case MemoryType::HOST_MAPPED:
                cudaFreeHost(ptr);
                break;
            case MemoryType::DEVICE_ONLY:
            case MemoryType::UNIFIED_MANAGED:
                cudaFree(ptr);
                break;
        }
    }
    
#ifdef PYTHON_INTEGRATION
    if (python_owner) {
        Py_DECREF(python_owner);
    }
#endif
}

void UnifiedMemoryManager::initialize(const UnifiedParallelManager::ParallelConfig& config) {
    if (initialized_) return;
    
    try {
        // Set maximum pool size based on available GPU memory
        if (config.cuda_devices_per_rank > 0) {
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            max_pool_size_ = config.gpu_memory_pool_size > 0 ? 
                           config.gpu_memory_pool_size : 
                           static_cast<size_t>(free_mem * 0.8); // Use 80% of available memory
        } else {
            max_pool_size_ = 1024 * 1024 * 1024; // 1GB default for CPU-only
        }
        
        // Initialize memory pools
        for (auto type : {MemoryType::HOST_PINNED, MemoryType::DEVICE_ONLY, 
                         MemoryType::UNIFIED_MANAGED, MemoryType::HOST_MAPPED}) {
            memory_pools_[type].max_size = max_pool_size_ / 4; // Distribute among types
        }
        
        initialized_ = true;
        
        std::cout << "UnifiedMemoryManager initialized with pool size: " 
                  << (max_pool_size_ / (1024 * 1024)) << " MB" << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize UnifiedMemoryManager: " << e.what() << std::endl;
        throw;
    }
}

void UnifiedMemoryManager::cleanup() {
    if (!initialized_) return;
    
    try {
        // Clear all memory pools
        for (auto& [type, pool] : memory_pools_) {
            std::lock_guard<std::mutex> lock(pool.mutex);
            pool.free_blocks.clear();
            pool.total_size = 0;
        }
        
        // Clear active blocks (they should clean themselves up via RAII)
        {
            std::unique_lock<std::shared_mutex> lock(active_blocks_mutex_);
            active_blocks_.clear();
        }
        
        initialized_ = false;
        
        std::cout << "UnifiedMemoryManager cleanup completed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during UnifiedMemoryManager cleanup: " << e.what() << std::endl;
    }
}

std::shared_ptr<UnifiedMemoryManager::MemoryBlock> 
UnifiedMemoryManager::allocate(size_t size, const std::string& tag, MemoryType preferred_type) {
    
    if (!initialized_) {
        throw std::runtime_error("UnifiedMemoryManager not initialized");
    }
    
    // Try to find a suitable block in the pool
    auto block = findInPool(size, preferred_type);
    if (block) {
        block->ref_count.store(1);
        block->tag = tag;
        registerBlock(block);
        return block;
    }
    
    // Allocate new block
    void* ptr = allocateRaw(size, preferred_type);
    if (!ptr) {
        throw std::runtime_error("Failed to allocate memory of size " + std::to_string(size));
    }
    
    auto new_block = std::make_shared<MemoryBlock>();
    new_block->ptr = ptr;
    new_block->size = size;
    new_block->type = preferred_type;
    new_block->tag = tag;
    
    int device_id = -1;
    if (preferred_type == MemoryType::DEVICE_ONLY || preferred_type == MemoryType::UNIFIED_MANAGED) {
        cudaGetDevice(&device_id);
        new_block->device_id = device_id;
    }
    
    registerBlock(new_block);
    
    // Update statistics
    total_allocated_.fetch_add(size);
    active_block_count_.fetch_add(1);
    
    size_t current_usage = total_allocated_.load();
    size_t current_peak = peak_usage_.load();
    while (current_usage > current_peak && 
           !peak_usage_.compare_exchange_weak(current_peak, current_usage)) {
        current_peak = peak_usage_.load();
    }
    
    return new_block;
}

void* UnifiedMemoryManager::allocateRaw(size_t size, MemoryType type) {
    void* ptr = nullptr;
    cudaError_t err = cudaSuccess;
    
    switch (type) {
        case MemoryType::HOST_PINNED:
            err = cudaMallocHost(&ptr, size);
            break;
            
        case MemoryType::DEVICE_ONLY:
            err = cudaMalloc(&ptr, size);
            break;
            
        case MemoryType::UNIFIED_MANAGED:
            err = cudaMallocManaged(&ptr, size);
            if (err == cudaSuccess) {
                // Set memory advice for optimal performance
                cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
            }
            break;
            
        case MemoryType::HOST_MAPPED:
            err = cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
            break;
    }
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    
    return ptr;
}

void UnifiedMemoryManager::copyAsync(std::shared_ptr<MemoryBlock> dst, 
                                    std::shared_ptr<MemoryBlock> src,
                                    cudaStream_t stream) {
    if (!dst || !src) {
        throw std::invalid_argument("Invalid memory blocks for copy operation");
    }
    
    cudaMemcpyKind kind = getMemcpyKind(src->type, dst->type);
    size_t copy_size = std::min(src->size, dst->size);
    
    cudaError_t err;
    if (stream == 0) {
        err = cudaMemcpy(dst->ptr, src->ptr, copy_size, kind);
    } else {
        err = cudaMemcpyAsync(dst->ptr, src->ptr, copy_size, kind, stream);
    }
    
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory copy failed: " + std::string(cudaGetErrorString(err)));
    }
}

cudaMemcpyKind UnifiedMemoryManager::getMemcpyKind(MemoryType src_type, MemoryType dst_type) {
    bool src_on_device = (src_type == MemoryType::DEVICE_ONLY || src_type == MemoryType::UNIFIED_MANAGED);
    bool dst_on_device = (dst_type == MemoryType::DEVICE_ONLY || dst_type == MemoryType::UNIFIED_MANAGED);
    
    if (src_on_device && dst_on_device) {
        return cudaMemcpyDeviceToDevice;
    } else if (src_on_device && !dst_on_device) {
        return cudaMemcpyDeviceToHost;
    } else if (!src_on_device && dst_on_device) {
        return cudaMemcpyHostToDevice;
    } else {
        return cudaMemcpyHostToHost;
    }
}

void UnifiedMemoryManager::registerBlock(std::shared_ptr<MemoryBlock> block) {
    std::unique_lock<std::shared_mutex> lock(active_blocks_mutex_);
    active_blocks_[block->ptr] = block;
}

void UnifiedMemoryManager::unregisterBlock(void* ptr) {
    std::unique_lock<std::shared_mutex> lock(active_blocks_mutex_);
    active_blocks_.erase(ptr);
}

std::shared_ptr<UnifiedMemoryManager::MemoryBlock> UnifiedMemoryManager::findBlock(void* ptr) {
    std::shared_lock<std::shared_mutex> lock(active_blocks_mutex_);
    auto it = active_blocks_.find(ptr);
    return (it != active_blocks_.end()) ? it->second : nullptr;
}

std::shared_ptr<UnifiedMemoryManager::MemoryBlock> 
UnifiedMemoryManager::findInPool(size_t size, MemoryType type) {
    auto& pool = memory_pools_[type];
    std::lock_guard<std::mutex> lock(pool.mutex);
    
    auto it = std::find_if(pool.free_blocks.begin(), pool.free_blocks.end(),
        [size](const std::shared_ptr<MemoryBlock>& block) {
            return block->size >= size && block->ref_count.load() == 0;
        });
    
    if (it != pool.free_blocks.end()) {
        auto block = *it;
        pool.free_blocks.erase(it);
        return block;
    }
    
    return nullptr;
}

std::shared_ptr<UnifiedMemoryManager::MemoryBlock>
UnifiedMemoryManager::wrapPythonArray(PyObject* array_obj) {
#ifdef PYTHON_INTEGRATION
    if (!PyArray_Check(array_obj)) {
        throw std::invalid_argument("Object is not a NumPy array");
    }

    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_obj);

    auto block = std::make_shared<MemoryBlock>();
    block->ptr = PyArray_DATA(array);
    block->size = PyArray_NBYTES(array);
    block->type = MemoryType::HOST_PINNED; // Assume host memory
    block->python_owner = array_obj;
    block->is_python_managed = true;

    // Increment Python reference count
    Py_INCREF(array_obj);

    registerBlock(block);
    python_block_count_.fetch_add(1);

    return block;
#else
    throw std::runtime_error("Python integration not enabled");
#endif
}

PyObject* UnifiedMemoryManager::createPythonArray(std::shared_ptr<MemoryBlock> block,
                                                 int ndim, const long* dims) {
#ifdef PYTHON_INTEGRATION
    if (!block) {
        throw std::invalid_argument("Invalid memory block");
    }

    // Determine NumPy data type (assuming double for now)
    npy_intp* numpy_dims = new npy_intp[ndim];
    for (int i = 0; i < ndim; ++i) {
        numpy_dims[i] = dims[i];
    }

    PyObject* array = PyArray_SimpleNewFromData(ndim, numpy_dims, NPY_DOUBLE, block->ptr);
    delete[] numpy_dims;

    if (!array) {
        throw std::runtime_error("Failed to create NumPy array");
    }

    // Set up custom destructor to handle memory cleanup
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(array),
                         PyCapsule_New(block.get(), nullptr,
                         [](PyObject* capsule) {
                             // Memory will be automatically freed when shared_ptr goes out of scope
                         }));

    block->python_owner = array;
    block->is_python_managed = true;
    python_block_count_.fetch_add(1);

    return array;
#else
    throw std::runtime_error("Python integration not enabled");
#endif
}

UnifiedMemoryManager::MemoryStats UnifiedMemoryManager::getStats() const {
    MemoryStats stats;
    stats.total_allocated = total_allocated_.load();
    stats.peak_usage = peak_usage_.load();
    stats.current_usage = total_allocated_.load(); // Simplified
    stats.active_blocks = active_block_count_.load();
    stats.python_blocks = python_block_count_.load();
    
    // Calculate pool size
    for (const auto& [type, pool] : memory_pools_) {
        stats.pool_size += pool.total_size;
    }
    
    return stats;
}

void UnifiedMemoryManager::printStats() const {
    auto stats = getStats();
    std::cout << "=== UnifiedMemoryManager Statistics ===" << std::endl;
    std::cout << "Total allocated: " << (stats.total_allocated / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Peak usage: " << (stats.peak_usage / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Active blocks: " << stats.active_blocks << std::endl;
    std::cout << "Python blocks: " << stats.python_blocks << std::endl;
    std::cout << "Pool size: " << (stats.pool_size / (1024 * 1024)) << " MB" << std::endl;
}

} // namespace QDSim
