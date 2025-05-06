/**
 * @file gpu_memory_pool.cpp
 * @brief Implementation of the GPUMemoryPool class.
 *
 * This file contains the implementation of the GPUMemoryPool class, which provides
 * efficient memory management for GPU operations by reusing memory allocations
 * and reducing the overhead of frequent allocations and deallocations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "gpu_memory_pool.h"
#include <sstream>
#include <iostream>

#ifdef USE_CUDA

// Get singleton instance
GPUMemoryPool& GPUMemoryPool::getInstance() {
    static GPUMemoryPool instance;
    return instance;
}

// Constructor
GPUMemoryPool::GPUMemoryPool()
    : total_allocated_(0), current_used_(0), allocation_count_(0), reuse_count_(0) {
    // Nothing to do here
}

// Destructor
GPUMemoryPool::~GPUMemoryPool() {
    freeAll();
}

// Allocate memory from the pool
void* GPUMemoryPool::allocate(size_t size, const std::string& tag) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Round up size to the nearest multiple of 256 bytes for better alignment
    size = (size + 255) & ~255;

    // Try to find a suitable memory block in the pool
    for (auto& block : memory_blocks_) {
        if (!block.in_use && block.size >= size) {
            // Found a suitable block
            block.in_use = true;
            block.tag = tag;
            current_used_ += size;
            reuse_count_++;
            return block.ptr;
        }
    }

    // No suitable block found, allocate a new one
    void* ptr = nullptr;
    cudaError_t cuda_status = cudaMalloc(&ptr, size);
    if (cuda_status != cudaSuccess) {
        std::stringstream ss;
        ss << "Failed to allocate GPU memory: " << cudaGetErrorString(cuda_status);
        throw std::runtime_error(ss.str());
    }

    // Add the new block to the pool
    memory_blocks_.push_back({ptr, size, true, tag});
    total_allocated_ += size;
    current_used_ += size;
    allocation_count_++;

    return ptr;
}

// Release memory back to the pool
void GPUMemoryPool::release(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Round up size to the nearest multiple of 256 bytes for better alignment
    size = (size + 255) & ~255;

    // Find the memory block in the pool
    for (auto& block : memory_blocks_) {
        if (block.ptr == ptr) {
            // Mark the block as available
            block.in_use = false;
            current_used_ -= size;
            return;
        }
    }

    // Block not found, this should not happen
    std::cerr << "Warning: Attempted to release a memory block that was not allocated by the pool" << std::endl;
}

// Free all memory in the pool
void GPUMemoryPool::freeAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free all memory blocks
    for (auto& block : memory_blocks_) {
        cudaFree(block.ptr);
    }

    // Clear the pool
    memory_blocks_.clear();
    total_allocated_ = 0;
    current_used_ = 0;
}

// Get statistics about the memory pool
std::string GPUMemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::stringstream ss;
    ss << "GPU Memory Pool Statistics:" << std::endl;
    ss << "  Total allocated: " << (total_allocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Current used: " << (current_used_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Allocation count: " << allocation_count_ << std::endl;
    ss << "  Reuse count: " << reuse_count_ << std::endl;
    ss << "  Reuse ratio: " << (reuse_count_ > 0 ? (reuse_count_ * 100.0 / (allocation_count_ + reuse_count_)) : 0.0) << "%" << std::endl;
    ss << "  Number of blocks: " << memory_blocks_.size() << std::endl;

    // Print details of each block for debugging
    ss << "  Memory blocks:" << std::endl;
    for (size_t i = 0; i < memory_blocks_.size(); ++i) {
        const auto& block = memory_blocks_[i];
        ss << "    Block " << i << ": " << block.ptr << ", " << (block.size / 1024.0) << " KB, "
           << (block.in_use ? "in use" : "available") << ", tag: " << block.tag << std::endl;
    }

    return ss.str();
}

#endif // USE_CUDA
