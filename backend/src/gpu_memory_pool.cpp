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
#include <algorithm>
#include <chrono>
#include <iomanip>

#ifdef USE_CUDA

// Get singleton instance
GPUMemoryPool& GPUMemoryPool::getInstance() {
    static GPUMemoryPool instance;
    return instance;
}

// Constructor
GPUMemoryPool::GPUMemoryPool()
    : total_allocated_(0), current_used_(0), allocation_count_(0), reuse_count_(0),
      trim_count_(0), oom_count_(0) {

    // Initialize configuration with default values
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    max_pool_size_ = free_mem * 0.9; // Use up to 90% of available GPU memory
    min_block_size_ = 1024; // 1 KB
    max_block_size_ = 256 * 1024 * 1024; // 256 MB
    growth_factor_ = 1.5f; // 50% growth
    block_ttl_ = std::chrono::seconds(60); // 1 minute TTL for unused blocks
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

    // Ensure size is at least the minimum block size
    size = std::max(size, min_block_size_);

    // Get the size category for faster lookup
    SizeCategory category = getSizeCategory(size);

    // Try to find a suitable memory block in the pool
    auto blocks = getBlocksByCategory(category);
    for (auto* block_ptr : blocks) {
        auto& block = *block_ptr;
        if (!block.in_use && block.size >= size) {
            // Found a suitable block
            block.in_use = true;
            block.tag = tag;
            block.last_used = std::chrono::steady_clock::now();
            current_used_ += size;
            reuse_count_++;
            return block.ptr;
        }
    }

    // Check if we need to trim the pool before allocating more memory
    if (shouldGrowPool(size)) {
        trimPool(size);
    }

    // No suitable block found, allocate a new one
    void* ptr = nullptr;
    cudaError_t cuda_status = cudaMalloc(&ptr, size);
    if (cuda_status != cudaSuccess) {
        // Try to recover by trimming the pool aggressively
        trimPool(size);

        // Try again
        cuda_status = cudaMalloc(&ptr, size);
        if (cuda_status != cudaSuccess) {
            oom_count_++;
            std::stringstream ss;
            ss << "Failed to allocate GPU memory: " << cudaGetErrorString(cuda_status)
               << " (requested size: " << (size / (1024.0 * 1024.0)) << " MB)";
            throw std::runtime_error(ss.str());
        }
    }

    // Add the new block to the pool
    MemoryBlock new_block = {
        ptr,
        size,
        true,
        tag,
        std::chrono::steady_clock::now()
    };

    // Add to the appropriate category
    memory_blocks_by_category_[category].push_back(new_block);

    // Add to the all blocks list
    all_memory_blocks_.push_back(&memory_blocks_by_category_[category].back());

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

    // Ensure size is at least the minimum block size
    size = std::max(size, min_block_size_);

    // Find the memory block in the pool
    for (auto* block_ptr : all_memory_blocks_) {
        auto& block = *block_ptr;
        if (block.ptr == ptr) {
            // Mark the block as available
            block.in_use = false;
            block.last_used = std::chrono::steady_clock::now();
            current_used_ -= size;

            // Periodically trim the pool
            if (trim_count_ % 100 == 0) {
                trim();
            }

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
    for (auto* block_ptr : all_memory_blocks_) {
        cudaFree(block_ptr->ptr);
    }

    // Clear the pool
    memory_blocks_by_category_.clear();
    all_memory_blocks_.clear();
    total_allocated_ = 0;
    current_used_ = 0;
}

// Set the maximum pool size
void GPUMemoryPool::setMaxPoolSize(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_pool_size_ = size;
}

// Set the minimum block size
void GPUMemoryPool::setMinBlockSize(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_block_size_ = size;
}

// Set the maximum block size
void GPUMemoryPool::setMaxBlockSize(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_block_size_ = size;
}

// Set the growth factor
void GPUMemoryPool::setGrowthFactor(float factor) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (factor > 1.0f) {
        growth_factor_ = factor;
    }
}

// Set the block TTL
void GPUMemoryPool::setBlockTTL(std::chrono::seconds ttl) {
    std::lock_guard<std::mutex> lock(mutex_);
    block_ttl_ = ttl;
}

// Prefetch memory blocks
void GPUMemoryPool::prefetch(const std::vector<size_t>& sizes) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t size : sizes) {
        // Round up size to the nearest multiple of 256 bytes for better alignment
        size = (size + 255) & ~255;

        // Ensure size is at least the minimum block size
        size = std::max(size, min_block_size_);

        // Get the size category
        SizeCategory category = getSizeCategory(size);

        // Check if we already have a suitable block
        bool found = false;
        auto blocks = getBlocksByCategory(category);
        for (auto* block_ptr : blocks) {
            auto& block = *block_ptr;
            if (!block.in_use && block.size >= size) {
                found = true;
                break;
            }
        }

        // If no suitable block found, allocate a new one
        if (!found) {
            void* ptr = nullptr;
            cudaError_t cuda_status = cudaMalloc(&ptr, size);
            if (cuda_status != cudaSuccess) {
                // Skip this size if allocation fails
                continue;
            }

            // Add the new block to the pool
            MemoryBlock new_block = {
                ptr,
                size,
                false, // Not in use
                "prefetch",
                std::chrono::steady_clock::now()
            };

            // Add to the appropriate category
            memory_blocks_by_category_[category].push_back(new_block);

            // Add to the all blocks list
            all_memory_blocks_.push_back(&memory_blocks_by_category_[category].back());

            total_allocated_ += size;
        }
    }
}

// Trim the memory pool
void GPUMemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::steady_clock::now();
    std::vector<MemoryBlock*> blocks_to_free;

    // Find blocks that have not been used for a while
    for (auto* block_ptr : all_memory_blocks_) {
        auto& block = *block_ptr;
        if (!block.in_use &&
            (now - block.last_used) > block_ttl_) {
            blocks_to_free.push_back(block_ptr);
        }
    }

    // Free the blocks
    for (auto* block_ptr : blocks_to_free) {
        auto& block = *block_ptr;
        cudaFree(block.ptr);
        total_allocated_ -= block.size;

        // Mark the block as invalid
        block.ptr = nullptr;
        block.size = 0;
    }

    // Remove freed blocks from the pool
    for (auto& category_pair : memory_blocks_by_category_) {
        auto& blocks = category_pair.second;
        blocks.erase(
            std::remove_if(blocks.begin(), blocks.end(),
                [](const MemoryBlock& block) { return block.ptr == nullptr; }),
            blocks.end());
    }

    // Rebuild the all_memory_blocks_ list
    all_memory_blocks_.clear();
    for (auto& category_pair : memory_blocks_by_category_) {
        auto& blocks = category_pair.second;
        for (auto& block : blocks) {
            all_memory_blocks_.push_back(&block);
        }
    }

    trim_count_++;
}

// Trim the pool to free memory when needed
void GPUMemoryPool::trimPool(size_t required_size) {
    // First, try to free blocks that have not been used for a while
    trim();

    // If that's not enough, free more aggressively
    if (total_allocated_ + required_size > max_pool_size_) {
        auto now = std::chrono::steady_clock::now();
        std::vector<MemoryBlock*> blocks_to_free;

        // Sort blocks by last used time (oldest first)
        std::vector<MemoryBlock*> unused_blocks;
        for (auto* block_ptr : all_memory_blocks_) {
            auto& block = *block_ptr;
            if (!block.in_use) {
                unused_blocks.push_back(block_ptr);
            }
        }

        std::sort(unused_blocks.begin(), unused_blocks.end(),
            [](const MemoryBlock* a, const MemoryBlock* b) {
                return a->last_used < b->last_used;
            });

        // Free blocks until we have enough memory
        size_t freed_memory = 0;
        for (auto* block_ptr : unused_blocks) {
            auto& block = *block_ptr;
            blocks_to_free.push_back(block_ptr);
            freed_memory += block.size;

            if (total_allocated_ - freed_memory + required_size <= max_pool_size_) {
                break;
            }
        }

        // Free the blocks
        for (auto* block_ptr : blocks_to_free) {
            auto& block = *block_ptr;
            cudaFree(block.ptr);
            total_allocated_ -= block.size;

            // Mark the block as invalid
            block.ptr = nullptr;
            block.size = 0;
        }

        // Remove freed blocks from the pool
        for (auto& category_pair : memory_blocks_by_category_) {
            auto& blocks = category_pair.second;
            blocks.erase(
                std::remove_if(blocks.begin(), blocks.end(),
                    [](const MemoryBlock& block) { return block.ptr == nullptr; }),
                blocks.end());
        }

        // Rebuild the all_memory_blocks_ list
        all_memory_blocks_.clear();
        for (auto& category_pair : memory_blocks_by_category_) {
            auto& blocks = category_pair.second;
            for (auto& block : blocks) {
                all_memory_blocks_.push_back(&block);
            }
        }
    }
}

// Check if we should grow the pool
bool GPUMemoryPool::shouldGrowPool(size_t required_size) const {
    return total_allocated_ + required_size > max_pool_size_;
}

// Get available GPU memory
size_t GPUMemoryPool::getAvailableGPUMemory() const {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

// Get the size category for a given size
GPUMemoryPool::SizeCategory GPUMemoryPool::getSizeCategory(size_t size) const {
    if (size < 1024) {
        return SizeCategory::TINY;
    } else if (size < 64 * 1024) {
        return SizeCategory::SMALL;
    } else if (size < 1024 * 1024) {
        return SizeCategory::MEDIUM;
    } else if (size < 16 * 1024 * 1024) {
        return SizeCategory::LARGE;
    } else if (size < 256 * 1024 * 1024) {
        return SizeCategory::HUGE;
    } else {
        return SizeCategory::ENORMOUS;
    }
}

// Get blocks by size category
std::vector<GPUMemoryPool::MemoryBlock*> GPUMemoryPool::getBlocksByCategory(SizeCategory category) {
    std::vector<MemoryBlock*> result;

    // Get blocks from the specified category
    auto it = memory_blocks_by_category_.find(category);
    if (it != memory_blocks_by_category_.end()) {
        for (auto& block : it->second) {
            result.push_back(&block);
        }
    }

    // For small categories, also include larger categories
    if (category != SizeCategory::ENORMOUS) {
        SizeCategory next_category = static_cast<SizeCategory>(static_cast<int>(category) + 1);
        auto next_blocks = getBlocksByCategory(next_category);
        result.insert(result.end(), next_blocks.begin(), next_blocks.end());
    }

    return result;
}

// Get the current memory usage
size_t GPUMemoryPool::getCurrentUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_used_;
}

// Get the total allocated memory
size_t GPUMemoryPool::getTotalAllocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

// Get statistics about the memory pool
std::string GPUMemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Get GPU memory info
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    std::stringstream ss;
    ss << "GPU Memory Pool Statistics:" << std::endl;
    ss << "  GPU Total Memory: " << (total_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  GPU Free Memory: " << (free_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  GPU Used Memory: " << ((total_mem - free_mem) / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Pool Max Size: " << (max_pool_size_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Pool Total Allocated: " << (total_allocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Pool Current Used: " << (current_used_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Allocation Count: " << allocation_count_ << std::endl;
    ss << "  Reuse Count: " << reuse_count_ << std::endl;
    ss << "  Trim Count: " << trim_count_ << std::endl;
    ss << "  OOM Count: " << oom_count_ << std::endl;
    ss << "  Reuse Ratio: " << (reuse_count_ > 0 ? (reuse_count_ * 100.0 / (allocation_count_ + reuse_count_)) : 0.0) << "%" << std::endl;
    ss << "  Number of Blocks: " << all_memory_blocks_.size() << std::endl;

    // Count blocks by category
    std::map<SizeCategory, size_t> blocks_by_category;
    std::map<SizeCategory, size_t> memory_by_category;
    std::map<SizeCategory, size_t> used_blocks_by_category;
    std::map<SizeCategory, size_t> used_memory_by_category;

    for (const auto* block_ptr : all_memory_blocks_) {
        const auto& block = *block_ptr;
        SizeCategory category = getSizeCategory(block.size);
        blocks_by_category[category]++;
        memory_by_category[category] += block.size;

        if (block.in_use) {
            used_blocks_by_category[category]++;
            used_memory_by_category[category] += block.size;
        }
    }

    // Print statistics by category
    ss << "  Memory by Category:" << std::endl;
    for (int i = static_cast<int>(SizeCategory::TINY); i <= static_cast<int>(SizeCategory::ENORMOUS); ++i) {
        SizeCategory category = static_cast<SizeCategory>(i);
        std::string category_name;
        switch (category) {
            case SizeCategory::TINY: category_name = "TINY"; break;
            case SizeCategory::SMALL: category_name = "SMALL"; break;
            case SizeCategory::MEDIUM: category_name = "MEDIUM"; break;
            case SizeCategory::LARGE: category_name = "LARGE"; break;
            case SizeCategory::HUGE: category_name = "HUGE"; break;
            case SizeCategory::ENORMOUS: category_name = "ENORMOUS"; break;
        }

        ss << "    " << category_name << ": "
           << blocks_by_category[category] << " blocks ("
           << used_blocks_by_category[category] << " in use), "
           << (memory_by_category[category] / (1024.0 * 1024.0)) << " MB ("
           << (used_memory_by_category[category] / (1024.0 * 1024.0)) << " MB in use)" << std::endl;
    }

    // Print details of each block for debugging (limit to 20 blocks)
    ss << "  Memory Blocks (top 20):" << std::endl;
    size_t block_count = std::min(all_memory_blocks_.size(), static_cast<size_t>(20));
    for (size_t i = 0; i < block_count; ++i) {
        const auto& block = *all_memory_blocks_[i];
        ss << "    Block " << i << ": " << block.ptr << ", "
           << (block.size < 1024 * 1024 ?
               std::to_string(block.size / 1024.0) + " KB" :
               std::to_string(block.size / (1024.0 * 1024.0)) + " MB") << ", "
           << (block.in_use ? "in use" : "available") << ", tag: " << block.tag << std::endl;
    }

    return ss.str();
}

#endif // USE_CUDA
