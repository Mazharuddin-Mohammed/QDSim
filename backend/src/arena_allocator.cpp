/**
 * @file arena_allocator.cpp
 * @brief Implementation of the arena allocator for QDSim.
 *
 * This file contains the implementation of the arena allocator
 * for temporary allocations in quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "custom_allocators.h"
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>

namespace Allocators {

// Get singleton instance
ArenaAllocator& ArenaAllocator::getInstance() {
    static ArenaAllocator instance;
    return instance;
}

// Constructor
ArenaAllocator::ArenaAllocator()
    : current_chunk_(nullptr), chunk_size_(1024 * 1024), max_arena_size_(1024 * 1024 * 1024), total_allocated_(0) {
    // Nothing to do here
}

// Destructor
ArenaAllocator::~ArenaAllocator() {
    clear();
}

// Allocate memory from the arena
void* ArenaAllocator::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Round up size to the alignment
    size = (size + alignment - 1) & ~(alignment - 1);
    
    // Check if we have a current chunk with enough space
    if (current_chunk_ == nullptr || current_chunk_->used + size > current_chunk_->size) {
        // Allocate a new chunk
        size_t chunk_size = std::max(chunk_size_, size);
        current_chunk_ = allocateChunk(chunk_size);
        
        if (current_chunk_ == nullptr) {
            throw std::bad_alloc();
        }
    }
    
    // Align the current offset
    size_t offset = current_chunk_->used;
    offset = (offset + alignment - 1) & ~(alignment - 1);
    
    // Check if we still have enough space after alignment
    if (offset + size > current_chunk_->size) {
        // Allocate a new chunk
        size_t chunk_size = std::max(chunk_size_, size);
        current_chunk_ = allocateChunk(chunk_size);
        
        if (current_chunk_ == nullptr) {
            throw std::bad_alloc();
        }
        
        // Align the offset in the new chunk
        offset = 0;
        offset = (offset + alignment - 1) & ~(alignment - 1);
    }
    
    // Allocate from the current chunk
    void* ptr = static_cast<char*>(current_chunk_->memory) + offset;
    current_chunk_->used = offset + size;
    
    // Store the allocation
    Allocation allocation;
    allocation.chunk = current_chunk_;
    allocation.offset = offset;
    allocation.size = size;
    allocations_[ptr] = allocation;
    
    return ptr;
}

// Deallocate memory from the arena
void ArenaAllocator::deallocate(void* ptr, size_t size) {
    if (ptr == nullptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find the allocation
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        // Not allocated by this arena
        return;
    }
    
    // Remove the allocation
    allocations_.erase(it);
}

// Reset the arena
void ArenaAllocator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Reset all chunks
    Chunk* chunk = current_chunk_;
    while (chunk != nullptr) {
        chunk->used = 0;
        chunk = chunk->next;
    }
    
    // Clear allocations
    allocations_.clear();
}

// Clear the arena
void ArenaAllocator::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all chunks
    Chunk* chunk = current_chunk_;
    while (chunk != nullptr) {
        Chunk* next = chunk->next;
        freeChunk(chunk);
        chunk = next;
    }
    
    // Reset state
    current_chunk_ = nullptr;
    allocations_.clear();
    total_allocated_ = 0;
}

// Get statistics about the arena
std::string ArenaAllocator::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Count chunks and memory usage
    size_t chunk_count = 0;
    size_t total_size = 0;
    size_t used_size = 0;
    
    Chunk* chunk = current_chunk_;
    while (chunk != nullptr) {
        chunk_count++;
        total_size += chunk->size;
        used_size += chunk->used;
        chunk = chunk->next;
    }
    
    // Format statistics
    std::ostringstream ss;
    ss << "Arena Allocator Statistics:" << std::endl;
    ss << "  Chunk size: " << (chunk_size_ / 1024.0) << " KB" << std::endl;
    ss << "  Maximum arena size: " << (max_arena_size_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Chunks: " << chunk_count << std::endl;
    ss << "  Total size: " << (total_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Used size: " << (used_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Utilization: " << (total_size > 0 ? (used_size * 100.0 / total_size) : 0.0) << "%" << std::endl;
    ss << "  Allocations: " << allocations_.size() << std::endl;
    
    return ss.str();
}

// Set the chunk size
void ArenaAllocator::setChunkSize(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    chunk_size_ = size;
}

// Get the chunk size
size_t ArenaAllocator::getChunkSize() const {
    return chunk_size_;
}

// Set the maximum arena size
void ArenaAllocator::setMaxArenaSize(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_arena_size_ = size;
}

// Get the maximum arena size
size_t ArenaAllocator::getMaxArenaSize() const {
    return max_arena_size_;
}

// Allocate a new chunk
ArenaAllocator::Chunk* ArenaAllocator::allocateChunk(size_t min_size) {
    // Check if we've reached the maximum arena size
    if (total_allocated_ + min_size > max_arena_size_) {
        return nullptr;
    }
    
    // Allocate memory for the chunk
    void* memory = ::operator new(min_size);
    
    // Create the chunk
    Chunk* chunk = new Chunk;
    chunk->memory = memory;
    chunk->size = min_size;
    chunk->used = 0;
    chunk->next = current_chunk_;
    
    // Update total allocated memory
    total_allocated_ += min_size;
    
    return chunk;
}

// Free a chunk
void ArenaAllocator::freeChunk(Chunk* chunk) {
    if (chunk == nullptr) {
        return;
    }
    
    // Free the memory
    ::operator delete(chunk->memory);
    
    // Update total allocated memory
    total_allocated_ -= chunk->size;
    
    // Delete the chunk
    delete chunk;
}

} // namespace Allocators
