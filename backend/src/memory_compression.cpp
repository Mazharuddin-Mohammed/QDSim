/**
 * @file memory_compression.cpp
 * @brief Implementation of memory compression utilities for QDSim.
 *
 * This file contains implementations of memory compression utilities
 * for reducing memory usage in large-scale quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "memory_compression.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

// Include compression libraries
#ifdef USE_LZ4
#include <lz4.h>
#include <lz4hc.h>
#endif

#ifdef USE_ZSTD
#include <zstd.h>
#endif

#ifdef USE_SNAPPY
#include <snappy.h>
#endif

#ifdef USE_BROTLI
#include <brotli/encode.h>
#include <brotli/decode.h>
#endif

namespace MemoryCompression {

// CompressedBlock implementation
CompressedBlock::CompressedBlock(CompressionAlgorithm algorithm, CompressionLevel level)
    : algorithm_(algorithm), level_(level), original_size_(0) {

    // Initialize statistics
    stats_.original_size = 0;
    stats_.compressed_size = 0;
    stats_.compression_ratio = 1.0;
    stats_.compression_time = 0.0;
    stats_.decompression_time = 0.0;
    stats_.algorithm = algorithm;
    stats_.level = level;
}

CompressedBlock::~CompressedBlock() {
    // Nothing to do here
}

bool CompressedBlock::compress(const void* data, size_t size) {
    if (data == nullptr || size == 0) {
        return false;
    }

    // Store original size
    original_size_ = size;
    stats_.original_size = size;

    // Measure compression time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Compress data using the selected algorithm
    compressed_data_ = compressData(data, size, algorithm_, level_, &stats_);

    // Calculate compression time
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.compression_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Check if compression was successful
    if (compressed_data_.empty()) {
        return false;
    }

    // Update statistics
    stats_.compressed_size = compressed_data_.size();
    stats_.compression_ratio = static_cast<double>(original_size_) / compressed_data_.size();

    return true;
}

bool CompressedBlock::decompress(void* dest, size_t dest_size) const {
    if (dest == nullptr || dest_size < original_size_ || compressed_data_.empty()) {
        return false;
    }

    // Measure decompression time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Decompress data using the selected algorithm
    auto decompressed_data = decompressData(compressed_data_.data(), compressed_data_.size(),
                                          original_size_, algorithm_, const_cast<CompressionStats*>(&stats_));

    // Calculate decompression time
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.decompression_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Check if decompression was successful
    if (decompressed_data.empty() || decompressed_data.size() != original_size_) {
        return false;
    }

    // Copy decompressed data to destination
    std::memcpy(dest, decompressed_data.data(), original_size_);

    return true;
}

size_t CompressedBlock::getOriginalSize() const {
    return original_size_;
}

size_t CompressedBlock::getCompressedSize() const {
    return compressed_data_.size();
}

double CompressedBlock::getCompressionRatio() const {
    if (compressed_data_.empty() || original_size_ == 0) {
        return 1.0;
    }
    return static_cast<double>(original_size_) / compressed_data_.size();
}

const CompressionStats& CompressedBlock::getStats() const {
    return stats_;
}

void CompressedBlock::setAlgorithm(CompressionAlgorithm algorithm) {
    algorithm_ = algorithm;
    stats_.algorithm = algorithm;
}

void CompressedBlock::setLevel(CompressionLevel level) {
    level_ = level;
    stats_.level = level;
}

// Additional methods for CompressedBlock
const uint8_t* CompressedBlock::getCompressedData() const {
    return compressed_data_.data();
}

// CompressedMemoryPool implementation
CompressedMemoryPool& CompressedMemoryPool::getInstance() {
    static CompressedMemoryPool instance;
    return instance;
}

CompressedMemoryPool::CompressedMemoryPool()
    : memory_pool_(CPUMemoryPool::getInstance()),
      algorithm_(CompressionAlgorithm::LZ4),
      level_(CompressionLevel::BALANCED),
      compression_threshold_(4096),  // 4 KB
      compression_ratio_threshold_(1.2),  // 20% compression
      auto_compression_(true),
      total_allocated_(0),
      current_used_(0),
      compressed_size_(0),
      original_size_(0),
      compression_count_(0),
      decompression_count_(0) {
}

CompressedMemoryPool::~CompressedMemoryPool() {
    freeAll();
}

void* CompressedMemoryPool::allocate(size_t size, const std::string& tag) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Try to find a suitable memory block in the pool
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            // Found a suitable block
            block.in_use = true;
            block.tag = tag;

            // Decompress the block if it's compressed
            if (block.compressed) {
                // Allocate memory for the decompressed data
                void* decompressed_ptr = memory_pool_.allocate(block.size, "decompressed_" + tag);

                // Decompress the data
                if (!block.compressed_data->decompress(decompressed_ptr, block.size)) {
                    // Decompression failed, free the memory and continue
                    memory_pool_.release(decompressed_ptr, block.size);
                    continue;
                }

                // Update statistics
                compressed_size_ -= block.compressed_data->getCompressedSize();
                original_size_ -= block.size;
                decompression_count_++;

                // Replace the compressed data with the decompressed data
                void* old_ptr = block.ptr;
                block.ptr = decompressed_ptr;
                block.compressed = false;
                block.compressed_data.reset();

                // Free the old memory
                memory_pool_.release(old_ptr, block.compressed_data->getCompressedSize());
            }

            current_used_ += size;
            return block.ptr;
        }
    }

    // No suitable block found, allocate a new one
    void* ptr = memory_pool_.allocate(size, tag);

    // Add the new block to the pool
    MemoryBlock block;
    block.ptr = ptr;
    block.size = size;
    block.in_use = true;
    block.tag = tag;
    block.compressed = false;
    blocks_.push_back(block);

    total_allocated_ += size;
    current_used_ += size;

    return ptr;
}

void CompressedMemoryPool::release(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find the memory block in the pool
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            // Mark the block as available
            block.in_use = false;
            current_used_ -= size;

            // Compress the block if auto-compression is enabled and the block is large enough
            if (auto_compression_ && !block.compressed && block.size >= compression_threshold_) {
                compressBlock(block);
            }

            return;
        }
    }

    // Block not found, this should not happen
    std::cerr << "Warning: Attempted to release a memory block that was not allocated by the pool" << std::endl;
}

void CompressedMemoryPool::freeAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free all memory blocks
    for (auto& block : blocks_) {
        if (block.compressed) {
            memory_pool_.release(block.ptr, block.compressed_data->getCompressedSize());
        } else {
            memory_pool_.release(block.ptr, block.size);
        }
    }

    // Clear the pool
    blocks_.clear();
    total_allocated_ = 0;
    current_used_ = 0;
    compressed_size_ = 0;
    original_size_ = 0;
}

std::string CompressedMemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream ss;
    ss << "Compressed Memory Pool Statistics:" << std::endl;
    ss << "  Total allocated: " << (total_allocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Current used: " << (current_used_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Compressed size: " << (compressed_size_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Original size: " << (original_size_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Memory savings: " << (getMemorySavings() / (1024.0 * 1024.0)) << " MB" << std::endl;
    ss << "  Compression count: " << compression_count_ << std::endl;
    ss << "  Decompression count: " << decompression_count_ << std::endl;
    ss << "  Number of blocks: " << blocks_.size() << std::endl;
    ss << "  Compression algorithm: " << getAlgorithmName(algorithm_) << std::endl;
    ss << "  Compression level: " << getLevelName(level_) << std::endl;
    ss << "  Compression threshold: " << (compression_threshold_ / 1024.0) << " KB" << std::endl;
    ss << "  Compression ratio threshold: " << compression_ratio_threshold_ << std::endl;
    ss << "  Auto-compression: " << (auto_compression_ ? "enabled" : "disabled") << std::endl;

    return ss.str();
}

void CompressedMemoryPool::setAlgorithm(CompressionAlgorithm algorithm) {
    std::lock_guard<std::mutex> lock(mutex_);
    algorithm_ = algorithm;
}

void CompressedMemoryPool::setLevel(CompressionLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

void CompressedMemoryPool::setCompressionThreshold(size_t threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    compression_threshold_ = threshold;
}

void CompressedMemoryPool::setCompressionRatioThreshold(double threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    compression_ratio_threshold_ = threshold;
}

void CompressedMemoryPool::setAutoCompression(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto_compression_ = enabled;
}

void CompressedMemoryPool::compressUnusedBlocks() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Compress all unused blocks
    for (auto& block : blocks_) {
        if (!block.in_use && !block.compressed && block.size >= compression_threshold_) {
            compressBlock(block);
        }
    }
}

size_t CompressedMemoryPool::getMemorySavings() const {
    if (original_size_ <= compressed_size_) {
        return 0;
    }
    return original_size_ - compressed_size_;
}

void CompressedMemoryPool::compressBlock(MemoryBlock& block) {
    // Create a new compressed block
    auto compressed_block = std::make_unique<CompressedBlock>(algorithm_, level_);

    // Compress the data
    if (!compressed_block->compress(block.ptr, block.size)) {
        // Compression failed, keep the block uncompressed
        return;
    }

    // Check if the compression ratio is good enough
    if (compressed_block->getCompressionRatio() < compression_ratio_threshold_) {
        // Compression ratio is too low, keep the block uncompressed
        return;
    }

    // Allocate memory for the compressed data
    void* compressed_ptr = memory_pool_.allocate(compressed_block->getCompressedSize(), "compressed_" + block.tag);

    // Copy the compressed data
    std::memcpy(compressed_ptr, compressed_block->getCompressedData(), compressed_block->getCompressedSize());

    // Update the block
    void* old_ptr = block.ptr;
    block.ptr = compressed_ptr;
    block.compressed = true;
    block.compressed_data = std::move(compressed_block);

    // Update statistics
    compressed_size_ += block.compressed_data->getCompressedSize();
    original_size_ += block.size;
    compression_count_++;

    // Free the old memory
    memory_pool_.release(old_ptr, block.size);
}

// Utility functions
std::vector<uint8_t> compressData(const void* data, size_t size,
                                CompressionAlgorithm algorithm,
                                CompressionLevel level,
                                CompressionStats* stats) {
    if (data == nullptr || size == 0) {
        return {};
    }

    // Prepare output buffer
    std::vector<uint8_t> compressed_data;

    // Compress data using the selected algorithm
    switch (algorithm) {
#ifdef USE_LZ4
        case CompressionAlgorithm::LZ4: {
            // Get the maximum compressed size
            int max_compressed_size = LZ4_compressBound(size);
            compressed_data.resize(max_compressed_size);

            // Compress the data
            int compressed_size = 0;
            if (level == CompressionLevel::FAST) {
                compressed_size = LZ4_compress_default(
                    static_cast<const char*>(data),
                    reinterpret_cast<char*>(compressed_data.data()),
                    size,
                    max_compressed_size
                );
            } else {
                // Map compression level to LZ4HC level
                int lz4hc_level = 0;
                switch (level) {
                    case CompressionLevel::BALANCED:
                        lz4hc_level = 6;
                        break;
                    case CompressionLevel::HIGH:
                        lz4hc_level = 9;
                        break;
                    case CompressionLevel::MAXIMUM:
                        lz4hc_level = 12;
                        break;
                    default:
                        lz4hc_level = 6;
                        break;
                }

                compressed_size = LZ4_compress_HC(
                    static_cast<const char*>(data),
                    reinterpret_cast<char*>(compressed_data.data()),
                    size,
                    max_compressed_size,
                    lz4hc_level
                );
            }

            // Check if compression was successful
            if (compressed_size <= 0) {
                return {};
            }

            // Resize the output buffer to the actual compressed size
            compressed_data.resize(compressed_size);
            break;
        }
#endif

#ifdef USE_ZSTD
        case CompressionAlgorithm::ZSTD: {
            // Map compression level to ZSTD level
            int zstd_level = 0;
            switch (level) {
                case CompressionLevel::FAST:
                    zstd_level = 1;
                    break;
                case CompressionLevel::BALANCED:
                    zstd_level = 3;
                    break;
                case CompressionLevel::HIGH:
                    zstd_level = 9;
                    break;
                case CompressionLevel::MAXIMUM:
                    zstd_level = 22;
                    break;
                default:
                    zstd_level = 3;
                    break;
            }

            // Get the maximum compressed size
            size_t max_compressed_size = ZSTD_compressBound(size);
            compressed_data.resize(max_compressed_size);

            // Compress the data
            size_t compressed_size = ZSTD_compress(
                compressed_data.data(),
                max_compressed_size,
                data,
                size,
                zstd_level
            );

            // Check if compression was successful
            if (ZSTD_isError(compressed_size)) {
                return {};
            }

            // Resize the output buffer to the actual compressed size
            compressed_data.resize(compressed_size);
            break;
        }
#endif

#ifdef USE_SNAPPY
        case CompressionAlgorithm::SNAPPY: {
            // Get the maximum compressed size
            size_t max_compressed_size = snappy::MaxCompressedLength(size);
            compressed_data.resize(max_compressed_size);

            // Compress the data
            size_t compressed_size = 0;
            snappy::RawCompress(
                static_cast<const char*>(data),
                size,
                reinterpret_cast<char*>(compressed_data.data()),
                &compressed_size
            );

            // Resize the output buffer to the actual compressed size
            compressed_data.resize(compressed_size);
            break;
        }
#endif

#ifdef USE_BROTLI
        case CompressionAlgorithm::BROTLI: {
            // Map compression level to Brotli quality
            int brotli_quality = 0;
            switch (level) {
                case CompressionLevel::FAST:
                    brotli_quality = 1;
                    break;
                case CompressionLevel::BALANCED:
                    brotli_quality = 6;
                    break;
                case CompressionLevel::HIGH:
                    brotli_quality = 9;
                    break;
                case CompressionLevel::MAXIMUM:
                    brotli_quality = 11;
                    break;
                default:
                    brotli_quality = 6;
                    break;
            }

            // Get the maximum compressed size
            size_t max_compressed_size = BrotliEncoderMaxCompressedSize(size);
            compressed_data.resize(max_compressed_size);

            // Compress the data
            size_t compressed_size = max_compressed_size;
            if (!BrotliEncoderCompress(
                brotli_quality,
                BROTLI_DEFAULT_WINDOW,
                BROTLI_DEFAULT_MODE,
                size,
                static_cast<const uint8_t*>(data),
                &compressed_size,
                compressed_data.data()
            )) {
                return {};
            }

            // Resize the output buffer to the actual compressed size
            compressed_data.resize(compressed_size);
            break;
        }
#endif

        case CompressionAlgorithm::DEFLATE:
        default:
            // Not implemented or not available
            return {};
    }

    // Update statistics if provided
    if (stats != nullptr) {
        stats->original_size = size;
        stats->compressed_size = compressed_data.size();
        stats->compression_ratio = static_cast<double>(size) / compressed_data.size();
        stats->algorithm = algorithm;
        stats->level = level;
    }

    return compressed_data;
}

std::vector<uint8_t> decompressData(const void* compressed_data, size_t compressed_size,
                                  size_t original_size,
                                  CompressionAlgorithm algorithm,
                                  CompressionStats* stats) {
    if (compressed_data == nullptr || compressed_size == 0 || original_size == 0) {
        return {};
    }

    // Prepare output buffer
    std::vector<uint8_t> decompressed_data(original_size);

    // Decompress data using the selected algorithm
    switch (algorithm) {
#ifdef USE_LZ4
        case CompressionAlgorithm::LZ4: {
            // Decompress the data
            int decompressed_size = LZ4_decompress_safe(
                static_cast<const char*>(compressed_data),
                reinterpret_cast<char*>(decompressed_data.data()),
                compressed_size,
                original_size
            );

            // Check if decompression was successful
            if (decompressed_size <= 0 || static_cast<size_t>(decompressed_size) != original_size) {
                return {};
            }
            break;
        }
#endif

#ifdef USE_ZSTD
        case CompressionAlgorithm::ZSTD: {
            // Decompress the data
            size_t decompressed_size = ZSTD_decompress(
                decompressed_data.data(),
                original_size,
                compressed_data,
                compressed_size
            );

            // Check if decompression was successful
            if (ZSTD_isError(decompressed_size) || decompressed_size != original_size) {
                return {};
            }
            break;
        }
#endif

#ifdef USE_SNAPPY
        case CompressionAlgorithm::SNAPPY: {
            // Decompress the data
            if (!snappy::RawUncompress(
                static_cast<const char*>(compressed_data),
                compressed_size,
                reinterpret_cast<char*>(decompressed_data.data())
            )) {
                return {};
            }
            break;
        }
#endif

#ifdef USE_BROTLI
        case CompressionAlgorithm::BROTLI: {
            // Decompress the data
            size_t decompressed_size = original_size;
            if (!BrotliDecoderDecompress(
                compressed_size,
                static_cast<const uint8_t*>(compressed_data),
                &decompressed_size,
                decompressed_data.data()
            ) || decompressed_size != original_size) {
                return {};
            }
            break;
        }
#endif

        case CompressionAlgorithm::DEFLATE:
        default:
            // Not implemented or not available
            return {};
    }

    // Update statistics if provided
    if (stats != nullptr) {
        stats->original_size = original_size;
        stats->compressed_size = compressed_size;
        stats->compression_ratio = static_cast<double>(original_size) / compressed_size;
        stats->algorithm = algorithm;
    }

    return decompressed_data;
}

std::string getAlgorithmName(CompressionAlgorithm algorithm) {
    switch (algorithm) {
        case CompressionAlgorithm::LZ4:
            return "LZ4";
        case CompressionAlgorithm::ZSTD:
            return "Zstandard";
        case CompressionAlgorithm::SNAPPY:
            return "Snappy";
        case CompressionAlgorithm::DEFLATE:
            return "Deflate";
        case CompressionAlgorithm::BROTLI:
            return "Brotli";
        default:
            return "Unknown";
    }
}

std::string getLevelName(CompressionLevel level) {
    switch (level) {
        case CompressionLevel::NONE:
            return "None";
        case CompressionLevel::FAST:
            return "Fast";
        case CompressionLevel::BALANCED:
            return "Balanced";
        case CompressionLevel::HIGH:
            return "High";
        case CompressionLevel::MAXIMUM:
            return "Maximum";
        default:
            return "Unknown";
    }
}

} // namespace MemoryCompression
