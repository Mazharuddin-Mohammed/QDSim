#pragma once
/**
 * @file memory_compression.h
 * @brief Defines memory compression utilities for QDSim.
 *
 * This file contains declarations of memory compression utilities
 * for reducing memory usage in large-scale quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <functional>
#include "cpu_memory_pool.h"

namespace MemoryCompression {

/**
 * @enum CompressionLevel
 * @brief Enumeration of compression levels.
 */
enum class CompressionLevel {
    NONE,       ///< No compression
    FAST,       ///< Fast compression (low ratio, high speed)
    BALANCED,   ///< Balanced compression (medium ratio, medium speed)
    HIGH,       ///< High compression (high ratio, low speed)
    MAXIMUM     ///< Maximum compression (highest ratio, lowest speed)
};

/**
 * @enum CompressionAlgorithm
 * @brief Enumeration of compression algorithms.
 */
enum class CompressionAlgorithm {
    LZ4,        ///< LZ4 compression algorithm
    ZSTD,       ///< Zstandard compression algorithm
    SNAPPY,     ///< Snappy compression algorithm
    DEFLATE,    ///< Deflate compression algorithm
    BROTLI      ///< Brotli compression algorithm
};

/**
 * @struct CompressionStats
 * @brief Structure to hold compression statistics.
 */
struct CompressionStats {
    size_t original_size;       ///< Original size in bytes
    size_t compressed_size;     ///< Compressed size in bytes
    double compression_ratio;   ///< Compression ratio (original_size / compressed_size)
    double compression_time;    ///< Compression time in milliseconds
    double decompression_time;  ///< Decompression time in milliseconds
    CompressionAlgorithm algorithm; ///< Compression algorithm used
    CompressionLevel level;     ///< Compression level used
};

/**
 * @class CompressedBlock
 * @brief Class for a compressed memory block.
 *
 * The CompressedBlock class represents a compressed memory block.
 * It provides methods for compressing and decompressing data.
 */
class CompressedBlock {
public:
    /**
     * @brief Constructs a new CompressedBlock object.
     *
     * @param algorithm The compression algorithm to use
     * @param level The compression level to use
     */
    CompressedBlock(CompressionAlgorithm algorithm = CompressionAlgorithm::LZ4,
                   CompressionLevel level = CompressionLevel::BALANCED);

    /**
     * @brief Destructor for the CompressedBlock object.
     */
    ~CompressedBlock();

    /**
     * @brief Compresses data.
     *
     * @param data Pointer to the data to compress
     * @param size Size of the data in bytes
     * @return True if compression was successful, false otherwise
     */
    bool compress(const void* data, size_t size);

    /**
     * @brief Decompresses data.
     *
     * @param dest Pointer to the destination buffer
     * @param dest_size Size of the destination buffer in bytes
     * @return True if decompression was successful, false otherwise
     */
    bool decompress(void* dest, size_t dest_size) const;

    /**
     * @brief Gets the original size of the data.
     *
     * @return The original size in bytes
     */
    size_t getOriginalSize() const;

    /**
     * @brief Gets the compressed size of the data.
     *
     * @return The compressed size in bytes
     */
    size_t getCompressedSize() const;

    /**
     * @brief Gets the compression ratio.
     *
     * @return The compression ratio (original_size / compressed_size)
     */
    double getCompressionRatio() const;

    /**
     * @brief Gets the compression statistics.
     *
     * @return The compression statistics
     */
    const CompressionStats& getStats() const;

    /**
     * @brief Gets the compressed data.
     *
     * @return Pointer to the compressed data
     */
    const uint8_t* getCompressedData() const;

    /**
     * @brief Sets the compression algorithm.
     *
     * @param algorithm The compression algorithm to use
     */
    void setAlgorithm(CompressionAlgorithm algorithm);

    /**
     * @brief Sets the compression level.
     *
     * @param level The compression level to use
     */
    void setLevel(CompressionLevel level);

private:
    CompressionAlgorithm algorithm_; ///< Compression algorithm
    CompressionLevel level_;         ///< Compression level
    std::vector<uint8_t> compressed_data_; ///< Compressed data
    size_t original_size_;           ///< Original size in bytes
    CompressionStats stats_;         ///< Compression statistics
};

/**
 * @class CompressedMemoryPool
 * @brief Memory pool with compression support.
 *
 * The CompressedMemoryPool class provides a memory pool with compression support.
 * It compresses memory blocks that are not in use to reduce memory usage.
 */
class CompressedMemoryPool {
public:
    /**
     * @brief Gets the singleton instance of the CompressedMemoryPool.
     *
     * @return The singleton instance
     */
    static CompressedMemoryPool& getInstance();

    /**
     * @brief Allocates memory from the pool.
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
     * @param ptr The pointer to the memory block to release
     * @param size The size of the memory block in bytes
     */
    void release(void* ptr, size_t size);

    /**
     * @brief Frees all memory in the pool.
     */
    void freeAll();

    /**
     * @brief Gets statistics about the memory pool.
     *
     * @return A string containing statistics about the memory pool
     */
    std::string getStats() const;

    /**
     * @brief Sets the compression algorithm.
     *
     * @param algorithm The compression algorithm to use
     */
    void setAlgorithm(CompressionAlgorithm algorithm);

    /**
     * @brief Sets the compression level.
     *
     * @param level The compression level to use
     */
    void setLevel(CompressionLevel level);

    /**
     * @brief Sets the compression threshold.
     *
     * @param threshold The compression threshold in bytes (blocks smaller than this will not be compressed)
     */
    void setCompressionThreshold(size_t threshold);

    /**
     * @brief Sets the compression ratio threshold.
     *
     * @param threshold The compression ratio threshold (blocks with ratio below this will not be kept compressed)
     */
    void setCompressionRatioThreshold(double threshold);

    /**
     * @brief Enables or disables automatic compression.
     *
     * @param enabled Whether automatic compression is enabled
     */
    void setAutoCompression(bool enabled);

    /**
     * @brief Compresses all unused blocks in the pool.
     */
    void compressUnusedBlocks();

    /**
     * @brief Gets the memory savings from compression.
     *
     * @return The memory savings in bytes
     */
    size_t getMemorySavings() const;

private:
    /**
     * @brief Constructs a new CompressedMemoryPool object.
     */
    CompressedMemoryPool();

    /**
     * @brief Destructor for the CompressedMemoryPool object.
     */
    ~CompressedMemoryPool();

    // Prevent copying and assignment
    CompressedMemoryPool(const CompressedMemoryPool&) = delete;
    CompressedMemoryPool& operator=(const CompressedMemoryPool&) = delete;

    // Structure to represent a memory block
    struct MemoryBlock {
        void* ptr;                      ///< Pointer to the memory block
        size_t size;                    ///< Size of the memory block in bytes
        bool in_use;                    ///< Whether the block is in use
        std::string tag;                ///< Tag for debugging
        bool compressed;                ///< Whether the block is compressed
        std::unique_ptr<CompressedBlock> compressed_data; ///< Compressed data
    };

    CPUMemoryPool& memory_pool_;        ///< CPU memory pool
    std::vector<MemoryBlock> blocks_;   ///< Memory blocks
    mutable std::mutex mutex_;          ///< Mutex for thread safety

    CompressionAlgorithm algorithm_;    ///< Compression algorithm
    CompressionLevel level_;            ///< Compression level
    size_t compression_threshold_;      ///< Compression threshold in bytes
    double compression_ratio_threshold_; ///< Compression ratio threshold
    bool auto_compression_;             ///< Whether automatic compression is enabled

    /**
     * @brief Compresses a memory block.
     *
     * @param block The memory block to compress
     */
    void compressBlock(MemoryBlock& block);

    // Statistics
    size_t total_allocated_;            ///< Total memory allocated
    size_t current_used_;               ///< Current memory in use
    size_t compressed_size_;            ///< Size of compressed data
    size_t original_size_;              ///< Original size of compressed data
    size_t compression_count_;          ///< Number of compressions
    size_t decompression_count_;        ///< Number of decompressions
};

/**
 * @brief Compresses data using the specified algorithm and level.
 *
 * @param data Pointer to the data to compress
 * @param size Size of the data in bytes
 * @param algorithm The compression algorithm to use
 * @param level The compression level to use
 * @param stats Optional pointer to store compression statistics
 * @return The compressed data
 */
std::vector<uint8_t> compressData(const void* data, size_t size,
                                 CompressionAlgorithm algorithm = CompressionAlgorithm::LZ4,
                                 CompressionLevel level = CompressionLevel::BALANCED,
                                 CompressionStats* stats = nullptr);

/**
 * @brief Decompresses data using the specified algorithm.
 *
 * @param compressed_data Pointer to the compressed data
 * @param compressed_size Size of the compressed data in bytes
 * @param original_size Size of the original data in bytes
 * @param algorithm The compression algorithm to use
 * @param stats Optional pointer to store compression statistics
 * @return The decompressed data
 */
std::vector<uint8_t> decompressData(const void* compressed_data, size_t compressed_size,
                                   size_t original_size,
                                   CompressionAlgorithm algorithm = CompressionAlgorithm::LZ4,
                                   CompressionStats* stats = nullptr);

/**
 * @brief Gets the name of a compression algorithm.
 *
 * @param algorithm The compression algorithm
 * @return The name of the algorithm
 */
std::string getAlgorithmName(CompressionAlgorithm algorithm);

/**
 * @brief Gets the name of a compression level.
 *
 * @param level The compression level
 * @return The name of the level
 */
std::string getLevelName(CompressionLevel level);

} // namespace MemoryCompression
