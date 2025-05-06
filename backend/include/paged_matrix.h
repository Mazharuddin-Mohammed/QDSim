#pragma once
/**
 * @file paged_matrix.h
 * @brief Defines the PagedMatrix class for handling very large matrices.
 *
 * This file contains the declaration of the PagedMatrix class, which provides
 * a memory-efficient implementation of a matrix that uses paging to handle
 * very large matrices that don't fit in memory.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <atomic>

/**
 * @class PagedMatrix
 * @brief A matrix implementation that uses paging for very large matrices.
 *
 * The PagedMatrix class provides a memory-efficient implementation of a matrix
 * that uses paging to handle very large matrices that don't fit in memory.
 * It divides the matrix into blocks and loads/unloads blocks as needed.
 *
 * @tparam Scalar The scalar type of the matrix elements
 */
template <typename Scalar>
class PagedMatrix {
public:
    /**
     * @brief Constructs a new PagedMatrix object.
     *
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param block_size The size of each block (default: 1024)
     * @param max_memory_blocks The maximum number of blocks to keep in memory (default: 16)
     * @param temp_dir The directory for temporary files (default: system temp directory)
     */
    PagedMatrix(int rows, int cols, int block_size = 1024, int max_memory_blocks = 16,
               const std::string& temp_dir = "");

    /**
     * @brief Destructor for the PagedMatrix object.
     */
    ~PagedMatrix();

    /**
     * @brief Gets the number of rows in the matrix.
     *
     * @return The number of rows
     */
    int rows() const;

    /**
     * @brief Gets the number of columns in the matrix.
     *
     * @return The number of columns
     */
    int cols() const;

    /**
     * @brief Gets the block size.
     *
     * @return The block size
     */
    int blockSize() const;

    /**
     * @brief Gets the maximum number of memory blocks.
     *
     * @return The maximum number of memory blocks
     */
    int maxMemoryBlocks() const;

    /**
     * @brief Sets a value in the matrix.
     *
     * @param row The row index
     * @param col The column index
     * @param value The value to set
     */
    void set(int row, int col, const Scalar& value);

    /**
     * @brief Gets a value from the matrix.
     *
     * @param row The row index
     * @param col The column index
     * @return The value at the specified position
     */
    Scalar get(int row, int col) const;

    /**
     * @brief Sets a block in the matrix.
     *
     * @param block_row The block row index
     * @param block_col The block column index
     * @param block The block to set
     */
    void setBlock(int block_row, int block_col, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& block);

    /**
     * @brief Gets a block from the matrix.
     *
     * @param block_row The block row index
     * @param block_col The block column index
     * @return The block at the specified position
     */
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> getBlock(int block_row, int block_col) const;

    /**
     * @brief Multiplies the matrix by a vector.
     *
     * @param x The vector to multiply
     * @param y The result vector
     */
    void multiply(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
                 Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y) const;

    /**
     * @brief Adds another matrix to this matrix.
     *
     * @param other The matrix to add
     */
    void add(const PagedMatrix<Scalar>& other);

    /**
     * @brief Subtracts another matrix from this matrix.
     *
     * @param other The matrix to subtract
     */
    void subtract(const PagedMatrix<Scalar>& other);

    /**
     * @brief Multiplies the matrix by a scalar.
     *
     * @param scalar The scalar to multiply by
     */
    void scale(const Scalar& scalar);

    /**
     * @brief Computes the transpose of the matrix.
     *
     * @return The transposed matrix
     */
    PagedMatrix<Scalar> transpose() const;

    /**
     * @brief Computes the Frobenius norm of the matrix.
     *
     * @return The Frobenius norm
     */
    double norm() const;

    /**
     * @brief Saves the matrix to a file.
     *
     * @param filename The file name
     * @return True if the matrix was saved successfully, false otherwise
     */
    bool save(const std::string& filename) const;

    /**
     * @brief Loads the matrix from a file.
     *
     * @param filename The file name
     * @return True if the matrix was loaded successfully, false otherwise
     */
    bool load(const std::string& filename);

    /**
     * @brief Flushes all in-memory blocks to disk.
     */
    void flush();

    /**
     * @brief Prefetches blocks into memory.
     *
     * @param block_rows The block row indices to prefetch
     * @param block_cols The block column indices to prefetch
     */
    void prefetch(const std::vector<int>& block_rows, const std::vector<int>& block_cols);

    /**
     * @brief Gets the number of cache hits.
     *
     * @return The number of cache hits
     */
    size_t cacheHits() const;

    /**
     * @brief Gets the number of cache misses.
     *
     * @return The number of cache misses
     */
    size_t cacheMisses() const;

    /**
     * @brief Gets the number of disk reads.
     *
     * @return The number of disk reads
     */
    size_t diskReads() const;

    /**
     * @brief Gets the number of disk writes.
     *
     * @return The number of disk writes
     */
    size_t diskWrites() const;

private:
    // Structure to represent a block key
    struct BlockKey {
        int row;
        int col;

        bool operator==(const BlockKey& other) const {
            return row == other.row && col == other.col;
        }
    };

    // Hash function for BlockKey
    struct BlockKeyHash {
        std::size_t operator()(const BlockKey& key) const {
            return std::hash<int>()(key.row) ^ std::hash<int>()(key.col);
        }
    };

    // Structure to represent a block
    struct Block {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> data;
        bool dirty;
        std::chrono::steady_clock::time_point last_used;
    };

    int rows_;                  ///< Number of rows
    int cols_;                  ///< Number of columns
    int block_size_;            ///< Block size
    int max_memory_blocks_;     ///< Maximum number of blocks in memory
    std::string temp_dir_;      ///< Temporary directory for block files

    int num_block_rows_;        ///< Number of block rows
    int num_block_cols_;        ///< Number of block columns

    mutable std::unordered_map<BlockKey, Block, BlockKeyHash> memory_blocks_; ///< Blocks in memory
    mutable std::mutex mutex_;  ///< Mutex for thread safety

    mutable std::atomic<size_t> cache_hits_;    ///< Number of cache hits
    mutable std::atomic<size_t> cache_misses_;  ///< Number of cache misses
    mutable std::atomic<size_t> disk_reads_;    ///< Number of disk reads
    mutable std::atomic<size_t> disk_writes_;   ///< Number of disk writes

    /**
     * @brief Gets the file name for a block.
     *
     * @param row The block row index
     * @param col The block column index
     * @return The file name
     */
    std::string getBlockFileName(int row, int col) const;

    /**
     * @brief Loads a block from disk.
     *
     * @param row The block row index
     * @param col The block column index
     * @return The loaded block
     */
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> loadBlock(int row, int col) const;

    /**
     * @brief Saves a block to disk.
     *
     * @param row The block row index
     * @param col The block column index
     * @param block The block to save
     */
    void saveBlock(int row, int col, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& block) const;

    /**
     * @brief Gets a block from memory or loads it from disk.
     *
     * @param row The block row index
     * @param col The block column index
     * @return The block
     */
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& getBlockFromCache(int row, int col) const;

    /**
     * @brief Evicts blocks from memory if needed.
     */
    void evictBlocksIfNeeded() const;
};

// Type aliases
using PagedMatrixd = PagedMatrix<double>;
using PagedMatrixcd = PagedMatrix<std::complex<double>>;
