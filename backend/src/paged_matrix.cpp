/**
 * @file paged_matrix.cpp
 * @brief Implementation of the PagedMatrix class.
 *
 * This file contains the implementation of the PagedMatrix class, which provides
 * a memory-efficient implementation of a matrix that uses paging to handle
 * very large matrices that don't fit in memory.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "paged_matrix.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <filesystem>

// Constructor
template <typename Scalar>
PagedMatrix<Scalar>::PagedMatrix(int rows, int cols, int block_size, int max_memory_blocks,
                               const std::string& temp_dir)
    : rows_(rows), cols_(cols), block_size_(block_size), max_memory_blocks_(max_memory_blocks),
      temp_dir_(temp_dir), cache_hits_(0), cache_misses_(0), disk_reads_(0), disk_writes_(0) {

    // Calculate number of block rows and columns
    num_block_rows_ = (rows_ + block_size_ - 1) / block_size_;
    num_block_cols_ = (cols_ + block_size_ - 1) / block_size_;

    // Create temporary directory if not specified
    if (temp_dir_.empty()) {
        temp_dir_ = std::filesystem::temp_directory_path().string() + "/paged_matrix_" +
                   std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    }

    // Create directory if it doesn't exist
    if (!std::filesystem::exists(temp_dir_)) {
        std::filesystem::create_directories(temp_dir_);
    }
}

// Destructor
template <typename Scalar>
PagedMatrix<Scalar>::~PagedMatrix() {
    // Flush all blocks to disk
    flush();

    // Remove temporary directory and files
    try {
        std::filesystem::remove_all(temp_dir_);
    } catch (const std::exception& e) {
        std::cerr << "Error removing temporary directory: " << e.what() << std::endl;
    }
}

// Get number of rows
template <typename Scalar>
int PagedMatrix<Scalar>::rows() const {
    return rows_;
}

// Get number of columns
template <typename Scalar>
int PagedMatrix<Scalar>::cols() const {
    return cols_;
}

// Get block size
template <typename Scalar>
int PagedMatrix<Scalar>::blockSize() const {
    return block_size_;
}

// Get maximum number of memory blocks
template <typename Scalar>
int PagedMatrix<Scalar>::maxMemoryBlocks() const {
    return max_memory_blocks_;
}

// Set a value in the matrix
template <typename Scalar>
void PagedMatrix<Scalar>::set(int row, int col, const Scalar& value) {
    // Check if indices are valid
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }

    // Calculate block indices
    int block_row = row / block_size_;
    int block_col = col / block_size_;

    // Calculate local indices within the block
    int local_row = row % block_size_;
    int local_col = col % block_size_;

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Get the block
    BlockKey key = {block_row, block_col};
    auto it = memory_blocks_.find(key);

    if (it != memory_blocks_.end()) {
        // Block is in memory
        it->second.data(local_row, local_col) = value;
        it->second.dirty = true;
        it->second.last_used = std::chrono::steady_clock::now();
        cache_hits_++;
    } else {
        // Block is not in memory, load it
        cache_misses_++;

        // Evict blocks if needed
        evictBlocksIfNeeded();

        // Load the block
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block;

        // Check if the block file exists
        std::string filename = getBlockFileName(block_row, block_col);
        if (std::filesystem::exists(filename)) {
            // Load the block from disk
            block = loadBlock(block_row, block_col);
            disk_reads_++;
        } else {
            // Create a new block
            int actual_block_rows = std::min(block_size_, rows_ - block_row * block_size_);
            int actual_block_cols = std::min(block_size_, cols_ - block_col * block_size_);
            block = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(actual_block_rows, actual_block_cols);
        }

        // Set the value
        block(local_row, local_col) = value;

        // Add the block to memory
        Block memory_block;
        memory_block.data = block;
        memory_block.dirty = true;
        memory_block.last_used = std::chrono::steady_clock::now();
        memory_blocks_[key] = memory_block;
    }
}

// Get a value from the matrix
template <typename Scalar>
Scalar PagedMatrix<Scalar>::get(int row, int col) const {
    // Check if indices are valid
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }

    // Calculate block indices
    int block_row = row / block_size_;
    int block_col = col / block_size_;

    // Calculate local indices within the block
    int local_row = row % block_size_;
    int local_col = col % block_size_;

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Get the block
    const auto& block = getBlockFromCache(block_row, block_col);

    // Return the value
    return block(local_row, local_col);
}

// Set a block in the matrix
template <typename Scalar>
void PagedMatrix<Scalar>::setBlock(int block_row, int block_col,
                                 const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& block) {
    // Check if block indices are valid
    if (block_row < 0 || block_row >= num_block_rows_ || block_col < 0 || block_col >= num_block_cols_) {
        throw std::out_of_range("Block indices out of range");
    }

    // Calculate actual block dimensions
    int actual_block_rows = std::min(block_size_, rows_ - block_row * block_size_);
    int actual_block_cols = std::min(block_size_, cols_ - block_col * block_size_);

    // Check if block dimensions match
    if (block.rows() != actual_block_rows || block.cols() != actual_block_cols) {
        throw std::invalid_argument("Block dimensions mismatch");
    }

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Evict blocks if needed
    evictBlocksIfNeeded();

    // Add the block to memory
    BlockKey key = {block_row, block_col};
    Block memory_block;
    memory_block.data = block;
    memory_block.dirty = true;
    memory_block.last_used = std::chrono::steady_clock::now();
    memory_blocks_[key] = memory_block;
}

// Get a block from the matrix
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> PagedMatrix<Scalar>::getBlock(int block_row, int block_col) const {
    // Check if block indices are valid
    if (block_row < 0 || block_row >= num_block_rows_ || block_col < 0 || block_col >= num_block_cols_) {
        throw std::out_of_range("Block indices out of range");
    }

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Get the block
    return getBlockFromCache(block_row, block_col);
}

// Multiply matrix by vector
template <typename Scalar>
void PagedMatrix<Scalar>::multiply(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
                                 Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y) const {
    // Check dimensions
    if (x.size() != cols_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    // Resize result vector if needed
    if (y.size() != rows_) {
        y.resize(rows_);
    }

    // Initialize result vector to zero
    y.setZero();

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Perform matrix-vector multiplication block by block
    for (int block_row = 0; block_row < num_block_rows_; ++block_row) {
        int row_offset = block_row * block_size_;
        int actual_block_rows = std::min(block_size_, rows_ - row_offset);

        for (int block_col = 0; block_col < num_block_cols_; ++block_col) {
            int col_offset = block_col * block_size_;
            int actual_block_cols = std::min(block_size_, cols_ - col_offset);

            // Get the block
            const auto& block = getBlockFromCache(block_row, block_col);

            // Multiply block by corresponding part of x
            for (int i = 0; i < actual_block_rows; ++i) {
                for (int j = 0; j < actual_block_cols; ++j) {
                    y(row_offset + i) += block(i, j) * x(col_offset + j);
                }
            }
        }
    }
}

// Add another matrix
template <typename Scalar>
void PagedMatrix<Scalar>::add(const PagedMatrix<Scalar>& other) {
    // Check dimensions
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Add block by block
    for (int block_row = 0; block_row < num_block_rows_; ++block_row) {
        for (int block_col = 0; block_col < num_block_cols_; ++block_col) {
            // Get blocks
            auto this_block = getBlockFromCache(block_row, block_col);
            const auto& other_block = other.getBlockFromCache(block_row, block_col);

            // Add blocks
            this_block += other_block;

            // Update block
            setBlock(block_row, block_col, this_block);
        }
    }
}

// Subtract another matrix
template <typename Scalar>
void PagedMatrix<Scalar>::subtract(const PagedMatrix<Scalar>& other) {
    // Check dimensions
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Subtract block by block
    for (int block_row = 0; block_row < num_block_rows_; ++block_row) {
        for (int block_col = 0; block_col < num_block_cols_; ++block_col) {
            // Get blocks
            auto this_block = getBlockFromCache(block_row, block_col);
            const auto& other_block = other.getBlockFromCache(block_row, block_col);

            // Subtract blocks
            this_block -= other_block;

            // Update block
            setBlock(block_row, block_col, this_block);
        }
    }
}

// Scale by a scalar
template <typename Scalar>
void PagedMatrix<Scalar>::scale(const Scalar& scalar) {
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Scale block by block
    for (int block_row = 0; block_row < num_block_rows_; ++block_row) {
        for (int block_col = 0; block_col < num_block_cols_; ++block_col) {
            // Get block
            auto block = getBlockFromCache(block_row, block_col);

            // Scale block
            block *= scalar;

            // Update block
            setBlock(block_row, block_col, block);
        }
    }
}

// Compute transpose
template <typename Scalar>
PagedMatrix<Scalar> PagedMatrix<Scalar>::transpose() const {
    // Create transposed matrix
    PagedMatrix<Scalar> result(cols_, rows_, block_size_, max_memory_blocks_, temp_dir_);

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Transpose block by block
    for (int block_row = 0; block_row < num_block_rows_; ++block_row) {
        for (int block_col = 0; block_col < num_block_cols_; ++block_col) {
            // Get block
            const auto& block = getBlockFromCache(block_row, block_col);

            // Transpose block
            auto transposed_block = block.transpose();

            // Set block in result
            result.setBlock(block_col, block_row, transposed_block);
        }
    }

    return result;
}

// Compute Frobenius norm
template <typename Scalar>
double PagedMatrix<Scalar>::norm() const {
    double sum_squares = 0.0;

    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Compute norm block by block
    for (int block_row = 0; block_row < num_block_rows_; ++block_row) {
        for (int block_col = 0; block_col < num_block_cols_; ++block_col) {
            // Get block
            const auto& block = getBlockFromCache(block_row, block_col);

            // Add squares of elements
            for (int i = 0; i < block.rows(); ++i) {
                for (int j = 0; j < block.cols(); ++j) {
                    sum_squares += std::norm(block(i, j));
                }
            }
        }
    }

    return std::sqrt(sum_squares);
}

// Save matrix to file
template <typename Scalar>
bool PagedMatrix<Scalar>::save(const std::string& filename) const {
    // Flush all blocks to disk
    flush();

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    // Write dimensions and parameters
    file.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
    file.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
    file.write(reinterpret_cast<const char*>(&block_size_), sizeof(block_size_));
    file.write(reinterpret_cast<const char*>(&max_memory_blocks_), sizeof(max_memory_blocks_));

    // Write number of blocks
    file.write(reinterpret_cast<const char*>(&num_block_rows_), sizeof(num_block_rows_));
    file.write(reinterpret_cast<const char*>(&num_block_cols_), sizeof(num_block_cols_));

    // Write all blocks
    for (int block_row = 0; block_row < num_block_rows_; ++block_row) {
        for (int block_col = 0; block_col < num_block_cols_; ++block_col) {
            // Get block
            const auto& block = getBlockFromCache(block_row, block_col);

            // Write block dimensions
            int block_rows = block.rows();
            int block_cols = block.cols();
            file.write(reinterpret_cast<const char*>(&block_rows), sizeof(block_rows));
            file.write(reinterpret_cast<const char*>(&block_cols), sizeof(block_cols));

            // Write block data
            file.write(reinterpret_cast<const char*>(block.data()), block_rows * block_cols * sizeof(Scalar));
        }
    }

    return true;
}

// Load matrix from file
template <typename Scalar>
bool PagedMatrix<Scalar>::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    // Read dimensions and parameters
    file.read(reinterpret_cast<char*>(&rows_), sizeof(rows_));
    file.read(reinterpret_cast<char*>(&cols_), sizeof(cols_));
    file.read(reinterpret_cast<char*>(&block_size_), sizeof(block_size_));
    file.read(reinterpret_cast<char*>(&max_memory_blocks_), sizeof(max_memory_blocks_));

    // Read number of blocks
    file.read(reinterpret_cast<char*>(&num_block_rows_), sizeof(num_block_rows_));
    file.read(reinterpret_cast<char*>(&num_block_cols_), sizeof(num_block_cols_));

    // Clear memory blocks
    memory_blocks_.clear();

    // Read all blocks
    for (int block_row = 0; block_row < num_block_rows_; ++block_row) {
        for (int block_col = 0; block_col < num_block_cols_; ++block_col) {
            // Read block dimensions
            int block_rows, block_cols;
            file.read(reinterpret_cast<char*>(&block_rows), sizeof(block_rows));
            file.read(reinterpret_cast<char*>(&block_cols), sizeof(block_cols));

            // Create block
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block(block_rows, block_cols);

            // Read block data
            file.read(reinterpret_cast<char*>(block.data()), block_rows * block_cols * sizeof(Scalar));

            // Save block to disk
            saveBlock(block_row, block_col, block);
        }
    }

    return true;
}

// Flush all in-memory blocks to disk
template <typename Scalar>
void PagedMatrix<Scalar>::flush() {
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Save all dirty blocks to disk
    for (auto& pair : memory_blocks_) {
        if (pair.second.dirty) {
            saveBlock(pair.first.row, pair.first.col, pair.second.data);
            pair.second.dirty = false;
            disk_writes_++;
        }
    }
}

// Prefetch blocks into memory
template <typename Scalar>
void PagedMatrix<Scalar>::prefetch(const std::vector<int>& block_rows, const std::vector<int>& block_cols) {
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(mutex_);

    // Prefetch specified blocks
    for (int block_row : block_rows) {
        for (int block_col : block_cols) {
            // Check if block indices are valid
            if (block_row >= 0 && block_row < num_block_rows_ && block_col >= 0 && block_col < num_block_cols_) {
                // Get the block (this will load it into memory)
                getBlockFromCache(block_row, block_col);
            }
        }
    }
}

// Get number of cache hits
template <typename Scalar>
size_t PagedMatrix<Scalar>::cacheHits() const {
    return cache_hits_;
}

// Get number of cache misses
template <typename Scalar>
size_t PagedMatrix<Scalar>::cacheMisses() const {
    return cache_misses_;
}

// Get number of disk reads
template <typename Scalar>
size_t PagedMatrix<Scalar>::diskReads() const {
    return disk_reads_;
}

// Get number of disk writes
template <typename Scalar>
size_t PagedMatrix<Scalar>::diskWrites() const {
    return disk_writes_;
}

// Get block file name
template <typename Scalar>
std::string PagedMatrix<Scalar>::getBlockFileName(int row, int col) const {
    return temp_dir_ + "/block_" + std::to_string(row) + "_" + std::to_string(col) + ".bin";
}

// Load block from disk
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> PagedMatrix<Scalar>::loadBlock(int row, int col) const {
    std::string filename = getBlockFileName(row, col);
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        // Block file doesn't exist, create a zero block
        int actual_block_rows = std::min(block_size_, rows_ - row * block_size_);
        int actual_block_cols = std::min(block_size_, cols_ - col * block_size_);
        return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(actual_block_rows, actual_block_cols);
    }

    // Read block dimensions
    int block_rows, block_cols;
    file.read(reinterpret_cast<char*>(&block_rows), sizeof(block_rows));
    file.read(reinterpret_cast<char*>(&block_cols), sizeof(block_cols));

    // Create block
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block(block_rows, block_cols);

    // Read block data
    file.read(reinterpret_cast<char*>(block.data()), block_rows * block_cols * sizeof(Scalar));

    return block;
}

// Save block to disk
template <typename Scalar>
void PagedMatrix<Scalar>::saveBlock(int row, int col, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& block) const {
    std::string filename = getBlockFileName(row, col);
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to save block to disk: " + filename);
    }

    // Write block dimensions
    int block_rows = block.rows();
    int block_cols = block.cols();
    file.write(reinterpret_cast<const char*>(&block_rows), sizeof(block_rows));
    file.write(reinterpret_cast<const char*>(&block_cols), sizeof(block_cols));

    // Write block data
    file.write(reinterpret_cast<const char*>(block.data()), block_rows * block_cols * sizeof(Scalar));
}

// Get block from cache or load from disk
template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& PagedMatrix<Scalar>::getBlockFromCache(int row, int col) const {
    BlockKey key = {row, col};
    auto it = memory_blocks_.find(key);

    if (it != memory_blocks_.end()) {
        // Block is in memory
        it->second.last_used = std::chrono::steady_clock::now();
        cache_hits_++;
        return it->second.data;
    } else {
        // Block is not in memory, load it
        cache_misses_++;

        // Evict blocks if needed
        evictBlocksIfNeeded();

        // Load the block
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block = loadBlock(row, col);
        disk_reads_++;

        // Add the block to memory
        Block memory_block;
        memory_block.data = block;
        memory_block.dirty = false;
        memory_block.last_used = std::chrono::steady_clock::now();
        auto result = memory_blocks_.insert(std::make_pair(key, memory_block));

        return result.first->second.data;
    }
}

// Evict blocks if needed
template <typename Scalar>
void PagedMatrix<Scalar>::evictBlocksIfNeeded() const {
    // Check if we need to evict blocks
    if (memory_blocks_.size() >= static_cast<size_t>(max_memory_blocks_)) {
        // Find the least recently used blocks
        std::vector<std::pair<BlockKey, std::chrono::steady_clock::time_point>> blocks;
        for (const auto& pair : memory_blocks_) {
            blocks.push_back(std::make_pair(pair.first, pair.second.last_used));
        }

        // Sort by last used time (oldest first)
        std::sort(blocks.begin(), blocks.end(),
                 [](const auto& a, const auto& b) {
                     return a.second < b.second;
                 });

        // Evict blocks until we have enough space
        int blocks_to_evict = memory_blocks_.size() - max_memory_blocks_ + 1;
        for (int i = 0; i < blocks_to_evict && i < static_cast<int>(blocks.size()); ++i) {
            const auto& key = blocks[i].first;
            auto it = memory_blocks_.find(key);

            if (it != memory_blocks_.end()) {
                // Save block to disk if dirty
                if (it->second.dirty) {
                    saveBlock(key.row, key.col, it->second.data);
                    disk_writes_++;
                }

                // Remove block from memory
                memory_blocks_.erase(it);
            }
        }
    }
}

// Explicit instantiations
template class PagedMatrix<double>;
template class PagedMatrix<std::complex<double>>;
