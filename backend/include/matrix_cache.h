#pragma once
/**
 * @file matrix_cache.h
 * @brief Defines a specialized cache for matrix data in QDSim.
 *
 * This file contains declarations of a specialized cache for matrix data
 * to improve performance in large-scale quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "multi_level_cache.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <string>
#include <memory>
#include <functional>

namespace Cache {

/**
 * @struct MatrixCacheKey
 * @brief Structure for a matrix cache key.
 */
struct MatrixCacheKey {
    std::string name;       ///< Matrix name
    int rows;               ///< Number of rows
    int cols;               ///< Number of columns
    int block_row;          ///< Block row index (for blocked matrices)
    int block_col;          ///< Block column index (for blocked matrices)
    
    /**
     * @brief Equality operator.
     *
     * @param other The other key
     * @return True if the keys are equal, false otherwise
     */
    bool operator==(const MatrixCacheKey& other) const {
        return name == other.name && rows == other.rows && cols == other.cols &&
               block_row == other.block_row && block_col == other.block_col;
    }
};

/**
 * @struct MatrixCacheKeyHash
 * @brief Hash function for MatrixCacheKey.
 */
struct MatrixCacheKeyHash {
    /**
     * @brief Computes the hash of a MatrixCacheKey.
     *
     * @param key The key
     * @return The hash value
     */
    std::size_t operator()(const MatrixCacheKey& key) const {
        std::size_t h1 = std::hash<std::string>()(key.name);
        std::size_t h2 = std::hash<int>()(key.rows);
        std::size_t h3 = std::hash<int>()(key.cols);
        std::size_t h4 = std::hash<int>()(key.block_row);
        std::size_t h5 = std::hash<int>()(key.block_col);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
};

/**
 * @class DenseMatrixCache
 * @brief Specialized cache for dense matrices.
 *
 * The DenseMatrixCache class provides a specialized cache for dense matrices
 * using the multi-level cache system.
 *
 * @tparam Scalar The scalar type of the matrix elements
 */
template <typename Scalar>
class DenseMatrixCache {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    
    /**
     * @brief Constructs a new DenseMatrixCache object.
     *
     * @param l1_capacity The capacity of the L1 cache in bytes
     * @param l2_capacity The capacity of the L2 cache in bytes
     * @param l3_capacity The capacity of the L3 cache in bytes
     */
    DenseMatrixCache(size_t l1_capacity = 64 * 1024 * 1024,    // 64 MB
                    size_t l2_capacity = 256 * 1024 * 1024,   // 256 MB
                    size_t l3_capacity = 1024 * 1024 * 1024)  // 1 GB
        : cache_(l1_capacity, l2_capacity, l3_capacity) {
        
        // Set size estimator
        cache_.setSizeEstimator([](const MatrixType& matrix) {
            return matrix.rows() * matrix.cols() * sizeof(Scalar);
        });
    }
    
    /**
     * @brief Inserts a matrix into the cache.
     *
     * @param name The matrix name
     * @param matrix The matrix
     * @param block_row The block row index (default: 0)
     * @param block_col The block column index (default: 0)
     * @param level The cache level to insert into (default: ALL)
     * @return True if the matrix was inserted, false otherwise
     */
    bool insert(const std::string& name, const MatrixType& matrix,
               int block_row = 0, int block_col = 0,
               CacheLevel level = CacheLevel::ALL) {
        
        MatrixCacheKey key = {name, matrix.rows(), matrix.cols(), block_row, block_col};
        return cache_.insert(key, matrix, 0, level);
    }
    
    /**
     * @brief Retrieves a matrix from the cache.
     *
     * @param name The matrix name
     * @param rows The number of rows
     * @param cols The number of columns
     * @param matrix Reference to store the matrix
     * @param block_row The block row index (default: 0)
     * @param block_col The block column index (default: 0)
     * @return True if the matrix was found, false otherwise
     */
    bool get(const std::string& name, int rows, int cols, MatrixType& matrix,
            int block_row = 0, int block_col = 0) {
        
        MatrixCacheKey key = {name, rows, cols, block_row, block_col};
        return cache_.get(key, matrix);
    }
    
    /**
     * @brief Removes a matrix from the cache.
     *
     * @param name The matrix name
     * @param rows The number of rows
     * @param cols The number of columns
     * @param block_row The block row index (default: 0)
     * @param block_col The block column index (default: 0)
     * @param level The cache level to remove from (default: ALL)
     * @return True if the matrix was removed, false otherwise
     */
    bool remove(const std::string& name, int rows, int cols,
               int block_row = 0, int block_col = 0,
               CacheLevel level = CacheLevel::ALL) {
        
        MatrixCacheKey key = {name, rows, cols, block_row, block_col};
        return cache_.remove(key, level);
    }
    
    /**
     * @brief Clears the cache.
     *
     * @param level The cache level to clear (default: ALL)
     */
    void clear(CacheLevel level = CacheLevel::ALL) {
        cache_.clear(level);
    }
    
    /**
     * @brief Gets the cache statistics.
     *
     * @param level The cache level to get statistics for
     * @return The cache statistics
     */
    CacheStats getStats(CacheLevel level = CacheLevel::ALL) const {
        return cache_.getStats(level);
    }
    
    /**
     * @brief Sets the capacity of a cache level.
     *
     * @param level The cache level
     * @param capacity The capacity in bytes
     */
    void setCapacity(CacheLevel level, size_t capacity) {
        cache_.setCapacity(level, capacity);
    }
    
    /**
     * @brief Sets the cache replacement policy for a cache level.
     *
     * @param level The cache level
     * @param policy The cache replacement policy
     */
    void setPolicy(CacheLevel level, CachePolicy policy) {
        cache_.setPolicy(level, policy);
    }
    
private:
    MultiLevelCache<MatrixCacheKey, MatrixType, MatrixCacheKeyHash> cache_;  ///< Multi-level cache
};

/**
 * @class SparseMatrixCache
 * @brief Specialized cache for sparse matrices.
 *
 * The SparseMatrixCache class provides a specialized cache for sparse matrices
 * using the multi-level cache system.
 *
 * @tparam Scalar The scalar type of the matrix elements
 */
template <typename Scalar>
class SparseMatrixCache {
public:
    using MatrixType = Eigen::SparseMatrix<Scalar>;
    
    /**
     * @brief Constructs a new SparseMatrixCache object.
     *
     * @param l1_capacity The capacity of the L1 cache in bytes
     * @param l2_capacity The capacity of the L2 cache in bytes
     * @param l3_capacity The capacity of the L3 cache in bytes
     */
    SparseMatrixCache(size_t l1_capacity = 64 * 1024 * 1024,    // 64 MB
                     size_t l2_capacity = 256 * 1024 * 1024,   // 256 MB
                     size_t l3_capacity = 1024 * 1024 * 1024)  // 1 GB
        : cache_(l1_capacity, l2_capacity, l3_capacity) {
        
        // Set size estimator
        cache_.setSizeEstimator([](const MatrixType& matrix) {
            return matrix.nonZeros() * (sizeof(Scalar) + sizeof(int)) + (matrix.rows() + 1) * sizeof(int);
        });
    }
    
    /**
     * @brief Inserts a matrix into the cache.
     *
     * @param name The matrix name
     * @param matrix The matrix
     * @param block_row The block row index (default: 0)
     * @param block_col The block column index (default: 0)
     * @param level The cache level to insert into (default: ALL)
     * @return True if the matrix was inserted, false otherwise
     */
    bool insert(const std::string& name, const MatrixType& matrix,
               int block_row = 0, int block_col = 0,
               CacheLevel level = CacheLevel::ALL) {
        
        MatrixCacheKey key = {name, matrix.rows(), matrix.cols(), block_row, block_col};
        return cache_.insert(key, matrix, 0, level);
    }
    
    /**
     * @brief Retrieves a matrix from the cache.
     *
     * @param name The matrix name
     * @param rows The number of rows
     * @param cols The number of columns
     * @param matrix Reference to store the matrix
     * @param block_row The block row index (default: 0)
     * @param block_col The block column index (default: 0)
     * @return True if the matrix was found, false otherwise
     */
    bool get(const std::string& name, int rows, int cols, MatrixType& matrix,
            int block_row = 0, int block_col = 0) {
        
        MatrixCacheKey key = {name, rows, cols, block_row, block_col};
        return cache_.get(key, matrix);
    }
    
    /**
     * @brief Removes a matrix from the cache.
     *
     * @param name The matrix name
     * @param rows The number of rows
     * @param cols The number of columns
     * @param block_row The block row index (default: 0)
     * @param block_col The block column index (default: 0)
     * @param level The cache level to remove from (default: ALL)
     * @return True if the matrix was removed, false otherwise
     */
    bool remove(const std::string& name, int rows, int cols,
               int block_row = 0, int block_col = 0,
               CacheLevel level = CacheLevel::ALL) {
        
        MatrixCacheKey key = {name, rows, cols, block_row, block_col};
        return cache_.remove(key, level);
    }
    
    /**
     * @brief Clears the cache.
     *
     * @param level The cache level to clear (default: ALL)
     */
    void clear(CacheLevel level = CacheLevel::ALL) {
        cache_.clear(level);
    }
    
    /**
     * @brief Gets the cache statistics.
     *
     * @param level The cache level to get statistics for
     * @return The cache statistics
     */
    CacheStats getStats(CacheLevel level = CacheLevel::ALL) const {
        return cache_.getStats(level);
    }
    
    /**
     * @brief Sets the capacity of a cache level.
     *
     * @param level The cache level
     * @param capacity The capacity in bytes
     */
    void setCapacity(CacheLevel level, size_t capacity) {
        cache_.setCapacity(level, capacity);
    }
    
    /**
     * @brief Sets the cache replacement policy for a cache level.
     *
     * @param level The cache level
     * @param policy The cache replacement policy
     */
    void setPolicy(CacheLevel level, CachePolicy policy) {
        cache_.setPolicy(level, policy);
    }
    
private:
    MultiLevelCache<MatrixCacheKey, MatrixType, MatrixCacheKeyHash> cache_;  ///< Multi-level cache
};

// Type aliases
using DenseMatrixCached = DenseMatrixCache<double>;
using DenseMatrixCachecd = DenseMatrixCache<std::complex<double>>;
using SparseMatrixCached = SparseMatrixCache<double>;
using SparseMatrixCachecd = SparseMatrixCache<std::complex<double>>;

} // namespace Cache
