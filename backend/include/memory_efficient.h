#pragma once
/**
 * @file memory_efficient.h
 * @brief Defines memory-efficient data structures for large-scale simulations.
 *
 * This file contains declarations of memory-efficient data structures for
 * large-scale quantum simulations, including out-of-core matrices and
 * distributed memory vectors.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Sparse>
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <fstream>

// Conditional compilation for MPI support
#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @class OutOfCoreMatrix
 * @brief A memory-efficient matrix that stores data on disk.
 *
 * The OutOfCoreMatrix class provides a memory-efficient implementation of a
 * matrix that stores data on disk and loads only the required parts into memory.
 * It is useful for large-scale simulations where the matrices do not fit in memory.
 *
 * @tparam Scalar The scalar type of the matrix elements
 */
template <typename Scalar>
class OutOfCoreMatrix {
public:
    /**
     * @brief Constructs a new OutOfCoreMatrix object.
     *
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param block_size The size of the blocks to load into memory
     * @param filename The name of the file to store the matrix data
     *
     * @throws std::runtime_error If the file cannot be created
     */
    OutOfCoreMatrix(int rows, int cols, int block_size = 1000, const std::string& filename = "");

    /**
     * @brief Destructor for the OutOfCoreMatrix object.
     *
     * Cleans up resources used by the matrix.
     */
    ~OutOfCoreMatrix();

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
     * @brief Gets the block size used for loading data into memory.
     *
     * @return The block size
     */
    int block_size() const;

    /**
     * @brief Gets the filename used for storing the matrix data.
     *
     * @return The filename
     */
    std::string filename() const;

    /**
     * @brief Sets a matrix element.
     *
     * @param row The row index
     * @param col The column index
     * @param value The value to set
     *
     * @throws std::out_of_range If the indices are out of range
     * @throws std::runtime_error If the file cannot be written
     */
    void set(int row, int col, const Scalar& value);

    /**
     * @brief Gets a matrix element.
     *
     * @param row The row index
     * @param col The column index
     *
     * @return The matrix element
     *
     * @throws std::out_of_range If the indices are out of range
     * @throws std::runtime_error If the file cannot be read
     */
    Scalar get(int row, int col) const;

    /**
     * @brief Loads a block of the matrix into memory.
     *
     * @param row_start The starting row index
     * @param col_start The starting column index
     * @param row_size The number of rows to load
     * @param col_size The number of columns to load
     *
     * @return The loaded block as an Eigen matrix
     *
     * @throws std::out_of_range If the indices are out of range
     * @throws std::runtime_error If the file cannot be read
     */
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> load_block(int row_start, int col_start, int row_size, int col_size) const;

    /**
     * @brief Saves a block of the matrix to disk.
     *
     * @param row_start The starting row index
     * @param col_start The starting column index
     * @param block The block to save
     *
     * @throws std::out_of_range If the indices are out of range
     * @throws std::runtime_error If the file cannot be written
     */
    void save_block(int row_start, int col_start, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& block);

    /**
     * @brief Performs matrix-vector multiplication.
     *
     * @param x The vector to multiply
     * @param y The result vector
     *
     * @throws std::invalid_argument If the vector dimensions are invalid
     * @throws std::runtime_error If the file cannot be read
     */
    void multiply(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y) const;

    /**
     * @brief Converts the matrix to an Eigen sparse matrix.
     *
     * @return The Eigen sparse matrix
     *
     * @throws std::runtime_error If the file cannot be read
     */
    Eigen::SparseMatrix<Scalar> to_sparse() const;

private:
    int rows_;                  ///< The number of rows in the matrix
    int cols_;                  ///< The number of columns in the matrix
    int block_size_;            ///< The size of the blocks to load into memory
    std::string filename_;      ///< The name of the file to store the matrix data
    mutable std::fstream file_; ///< The file stream for reading/writing matrix data

    /**
     * @brief Gets the file offset for a matrix element.
     *
     * @param row The row index
     * @param col The column index
     *
     * @return The file offset
     */
    std::streampos get_offset(int row, int col) const;

    /**
     * @brief Checks if the indices are valid.
     *
     * @param row The row index
     * @param col The column index
     *
     * @throws std::out_of_range If the indices are out of range
     */
    void check_indices(int row, int col) const;
};

/**
 * @class DistributedVector
 * @brief A memory-efficient vector that is distributed across multiple processes.
 *
 * The DistributedVector class provides a memory-efficient implementation of a
 * vector that is distributed across multiple processes using MPI. It is useful
 * for large-scale simulations where the vectors do not fit in the memory of a
 * single process.
 *
 * @tparam Scalar The scalar type of the vector elements
 */
template <typename Scalar>
class DistributedVector {
public:
    /**
     * @brief Constructs a new DistributedVector object.
     *
     * @param size The global size of the vector
     * @param comm The MPI communicator to use
     *
     * @throws std::runtime_error If MPI is not available
     */
    DistributedVector(int size, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Destructor for the DistributedVector object.
     *
     * Cleans up resources used by the vector.
     */
    ~DistributedVector();

    /**
     * @brief Gets the global size of the vector.
     *
     * @return The global size
     */
    int size() const;

    /**
     * @brief Gets the local size of the vector on this process.
     *
     * @return The local size
     */
    int local_size() const;

    /**
     * @brief Gets the global index of the first element on this process.
     *
     * @return The global index
     */
    int global_start() const;

    /**
     * @brief Sets a vector element.
     *
     * @param index The global index
     * @param value The value to set
     *
     * @throws std::out_of_range If the index is out of range
     */
    void set(int index, const Scalar& value);

    /**
     * @brief Gets a vector element.
     *
     * @param index The global index
     *
     * @return The vector element
     *
     * @throws std::out_of_range If the index is out of range
     */
    Scalar get(int index) const;

    /**
     * @brief Gets the local data on this process.
     *
     * @return The local data
     */
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& local_data() const;

    /**
     * @brief Gets the local data on this process (mutable).
     *
     * @return The local data
     */
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& local_data();

    /**
     * @brief Gathers the vector to all processes.
     *
     * @return The gathered vector
     *
     * @throws std::runtime_error If MPI is not available
     */
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> gather() const;

    /**
     * @brief Scatters a vector from the root process to all processes.
     *
     * @param global_vec The global vector on the root process
     * @param root The root process
     *
     * @throws std::runtime_error If MPI is not available
     */
    void scatter(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& global_vec, int root = 0);

    /**
     * @brief Performs a dot product with another distributed vector.
     *
     * @param other The other vector
     *
     * @return The dot product
     *
     * @throws std::invalid_argument If the vector dimensions are invalid
     * @throws std::runtime_error If MPI is not available
     */
    Scalar dot(const DistributedVector<Scalar>& other) const;

    /**
     * @brief Performs a norm calculation.
     *
     * @return The norm
     *
     * @throws std::runtime_error If MPI is not available
     */
    double norm() const;

private:
    int size_;                  ///< The global size of the vector
    int local_size_;            ///< The local size of the vector on this process
    int global_start_;          ///< The global index of the first element on this process
    MPI_Comm comm_;             ///< The MPI communicator
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> local_data_; ///< The local data on this process

    /**
     * @brief Checks if the index is valid.
     *
     * @param index The global index
     *
     * @throws std::out_of_range If the index is out of range
     */
    void check_index(int index) const;

    /**
     * @brief Converts a global index to a local index.
     *
     * @param global_index The global index
     *
     * @return The local index
     *
     * @throws std::out_of_range If the index is out of range
     */
    int global_to_local(int global_index) const;

    /**
     * @brief Converts a local index to a global index.
     *
     * @param local_index The local index
     *
     * @return The global index
     *
     * @throws std::out_of_range If the index is out of range
     */
    int local_to_global(int local_index) const;
};
