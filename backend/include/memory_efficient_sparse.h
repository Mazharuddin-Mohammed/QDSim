#pragma once
/**
 * @file memory_efficient_sparse.h
 * @brief Defines memory-efficient sparse matrix operations.
 *
 * This file contains declarations of memory-efficient sparse matrix operations
 * for large-scale quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Sparse>
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "cpu_memory_pool.h"

/**
 * @class MemoryEfficientSparseMatrix
 * @brief A memory-efficient sparse matrix implementation.
 *
 * The MemoryEfficientSparseMatrix class provides a memory-efficient implementation
 * of a sparse matrix that uses memory pooling and other optimization techniques
 * to reduce memory usage and improve performance.
 *
 * @tparam Scalar The scalar type of the matrix elements
 */
template <typename Scalar>
class MemoryEfficientSparseMatrix {
public:
    /**
     * @brief Constructs a new MemoryEfficientSparseMatrix object.
     *
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param estimated_non_zeros The estimated number of non-zero elements
     */
    MemoryEfficientSparseMatrix(int rows, int cols, int estimated_non_zeros = 0);

    /**
     * @brief Constructs a new MemoryEfficientSparseMatrix object from an Eigen sparse matrix.
     *
     * @param matrix The Eigen sparse matrix
     */
    MemoryEfficientSparseMatrix(const Eigen::SparseMatrix<Scalar>& matrix);

    /**
     * @brief Copy constructor for the MemoryEfficientSparseMatrix object.
     *
     * @param other The matrix to copy
     */
    MemoryEfficientSparseMatrix(const MemoryEfficientSparseMatrix<Scalar>& other);

    /**
     * @brief Move constructor for the MemoryEfficientSparseMatrix object.
     *
     * @param other The matrix to move
     */
    MemoryEfficientSparseMatrix(MemoryEfficientSparseMatrix<Scalar>&& other) noexcept;

    /**
     * @brief Copy assignment operator for the MemoryEfficientSparseMatrix object.
     *
     * @param other The matrix to copy
     * @return A reference to this matrix
     */
    MemoryEfficientSparseMatrix<Scalar>& operator=(const MemoryEfficientSparseMatrix<Scalar>& other);

    /**
     * @brief Move assignment operator for the MemoryEfficientSparseMatrix object.
     *
     * @param other The matrix to move
     * @return A reference to this matrix
     */
    MemoryEfficientSparseMatrix<Scalar>& operator=(MemoryEfficientSparseMatrix<Scalar>&& other) noexcept;

    /**
     * @brief Destructor for the MemoryEfficientSparseMatrix object.
     */
    ~MemoryEfficientSparseMatrix();

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
     * @brief Gets the number of non-zero elements in the matrix.
     *
     * @return The number of non-zero elements
     */
    int nonZeros() const;

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
     * @brief Converts the matrix to an Eigen sparse matrix.
     *
     * @return The Eigen sparse matrix
     */
    Eigen::SparseMatrix<Scalar> toEigen() const;

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
    void add(const MemoryEfficientSparseMatrix<Scalar>& other);

    /**
     * @brief Subtracts another matrix from this matrix.
     *
     * @param other The matrix to subtract
     */
    void subtract(const MemoryEfficientSparseMatrix<Scalar>& other);

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
    MemoryEfficientSparseMatrix<Scalar> transpose() const;

    /**
     * @brief Computes the conjugate transpose of the matrix.
     *
     * @return The conjugate transposed matrix
     */
    MemoryEfficientSparseMatrix<Scalar> adjoint() const;

    /**
     * @brief Computes the matrix-matrix product.
     *
     * @param other The matrix to multiply with
     * @return The product matrix
     */
    MemoryEfficientSparseMatrix<Scalar> multiply(const MemoryEfficientSparseMatrix<Scalar>& other) const;

    /**
     * @brief Computes the Frobenius norm of the matrix.
     *
     * @return The Frobenius norm
     */
    double norm() const;

    /**
     * @brief Computes the trace of the matrix.
     *
     * @return The trace
     */
    Scalar trace() const;

    /**
     * @brief Computes the determinant of the matrix.
     *
     * @return The determinant
     */
    Scalar determinant() const;

    /**
     * @brief Check if a matrix is Hermitian (self-adjoint).
     *
     * @param matrix The matrix to check
     * @return True if the matrix is Hermitian, false otherwise
     */
    bool isHermitian(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix) const;

    /**
     * @brief Computes the inverse of the matrix.
     *
     * @return The inverse matrix
     */
    MemoryEfficientSparseMatrix<Scalar> inverse() const;

    /**
     * @brief Solves the linear system Ax = b.
     *
     * @param b The right-hand side vector
     * @return The solution vector
     */
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b) const;

    /**
     * @brief Computes the eigenvalues and eigenvectors of the matrix.
     *
     * @param eigenvalues The eigenvalues
     * @param eigenvectors The eigenvectors
     * @param num_eigenvalues The number of eigenvalues to compute
     */
    void eigensolve(std::vector<Scalar>& eigenvalues,
                   std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& eigenvectors,
                   int num_eigenvalues = 0) const;

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

private:
    int rows_;                  ///< Number of rows
    int cols_;                  ///< Number of columns
    int non_zeros_;             ///< Number of non-zero elements

    // Compressed Sparse Row (CSR) format
    std::vector<int> row_ptr_;  ///< Row pointers
    std::vector<int> col_ind_;  ///< Column indices
    std::vector<Scalar> values_; ///< Non-zero values

    // Memory pool for temporary allocations
    CPUMemoryPool& memory_pool_; ///< Memory pool
};

// Explicit instantiations
extern template class MemoryEfficientSparseMatrix<double>;
extern template class MemoryEfficientSparseMatrix<std::complex<double>>;

// Type aliases
using MemoryEfficientSparseMatrixd = MemoryEfficientSparseMatrix<double>;
using MemoryEfficientSparseMatrixcd = MemoryEfficientSparseMatrix<std::complex<double>>;
