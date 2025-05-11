/**
 * @file memory_efficient_sparse.cpp
 * @brief Implementation of memory-efficient sparse matrix operations.
 *
 * This file contains implementations of memory-efficient sparse matrix operations
 * for large-scale quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "memory_efficient_sparse.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

// Constructor with dimensions
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar>::MemoryEfficientSparseMatrix(int rows, int cols, int estimated_non_zeros)
    : rows_(rows), cols_(cols), non_zeros_(0), memory_pool_(CPUMemoryPool::getInstance()) {

    // Initialize CSR format
    row_ptr_.resize(rows_ + 1, 0);

    // Reserve space for non-zero elements if an estimate is provided
    if (estimated_non_zeros > 0) {
        col_ind_.reserve(estimated_non_zeros);
        values_.reserve(estimated_non_zeros);
    }
}

// Constructor from Eigen sparse matrix
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar>::MemoryEfficientSparseMatrix(const Eigen::SparseMatrix<Scalar>& matrix)
    : rows_(matrix.rows()), cols_(matrix.cols()), non_zeros_(matrix.nonZeros()),
      memory_pool_(CPUMemoryPool::getInstance()) {

    // Ensure the input matrix is in compressed format
    Eigen::SparseMatrix<Scalar> compressed_matrix = matrix;
    if (!compressed_matrix.isCompressed()) {
        compressed_matrix.makeCompressed();
    }

    // Copy CSR format from Eigen sparse matrix
    row_ptr_.resize(rows_ + 1);
    col_ind_.resize(non_zeros_);
    values_.resize(non_zeros_);

    // Copy row pointers
    for (int i = 0; i <= rows_; ++i) {
        row_ptr_[i] = compressed_matrix.outerIndexPtr()[i];
    }

    // Copy column indices and values
    for (int i = 0; i < non_zeros_; ++i) {
        col_ind_[i] = compressed_matrix.innerIndexPtr()[i];
        values_[i] = compressed_matrix.valuePtr()[i];
    }
}

// Copy constructor
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar>::MemoryEfficientSparseMatrix(const MemoryEfficientSparseMatrix<Scalar>& other)
    : rows_(other.rows_), cols_(other.cols_), non_zeros_(other.non_zeros_),
      row_ptr_(other.row_ptr_), col_ind_(other.col_ind_), values_(other.values_),
      memory_pool_(CPUMemoryPool::getInstance()) {
}

// Move constructor
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar>::MemoryEfficientSparseMatrix(MemoryEfficientSparseMatrix<Scalar>&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), non_zeros_(other.non_zeros_),
      memory_pool_(CPUMemoryPool::getInstance()) {

    // Move data from other
    row_ptr_ = std::move(other.row_ptr_);
    col_ind_ = std::move(other.col_ind_);
    values_ = std::move(other.values_);

    // Reset other
    other.rows_ = 0;
    other.cols_ = 0;
    other.non_zeros_ = 0;
}

// Copy assignment operator
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar>& MemoryEfficientSparseMatrix<Scalar>::operator=(const MemoryEfficientSparseMatrix<Scalar>& other) {
    if (this != &other) {
        // Copy data from other
        rows_ = other.rows_;
        cols_ = other.cols_;
        non_zeros_ = other.non_zeros_;
        row_ptr_ = other.row_ptr_;
        col_ind_ = other.col_ind_;
        values_ = other.values_;
    }
    return *this;
}

// Move assignment operator
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar>& MemoryEfficientSparseMatrix<Scalar>::operator=(MemoryEfficientSparseMatrix<Scalar>&& other) noexcept {
    if (this != &other) {
        // Move data from other
        rows_ = other.rows_;
        cols_ = other.cols_;
        non_zeros_ = other.non_zeros_;
        row_ptr_ = std::move(other.row_ptr_);
        col_ind_ = std::move(other.col_ind_);
        values_ = std::move(other.values_);

        // Reset other
        other.rows_ = 0;
        other.cols_ = 0;
        other.non_zeros_ = 0;
    }
    return *this;
}

// Destructor
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar>::~MemoryEfficientSparseMatrix() {
    // Nothing to do here, vectors will clean up automatically
}

// Get number of rows
template <typename Scalar>
int MemoryEfficientSparseMatrix<Scalar>::rows() const {
    return rows_;
}

// Get number of columns
template <typename Scalar>
int MemoryEfficientSparseMatrix<Scalar>::cols() const {
    return cols_;
}

// Get number of non-zero elements
template <typename Scalar>
int MemoryEfficientSparseMatrix<Scalar>::nonZeros() const {
    return non_zeros_;
}

// Set a value in the matrix
template <typename Scalar>
void MemoryEfficientSparseMatrix<Scalar>::set(int row, int col, const Scalar& value) {
    // Check if indices are valid
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }

    // Check if the element already exists
    for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
        if (col_ind_[i] == col) {
            // Element exists, update its value
            values_[i] = value;
            return;
        }
    }

    // Element doesn't exist, insert it
    if (value != Scalar(0)) {
        // Find the position to insert the new element
        int insert_pos = row_ptr_[row];
        while (insert_pos < row_ptr_[row + 1] && col_ind_[insert_pos] < col) {
            ++insert_pos;
        }

        // Insert the new element
        col_ind_.insert(col_ind_.begin() + insert_pos, col);
        values_.insert(values_.begin() + insert_pos, value);

        // Update row pointers
        for (int i = row + 1; i <= rows_; ++i) {
            ++row_ptr_[i];
        }

        // Increment non-zero count
        ++non_zeros_;
    }
}

// Get a value from the matrix
template <typename Scalar>
Scalar MemoryEfficientSparseMatrix<Scalar>::get(int row, int col) const {
    // Check if indices are valid
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }

    // Search for the element
    for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
        if (col_ind_[i] == col) {
            return values_[i];
        }
    }

    // Element not found, return zero
    return Scalar(0);
}

// Convert to Eigen sparse matrix
template <typename Scalar>
Eigen::SparseMatrix<Scalar> MemoryEfficientSparseMatrix<Scalar>::toEigen() const {
    // Create a sparse matrix in triplet format
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> triplets;
    triplets.reserve(non_zeros_);

    // Fill triplets
    for (int row = 0; row < rows_; ++row) {
        for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            int col = col_ind_[i];
            Scalar value = values_[i];
            triplets.push_back(T(row, col, value));
        }
    }

    // Create the sparse matrix
    Eigen::SparseMatrix<Scalar> matrix(rows_, cols_);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();

    return matrix;
}

// Multiply matrix by vector
template <typename Scalar>
void MemoryEfficientSparseMatrix<Scalar>::multiply(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
                                                 Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y) const {
    // Check dimensions
    if (x.size() != cols_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    // Resize result vector if needed
    if (y.size() != rows_) {
        y.resize(rows_);
    }

    // Perform matrix-vector multiplication
    for (int row = 0; row < rows_; ++row) {
        y[row] = Scalar(0);
        for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            int col = col_ind_[i];
            Scalar value = values_[i];
            y[row] += value * x[col];
        }
    }
}

// Add another matrix
template <typename Scalar>
void MemoryEfficientSparseMatrix<Scalar>::add(const MemoryEfficientSparseMatrix<Scalar>& other) {
    // Check dimensions
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }

    // Create a new matrix in triplet format
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> triplets;
    triplets.reserve(non_zeros_ + other.non_zeros_);

    // Add elements from this matrix
    for (int row = 0; row < rows_; ++row) {
        for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            int col = col_ind_[i];
            Scalar value = values_[i];
            triplets.push_back(T(row, col, value));
        }
    }

    // Add elements from the other matrix
    for (int row = 0; row < other.rows_; ++row) {
        for (int i = other.row_ptr_[row]; i < other.row_ptr_[row + 1]; ++i) {
            int col = other.col_ind_[i];
            Scalar value = other.values_[i];
            triplets.push_back(T(row, col, value));
        }
    }

    // Create an Eigen sparse matrix
    Eigen::SparseMatrix<Scalar> result(rows_, cols_);
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();

    // Convert back to our format
    *this = MemoryEfficientSparseMatrix<Scalar>(result);
}

// Subtract another matrix
template <typename Scalar>
void MemoryEfficientSparseMatrix<Scalar>::subtract(const MemoryEfficientSparseMatrix<Scalar>& other) {
    // Check dimensions
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }

    // Create a new matrix in triplet format
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> triplets;
    triplets.reserve(non_zeros_ + other.non_zeros_);

    // Add elements from this matrix
    for (int row = 0; row < rows_; ++row) {
        for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            int col = col_ind_[i];
            Scalar value = values_[i];
            triplets.push_back(T(row, col, value));
        }
    }

    // Subtract elements from the other matrix
    for (int row = 0; row < other.rows_; ++row) {
        for (int i = other.row_ptr_[row]; i < other.row_ptr_[row + 1]; ++i) {
            int col = other.col_ind_[i];
            Scalar value = -other.values_[i];
            triplets.push_back(T(row, col, value));
        }
    }

    // Create an Eigen sparse matrix
    Eigen::SparseMatrix<Scalar> result(rows_, cols_);
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();

    // Convert back to our format
    *this = MemoryEfficientSparseMatrix<Scalar>(result);
}

// Scale by a scalar
template <typename Scalar>
void MemoryEfficientSparseMatrix<Scalar>::scale(const Scalar& scalar) {
    // Scale all non-zero values
    for (int i = 0; i < non_zeros_; ++i) {
        values_[i] *= scalar;
    }
}

// Compute transpose
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar> MemoryEfficientSparseMatrix<Scalar>::transpose() const {
    // Create a new matrix in triplet format
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> triplets;
    triplets.reserve(non_zeros_);

    // Fill triplets with transposed elements
    for (int row = 0; row < rows_; ++row) {
        for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            int col = col_ind_[i];
            Scalar value = values_[i];
            triplets.push_back(T(col, row, value));
        }
    }

    // Create an Eigen sparse matrix
    Eigen::SparseMatrix<Scalar> result(cols_, rows_);
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();

    // Convert to our format
    return MemoryEfficientSparseMatrix<Scalar>(result);
}

// Compute conjugate transpose
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar> MemoryEfficientSparseMatrix<Scalar>::adjoint() const {
    // Create a new matrix in triplet format
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> triplets;
    triplets.reserve(non_zeros_);

    // Fill triplets with conjugate transposed elements
    for (int row = 0; row < rows_; ++row) {
        for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            int col = col_ind_[i];
            Scalar value = std::conj(values_[i]);
            triplets.push_back(T(col, row, value));
        }
    }

    // Create an Eigen sparse matrix
    Eigen::SparseMatrix<Scalar> result(cols_, rows_);
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();

    // Convert to our format
    return MemoryEfficientSparseMatrix<Scalar>(result);
}

// Compute matrix-matrix product
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar> MemoryEfficientSparseMatrix<Scalar>::multiply(
    const MemoryEfficientSparseMatrix<Scalar>& other) const {

    // Check dimensions
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
    }

    // Convert to Eigen sparse matrices for multiplication
    Eigen::SparseMatrix<Scalar> a = this->toEigen();
    Eigen::SparseMatrix<Scalar> b = other.toEigen();

    // Compute product
    Eigen::SparseMatrix<Scalar> result = a * b;

    // Convert back to our format
    return MemoryEfficientSparseMatrix<Scalar>(result);
}

// Compute Frobenius norm
template <typename Scalar>
double MemoryEfficientSparseMatrix<Scalar>::norm() const {
    double sum_squares = 0.0;

    // Sum squares of all non-zero elements
    for (int i = 0; i < non_zeros_; ++i) {
        sum_squares += std::norm(values_[i]);
    }

    return std::sqrt(sum_squares);
}

// Compute trace
template <typename Scalar>
Scalar MemoryEfficientSparseMatrix<Scalar>::trace() const {
    // Check if the matrix is square
    if (rows_ != cols_) {
        throw std::invalid_argument("Trace is only defined for square matrices");
    }

    Scalar trace_value = Scalar(0);

    // Sum diagonal elements
    for (int row = 0; row < rows_; ++row) {
        for (int i = row_ptr_[row]; i < row_ptr_[row + 1]; ++i) {
            if (col_ind_[i] == row) {
                trace_value += values_[i];
                break;
            }
        }
    }

    return trace_value;
}

// Compute determinant
template <typename Scalar>
Scalar MemoryEfficientSparseMatrix<Scalar>::determinant() const {
    // Check if the matrix is square
    if (rows_ != cols_) {
        throw std::invalid_argument("Determinant is only defined for square matrices");
    }

    // Convert to Eigen sparse matrix
    Eigen::SparseMatrix<Scalar> eigen_matrix = this->toEigen();

    // For small matrices, convert to dense and use LU decomposition
    if (rows_ <= 100) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense_matrix = eigen_matrix;
        return dense_matrix.determinant();
    } else {
        // For larger matrices, use sparse LU decomposition
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(eigen_matrix);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Failed to compute determinant");
        }

        // The determinant is the product of the diagonal elements of U
        Scalar det = Scalar(1);
        for (int i = 0; i < rows_; ++i) {
            det *= solver.matrixU().coeff(i, i);
        }

        return det;
    }
}

// Compute inverse
template <typename Scalar>
MemoryEfficientSparseMatrix<Scalar> MemoryEfficientSparseMatrix<Scalar>::inverse() const {
    // Check if the matrix is square
    if (rows_ != cols_) {
        throw std::invalid_argument("Inverse is only defined for square matrices");
    }

    // Convert to Eigen sparse matrix
    Eigen::SparseMatrix<Scalar> eigen_matrix = this->toEigen();

    // For small matrices, convert to dense and use LU decomposition
    if (rows_ <= 100) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense_matrix = eigen_matrix;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inverse_matrix = dense_matrix.inverse();

        // Convert back to sparse
        Eigen::SparseMatrix<Scalar> sparse_inverse = inverse_matrix.sparseView();

        return MemoryEfficientSparseMatrix<Scalar>(sparse_inverse);
    } else {
        // For larger matrices, use sparse LU decomposition
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(eigen_matrix);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Failed to compute inverse");
        }

        // Create identity matrix
        Eigen::SparseMatrix<Scalar> identity(rows_, cols_);
        identity.setIdentity();

        // Solve for each column of the inverse
        Eigen::SparseMatrix<Scalar> inverse(rows_, cols_);
        for (int col = 0; col < cols_; ++col) {
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> e = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(cols_);
            e(col) = Scalar(1);

            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x = solver.solve(e);

            if (solver.info() != Eigen::Success) {
                throw std::runtime_error("Failed to compute inverse");
            }

            // Set column of inverse
            for (int row = 0; row < rows_; ++row) {
                if (x(row) != Scalar(0)) {
                    inverse.insert(row, col) = x(row);
                }
            }
        }

        inverse.makeCompressed();

        return MemoryEfficientSparseMatrix<Scalar>(inverse);
    }
}

// Solve linear system Ax = b
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> MemoryEfficientSparseMatrix<Scalar>::solve(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b) const {

    // Check dimensions
    if (rows_ != b.size()) {
        throw std::invalid_argument("Vector dimension mismatch for solving");
    }

    // Check if the matrix is square
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square for solving");
    }

    // Convert to Eigen sparse matrix
    Eigen::SparseMatrix<Scalar> eigen_matrix = this->toEigen();

    // For small matrices, convert to dense and use LU decomposition
    if (rows_ <= 100) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense_matrix = eigen_matrix;
        return dense_matrix.lu().solve(b);
    } else {
        // For larger matrices, use sparse LU decomposition
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(eigen_matrix);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Failed to factorize matrix for solving");
        }

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x = solver.solve(b);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Failed to solve linear system");
        }

        return x;
    }
}

// Compute eigenvalues and eigenvectors
template <typename Scalar>
void MemoryEfficientSparseMatrix<Scalar>::eigensolve(
    std::vector<Scalar>& eigenvalues,
    std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& eigenvectors,
    int num_eigenvalues) const {

    // Check if the matrix is square
    if (rows_ != cols_) {
        throw std::invalid_argument("Eigendecomposition is only defined for square matrices");
    }

    // If num_eigenvalues is not specified, compute all eigenvalues
    if (num_eigenvalues <= 0 || num_eigenvalues > rows_) {
        num_eigenvalues = rows_;
    }

    // Convert to Eigen sparse matrix
    Eigen::SparseMatrix<Scalar> eigen_matrix = this->toEigen();

    // Convert to dense matrix and use Eigen's built-in eigensolvers
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense_matrix = eigen_matrix;

    // Use Eigen's SelfAdjointEigenSolver for Hermitian matrices
    // or EigenSolver for general matrices
    if (isHermitian(dense_matrix)) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver(dense_matrix);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Failed to compute eigendecomposition");
        }

        // Get eigenvalues and eigenvectors
        eigenvalues.resize(num_eigenvalues);
        eigenvectors.resize(num_eigenvalues);

        for (int i = 0; i < num_eigenvalues; ++i) {
            eigenvalues[i] = solver.eigenvalues()(i);
            eigenvectors[i] = solver.eigenvectors().col(i);
        }
    } else {
        Eigen::ComplexEigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver(dense_matrix);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Failed to compute eigendecomposition");
        }

        // Get eigenvalues and eigenvectors
        eigenvalues.resize(num_eigenvalues);
        eigenvectors.resize(num_eigenvalues);

        for (int i = 0; i < num_eigenvalues; ++i) {
            eigenvalues[i] = solver.eigenvalues()(i);
            eigenvectors[i] = solver.eigenvectors().col(i);
        }
    }
}

// Save matrix to file
template <typename Scalar>
bool MemoryEfficientSparseMatrix<Scalar>::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    // Write dimensions
    file.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
    file.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
    file.write(reinterpret_cast<const char*>(&non_zeros_), sizeof(non_zeros_));

    // Write row pointers
    int row_ptr_size = static_cast<int>(row_ptr_.size());
    file.write(reinterpret_cast<const char*>(&row_ptr_size), sizeof(row_ptr_size));
    file.write(reinterpret_cast<const char*>(row_ptr_.data()), row_ptr_size * sizeof(int));

    // Write column indices
    int col_ind_size = static_cast<int>(col_ind_.size());
    file.write(reinterpret_cast<const char*>(&col_ind_size), sizeof(col_ind_size));
    file.write(reinterpret_cast<const char*>(col_ind_.data()), col_ind_size * sizeof(int));

    // Write values
    int values_size = static_cast<int>(values_.size());
    file.write(reinterpret_cast<const char*>(&values_size), sizeof(values_size));
    file.write(reinterpret_cast<const char*>(values_.data()), values_size * sizeof(Scalar));

    return true;
}

// Load matrix from file
template <typename Scalar>
bool MemoryEfficientSparseMatrix<Scalar>::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    // Read dimensions
    file.read(reinterpret_cast<char*>(&rows_), sizeof(rows_));
    file.read(reinterpret_cast<char*>(&cols_), sizeof(cols_));
    file.read(reinterpret_cast<char*>(&non_zeros_), sizeof(non_zeros_));

    // Read row pointers
    int row_ptr_size;
    file.read(reinterpret_cast<char*>(&row_ptr_size), sizeof(row_ptr_size));
    row_ptr_.resize(row_ptr_size);
    file.read(reinterpret_cast<char*>(row_ptr_.data()), row_ptr_size * sizeof(int));

    // Read column indices
    int col_ind_size;
    file.read(reinterpret_cast<char*>(&col_ind_size), sizeof(col_ind_size));
    col_ind_.resize(col_ind_size);
    file.read(reinterpret_cast<char*>(col_ind_.data()), col_ind_size * sizeof(int));

    // Read values
    int values_size;
    file.read(reinterpret_cast<char*>(&values_size), sizeof(values_size));
    values_.resize(values_size);
    file.read(reinterpret_cast<char*>(values_.data()), values_size * sizeof(Scalar));

    return true;
}

// Check if a matrix is Hermitian (self-adjoint)
template <typename Scalar>
bool MemoryEfficientSparseMatrix<Scalar>::isHermitian(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix) const {
    // Check if the matrix is square
    if (matrix.rows() != matrix.cols()) {
        return false;
    }

    // Check if the matrix is equal to its conjugate transpose
    const double tolerance = 1e-10;
    return (matrix - matrix.adjoint()).norm() < tolerance;
}

// Explicit instantiations
template class MemoryEfficientSparseMatrix<double>;
template class MemoryEfficientSparseMatrix<std::complex<double>>;
