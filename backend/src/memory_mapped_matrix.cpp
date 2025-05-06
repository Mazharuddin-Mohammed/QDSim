/**
 * @file memory_mapped_matrix.cpp
 * @brief Implementation of memory-mapped matrices for QDSim.
 *
 * This file contains implementations of memory-mapped matrices
 * for handling extremely large matrices in quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "memory_mapped_matrix.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <complex>

namespace MemoryMapped {

// MemoryMappedMatrix implementation

// Constructor
template <typename Scalar>
MemoryMappedMatrix<Scalar>::MemoryMappedMatrix(const std::string& filename, int rows, int cols, MapMode mode)
    : rows_(rows), cols_(cols) {
    
    // Check dimensions
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    
    // Calculate file size
    size_t size = static_cast<size_t>(rows) * cols * sizeof(Scalar);
    
    // Create memory-mapped file
    file_ = std::make_unique<MemoryMappedFile>(filename, mode, size);
    
    if (!file_->isMapped()) {
        throw std::runtime_error("Failed to map file: " + filename);
    }
}

// Destructor
template <typename Scalar>
MemoryMappedMatrix<Scalar>::~MemoryMappedMatrix() {
    // Nothing to do here, file_ will be automatically destroyed
}

// Get number of rows
template <typename Scalar>
int MemoryMappedMatrix<Scalar>::rows() const {
    return rows_;
}

// Get number of columns
template <typename Scalar>
int MemoryMappedMatrix<Scalar>::cols() const {
    return cols_;
}

// Get a value from the matrix
template <typename Scalar>
Scalar MemoryMappedMatrix<Scalar>::get(int row, int col) const {
    // Check if indices are valid
    if (!isValidIndex(row, col)) {
        throw std::out_of_range("Matrix indices out of range");
    }
    
    // Calculate offset
    size_t offset = getOffset(row, col);
    
    // Read value
    Scalar value;
    file_->read(offset, &value, sizeof(Scalar));
    
    return value;
}

// Set a value in the matrix
template <typename Scalar>
void MemoryMappedMatrix<Scalar>::set(int row, int col, const Scalar& value) {
    // Check if indices are valid
    if (!isValidIndex(row, col)) {
        throw std::out_of_range("Matrix indices out of range");
    }
    
    // Calculate offset
    size_t offset = getOffset(row, col);
    
    // Write value
    file_->write(offset, &value, sizeof(Scalar));
}

// Get a block from the matrix
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MemoryMappedMatrix<Scalar>::getBlock(
    int start_row, int start_col, int block_rows, int block_cols) const {
    
    // Check if indices are valid
    if (start_row < 0 || start_row + block_rows > rows_ ||
        start_col < 0 || start_col + block_cols > cols_) {
        throw std::out_of_range("Block indices out of range");
    }
    
    // Create result matrix
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(block_rows, block_cols);
    
    // Read block row by row
    for (int i = 0; i < block_rows; ++i) {
        int row = start_row + i;
        size_t offset = getOffset(row, start_col);
        file_->read(offset, result.row(i).data(), block_cols * sizeof(Scalar));
    }
    
    return result;
}

// Set a block in the matrix
template <typename Scalar>
void MemoryMappedMatrix<Scalar>::setBlock(int start_row, int start_col,
                                        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& block) {
    
    // Check if indices are valid
    if (start_row < 0 || start_row + block.rows() > rows_ ||
        start_col < 0 || start_col + block.cols() > cols_) {
        throw std::out_of_range("Block indices out of range");
    }
    
    // Write block row by row
    for (int i = 0; i < block.rows(); ++i) {
        int row = start_row + i;
        size_t offset = getOffset(row, start_col);
        file_->write(offset, block.row(i).data(), block.cols() * sizeof(Scalar));
    }
}

// Multiply the matrix by a vector
template <typename Scalar>
void MemoryMappedMatrix<Scalar>::multiply(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
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
    
    // Perform matrix-vector multiplication block by block
    const int block_size = 1024;  // Block size for better cache utilization
    
    for (int i = 0; i < rows_; i += block_size) {
        int block_rows = std::min(block_size, rows_ - i);
        
        for (int j = 0; j < cols_; j += block_size) {
            int block_cols = std::min(block_size, cols_ - j);
            
            // Get block
            auto block = getBlock(i, j, block_rows, block_cols);
            
            // Multiply block by corresponding part of x
            y.segment(i, block_rows) += block * x.segment(j, block_cols);
        }
    }
}

// Flush changes to disk
template <typename Scalar>
bool MemoryMappedMatrix<Scalar>::flush(bool async) {
    return file_->flush(async);
}

// Prefault pages into memory
template <typename Scalar>
bool MemoryMappedMatrix<Scalar>::prefault(int start_row, int start_col, int block_rows, int block_cols) {
    // Check if indices are valid
    if (start_row < 0 || start_row + block_rows > rows_ ||
        start_col < 0 || start_col + block_cols > cols_) {
        throw std::out_of_range("Block indices out of range");
    }
    
    // Calculate offset and size
    size_t offset = getOffset(start_row, start_col);
    size_t size = static_cast<size_t>(block_rows) * block_cols * sizeof(Scalar);
    
    return file_->prefault(offset, size);
}

// Get mapping statistics
template <typename Scalar>
MappingStats MemoryMappedMatrix<Scalar>::getStats() const {
    return file_->getStats();
}

// Get offset for a matrix element
template <typename Scalar>
size_t MemoryMappedMatrix<Scalar>::getOffset(int row, int col) const {
    return (static_cast<size_t>(row) * cols_ + col) * sizeof(Scalar);
}

// Check if indices are valid
template <typename Scalar>
bool MemoryMappedMatrix<Scalar>::isValidIndex(int row, int col) const {
    return row >= 0 && row < rows_ && col >= 0 && col < cols_;
}

// MemoryMappedSparseMatrix implementation

// Constructor
template <typename Scalar>
MemoryMappedSparseMatrix<Scalar>::MemoryMappedSparseMatrix(const std::string& filename,
                                                         int rows, int cols,
                                                         int estimated_non_zeros,
                                                         MapMode mode)
    : rows_(rows), cols_(cols), non_zeros_(0) {
    
    // Check dimensions
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    
    // Ensure estimated_non_zeros is positive
    if (estimated_non_zeros <= 0) {
        estimated_non_zeros = std::max(10, rows * cols / 100);  // Default to 1% density
    }
    
    // Calculate file sizes
    size_t row_ptr_size = static_cast<size_t>(rows + 1) * sizeof(int);
    size_t col_ind_size = static_cast<size_t>(estimated_non_zeros) * sizeof(int);
    size_t values_size = static_cast<size_t>(estimated_non_zeros) * sizeof(Scalar);
    
    // Create memory-mapped files
    row_ptr_file_ = std::make_unique<MemoryMappedFile>(filename + ".row_ptr", mode, row_ptr_size);
    col_ind_file_ = std::make_unique<MemoryMappedFile>(filename + ".col_ind", mode, col_ind_size);
    values_file_ = std::make_unique<MemoryMappedFile>(filename + ".values", mode, values_size);
    
    if (!row_ptr_file_->isMapped() || !col_ind_file_->isMapped() || !values_file_->isMapped()) {
        throw std::runtime_error("Failed to map files for sparse matrix: " + filename);
    }
    
    // Initialize row pointers to zero
    int zero = 0;
    for (int i = 0; i <= rows_; ++i) {
        row_ptr_file_->write(i * sizeof(int), &zero, sizeof(int));
    }
}

// Destructor
template <typename Scalar>
MemoryMappedSparseMatrix<Scalar>::~MemoryMappedSparseMatrix() {
    // Nothing to do here, files will be automatically destroyed
}

// Get number of rows
template <typename Scalar>
int MemoryMappedSparseMatrix<Scalar>::rows() const {
    return rows_;
}

// Get number of columns
template <typename Scalar>
int MemoryMappedSparseMatrix<Scalar>::cols() const {
    return cols_;
}

// Get number of non-zero elements
template <typename Scalar>
int MemoryMappedSparseMatrix<Scalar>::nonZeros() const {
    return non_zeros_;
}

// Get a value from the matrix
template <typename Scalar>
Scalar MemoryMappedSparseMatrix<Scalar>::get(int row, int col) const {
    // Check if indices are valid
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    
    // Get row pointers
    int row_start, row_end;
    row_ptr_file_->read(row * sizeof(int), &row_start, sizeof(int));
    row_ptr_file_->read((row + 1) * sizeof(int), &row_end, sizeof(int));
    
    // Search for the column index
    for (int i = row_start; i < row_end; ++i) {
        int col_idx;
        col_ind_file_->read(i * sizeof(int), &col_idx, sizeof(int));
        
        if (col_idx == col) {
            // Found the element
            Scalar value;
            values_file_->read(i * sizeof(Scalar), &value, sizeof(Scalar));
            return value;
        }
    }
    
    // Element not found, return zero
    return Scalar(0);
}

// Set a value in the matrix
template <typename Scalar>
void MemoryMappedSparseMatrix<Scalar>::set(int row, int col, const Scalar& value) {
    // Check if indices are valid
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    
    // Get row pointers
    int row_start, row_end;
    row_ptr_file_->read(row * sizeof(int), &row_start, sizeof(int));
    row_ptr_file_->read((row + 1) * sizeof(int), &row_end, sizeof(int));
    
    // Search for the column index
    for (int i = row_start; i < row_end; ++i) {
        int col_idx;
        col_ind_file_->read(i * sizeof(int), &col_idx, sizeof(int));
        
        if (col_idx == col) {
            // Found the element, update its value
            if (value == Scalar(0)) {
                // Remove the element
                // Shift all elements after this one
                for (int j = i + 1; j < non_zeros_; ++j) {
                    int next_col;
                    col_ind_file_->read(j * sizeof(int), &next_col, sizeof(int));
                    col_ind_file_->write((j - 1) * sizeof(int), &next_col, sizeof(int));
                    
                    Scalar next_value;
                    values_file_->read(j * sizeof(Scalar), &next_value, sizeof(Scalar));
                    values_file_->write((j - 1) * sizeof(Scalar), &next_value, sizeof(Scalar));
                }
                
                // Update row pointers
                for (int j = row + 1; j <= rows_; ++j) {
                    int ptr;
                    row_ptr_file_->read(j * sizeof(int), &ptr, sizeof(int));
                    ptr--;
                    row_ptr_file_->write(j * sizeof(int), &ptr, sizeof(int));
                }
                
                non_zeros_--;
            } else {
                // Update the value
                values_file_->write(i * sizeof(Scalar), &value, sizeof(Scalar));
            }
            
            return;
        }
    }
    
    // Element not found, insert it if value is non-zero
    if (value != Scalar(0)) {
        // Check if we need to resize the files
        if (non_zeros_ >= static_cast<int>(col_ind_file_->getSize() / sizeof(int))) {
            // Resize files
            size_t new_size = col_ind_file_->getSize() * 2;
            col_ind_file_->resize(new_size);
            values_file_->resize(new_size / sizeof(int) * sizeof(Scalar));
        }
        
        // Find the position to insert the new element
        int insert_pos = row_start;
        while (insert_pos < row_end) {
            int col_idx;
            col_ind_file_->read(insert_pos * sizeof(int), &col_idx, sizeof(int));
            
            if (col_idx > col) {
                break;
            }
            
            insert_pos++;
        }
        
        // Shift all elements after the insertion point
        for (int i = non_zeros_ - 1; i >= insert_pos; --i) {
            int col_idx;
            col_ind_file_->read(i * sizeof(int), &col_idx, sizeof(int));
            col_ind_file_->write((i + 1) * sizeof(int), &col_idx, sizeof(int));
            
            Scalar val;
            values_file_->read(i * sizeof(Scalar), &val, sizeof(Scalar));
            values_file_->write((i + 1) * sizeof(Scalar), &val, sizeof(Scalar));
        }
        
        // Insert the new element
        col_ind_file_->write(insert_pos * sizeof(int), &col, sizeof(int));
        values_file_->write(insert_pos * sizeof(Scalar), &value, sizeof(Scalar));
        
        // Update row pointers
        for (int i = row + 1; i <= rows_; ++i) {
            int ptr;
            row_ptr_file_->read(i * sizeof(int), &ptr, sizeof(int));
            ptr++;
            row_ptr_file_->write(i * sizeof(int), &ptr, sizeof(int));
        }
        
        non_zeros_++;
    }
}

// Convert to Eigen sparse matrix
template <typename Scalar>
Eigen::SparseMatrix<Scalar> MemoryMappedSparseMatrix<Scalar>::toEigen() const {
    // Create triplet list
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(non_zeros_);
    
    // Read row pointers
    std::vector<int> row_ptr(rows_ + 1);
    for (int i = 0; i <= rows_; ++i) {
        row_ptr_file_->read(i * sizeof(int), &row_ptr[i], sizeof(int));
    }
    
    // Read column indices and values
    for (int row = 0; row < rows_; ++row) {
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
            int col;
            col_ind_file_->read(i * sizeof(int), &col, sizeof(int));
            
            Scalar value;
            values_file_->read(i * sizeof(Scalar), &value, sizeof(Scalar));
            
            triplets.push_back(Eigen::Triplet<Scalar>(row, col, value));
        }
    }
    
    // Create sparse matrix
    Eigen::SparseMatrix<Scalar> result(rows_, cols_);
    result.setFromTriplets(triplets.begin(), triplets.end());
    
    return result;
}

// Multiply the matrix by a vector
template <typename Scalar>
void MemoryMappedSparseMatrix<Scalar>::multiply(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
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
    
    // Read row pointers
    std::vector<int> row_ptr(rows_ + 1);
    for (int i = 0; i <= rows_; ++i) {
        row_ptr_file_->read(i * sizeof(int), &row_ptr[i], sizeof(int));
    }
    
    // Perform matrix-vector multiplication
    for (int row = 0; row < rows_; ++row) {
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
            int col;
            col_ind_file_->read(i * sizeof(int), &col, sizeof(int));
            
            Scalar value;
            values_file_->read(i * sizeof(Scalar), &value, sizeof(Scalar));
            
            y(row) += value * x(col);
        }
    }
}

// Flush changes to disk
template <typename Scalar>
bool MemoryMappedSparseMatrix<Scalar>::flush(bool async) {
    return row_ptr_file_->flush(async) && col_ind_file_->flush(async) && values_file_->flush(async);
}

// Get mapping statistics
template <typename Scalar>
MappingStats MemoryMappedSparseMatrix<Scalar>::getStats() const {
    MappingStats row_ptr_stats = row_ptr_file_->getStats();
    MappingStats col_ind_stats = col_ind_file_->getStats();
    MappingStats values_stats = values_file_->getStats();
    
    // Combine statistics
    MappingStats stats;
    stats.file_size = row_ptr_stats.file_size + col_ind_stats.file_size + values_stats.file_size;
    stats.mapped_size = row_ptr_stats.mapped_size + col_ind_stats.mapped_size + values_stats.mapped_size;
    stats.page_size = row_ptr_stats.page_size;
    stats.page_faults = row_ptr_stats.page_faults + col_ind_stats.page_faults + values_stats.page_faults;
    stats.read_operations = row_ptr_stats.read_operations + col_ind_stats.read_operations + values_stats.read_operations;
    stats.write_operations = row_ptr_stats.write_operations + col_ind_stats.write_operations + values_stats.write_operations;
    stats.avg_read_time = (row_ptr_stats.avg_read_time + col_ind_stats.avg_read_time + values_stats.avg_read_time) / 3.0;
    stats.avg_write_time = (row_ptr_stats.avg_write_time + col_ind_stats.avg_write_time + values_stats.avg_write_time) / 3.0;
    
    return stats;
}

// Update the number of non-zero elements
template <typename Scalar>
void MemoryMappedSparseMatrix<Scalar>::updateNonZeros() {
    // Read the last row pointer
    int last_ptr;
    row_ptr_file_->read(rows_ * sizeof(int), &last_ptr, sizeof(int));
    non_zeros_ = last_ptr;
}

// Explicit instantiations
template class MemoryMappedMatrix<double>;
template class MemoryMappedMatrix<std::complex<double>>;
template class MemoryMappedSparseMatrix<double>;
template class MemoryMappedSparseMatrix<std::complex<double>>;

} // namespace MemoryMapped
