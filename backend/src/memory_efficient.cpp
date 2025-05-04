/**
 * @file memory_efficient.cpp
 * @brief Implementation of memory-efficient data structures for large-scale simulations.
 *
 * This file contains implementations of memory-efficient data structures for
 * large-scale quantum simulations, including out-of-core matrices and
 * distributed memory vectors.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "memory_efficient.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>

// OutOfCoreMatrix implementation

template <typename Scalar>
OutOfCoreMatrix<Scalar>::OutOfCoreMatrix(int rows, int cols, int block_size, const std::string& filename)
    : rows_(rows), cols_(cols), block_size_(block_size) {
    
    // Generate a temporary filename if none is provided
    if (filename.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 999999);
        filename_ = "matrix_" + std::to_string(dis(gen)) + ".bin";
    } else {
        filename_ = filename;
    }
    
    // Open the file for reading and writing
    file_.open(filename_, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to create file: " + filename_);
    }
    
    // Allocate space for the matrix
    Scalar zero = Scalar(0);
    for (int i = 0; i < rows_ * cols_; ++i) {
        file_.write(reinterpret_cast<const char*>(&zero), sizeof(Scalar));
    }
    
    // Flush the file
    file_.flush();
}

template <typename Scalar>
OutOfCoreMatrix<Scalar>::~OutOfCoreMatrix() {
    // Close the file
    if (file_.is_open()) {
        file_.close();
    }
    
    // Delete the file
    std::remove(filename_.c_str());
}

template <typename Scalar>
int OutOfCoreMatrix<Scalar>::rows() const {
    return rows_;
}

template <typename Scalar>
int OutOfCoreMatrix<Scalar>::cols() const {
    return cols_;
}

template <typename Scalar>
int OutOfCoreMatrix<Scalar>::block_size() const {
    return block_size_;
}

template <typename Scalar>
std::string OutOfCoreMatrix<Scalar>::filename() const {
    return filename_;
}

template <typename Scalar>
void OutOfCoreMatrix<Scalar>::set(int row, int col, const Scalar& value) {
    // Check if indices are valid
    check_indices(row, col);
    
    // Get the file offset
    std::streampos offset = get_offset(row, col);
    
    // Seek to the offset
    file_.seekp(offset);
    
    // Write the value
    file_.write(reinterpret_cast<const char*>(&value), sizeof(Scalar));
    
    // Check for errors
    if (file_.fail()) {
        throw std::runtime_error("Failed to write to file: " + filename_);
    }
}

template <typename Scalar>
Scalar OutOfCoreMatrix<Scalar>::get(int row, int col) const {
    // Check if indices are valid
    check_indices(row, col);
    
    // Get the file offset
    std::streampos offset = get_offset(row, col);
    
    // Seek to the offset
    file_.seekg(offset);
    
    // Read the value
    Scalar value;
    file_.read(reinterpret_cast<char*>(&value), sizeof(Scalar));
    
    // Check for errors
    if (file_.fail()) {
        throw std::runtime_error("Failed to read from file: " + filename_);
    }
    
    return value;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> OutOfCoreMatrix<Scalar>::load_block(
    int row_start, int col_start, int row_size, int col_size) const {
    
    // Check if indices are valid
    if (row_start < 0 || row_start + row_size > rows_ ||
        col_start < 0 || col_start + col_size > cols_) {
        throw std::out_of_range("Block indices out of range");
    }
    
    // Create a matrix to store the block
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block(row_size, col_size);
    
    // Load the block element by element
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j) {
            block(i, j) = get(row_start + i, col_start + j);
        }
    }
    
    return block;
}

template <typename Scalar>
void OutOfCoreMatrix<Scalar>::save_block(
    int row_start, int col_start, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& block) {
    
    // Check if indices are valid
    if (row_start < 0 || row_start + block.rows() > rows_ ||
        col_start < 0 || col_start + block.cols() > cols_) {
        throw std::out_of_range("Block indices out of range");
    }
    
    // Save the block element by element
    for (int i = 0; i < block.rows(); ++i) {
        for (int j = 0; j < block.cols(); ++j) {
            set(row_start + i, col_start + j, block(i, j));
        }
    }
}

template <typename Scalar>
void OutOfCoreMatrix<Scalar>::multiply(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y) const {
    
    // Check if dimensions are valid
    if (x.size() != cols_) {
        throw std::invalid_argument("Invalid vector dimension");
    }
    
    // Resize the result vector
    y.resize(rows_);
    y.setZero();
    
    // Perform the multiplication block by block
    for (int i = 0; i < rows_; i += block_size_) {
        int row_size = std::min(block_size_, rows_ - i);
        
        for (int j = 0; j < cols_; j += block_size_) {
            int col_size = std::min(block_size_, cols_ - j);
            
            // Load a block of the matrix
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block = load_block(i, j, row_size, col_size);
            
            // Multiply the block with the corresponding part of x
            y.segment(i, row_size) += block * x.segment(j, col_size);
        }
    }
}

template <typename Scalar>
Eigen::SparseMatrix<Scalar> OutOfCoreMatrix<Scalar>::to_sparse() const {
    // Create a sparse matrix
    Eigen::SparseMatrix<Scalar> sparse(rows_, cols_);
    
    // Reserve space for non-zero elements
    // Assuming 1% of the matrix is non-zero
    sparse.reserve(Eigen::VectorXi::Constant(cols_, rows_ * cols_ / 100));
    
    // Fill the sparse matrix
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            Scalar value = get(i, j);
            if (value != Scalar(0)) {
                sparse.insert(i, j) = value;
            }
        }
    }
    
    // Compress the matrix
    sparse.makeCompressed();
    
    return sparse;
}

template <typename Scalar>
std::streampos OutOfCoreMatrix<Scalar>::get_offset(int row, int col) const {
    return static_cast<std::streampos>((row * cols_ + col) * sizeof(Scalar));
}

template <typename Scalar>
void OutOfCoreMatrix<Scalar>::check_indices(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }
}

// DistributedVector implementation

template <typename Scalar>
DistributedVector<Scalar>::DistributedVector(int size, MPI_Comm comm)
    : size_(size), comm_(comm) {
    
#ifdef USE_MPI
    // Get MPI rank and size
    int rank, num_procs;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &num_procs);
    
    // Calculate local size and global start
    local_size_ = size_ / num_procs;
    if (rank < size_ % num_procs) {
        local_size_++;
    }
    
    global_start_ = rank * (size_ / num_procs);
    if (rank < size_ % num_procs) {
        global_start_ += rank;
    } else {
        global_start_ += size_ % num_procs;
    }
    
    // Resize local data
    local_data_.resize(local_size_);
    local_data_.setZero();
#else
    // If MPI is not available, use the entire vector locally
    local_size_ = size_;
    global_start_ = 0;
    
    // Resize local data
    local_data_.resize(local_size_);
    local_data_.setZero();
#endif
}

template <typename Scalar>
DistributedVector<Scalar>::~DistributedVector() {
    // Nothing to clean up
}

template <typename Scalar>
int DistributedVector<Scalar>::size() const {
    return size_;
}

template <typename Scalar>
int DistributedVector<Scalar>::local_size() const {
    return local_size_;
}

template <typename Scalar>
int DistributedVector<Scalar>::global_start() const {
    return global_start_;
}

template <typename Scalar>
void DistributedVector<Scalar>::set(int index, const Scalar& value) {
    // Check if index is valid
    check_index(index);
    
    // Convert global index to local index
    int local_index = global_to_local(index);
    
    // Set the value
    if (local_index >= 0 && local_index < local_size_) {
        local_data_[local_index] = value;
    }
}

template <typename Scalar>
Scalar DistributedVector<Scalar>::get(int index) const {
    // Check if index is valid
    check_index(index);
    
    // Convert global index to local index
    int local_index = global_to_local(index);
    
    // Get the value
    if (local_index >= 0 && local_index < local_size_) {
        return local_data_[local_index];
    } else {
        // If the index is not on this process, return 0
        return Scalar(0);
    }
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& DistributedVector<Scalar>::local_data() const {
    return local_data_;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& DistributedVector<Scalar>::local_data() {
    return local_data_;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DistributedVector<Scalar>::gather() const {
#ifdef USE_MPI
    // Create a vector to store the global data
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> global_data(size_);
    
    // Get MPI rank and size
    int rank, num_procs;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &num_procs);
    
    // Gather the local sizes
    std::vector<int> local_sizes(num_procs);
    MPI_Allgather(&local_size_, 1, MPI_INT, local_sizes.data(), 1, MPI_INT, comm_);
    
    // Gather the global starts
    std::vector<int> global_starts(num_procs);
    MPI_Allgather(&global_start_, 1, MPI_INT, global_starts.data(), 1, MPI_INT, comm_);
    
    // Gather the local data
    for (int i = 0; i < num_procs; ++i) {
        MPI_Bcast(global_data.data() + global_starts[i], local_sizes[i] * sizeof(Scalar), MPI_BYTE, i, comm_);
    }
    
    return global_data;
#else
    // If MPI is not available, return the local data
    return local_data_;
#endif
}

template <typename Scalar>
void DistributedVector<Scalar>::scatter(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& global_vec, int root) {
#ifdef USE_MPI
    // Check if dimensions are valid
    if (global_vec.size() != size_) {
        throw std::invalid_argument("Invalid vector dimension");
    }
    
    // Get MPI rank and size
    int rank, num_procs;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &num_procs);
    
    // Scatter the global data
    if (rank == root) {
        // Send data to each process
        for (int i = 0; i < num_procs; ++i) {
            if (i != root) {
                // Calculate local size and global start for process i
                int local_size_i = size_ / num_procs;
                if (i < size_ % num_procs) {
                    local_size_i++;
                }
                
                int global_start_i = i * (size_ / num_procs);
                if (i < size_ % num_procs) {
                    global_start_i += i;
                } else {
                    global_start_i += size_ % num_procs;
                }
                
                // Send data to process i
                MPI_Send(global_vec.data() + global_start_i, local_size_i * sizeof(Scalar), MPI_BYTE, i, 0, comm_);
            }
        }
        
        // Copy data for the root process
        local_data_ = global_vec.segment(global_start_, local_size_);
    } else {
        // Receive data from the root process
        MPI_Recv(local_data_.data(), local_size_ * sizeof(Scalar), MPI_BYTE, root, 0, comm_, MPI_STATUS_IGNORE);
    }
#else
    // If MPI is not available, copy the global data
    local_data_ = global_vec;
#endif
}

template <typename Scalar>
Scalar DistributedVector<Scalar>::dot(const DistributedVector<Scalar>& other) const {
    // Check if dimensions are valid
    if (size_ != other.size_) {
        throw std::invalid_argument("Invalid vector dimension");
    }
    
    // Compute local dot product
    Scalar local_dot = local_data_.dot(other.local_data_);
    
#ifdef USE_MPI
    // Reduce the local dot products
    Scalar global_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return global_dot;
#else
    // If MPI is not available, return the local dot product
    return local_dot;
#endif
}

template <typename Scalar>
double DistributedVector<Scalar>::norm() const {
    // Compute local norm squared
    double local_norm_squared = local_data_.squaredNorm();
    
#ifdef USE_MPI
    // Reduce the local norms
    double global_norm_squared;
    MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return std::sqrt(global_norm_squared);
#else
    // If MPI is not available, return the local norm
    return std::sqrt(local_norm_squared);
#endif
}

template <typename Scalar>
void DistributedVector<Scalar>::check_index(int index) const {
    if (index < 0 || index >= size_) {
        throw std::out_of_range("Vector index out of range");
    }
}

template <typename Scalar>
int DistributedVector<Scalar>::global_to_local(int global_index) const {
    if (global_index >= global_start_ && global_index < global_start_ + local_size_) {
        return global_index - global_start_;
    } else {
        return -1;
    }
}

template <typename Scalar>
int DistributedVector<Scalar>::local_to_global(int local_index) const {
    if (local_index >= 0 && local_index < local_size_) {
        return global_start_ + local_index;
    } else {
        throw std::out_of_range("Local index out of range");
    }
}

// Explicit template instantiations
template class OutOfCoreMatrix<double>;
template class OutOfCoreMatrix<std::complex<double>>;
template class DistributedVector<double>;
template class DistributedVector<std::complex<double>>;
