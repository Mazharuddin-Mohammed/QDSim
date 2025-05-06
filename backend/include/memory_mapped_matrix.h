#pragma once
/**
 * @file memory_mapped_matrix.h
 * @brief Defines memory-mapped matrix classes for QDSim.
 *
 * This file contains declarations of memory-mapped matrix classes
 * for handling extremely large matrices in quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace MemoryMapped {

/**
 * @enum MapMode
 * @brief Enumeration of memory mapping modes.
 */
enum class MapMode {
    READ_ONLY,   ///< Read-only mapping
    READ_WRITE,  ///< Read-write mapping
    COPY_ON_WRITE ///< Copy-on-write mapping
};

/**
 * @struct MappingStats
 * @brief Structure to hold memory mapping statistics.
 */
struct MappingStats {
    size_t file_size;           ///< File size in bytes
    size_t mapped_size;         ///< Mapped size in bytes
    size_t page_size;           ///< Page size in bytes
    size_t page_faults;         ///< Number of page faults
    size_t read_operations;     ///< Number of read operations
    size_t write_operations;    ///< Number of write operations
    double avg_read_time;       ///< Average read time in microseconds
    double avg_write_time;      ///< Average write time in microseconds
};

/**
 * @class MemoryMappedFile
 * @brief Class for memory-mapped file I/O.
 *
 * The MemoryMappedFile class provides memory-mapped file I/O
 * for efficient access to large files.
 */
class MemoryMappedFile {
public:
    /**
     * @brief Constructs a new MemoryMappedFile object.
     *
     * @param filename The file name
     * @param mode The mapping mode
     * @param size The file size (for new files)
     */
    MemoryMappedFile(const std::string& filename, MapMode mode = MapMode::READ_ONLY, size_t size = 0);
    
    /**
     * @brief Destructor for the MemoryMappedFile object.
     */
    ~MemoryMappedFile();
    
    /**
     * @brief Gets a pointer to the mapped memory.
     *
     * @return Pointer to the mapped memory
     */
    void* getData() const;
    
    /**
     * @brief Gets the size of the mapped memory.
     *
     * @return The size in bytes
     */
    size_t getSize() const;
    
    /**
     * @brief Flushes changes to disk.
     *
     * @param async Whether to flush asynchronously
     * @return True if successful, false otherwise
     */
    bool flush(bool async = false);
    
    /**
     * @brief Resizes the mapped file.
     *
     * @param new_size The new size in bytes
     * @return True if successful, false otherwise
     */
    bool resize(size_t new_size);
    
    /**
     * @brief Checks if the file is mapped.
     *
     * @return True if mapped, false otherwise
     */
    bool isMapped() const;
    
    /**
     * @brief Gets the mapping statistics.
     *
     * @return The mapping statistics
     */
    MappingStats getStats() const;
    
    /**
     * @brief Reads data from the mapped memory.
     *
     * @param offset The offset in bytes
     * @param buffer The buffer to read into
     * @param size The number of bytes to read
     * @return The number of bytes read
     */
    size_t read(size_t offset, void* buffer, size_t size);
    
    /**
     * @brief Writes data to the mapped memory.
     *
     * @param offset The offset in bytes
     * @param buffer The buffer to write from
     * @param size The number of bytes to write
     * @return The number of bytes written
     */
    size_t write(size_t offset, const void* buffer, size_t size);
    
    /**
     * @brief Prefaults pages into memory.
     *
     * @param offset The offset in bytes
     * @param size The number of bytes to prefault
     * @return True if successful, false otherwise
     */
    bool prefault(size_t offset, size_t size);
    
private:
    std::string filename_;      ///< File name
    MapMode mode_;              ///< Mapping mode
    size_t size_;               ///< File size
    void* data_;                ///< Mapped memory
    bool is_mapped_;            ///< Whether the file is mapped
    
#ifdef _WIN32
    HANDLE file_handle_;        ///< File handle
    HANDLE mapping_handle_;     ///< Mapping handle
#else
    int file_descriptor_;       ///< File descriptor
#endif
    
    // Statistics
    mutable std::mutex stats_mutex_;        ///< Mutex for statistics
    std::atomic<size_t> page_faults_;       ///< Number of page faults
    std::atomic<size_t> read_operations_;   ///< Number of read operations
    std::atomic<size_t> write_operations_;  ///< Number of write operations
    std::atomic<double> total_read_time_;   ///< Total read time in microseconds
    std::atomic<size_t> read_count_;        ///< Number of reads
    std::atomic<double> total_write_time_;  ///< Total write time in microseconds
    std::atomic<size_t> write_count_;       ///< Number of writes
    
    /**
     * @brief Maps the file into memory.
     *
     * @return True if successful, false otherwise
     */
    bool map();
    
    /**
     * @brief Unmaps the file from memory.
     */
    void unmap();
};

/**
 * @class MemoryMappedMatrix
 * @brief Class for memory-mapped matrices.
 *
 * The MemoryMappedMatrix class provides memory-mapped storage
 * for extremely large dense matrices.
 *
 * @tparam Scalar The scalar type of the matrix elements
 */
template <typename Scalar>
class MemoryMappedMatrix {
public:
    /**
     * @brief Constructs a new MemoryMappedMatrix object.
     *
     * @param filename The file name
     * @param rows The number of rows
     * @param cols The number of columns
     * @param mode The mapping mode
     */
    MemoryMappedMatrix(const std::string& filename, int rows, int cols,
                      MapMode mode = MapMode::READ_WRITE);
    
    /**
     * @brief Destructor for the MemoryMappedMatrix object.
     */
    ~MemoryMappedMatrix();
    
    /**
     * @brief Gets the number of rows.
     *
     * @return The number of rows
     */
    int rows() const;
    
    /**
     * @brief Gets the number of columns.
     *
     * @return The number of columns
     */
    int cols() const;
    
    /**
     * @brief Gets a value from the matrix.
     *
     * @param row The row index
     * @param col The column index
     * @return The value at the specified position
     */
    Scalar get(int row, int col) const;
    
    /**
     * @brief Sets a value in the matrix.
     *
     * @param row The row index
     * @param col The column index
     * @param value The value to set
     */
    void set(int row, int col, const Scalar& value);
    
    /**
     * @brief Gets a block from the matrix.
     *
     * @param start_row The starting row index
     * @param start_col The starting column index
     * @param block_rows The number of rows in the block
     * @param block_cols The number of columns in the block
     * @return The block as an Eigen matrix
     */
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> getBlock(
        int start_row, int start_col, int block_rows, int block_cols) const;
    
    /**
     * @brief Sets a block in the matrix.
     *
     * @param start_row The starting row index
     * @param start_col The starting column index
     * @param block The block to set
     */
    void setBlock(int start_row, int start_col,
                 const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& block);
    
    /**
     * @brief Multiplies the matrix by a vector.
     *
     * @param x The vector to multiply
     * @param y The result vector
     */
    void multiply(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
                 Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y) const;
    
    /**
     * @brief Flushes changes to disk.
     *
     * @param async Whether to flush asynchronously
     * @return True if successful, false otherwise
     */
    bool flush(bool async = false);
    
    /**
     * @brief Prefaults pages into memory.
     *
     * @param start_row The starting row index
     * @param start_col The starting column index
     * @param block_rows The number of rows to prefault
     * @param block_cols The number of columns to prefault
     * @return True if successful, false otherwise
     */
    bool prefault(int start_row, int start_col, int block_rows, int block_cols);
    
    /**
     * @brief Gets the mapping statistics.
     *
     * @return The mapping statistics
     */
    MappingStats getStats() const;
    
private:
    int rows_;                          ///< Number of rows
    int cols_;                          ///< Number of columns
    std::unique_ptr<MemoryMappedFile> file_; ///< Memory-mapped file
    
    /**
     * @brief Gets the offset for a matrix element.
     *
     * @param row The row index
     * @param col The column index
     * @return The offset in bytes
     */
    size_t getOffset(int row, int col) const;
    
    /**
     * @brief Checks if indices are valid.
     *
     * @param row The row index
     * @param col The column index
     * @return True if valid, false otherwise
     */
    bool isValidIndex(int row, int col) const;
};

/**
 * @class MemoryMappedSparseMatrix
 * @brief Class for memory-mapped sparse matrices.
 *
 * The MemoryMappedSparseMatrix class provides memory-mapped storage
 * for extremely large sparse matrices.
 *
 * @tparam Scalar The scalar type of the matrix elements
 */
template <typename Scalar>
class MemoryMappedSparseMatrix {
public:
    /**
     * @brief Constructs a new MemoryMappedSparseMatrix object.
     *
     * @param filename The file name
     * @param rows The number of rows
     * @param cols The number of columns
     * @param estimated_non_zeros The estimated number of non-zero elements
     * @param mode The mapping mode
     */
    MemoryMappedSparseMatrix(const std::string& filename, int rows, int cols,
                            int estimated_non_zeros = 0,
                            MapMode mode = MapMode::READ_WRITE);
    
    /**
     * @brief Destructor for the MemoryMappedSparseMatrix object.
     */
    ~MemoryMappedSparseMatrix();
    
    /**
     * @brief Gets the number of rows.
     *
     * @return The number of rows
     */
    int rows() const;
    
    /**
     * @brief Gets the number of columns.
     *
     * @return The number of columns
     */
    int cols() const;
    
    /**
     * @brief Gets the number of non-zero elements.
     *
     * @return The number of non-zero elements
     */
    int nonZeros() const;
    
    /**
     * @brief Gets a value from the matrix.
     *
     * @param row The row index
     * @param col The column index
     * @return The value at the specified position
     */
    Scalar get(int row, int col) const;
    
    /**
     * @brief Sets a value in the matrix.
     *
     * @param row The row index
     * @param col The column index
     * @param value The value to set
     */
    void set(int row, int col, const Scalar& value);
    
    /**
     * @brief Converts to an Eigen sparse matrix.
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
     * @brief Flushes changes to disk.
     *
     * @param async Whether to flush asynchronously
     * @return True if successful, false otherwise
     */
    bool flush(bool async = false);
    
    /**
     * @brief Gets the mapping statistics.
     *
     * @return The mapping statistics
     */
    MappingStats getStats() const;
    
private:
    int rows_;                          ///< Number of rows
    int cols_;                          ///< Number of columns
    int non_zeros_;                     ///< Number of non-zero elements
    std::unique_ptr<MemoryMappedFile> row_ptr_file_; ///< Row pointers file
    std::unique_ptr<MemoryMappedFile> col_ind_file_; ///< Column indices file
    std::unique_ptr<MemoryMappedFile> values_file_;  ///< Values file
    
    /**
     * @brief Updates the number of non-zero elements.
     */
    void updateNonZeros();
};

// Type aliases
using MemoryMappedMatrixd = MemoryMappedMatrix<double>;
using MemoryMappedMatrixcd = MemoryMappedMatrix<std::complex<double>>;
using MemoryMappedSparseMatrixd = MemoryMappedSparseMatrix<double>;
using MemoryMappedSparseMatrixcd = MemoryMappedSparseMatrix<std::complex<double>>;

} // namespace MemoryMapped
