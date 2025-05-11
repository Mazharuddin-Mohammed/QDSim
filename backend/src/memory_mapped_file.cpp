/**
 * @file memory_mapped_file.cpp
 * @brief Implementation of memory-mapped file I/O for QDSim.
 *
 * This file contains implementations of memory-mapped file I/O
 * for efficient access to large files in quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "memory_mapped_matrix.h"
#include <chrono>
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace MemoryMapped {

// Constructor
MemoryMappedFile::MemoryMappedFile(const std::string& filename, MapMode mode, size_t size)
    : filename_(filename), mode_(mode), size_(size), data_(nullptr), is_mapped_(false),
      page_faults_(0), read_operations_(0), write_operations_(0),
      total_read_time_(0), read_count_(0), total_write_time_(0), write_count_(0) {

#ifdef _WIN32
    file_handle_ = INVALID_HANDLE_VALUE;
    mapping_handle_ = NULL;
#else
    file_descriptor_ = -1;
#endif

    // Map the file
    map();
}

// Destructor
MemoryMappedFile::~MemoryMappedFile() {
    unmap();
}

// Get data pointer
void* MemoryMappedFile::getData() const {
    return data_;
}

// Get size
size_t MemoryMappedFile::getSize() const {
    return size_;
}

// Flush changes to disk
bool MemoryMappedFile::flush(bool async) {
    if (!is_mapped_ || data_ == nullptr) {
        return false;
    }

#ifdef _WIN32
    return FlushViewOfFile(data_, size_) != 0;
#else
    int flags = async ? MS_ASYNC : MS_SYNC;
    return msync(data_, size_, flags) == 0;
#endif
}

// Resize the mapped file
bool MemoryMappedFile::resize(size_t new_size) {
    if (mode_ == MapMode::READ_ONLY) {
        return false;
    }

    // Unmap the file
    unmap();

    // Resize the file
#ifdef _WIN32
    file_handle_ = CreateFile(filename_.c_str(), GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, NULL);

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER file_size;
    file_size.QuadPart = new_size;

    if (!SetFilePointerEx(file_handle_, file_size, NULL, FILE_BEGIN) ||
        !SetEndOfFile(file_handle_)) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
#else
    int fd = open(filename_.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);

    if (fd == -1) {
        return false;
    }

    if (ftruncate(fd, new_size) != 0) {
        close(fd);
        return false;
    }

    close(fd);
#endif

    // Update size
    size_ = new_size;

    // Map the file again
    return map();
}

// Check if the file is mapped
bool MemoryMappedFile::isMapped() const {
    return is_mapped_;
}

// Get mapping statistics
MappingStats MemoryMappedFile::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    MappingStats stats;
    stats.file_size = size_;
    stats.mapped_size = size_;

#ifdef _WIN32
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    stats.page_size = system_info.dwPageSize;
#else
    stats.page_size = sysconf(_SC_PAGESIZE);
#endif

    stats.page_faults = page_faults_;
    stats.read_operations = read_operations_;
    stats.write_operations = write_operations_;
    stats.avg_read_time = (read_count_ > 0) ? total_read_time_ / read_count_ : 0.0;
    stats.avg_write_time = (write_count_ > 0) ? total_write_time_ / write_count_ : 0.0;

    return stats;
}

// Read data from the mapped memory
size_t MemoryMappedFile::read(size_t offset, void* buffer, size_t size) {
    if (!is_mapped_ || data_ == nullptr || buffer == nullptr) {
        return 0;
    }

    // Check if the read is within bounds
    if (offset >= size_ || offset + size > size_) {
        return 0;
    }

    // Measure read time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Read data
    std::memcpy(buffer, static_cast<char*>(data_) + offset, size);

    // Update statistics
    read_operations_.fetch_add(1);

    // Calculate read time
    auto end_time = std::chrono::high_resolution_clock::now();
    double read_time = std::chrono::duration<double, std::micro>(end_time - start_time).count();

    // Update atomic variables safely
    double current_read_time = total_read_time_.load();
    while (!total_read_time_.compare_exchange_weak(current_read_time, current_read_time + read_time)) {
        // If the compare_exchange_weak fails, current_read_time is updated with the current value
    }

    read_count_.fetch_add(1);

    return size;
}

// Write data to the mapped memory
size_t MemoryMappedFile::write(size_t offset, const void* buffer, size_t size) {
    if (!is_mapped_ || data_ == nullptr || buffer == nullptr) {
        return 0;
    }

    // Check if the write is within bounds
    if (offset >= size_ || offset + size > size_) {
        return 0;
    }

    // Check if the mapping is read-only
    if (mode_ == MapMode::READ_ONLY) {
        return 0;
    }

    // Measure write time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Write data
    std::memcpy(static_cast<char*>(data_) + offset, buffer, size);

    // Update statistics
    write_operations_.fetch_add(1);

    // Calculate write time
    auto end_time = std::chrono::high_resolution_clock::now();
    double write_time = std::chrono::duration<double, std::micro>(end_time - start_time).count();

    // Update atomic variables safely
    double current_write_time = total_write_time_.load();
    while (!total_write_time_.compare_exchange_weak(current_write_time, current_write_time + write_time)) {
        // If the compare_exchange_weak fails, current_write_time is updated with the current value
    }

    write_count_.fetch_add(1);

    return size;
}

// Prefault pages into memory
bool MemoryMappedFile::prefault(size_t offset, size_t size) {
    if (!is_mapped_ || data_ == nullptr) {
        return false;
    }

    // Check if the prefault is within bounds
    if (offset >= size_ || offset + size > size_) {
        return false;
    }

    // Get page size
#ifdef _WIN32
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    size_t page_size = system_info.dwPageSize;
#else
    size_t page_size = sysconf(_SC_PAGESIZE);
#endif

    // Align offset to page boundary
    size_t aligned_offset = offset & ~(page_size - 1);
    size_t aligned_size = ((offset + size + page_size - 1) & ~(page_size - 1)) - aligned_offset;

    // Touch pages to prefault them
    volatile char* p = static_cast<char*>(data_) + aligned_offset;
    volatile char* end = p + aligned_size;

    while (p < end) {
        char c = *p;
        p += page_size;
    }

    return true;
}

// Map the file into memory
bool MemoryMappedFile::map() {
    // Check if already mapped
    if (is_mapped_) {
        return true;
    }

#ifdef _WIN32
    // Open the file
    DWORD access_flags = 0;
    DWORD map_flags = 0;

    switch (mode_) {
        case MapMode::READ_ONLY:
            access_flags = GENERIC_READ;
            map_flags = PAGE_READONLY;
            break;
        case MapMode::READ_WRITE:
            access_flags = GENERIC_READ | GENERIC_WRITE;
            map_flags = PAGE_READWRITE;
            break;
        case MapMode::COPY_ON_WRITE:
            access_flags = GENERIC_READ | GENERIC_WRITE;
            map_flags = PAGE_WRITECOPY;
            break;
    }

    file_handle_ = CreateFile(filename_.c_str(), access_flags,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, NULL);

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }

    // Get file size
    LARGE_INTEGER file_size;

    if (!GetFileSizeEx(file_handle_, &file_size)) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // If the file is empty and a size was specified, set the file size
    if (file_size.QuadPart == 0 && size_ > 0) {
        LARGE_INTEGER new_size;
        new_size.QuadPart = size_;

        if (!SetFilePointerEx(file_handle_, new_size, NULL, FILE_BEGIN) ||
            !SetEndOfFile(file_handle_)) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }

        file_size.QuadPart = size_;
    } else {
        size_ = file_size.QuadPart;
    }

    // Create file mapping
    mapping_handle_ = CreateFileMapping(file_handle_, NULL, map_flags, 0, 0, NULL);

    if (mapping_handle_ == NULL) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // Map the file
    DWORD view_flags = 0;

    switch (mode_) {
        case MapMode::READ_ONLY:
            view_flags = FILE_MAP_READ;
            break;
        case MapMode::READ_WRITE:
            view_flags = FILE_MAP_ALL_ACCESS;
            break;
        case MapMode::COPY_ON_WRITE:
            view_flags = FILE_MAP_COPY;
            break;
    }

    data_ = MapViewOfFile(mapping_handle_, view_flags, 0, 0, 0);

    if (data_ == nullptr) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = NULL;
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }
#else
    // Open the file
    int flags = 0;
    int prot = 0;

    switch (mode_) {
        case MapMode::READ_ONLY:
            flags = O_RDONLY;
            prot = PROT_READ;
            break;
        case MapMode::READ_WRITE:
            flags = O_RDWR;
            prot = PROT_READ | PROT_WRITE;
            break;
        case MapMode::COPY_ON_WRITE:
            flags = O_RDWR;
            prot = PROT_READ | PROT_WRITE;
            break;
    }

    file_descriptor_ = open(filename_.c_str(), flags | O_CREAT, S_IRUSR | S_IWUSR);

    if (file_descriptor_ == -1) {
        return false;
    }

    // Get file size
    struct stat file_stat;

    if (fstat(file_descriptor_, &file_stat) == -1) {
        close(file_descriptor_);
        file_descriptor_ = -1;
        return false;
    }

    // If the file is empty and a size was specified, set the file size
    if (file_stat.st_size == 0 && size_ > 0) {
        if (ftruncate(file_descriptor_, size_) == -1) {
            close(file_descriptor_);
            file_descriptor_ = -1;
            return false;
        }
    } else {
        size_ = file_stat.st_size;
    }

    // Map the file
    int map_flags = (mode_ == MapMode::COPY_ON_WRITE) ? MAP_PRIVATE : MAP_SHARED;

    data_ = mmap(nullptr, size_, prot, map_flags, file_descriptor_, 0);

    if (data_ == MAP_FAILED) {
        close(file_descriptor_);
        file_descriptor_ = -1;
        data_ = nullptr;
        return false;
    }
#endif

    is_mapped_ = true;
    return true;
}

// Unmap the file from memory
void MemoryMappedFile::unmap() {
    if (!is_mapped_) {
        return;
    }

#ifdef _WIN32
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }

    if (mapping_handle_ != NULL) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = NULL;
    }

    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }

    if (file_descriptor_ != -1) {
        close(file_descriptor_);
        file_descriptor_ = -1;
    }
#endif

    is_mapped_ = false;
}

} // namespace MemoryMapped
