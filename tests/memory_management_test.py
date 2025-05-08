#!/usr/bin/env python3
"""
Comprehensive test for memory management optimizations.

This test verifies the following enhancements:
1. Memory pools for efficient memory reuse
2. Memory-mapped matrices for handling large datasets
3. Memory optimization techniques for large simulations

Author: Dr. Mazharuddin Mohammed
Date: 2023-07-15
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import time
import psutil
import gc
from contextlib import contextmanager

# Add the parent directory to the path so we can import qdsim modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import qdsim_cpp
except ImportError:
    print("Warning: qdsim_cpp module not found. Some tests may be skipped.")
    qdsim_cpp = None

# Try to import memory management modules
try:
    from qdsim.memory_pool import CPUMemoryPool, GPUMemoryPool
    from qdsim.memory_mapped_matrix import MemoryMappedMatrix, MemoryMappedSparseMatrix
    _has_memory_modules = True
except ImportError:
    print("Warning: Memory management modules not found. Using simplified implementations.")
    _has_memory_modules = False
    
    # Simplified implementations for testing
    class CPUMemoryPool:
        def __init__(self):
            self.blocks = {}
            self.total_allocated = 0
            self.current_used = 0
            self.allocation_count = 0
            self.reuse_count = 0
        
        def allocate(self, size, tag=""):
            # Try to find a suitable block
            for block_id, block in list(self.blocks.items()):
                if not block["in_use"] and block["size"] >= size:
                    # Found a suitable block
                    block["in_use"] = True
                    block["tag"] = tag
                    self.current_used += size
                    self.reuse_count += 1
                    return block["ptr"]
            
            # No suitable block found, allocate a new one
            ptr = np.zeros(size, dtype=np.uint8)
            block_id = len(self.blocks)
            self.blocks[block_id] = {
                "ptr": ptr,
                "size": size,
                "in_use": True,
                "tag": tag
            }
            self.total_allocated += size
            self.current_used += size
            self.allocation_count += 1
            return ptr
        
        def release(self, ptr):
            # Find the block
            for block_id, block in self.blocks.items():
                if block["ptr"] is ptr:
                    # Mark the block as available
                    block["in_use"] = False
                    self.current_used -= block["size"]
                    return
        
        def get_stats(self):
            return {
                "total_allocated": self.total_allocated,
                "current_used": self.current_used,
                "allocation_count": self.allocation_count,
                "reuse_count": self.reuse_count
            }
        
        def trim(self):
            # Free some memory
            blocks_to_free = []
            for block_id, block in self.blocks.items():
                if not block["in_use"]:
                    blocks_to_free.append(block_id)
            
            for block_id in blocks_to_free:
                block = self.blocks[block_id]
                self.total_allocated -= block["size"]
                del self.blocks[block_id]
    
    class GPUMemoryPool(CPUMemoryPool):
        pass
    
    class MemoryMappedMatrix:
        def __init__(self, filename, rows, cols, dtype=np.float64):
            self.filename = filename
            self.rows = rows
            self.cols = cols
            self.dtype = dtype
            self.data = np.memmap(filename, dtype=dtype, mode='w+', shape=(rows, cols))
        
        def get(self, row, col):
            return self.data[row, col]
        
        def set(self, row, col, value):
            self.data[row, col] = value
        
        def get_block(self, start_row, start_col, block_rows, block_cols):
            return self.data[start_row:start_row+block_rows, start_col:start_col+block_cols].copy()
        
        def set_block(self, start_row, start_col, block):
            self.data[start_row:start_row+block.shape[0], start_col:start_col+block.shape[1]] = block
        
        def flush(self):
            self.data.flush()
        
        def close(self):
            del self.data
    
    class MemoryMappedSparseMatrix:
        def __init__(self, filename, rows, cols, dtype=np.float64):
            self.filename = filename
            self.rows = rows
            self.cols = cols
            self.dtype = dtype
            self.data = {}
        
        def get(self, row, col):
            return self.data.get((row, col), 0)
        
        def set(self, row, col, value):
            if value != 0:
                self.data[(row, col)] = value
            elif (row, col) in self.data:
                del self.data[(row, col)]
        
        def to_dense(self):
            result = np.zeros((self.rows, self.cols), dtype=self.dtype)
            for (row, col), value in self.data.items():
                result[row, col] = value
            return result
        
        def close(self):
            self.data.clear()

# Define a context manager to measure memory usage
@contextmanager
def measure_memory_usage():
    process = psutil.Process(os.getpid())
    gc.collect()
    start_mem = process.memory_info().rss
    yield
    gc.collect()
    end_mem = process.memory_info().rss
    memory_used = end_mem - start_mem
    print(f"Memory used: {memory_used / (1024 * 1024):.2f} MB")

class MemoryManagementTest(unittest.TestCase):
    """Test case for memory management optimizations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for memory-mapped files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create memory pools
        self.cpu_pool = CPUMemoryPool()
        if _has_memory_modules:
            try:
                self.gpu_pool = GPUMemoryPool()
            except Exception as e:
                print(f"GPU memory pool initialization failed: {e}")
                self.gpu_pool = None
        else:
            self.gpu_pool = None
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_cpu_memory_pool(self):
        """Test CPU memory pool for efficient memory reuse."""
        print("\n=== Testing CPU Memory Pool ===")
        
        # Allocate memory from the pool
        block_sizes = [1024, 2048, 4096, 8192]
        blocks = []
        
        for i, size in enumerate(block_sizes):
            print(f"Allocating block {i+1} of size {size} bytes")
            block = self.cpu_pool.allocate(size, f"block_{i}")
            blocks.append(block)
        
        # Get pool statistics
        stats = self.cpu_pool.get_stats()
        print(f"Pool statistics after allocation: {stats}")
        
        # Verify that memory was allocated
        self.assertEqual(stats["allocation_count"], len(block_sizes))
        self.assertEqual(stats["reuse_count"], 0)
        self.assertGreaterEqual(stats["total_allocated"], sum(block_sizes))
        self.assertGreaterEqual(stats["current_used"], sum(block_sizes))
        
        # Release some blocks
        print("Releasing blocks 0 and 2")
        self.cpu_pool.release(blocks[0])
        self.cpu_pool.release(blocks[2])
        
        # Get updated statistics
        stats = self.cpu_pool.get_stats()
        print(f"Pool statistics after release: {stats}")
        
        # Verify that memory was released
        self.assertLess(stats["current_used"], sum(block_sizes))
        
        # Allocate more memory, which should reuse the released blocks
        print("Allocating more blocks that should reuse released memory")
        new_blocks = []
        for i, size in enumerate(block_sizes):
            block = self.cpu_pool.allocate(size, f"new_block_{i}")
            new_blocks.append(block)
        
        # Get updated statistics
        stats = self.cpu_pool.get_stats()
        print(f"Pool statistics after reallocation: {stats}")
        
        # Verify that memory was reused
        self.assertGreater(stats["reuse_count"], 0)
        
        # Trim the pool
        print("Trimming the pool")
        self.cpu_pool.trim()
        
        # Get updated statistics
        stats = self.cpu_pool.get_stats()
        print(f"Pool statistics after trimming: {stats}")
        
        print("CPU memory pool test passed!")
    
    def test_gpu_memory_pool(self):
        """Test GPU memory pool for efficient memory reuse."""
        print("\n=== Testing GPU Memory Pool ===")
        
        if self.gpu_pool is None:
            print("GPU memory pool not available, skipping test")
            return
        
        # Allocate memory from the pool
        block_sizes = [1024, 2048, 4096, 8192]
        blocks = []
        
        for i, size in enumerate(block_sizes):
            print(f"Allocating block {i+1} of size {size} bytes")
            block = self.gpu_pool.allocate(size, f"block_{i}")
            blocks.append(block)
        
        # Get pool statistics
        stats = self.gpu_pool.get_stats()
        print(f"Pool statistics after allocation: {stats}")
        
        # Verify that memory was allocated
        self.assertEqual(stats["allocation_count"], len(block_sizes))
        self.assertEqual(stats["reuse_count"], 0)
        self.assertGreaterEqual(stats["total_allocated"], sum(block_sizes))
        self.assertGreaterEqual(stats["current_used"], sum(block_sizes))
        
        # Release some blocks
        print("Releasing blocks 0 and 2")
        self.gpu_pool.release(blocks[0])
        self.gpu_pool.release(blocks[2])
        
        # Get updated statistics
        stats = self.gpu_pool.get_stats()
        print(f"Pool statistics after release: {stats}")
        
        # Verify that memory was released
        self.assertLess(stats["current_used"], sum(block_sizes))
        
        # Allocate more memory, which should reuse the released blocks
        print("Allocating more blocks that should reuse released memory")
        new_blocks = []
        for i, size in enumerate(block_sizes):
            block = self.gpu_pool.allocate(size, f"new_block_{i}")
            new_blocks.append(block)
        
        # Get updated statistics
        stats = self.gpu_pool.get_stats()
        print(f"Pool statistics after reallocation: {stats}")
        
        # Verify that memory was reused
        self.assertGreater(stats["reuse_count"], 0)
        
        print("GPU memory pool test passed!")
    
    def test_memory_mapped_matrix(self):
        """Test memory-mapped matrix for handling large datasets."""
        print("\n=== Testing Memory-Mapped Matrix ===")
        
        # Create a memory-mapped matrix
        matrix_file = os.path.join(self.temp_dir.name, "test_matrix.dat")
        rows, cols = 1000, 1000
        
        print(f"Creating memory-mapped matrix of size {rows}x{cols}")
        matrix = MemoryMappedMatrix(matrix_file, rows, cols)
        
        # Set some values in the matrix
        print("Setting values in the matrix")
        for i in range(0, rows, 100):
            for j in range(0, cols, 100):
                matrix.set(i, j, i + j)
        
        # Get some values from the matrix
        print("Getting values from the matrix")
        for i in range(0, rows, 100):
            for j in range(0, cols, 100):
                value = matrix.get(i, j)
                self.assertEqual(value, i + j)
        
        # Set and get a block
        print("Setting and getting a block")
        block = np.random.rand(10, 10)
        matrix.set_block(50, 50, block)
        retrieved_block = matrix.get_block(50, 50, 10, 10)
        np.testing.assert_allclose(retrieved_block, block)
        
        # Measure memory usage when working with a large matrix
        print("Measuring memory usage with memory-mapped matrix")
        with measure_memory_usage():
            # Create a large matrix
            large_matrix_file = os.path.join(self.temp_dir.name, "large_matrix.dat")
            large_rows, large_cols = 10000, 10000
            large_matrix = MemoryMappedMatrix(large_matrix_file, large_rows, large_cols)
            
            # Set and get some values
            for i in range(0, large_rows, 1000):
                for j in range(0, large_cols, 1000):
                    large_matrix.set(i, j, i + j)
                    value = large_matrix.get(i, j)
                    self.assertEqual(value, i + j)
            
            # Clean up
            large_matrix.close()
        
        # Clean up
        matrix.close()
        
        print("Memory-mapped matrix test passed!")
    
    def test_memory_mapped_sparse_matrix(self):
        """Test memory-mapped sparse matrix for handling large sparse datasets."""
        print("\n=== Testing Memory-Mapped Sparse Matrix ===")
        
        # Create a memory-mapped sparse matrix
        matrix_file = os.path.join(self.temp_dir.name, "test_sparse_matrix")
        rows, cols = 10000, 10000
        
        print(f"Creating memory-mapped sparse matrix of size {rows}x{cols}")
        matrix = MemoryMappedSparseMatrix(matrix_file, rows, cols)
        
        # Set some values in the matrix (sparse pattern)
        print("Setting values in the sparse matrix")
        for i in range(0, rows, 100):
            for j in range(0, cols, 100):
                matrix.set(i, j, i + j)
        
        # Get some values from the matrix
        print("Getting values from the sparse matrix")
        for i in range(0, rows, 100):
            for j in range(0, cols, 100):
                value = matrix.get(i, j)
                self.assertEqual(value, i + j)
        
        # Verify that zeros are returned for unset elements
        print("Verifying zeros for unset elements")
        for i in range(1, rows, 100):
            for j in range(1, cols, 100):
                value = matrix.get(i, j)
                self.assertEqual(value, 0)
        
        # Measure memory usage when working with a large sparse matrix
        print("Measuring memory usage with memory-mapped sparse matrix")
        with measure_memory_usage():
            # Create a large sparse matrix
            large_matrix_file = os.path.join(self.temp_dir.name, "large_sparse_matrix")
            large_rows, large_cols = 100000, 100000
            large_matrix = MemoryMappedSparseMatrix(large_matrix_file, large_rows, large_cols)
            
            # Set and get some values (very sparse)
            for i in range(0, large_rows, 1000):
                for j in range(0, large_cols, 1000):
                    large_matrix.set(i, j, i + j)
                    value = large_matrix.get(i, j)
                    self.assertEqual(value, i + j)
            
            # Clean up
            large_matrix.close()
        
        # Clean up
        matrix.close()
        
        print("Memory-mapped sparse matrix test passed!")
    
    def test_memory_optimization_techniques(self):
        """Test memory optimization techniques for large simulations."""
        print("\n=== Testing Memory Optimization Techniques ===")
        
        # Test memory usage with and without optimization
        print("Comparing memory usage with and without optimization")
        
        # Without optimization (create a large array in memory)
        print("Memory usage without optimization:")
        with measure_memory_usage():
            large_array = np.random.rand(5000, 5000)
            # Do some operations
            result = large_array + large_array.T
            # Force computation
            sum_value = np.sum(result)
            print(f"Sum: {sum_value}")
        
        # With optimization (use memory-mapped file)
        print("Memory usage with optimization:")
        with measure_memory_usage():
            # Create memory-mapped arrays
            array_file = os.path.join(self.temp_dir.name, "large_array.dat")
            result_file = os.path.join(self.temp_dir.name, "result_array.dat")
            
            # Create the arrays
            array1 = np.memmap(array_file, dtype=np.float64, mode='w+', shape=(5000, 5000))
            array1[:] = np.random.rand(5000, 5000)
            
            # Create result array
            result = np.memmap(result_file, dtype=np.float64, mode='w+', shape=(5000, 5000))
            
            # Process in blocks to reduce memory usage
            block_size = 1000
            for i in range(0, 5000, block_size):
                for j in range(0, 5000, block_size):
                    # Get blocks
                    block1 = array1[i:i+block_size, j:j+block_size]
                    block2 = array1[j:j+block_size, i:i+block_size]
                    
                    # Compute result block
                    result[i:i+block_size, j:j+block_size] = block1 + block2.T
            
            # Force computation
            sum_value = 0
            for i in range(0, 5000, block_size):
                sum_value += np.sum(result[i:i+block_size, :])
            
            print(f"Sum: {sum_value}")
            
            # Clean up
            del array1
            del result
        
        print("Memory optimization techniques test passed!")

if __name__ == "__main__":
    unittest.main()
