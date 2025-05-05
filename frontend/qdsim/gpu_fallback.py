"""
GPU Fallback Module for QDSim

This module provides a fallback implementation for GPU acceleration when CuPy is not available.
It implements a subset of the CuPy API using NumPy, allowing code that uses CuPy to run
without modification, albeit without the performance benefits of GPU acceleration.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import warnings
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# Flag to indicate if CuPy is available
_has_cupy = False

# Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy.sparse import lil_matrix as cp_lil_matrix
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
    _has_cupy = True
    print("CuPy is available for GPU acceleration")
except ImportError:
    # Create fallback implementations
    print("CuPy not available, using NumPy fallback for 'GPU' operations")
    
    # Create a CuPy-like module using NumPy
    class CuPyFallback:
        """
        A fallback implementation of CuPy using NumPy.
        """
        def __init__(self):
            self.numpy = np
            
            # Copy all NumPy functions to this object
            for name in dir(np):
                if not name.startswith('_'):
                    setattr(self, name, getattr(np, name))
        
        def array(self, obj, dtype=None):
            """Create a NumPy array."""
            return np.array(obj, dtype=dtype)
        
        def asnumpy(self, array):
            """Convert array to NumPy array (no-op in fallback)."""
            return array
        
        def get_array_module(self, array):
            """Get the array module (always NumPy in fallback)."""
            return np
    
    # Create the fallback object
    cp = CuPyFallback()
    
    # Create fallback sparse matrix classes
    cp_lil_matrix = lil_matrix
    cp_csr_matrix = csr_matrix
    cp_eigsh = eigsh

# Define a function to check if GPU acceleration is available
def is_gpu_available():
    """
    Check if GPU acceleration is available.
    
    Returns:
        bool: True if GPU acceleration is available, False otherwise
    """
    return _has_cupy

# Define a function to get the appropriate array module
def get_array_module(array):
    """
    Get the appropriate array module for the given array.
    
    Args:
        array: The array to get the module for
        
    Returns:
        module: The array module (NumPy or CuPy)
    """
    if _has_cupy:
        return cp.get_array_module(array)
    return np

# Define a function to transfer data to the GPU
def to_gpu(array):
    """
    Transfer data to the GPU.
    
    Args:
        array: The array to transfer
        
    Returns:
        array: The array on the GPU (or the original array if GPU is not available)
    """
    if _has_cupy:
        return cp.array(array)
    return array

# Define a function to transfer data to the CPU
def to_cpu(array):
    """
    Transfer data to the CPU.
    
    Args:
        array: The array to transfer
        
    Returns:
        array: The array on the CPU
    """
    if _has_cupy and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array

# Define a function to create a sparse matrix
def create_sparse_matrix(shape, format='lil'):
    """
    Create a sparse matrix.
    
    Args:
        shape: The shape of the matrix
        format: The format of the matrix ('lil' or 'csr')
        
    Returns:
        matrix: The sparse matrix
    """
    if format == 'lil':
        if _has_cupy:
            return cp_lil_matrix(shape)
        return lil_matrix(shape)
    elif format == 'csr':
        if _has_cupy:
            return cp_csr_matrix(shape)
        return csr_matrix(shape)
    else:
        raise ValueError(f"Unsupported sparse matrix format: {format}")

# Define a function to solve the generalized eigenvalue problem
def solve_generalized_eigenvalue_problem(A, B, k, sigma=0.0, which='LM'):
    """
    Solve the generalized eigenvalue problem A·x = λ·B·x.
    
    Args:
        A: The A matrix
        B: The B matrix
        k: The number of eigenvalues to compute
        sigma: The shift value
        which: Which eigenvalues to compute
        
    Returns:
        eigenvalues: The eigenvalues
        eigenvectors: The eigenvectors
    """
    if _has_cupy and isinstance(A, cp_csr_matrix) and isinstance(B, cp_csr_matrix):
        try:
            eigenvalues, eigenvectors = cp_eigsh(A, k=k, M=B, sigma=sigma, which=which)
            return cp.asnumpy(eigenvalues), cp.asnumpy(eigenvectors)
        except Exception as e:
            warnings.warn(f"GPU eigenvalue solver failed: {e}. Falling back to CPU solver.")
            # Convert matrices to CPU
            A_cpu = lil_matrix(A.shape)
            B_cpu = lil_matrix(B.shape)
            
            # Copy data from GPU to CPU
            A_coo = A.tocoo()
            B_coo = B.tocoo()
            
            for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
                A_cpu[i, j] = v
            
            for i, j, v in zip(B_coo.row, B_coo.col, B_coo.data):
                B_cpu[i, j] = v
            
            A_cpu = A_cpu.tocsr()
            B_cpu = B_cpu.tocsr()
            
            # Solve on CPU
            return eigsh(A_cpu, k=k, M=B_cpu, sigma=sigma, which=which)
    
    # Solve on CPU
    return eigsh(A, k=k, M=B, sigma=sigma, which=which)

# Define a class for GPU-accelerated matrix operations
class GPUMatrix:
    """
    A class for GPU-accelerated matrix operations.
    """
    def __init__(self, shape, dtype=np.float64):
        """
        Initialize the GPU matrix.
        
        Args:
            shape: The shape of the matrix
            dtype: The data type of the matrix
        """
        self.shape = shape
        self.dtype = dtype
        
        if _has_cupy:
            self.data = cp.zeros(shape, dtype=dtype)
        else:
            self.data = np.zeros(shape, dtype=dtype)
    
    def to_cpu(self):
        """
        Transfer the matrix to the CPU.
        
        Returns:
            array: The matrix on the CPU
        """
        if _has_cupy and isinstance(self.data, cp.ndarray):
            return cp.asnumpy(self.data)
        return self.data
    
    def to_gpu(self):
        """
        Transfer the matrix to the GPU.
        
        Returns:
            array: The matrix on the GPU
        """
        if _has_cupy and not isinstance(self.data, cp.ndarray):
            self.data = cp.array(self.data)
        return self.data
    
    def __getitem__(self, key):
        """
        Get an item from the matrix.
        
        Args:
            key: The key to get
            
        Returns:
            item: The item
        """
        return self.data[key]
    
    def __setitem__(self, key, value):
        """
        Set an item in the matrix.
        
        Args:
            key: The key to set
            value: The value to set
        """
        self.data[key] = value
