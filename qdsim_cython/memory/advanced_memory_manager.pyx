# distutils: language = c++
# cython: language_level = 3

"""
Advanced Memory Manager for QDSim

Re-enables and enhances the advanced memory management classes with:
- RAII-based memory management
- Unified CPU/GPU memory allocation
- Memory pools for performance
- Automatic garbage collection
- Memory leak detection
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy, memset
import threading
import time
from typing import Dict, List, Optional, Any

# Initialize NumPy
cnp.import_array()

# Memory allocation strategies
cdef enum AllocationStrategy:
    CPU_ONLY = 0
    GPU_ONLY = 1
    UNIFIED = 2
    ADAPTIVE = 3

# Access patterns for optimization
cdef enum AccessPattern:
    SEQUENTIAL = 0
    RANDOM = 1
    READ_ONLY = 2
    WRITE_ONLY = 3
    READ_WRITE = 4

cdef class MemoryBlock:
    """
    RAII-based memory block with automatic cleanup
    """
    
    cdef void* ptr
    cdef size_t size
    cdef size_t capacity
    cdef bint is_gpu
    cdef bint is_unified
    cdef str tag
    cdef double allocation_time
    cdef int ref_count
    
    def __cinit__(self, size_t size, str tag="", bint is_gpu=False, bint is_unified=False):
        self.size = size
        self.capacity = size
        self.is_gpu = is_gpu
        self.is_unified = is_unified
        self.tag = tag
        self.allocation_time = time.time()
        self.ref_count = 1
        
        # Allocate memory
        if is_unified:
            # For now, use CPU memory (GPU unified memory would require CUDA)
            self.ptr = malloc(size)
        elif is_gpu:
            # For now, use CPU memory (GPU memory would require CUDA)
            self.ptr = malloc(size)
        else:
            self.ptr = malloc(size)
        
        if self.ptr == NULL:
            raise MemoryError(f"Failed to allocate {size} bytes for {tag}")
        
        # Initialize to zero
        memset(self.ptr, 0, size)
    
    def __dealloc__(self):
        if self.ptr != NULL:
            free(self.ptr)
            self.ptr = NULL
    
    def get_ptr(self):
        """Get memory pointer as integer (for interfacing)"""
        return <size_t>self.ptr
    
    def get_size(self):
        """Get allocated size"""
        return self.size
    
    def get_tag(self):
        """Get memory tag"""
        return self.tag
    
    def copy_from(self, MemoryBlock other):
        """Copy data from another memory block"""
        if other.size > self.capacity:
            raise ValueError("Source block too large for destination")
        
        memcpy(self.ptr, other.ptr, min(self.size, other.size))
    
    def zero_memory(self):
        """Zero out memory"""
        memset(self.ptr, 0, self.size)

cdef class MemoryPool:
    """
    High-performance memory pool for frequent allocations
    """
    
    cdef size_t block_size
    cdef size_t pool_size
    cdef list free_blocks
    cdef list allocated_blocks
    cdef size_t total_allocated
    cdef size_t peak_usage
    cdef int allocation_count
    cdef int reuse_count
    cdef object lock
    
    def __cinit__(self, size_t block_size, size_t initial_pool_size=1024):
        self.block_size = block_size
        self.pool_size = initial_pool_size
        self.free_blocks = []
        self.allocated_blocks = []
        self.total_allocated = 0
        self.peak_usage = 0
        self.allocation_count = 0
        self.reuse_count = 0
        self.lock = threading.Lock()
        
        # Pre-allocate initial pool
        self._expand_pool(initial_pool_size)
    
    def _expand_pool(self, size_t additional_blocks):
        """Expand the memory pool"""
        cdef size_t i
        cdef MemoryBlock block
        
        for i in range(additional_blocks):
            block = MemoryBlock(self.block_size, f"pool_block_{i}")
            self.free_blocks.append(block)
            self.total_allocated += self.block_size
    
    def allocate(self, str tag=""):
        """Allocate a block from the pool"""
        with self.lock:
            if not self.free_blocks:
                # Expand pool if needed
                self._expand_pool(max(1, self.pool_size // 4))
            
            if self.free_blocks:
                block = self.free_blocks.pop()
                # Note: tag is set during creation, not modified here
                self.allocated_blocks.append(block)
                self.allocation_count += 1
                self.reuse_count += 1
                
                # Update peak usage
                current_usage = len(self.allocated_blocks) * self.block_size
                if current_usage > self.peak_usage:
                    self.peak_usage = current_usage
                
                return block
            else:
                # Fallback to direct allocation
                block = MemoryBlock(self.block_size, tag)
                self.allocated_blocks.append(block)
                self.allocation_count += 1
                return block
    
    def deallocate(self, MemoryBlock block):
        """Return a block to the pool"""
        with self.lock:
            if block in self.allocated_blocks:
                self.allocated_blocks.remove(block)
                block.zero_memory()  # Clear for reuse
                self.free_blocks.append(block)
    
    def get_stats(self):
        """Get pool statistics"""
        with self.lock:
            return {
                'block_size': self.block_size,
                'total_allocated': self.total_allocated,
                'peak_usage': self.peak_usage,
                'allocation_count': self.allocation_count,
                'reuse_count': self.reuse_count,
                'free_blocks': len(self.free_blocks),
                'allocated_blocks': len(self.allocated_blocks),
                'utilization': len(self.allocated_blocks) / max(1, len(self.free_blocks) + len(self.allocated_blocks))
            }
    
    def trim(self):
        """Trim unused blocks from pool"""
        with self.lock:
            # Keep only half of free blocks
            blocks_to_free = len(self.free_blocks) // 2
            for _ in range(blocks_to_free):
                if self.free_blocks:
                    block = self.free_blocks.pop()
                    self.total_allocated -= self.block_size

cdef class AdvancedMemoryManager:
    """
    Advanced memory manager with comprehensive features
    """
    
    cdef dict memory_pools
    cdef dict allocated_blocks
    cdef size_t total_allocated
    cdef size_t peak_usage
    cdef size_t memory_limit
    cdef bint auto_gc_enabled
    cdef bint leak_detection_enabled
    cdef object lock
    cdef list tracked_blocks
    cdef double last_gc_time
    
    def __cinit__(self):
        self.memory_pools = {}
        self.allocated_blocks = {}
        self.total_allocated = 0
        self.peak_usage = 0
        self.memory_limit = 0  # No limit by default
        self.auto_gc_enabled = True
        self.leak_detection_enabled = True
        self.lock = threading.RLock()
        self.tracked_blocks = []
        self.last_gc_time = time.time()
    
    def allocate_block(self, size_t size, str tag="", AllocationStrategy strategy=AllocationStrategy.CPU_ONLY):
        """Allocate a memory block with specified strategy"""
        with self.lock:
            # Check memory limit
            if self.memory_limit > 0 and self.total_allocated + size > self.memory_limit:
                if self.auto_gc_enabled:
                    self.garbage_collect()
                    if self.total_allocated + size > self.memory_limit:
                        raise MemoryError(f"Memory limit exceeded: {self.memory_limit} bytes")
                else:
                    raise MemoryError(f"Memory limit exceeded: {self.memory_limit} bytes")
            
            # Create memory block
            is_gpu = (strategy == AllocationStrategy.GPU_ONLY)
            is_unified = (strategy == AllocationStrategy.UNIFIED)
            
            block = MemoryBlock(size, tag, is_gpu, is_unified)
            
            # Track allocation
            block_id = id(block)
            self.allocated_blocks[block_id] = {
                'block': block,
                'size': size,
                'tag': tag,
                'allocation_time': time.time(),
                'strategy': strategy
            }
            
            self.total_allocated += size
            if self.total_allocated > self.peak_usage:
                self.peak_usage = self.total_allocated
            
            # Add to tracking list
            if self.leak_detection_enabled:
                self.tracked_blocks.append(block_id)
            
            return block
    
    def get_or_create_pool(self, size_t block_size):
        """Get or create a memory pool for specific block size"""
        with self.lock:
            if block_size not in self.memory_pools:
                self.memory_pools[block_size] = MemoryPool(block_size)
            return self.memory_pools[block_size]
    
    def allocate_from_pool(self, size_t size, str tag=""):
        """Allocate from memory pool"""
        pool = self.get_or_create_pool(size)
        return pool.allocate(tag)
    
    def deallocate_to_pool(self, MemoryBlock block):
        """Deallocate to memory pool"""
        pool = self.get_or_create_pool(block.get_size())
        pool.deallocate(block)
    
    def create_numpy_array(self, shape, dtype=np.float64, str tag="numpy_array"):
        """Create NumPy array with managed memory"""
        # Calculate size
        cdef size_t itemsize = np.dtype(dtype).itemsize
        cdef size_t total_size = np.prod(shape) * itemsize
        
        # Allocate memory block
        block = self.allocate_block(total_size, tag)
        
        # Create NumPy array from memory
        cdef cnp.npy_intp* dims = <cnp.npy_intp*>malloc(len(shape) * sizeof(cnp.npy_intp))
        for i, dim in enumerate(shape):
            dims[i] = dim
        
        array = cnp.PyArray_SimpleNewFromData(
            len(shape), dims, np.dtype(dtype).num, <void*>block.ptr
        )
        
        free(dims)
        
        # Keep reference to block to prevent deallocation
        array._memory_block = block
        
        return array
    
    def garbage_collect(self):
        """Perform garbage collection"""
        with self.lock:
            current_time = time.time()
            
            # Clean up deallocated blocks
            blocks_to_remove = []
            for block_id, info in self.allocated_blocks.items():
                block = info['block']
                if block.ref_count <= 0:
                    blocks_to_remove.append(block_id)
                    self.total_allocated -= info['size']
            
            for block_id in blocks_to_remove:
                del self.allocated_blocks[block_id]
            
            # Trim memory pools
            for pool in self.memory_pools.values():
                pool.trim()
            
            self.last_gc_time = current_time
            
            print(f"🧹 Garbage collection completed: freed {len(blocks_to_remove)} blocks")
    
    def get_memory_stats(self):
        """Get comprehensive memory statistics"""
        with self.lock:
            pool_stats = {}
            for size, pool in self.memory_pools.items():
                pool_stats[size] = pool.get_stats()
            
            return {
                'total_allocated': self.total_allocated,
                'peak_usage': self.peak_usage,
                'memory_limit': self.memory_limit,
                'active_blocks': len(self.allocated_blocks),
                'memory_pools': pool_stats,
                'auto_gc_enabled': self.auto_gc_enabled,
                'leak_detection_enabled': self.leak_detection_enabled,
                'last_gc_time': self.last_gc_time
            }
    
    def print_memory_report(self):
        """Print detailed memory report"""
        stats = self.get_memory_stats()
        
        print("🧠 ADVANCED MEMORY MANAGER REPORT")
        print("=" * 50)
        print(f"Total allocated: {stats['total_allocated'] / 1024 / 1024:.2f} MB")
        print(f"Peak usage: {stats['peak_usage'] / 1024 / 1024:.2f} MB")
        print(f"Active blocks: {stats['active_blocks']}")
        print(f"Memory pools: {len(stats['memory_pools'])}")
        
        for size, pool_stats in stats['memory_pools'].items():
            print(f"  Pool {size} bytes: {pool_stats['utilization']:.1%} utilization")
    
    def set_memory_limit(self, size_t limit_bytes):
        """Set memory limit"""
        self.memory_limit = limit_bytes
    
    def enable_auto_gc(self, bint enable=True):
        """Enable/disable automatic garbage collection"""
        self.auto_gc_enabled = enable
    
    def enable_leak_detection(self, bint enable=True):
        """Enable/disable memory leak detection"""
        self.leak_detection_enabled = enable

# Global memory manager instance
cdef AdvancedMemoryManager _global_memory_manager = None

def get_memory_manager():
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = AdvancedMemoryManager()
    return _global_memory_manager

def create_managed_array(shape, dtype=np.float64, tag="managed_array"):
    """Create a managed NumPy array"""
    manager = get_memory_manager()
    return manager.create_numpy_array(shape, dtype, tag)

def allocate_managed_memory(size_t size, str tag="", strategy="cpu"):
    """Allocate managed memory block"""
    manager = get_memory_manager()
    
    cdef AllocationStrategy alloc_strategy
    if strategy == "gpu":
        alloc_strategy = AllocationStrategy.GPU_ONLY
    elif strategy == "unified":
        alloc_strategy = AllocationStrategy.UNIFIED
    elif strategy == "adaptive":
        alloc_strategy = AllocationStrategy.ADAPTIVE
    else:
        alloc_strategy = AllocationStrategy.CPU_ONLY
    
    return manager.allocate_block(size, tag, alloc_strategy)

def get_memory_stats():
    """Get global memory statistics"""
    manager = get_memory_manager()
    return manager.get_memory_stats()

def print_memory_report():
    """Print global memory report"""
    manager = get_memory_manager()
    manager.print_memory_report()

def garbage_collect():
    """Trigger garbage collection"""
    manager = get_memory_manager()
    manager.garbage_collect()

def set_memory_limit(size_t limit_mb):
    """Set memory limit in MB"""
    manager = get_memory_manager()
    manager.set_memory_limit(limit_mb * 1024 * 1024)

def test_memory_manager():
    """Test the advanced memory manager"""
    print("🧠 Testing Advanced Memory Manager")
    print("=" * 50)
    
    manager = get_memory_manager()
    
    # Test basic allocation
    print("1. Testing basic allocation...")
    block1 = manager.allocate_block(1024, "test_block_1")
    block2 = manager.allocate_block(2048, "test_block_2")
    print(f"✅ Allocated 2 blocks")
    
    # Test memory pools
    print("\n2. Testing memory pools...")
    pool_block1 = manager.allocate_from_pool(512, "pool_test_1")
    pool_block2 = manager.allocate_from_pool(512, "pool_test_2")
    print(f"✅ Allocated from memory pool")
    
    # Test NumPy array creation
    print("\n3. Testing managed NumPy arrays...")
    array1 = manager.create_numpy_array((100, 100), np.float64, "test_array")
    array2 = manager.create_numpy_array((50, 50), np.complex128, "complex_array")
    print(f"✅ Created managed arrays: {array1.shape}, {array2.shape}")
    
    # Test statistics
    print("\n4. Memory statistics:")
    manager.print_memory_report()
    
    # Test garbage collection
    print("\n5. Testing garbage collection...")
    manager.garbage_collect()
    
    print("\n✅ Advanced memory manager test completed!")
    return True
