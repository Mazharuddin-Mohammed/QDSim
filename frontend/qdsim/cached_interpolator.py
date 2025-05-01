"""
Cached implementation of finite element interpolation.
"""

import numpy as np
import os
import time
from functools import lru_cache
from collections import OrderedDict

class CachedInterpolator:
    """
    Cached Finite Element Interpolator for scalar fields on a mesh.
    
    This class provides methods to interpolate scalar fields (like potentials)
    at arbitrary points in the mesh with caching for frequently accessed points.
    """
    
    def __init__(self, interpolator, cache_size=1024):
        """
        Initialize the cached interpolator.
        
        Args:
            interpolator: The underlying interpolator to use
            cache_size: Maximum number of points to cache
        """
        self.interpolator = interpolator
        self.cache_size = cache_size
        
        # Create caches for interpolation results
        self.value_cache = LRUCache(cache_size)
        self.gradient_cache = LRUCache(cache_size)
        
        # Cache statistics
        self.value_hits = 0
        self.value_misses = 0
        self.gradient_hits = 0
        self.gradient_misses = 0
    
    def interpolate(self, x, y, field):
        """
        Interpolate a field at a point with caching.
        
        Args:
            x, y: Coordinates of the point
            field: Field values at mesh nodes
            
        Returns:
            Interpolated value at (x, y)
        """
        # Create a cache key
        key = self._make_key(x, y, field)
        
        # Check if the result is in the cache
        if key in self.value_cache:
            self.value_hits += 1
            return self.value_cache[key]
        
        # Compute the result
        self.value_misses += 1
        value = self.interpolator.interpolate(x, y, field)
        
        # Cache the result
        self.value_cache[key] = value
        
        return value
    
    def interpolate_with_gradient(self, x, y, field):
        """
        Interpolate a field and its gradient at a point with caching.
        
        Args:
            x, y: Coordinates of the point
            field: Field values at mesh nodes
            
        Returns:
            Tuple of (value, grad_x, grad_y)
        """
        # Create a cache key
        key = self._make_key(x, y, field)
        
        # Check if the result is in the cache
        if key in self.gradient_cache:
            self.gradient_hits += 1
            return self.gradient_cache[key]
        
        # Compute the result
        self.gradient_misses += 1
        result = self.interpolator.interpolate_with_gradient(x, y, field)
        
        # Cache the result
        self.gradient_cache[key] = result
        
        return result
    
    def interpolate_grid(self, x_min, x_max, y_min, y_max, nx, ny, field):
        """
        Interpolate a field on a regular grid.
        
        This method doesn't use caching since it's more efficient to compute
        the entire grid at once.
        
        Args:
            x_min, x_max: Range of x coordinates
            y_min, y_max: Range of y coordinates
            nx, ny: Number of grid points in x and y directions
            field: Field values at mesh nodes
            
        Returns:
            2D array of interpolated values with shape (ny, nx)
        """
        return self.interpolator.interpolate_grid(x_min, x_max, y_min, y_max, nx, ny, field)
    
    def interpolate_grid_with_gradient(self, x_min, x_max, y_min, y_max, nx, ny, field):
        """
        Interpolate a field and its gradient on a regular grid.
        
        This method doesn't use caching since it's more efficient to compute
        the entire grid at once.
        
        Args:
            x_min, x_max: Range of x coordinates
            y_min, y_max: Range of y coordinates
            nx, ny: Number of grid points in x and y directions
            field: Field values at mesh nodes
            
        Returns:
            Tuple of (values, grad_x, grad_y) where each is a 2D array with shape (ny, nx)
        """
        return self.interpolator.interpolate_grid_with_gradient(x_min, x_max, y_min, y_max, nx, ny, field)
    
    def get_cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        value_total = self.value_hits + self.value_misses
        gradient_total = self.gradient_hits + self.gradient_misses
        
        value_hit_rate = self.value_hits / value_total if value_total > 0 else 0
        gradient_hit_rate = self.gradient_hits / gradient_total if gradient_total > 0 else 0
        
        return {
            'value_hits': self.value_hits,
            'value_misses': self.value_misses,
            'value_hit_rate': value_hit_rate,
            'gradient_hits': self.gradient_hits,
            'gradient_misses': self.gradient_misses,
            'gradient_hit_rate': gradient_hit_rate,
            'value_cache_size': len(self.value_cache),
            'gradient_cache_size': len(self.gradient_cache),
            'max_cache_size': self.cache_size
        }
    
    def clear_caches(self):
        """Clear all caches."""
        self.value_cache.clear()
        self.gradient_cache.clear()
        self.value_hits = 0
        self.value_misses = 0
        self.gradient_hits = 0
        self.gradient_misses = 0
    
    def _make_key(self, x, y, field):
        """
        Create a cache key for a point and field.
        
        Args:
            x, y: Coordinates of the point
            field: Field values at mesh nodes
            
        Returns:
            Cache key
        """
        # We can't use the field array directly as a key because it's not hashable
        # Instead, we use its id and a hash of its first few values
        field_id = id(field)
        field_hash = hash(tuple(field[:min(10, len(field))]))
        
        # Round the coordinates to a reasonable precision to increase cache hits
        x_rounded = round(x, 10)
        y_rounded = round(y, 10)
        
        return (x_rounded, y_rounded, field_id, field_hash)


class LRUCache:
    """
    A simple LRU (Least Recently Used) cache.
    
    This class provides a dictionary-like interface with a maximum size.
    When the cache is full, the least recently used item is removed.
    """
    
    def __init__(self, capacity):
        """
        Initialize the cache.
        
        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def __getitem__(self, key):
        """Get an item from the cache and mark it as recently used."""
        if key not in self.cache:
            raise KeyError(key)
        
        # Move the item to the end of the OrderedDict (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        
        return value
    
    def __setitem__(self, key, value):
        """Add an item to the cache."""
        # If the key already exists, remove it first
        if key in self.cache:
            self.cache.pop(key)
        
        # If the cache is full, remove the least recently used item
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        
        # Add the new item
        self.cache[key] = value
    
    def __contains__(self, key):
        """Check if an item is in the cache."""
        return key in self.cache
    
    def __len__(self):
        """Get the number of items in the cache."""
        return len(self.cache)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
