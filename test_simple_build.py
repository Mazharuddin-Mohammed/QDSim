#!/usr/bin/env python3
"""
Simple test to build a minimal Cython extension

This script tests the basic Cython build process with minimal dependencies.
"""

import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

def test_simple_build():
    """Test building a simple Cython extension."""
    
    print("Testing simple Cython build...")
    
    # Create a minimal Cython file for testing
    test_pyx_content = '''
# distutils: language = c++
# cython: language_level = 3

"""
Minimal test Cython module
"""

import numpy as np
cimport numpy as cnp

def test_function():
    """Simple test function."""
    return "Hello from Cython!"

def test_numpy():
    """Test NumPy integration."""
    cdef cnp.ndarray[double, ndim=1] arr = np.array([1.0, 2.0, 3.0])
    return np.sum(arr)
'''
    
    # Write test file
    with open('test_cython_module.pyx', 'w') as f:
        f.write(test_pyx_content)
    
    try:
        # Create extension
        ext = Extension(
            "test_cython_module",
            sources=["test_cython_module.pyx"],
            include_dirs=[np.get_include()],
            language="c++"
        )
        
        # Build extension
        setup(
            name="test_cython",
            ext_modules=cythonize([ext], language_level=3),
            zip_safe=False
        )
        
        print("✓ Simple Cython build successful")
        return True
        
    except Exception as e:
        print(f"✗ Simple Cython build failed: {e}")
        return False
    
    finally:
        # Clean up
        if os.path.exists('test_cython_module.pyx'):
            os.remove('test_cython_module.pyx')

def test_import():
    """Test importing the built module."""
    try:
        import test_cython_module
        
        result1 = test_cython_module.test_function()
        print(f"test_function() returned: {result1}")
        
        result2 = test_cython_module.test_numpy()
        print(f"test_numpy() returned: {result2}")
        
        print("✓ Module import and function calls successful")
        return True
        
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        return False

if __name__ == "__main__":
    print("Running simple Cython build test...")
    
    # Test build
    build_success = test_simple_build()
    
    if build_success:
        # Test import
        import_success = test_import()
        
        if import_success:
            print("\n🎉 Simple Cython test successful!")
            sys.exit(0)
        else:
            print("\n⚠️ Import test failed")
            sys.exit(1)
    else:
        print("\n⚠️ Build test failed")
        sys.exit(1)
