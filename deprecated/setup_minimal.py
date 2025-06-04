#!/usr/bin/env python3
"""
Setup script for minimal materials module
"""

import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize

def main():
    """Build minimal materials module"""
    
    print("Building minimal materials module...")
    
    # Define the minimal materials extension
    materials_extension = Extension(
        name="materials_minimal",
        sources=["qdsim_cython/core/materials_minimal.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O2"],
        extra_link_args=["-std=c++17"],
    )
    
    # Cythonize
    try:
        cythonized_extensions = cythonize(
            [materials_extension],
            compiler_directives={
                'language_level': 3,
                'embedsignature': True,
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
            },
            build_dir="build_minimal",
        )
        
        print("✓ Cython compilation successful")
        
        # Setup
        setup(
            name="materials_minimal",
            ext_modules=cythonized_extensions,
            zip_safe=False,
        )
        
        print("✓ Minimal materials module built successfully")
        return True
        
    except Exception as e:
        print(f"❌ Cython compilation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
