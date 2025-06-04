#!/usr/bin/env python3
"""
Simplified setup script for materials module only
Tests Cython compilation for the materials module
"""

import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

# Enable debugging
Options.docstrings = True
Options.embed_pos_in_docstring = True

def main():
    """Build only the materials module"""
    
    print("Building Cython materials module...")
    
    # Include directories
    include_dirs = [
        np.get_include(),
        'qdsim_cython',
    ]
    
    # Define the materials extension
    materials_extension = Extension(
        name="qdsim_cython.core.materials",
        sources=["qdsim_cython/core/materials.pyx"],
        include_dirs=include_dirs,
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
            build_dir="build",
        )
        
        print("✓ Cython compilation successful")
        
        # Setup
        setup(
            name="qdsim_materials",
            ext_modules=cythonized_extensions,
            zip_safe=False,
        )
        
        print("✓ Materials module built successfully")
        return True
        
    except Exception as e:
        print(f"❌ Cython compilation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
