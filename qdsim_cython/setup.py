#!/usr/bin/env python3
"""
Setup script for QDSim Cython modules

This script builds all Cython extensions for high-performance
quantum device simulation.

Usage:
    python setup.py build_ext --inplace
    python setup.py install
"""

import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

# Enable Cython optimizations
Options.docstrings = False
Options.embed_pos_in_docstring = False

# Compiler and linker flags
extra_compile_args = [
    '-std=c++17',
    '-O3',
    '-march=native',
    '-ffast-math',
    '-fopenmp',
    '-DEIGEN_USE_MKL_ALL',
    '-DEIGEN_VECTORIZE_SSE4_2',
    '-DEIGEN_VECTORIZE_AVX',
    '-DEIGEN_VECTORIZE_AVX2',
    '-DEIGEN_VECTORIZE_FMA',
]

extra_link_args = [
    '-fopenmp',
    '-lmkl_intel_lp64',
    '-lmkl_gnu_thread',
    '-lmkl_core',
    '-lpthread',
    '-lm',
    '-ldl',
]

# CUDA support (optional)
cuda_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    cuda_available = True
    extra_compile_args.extend([
        '-DUSE_CUDA',
        '-DCUDA_ARCH_SM_70',  # Adjust based on target GPU
    ])
    extra_link_args.extend([
        '-lcudart',
        '-lcublas',
        '-lcusolver',
        '-lcufft',
    ])
except ImportError:
    print("CUDA not available, building CPU-only version")

# Include directories
include_dirs = [
    np.get_include(),
    '../backend/include',
    '../backend/external/eigen',
    '/usr/include/eigen3',
    '/opt/intel/mkl/include',
]

if cuda_available:
    include_dirs.extend([
        '/usr/local/cuda/include',
        '/opt/cuda/include',
    ])

# Library directories
library_dirs = [
    '../backend/build',
    '/opt/intel/mkl/lib/intel64',
]

if cuda_available:
    library_dirs.extend([
        '/usr/local/cuda/lib64',
        '/opt/cuda/lib64',
    ])

# Libraries to link
libraries = [
    'qdsim_backend',
    'mkl_intel_lp64',
    'mkl_gnu_thread',
    'mkl_core',
]

if cuda_available:
    libraries.extend([
        'cudart',
        'cublas',
        'cusolver',
        'cufft',
    ])

# Define Cython extensions
extensions = [
    # Core modules
    Extension(
        "qdsim_cython.core.materials",
        ["core/materials.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    Extension(
        "qdsim_cython.core.mesh",
        ["core/mesh.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    Extension(
        "qdsim_cython.core.physics",
        ["core/physics.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    Extension(
        "qdsim_cython.core.interpolator",
        ["core/interpolator.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    # Solver modules
    Extension(
        "qdsim_cython.solvers.poisson",
        ["solvers/poisson.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    Extension(
        "qdsim_cython.solvers.schrodinger",
        ["solvers/schrodinger.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

# Add GPU extensions if CUDA is available
if cuda_available:
    extensions.append(
        Extension(
            "qdsim_cython.gpu.cuda_solver",
            ["gpu/cuda_solver.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++",
        )
    )

# Cython compiler directives
compiler_directives = {
    'language_level': 3,
    'boundscheck': False,
    'wraparound': False,
    'initializedcheck': False,
    'cdivision': True,
    'embedsignature': True,
    'optimize.use_switch': True,
    'optimize.unpack_method_calls': True,
}

# Build configuration
def build_configuration():
    """Print build configuration information"""
    print("QDSim Cython Build Configuration")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"Number of extensions: {len(extensions)}")
    print(f"Compiler flags: {' '.join(extra_compile_args)}")
    print(f"Linker flags: {' '.join(extra_link_args)}")
    print("=" * 50)

if __name__ == "__main__":
    build_configuration()
    
    setup(
        name="qdsim_cython",
        version="1.0.0",
        description="High-performance Cython extensions for QDSim quantum device simulation",
        author="Dr. Mazharuddin Mohammed",
        author_email="mazhar@qdsim.org",
        url="https://github.com/qdsim/qdsim",
        
        packages=[
            "qdsim_cython",
            "qdsim_cython.core",
            "qdsim_cython.solvers",
            "qdsim_cython.gpu",
            "qdsim_cython.analysis",
            "qdsim_cython.visualization",
        ],
        
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=True,  # Generate HTML annotation files
            nthreads=4,     # Parallel compilation
        ),
        
        zip_safe=False,
        
        install_requires=[
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "matplotlib>=3.3.0",
            "cython>=0.29.0",
        ],
        
        extras_require={
            "cuda": ["pynvml", "cupy"],
            "mkl": ["mkl", "mkl-include"],
            "dev": ["pytest", "pytest-cov", "black", "flake8"],
        },
        
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Cython",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
        
        python_requires=">=3.8",
    )
