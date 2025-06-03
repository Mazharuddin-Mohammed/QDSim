#!/usr/bin/env python3
"""
Setup script for QDSim Cython bindings

This script builds the Cython extensions for the QDSim quantum dot simulation library.
It replaces the previous pybind11-based bindings with a unified Cython approach.

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# Check for required dependencies
def check_dependencies():
    """Check for required system dependencies."""
    required_libs = ['eigen3']
    optional_libs = ['cuda', 'mpi']
    
    found_libs = {}
    
    for lib in required_libs:
        try:
            if lib == 'eigen3':
                # Check for Eigen3
                import subprocess
                result = subprocess.run(['pkg-config', '--exists', 'eigen3'], 
                                      capture_output=True)
                if result.returncode == 0:
                    found_libs[lib] = True
                else:
                    # Try common paths
                    eigen_paths = ['/usr/include/eigen3', '/usr/local/include/eigen3',
                                 '/opt/homebrew/include/eigen3']
                    for path in eigen_paths:
                        if os.path.exists(path):
                            found_libs[lib] = path
                            break
                    else:
                        found_libs[lib] = False
            else:
                found_libs[lib] = True
        except:
            found_libs[lib] = False
    
    return found_libs

def get_cpp_sources():
    """Get list of C++ source files from backend."""
    backend_src = 'backend/src'
    cpp_sources = []
    
    # Core sources (excluding bindings.cpp as we're replacing it)
    core_files = [
        'mesh.cpp', 'fem.cpp', 'physics.cpp', 'solver.cpp',
        'adaptive_mesh.cpp', 'normalization.cpp', 'poisson.cpp',
        'self_consistent.cpp', 'materials.cpp', 'simple_mesh.cpp',
        'simple_interpolator.cpp', 'pn_junction.cpp', 'basic_solver.cpp',
        'improved_self_consistent.cpp', 'fe_interpolator.cpp',
        'simple_self_consistent.cpp', 'schrodinger.cpp',
        'gpu_accelerator.cpp', 'gpu_memory_pool.cpp',
        'full_poisson_dd_solver.cpp', 'callback_system.cpp'
    ]
    
    for file in core_files:
        filepath = os.path.join(backend_src, file)
        if os.path.exists(filepath):
            cpp_sources.append(filepath)
    
    # Advanced features (if available)
    advanced_files = [
        'error_estimator.cpp', 'mesh_quality.cpp', 'adaptive_refinement.cpp',
        'memory_efficient.cpp', 'parallel_eigensolver.cpp', 'spin_orbit.cpp',
        'error_handling.cpp', 'diagnostic_manager.cpp', 'error_visualizer.cpp',
        'cpu_memory_pool.cpp', 'memory_efficient_sparse.cpp', 'paged_matrix.cpp',
        'memory_compression.cpp', 'memory_mapped_file.cpp', 'memory_mapped_matrix.cpp',
        'numa_allocator.cpp', 'arena_allocator.cpp', 'carrier_statistics.cpp',
        'mobility_models.cpp', 'strain_effects.cpp', 'bandgap_models.cpp'
    ]
    
    for file in advanced_files:
        filepath = os.path.join(backend_src, file)
        if os.path.exists(filepath):
            cpp_sources.append(filepath)
    
    return cpp_sources

def get_include_dirs():
    """Get include directories."""
    include_dirs = [
        np.get_include(),
        'backend/include',
        'qdsim_cython'
    ]
    
    # Add Eigen3 include directory
    try:
        import subprocess
        result = subprocess.run(['pkg-config', '--cflags-only-I', 'eigen3'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            eigen_includes = result.stdout.strip().replace('-I', '').split()
            include_dirs.extend(eigen_includes)
        else:
            # Try common paths
            eigen_paths = ['/usr/include/eigen3', '/usr/local/include/eigen3',
                         '/opt/homebrew/include/eigen3']
            for path in eigen_paths:
                if os.path.exists(path):
                    include_dirs.append(path)
                    break
    except:
        # Fallback to common paths
        include_dirs.extend(['/usr/include/eigen3', '/usr/local/include/eigen3'])
    
    return include_dirs

def get_compile_args():
    """Get compiler arguments."""
    compile_args = [
        '-std=c++17',
        '-O3',
        '-DNDEBUG',
        '-ffast-math',
        '-march=native'
    ]
    
    # Add OpenMP support if available
    try:
        import subprocess
        result = subprocess.run(['pkg-config', '--exists', 'openmp'], 
                              capture_output=True)
        if result.returncode == 0:
            compile_args.append('-fopenmp')
    except:
        pass
    
    return compile_args

def get_link_args():
    """Get linker arguments."""
    link_args = []
    
    # Add OpenMP linking if available
    try:
        import subprocess
        result = subprocess.run(['pkg-config', '--exists', 'openmp'], 
                              capture_output=True)
        if result.returncode == 0:
            link_args.append('-fopenmp')
    except:
        pass
    
    return link_args

def create_extensions():
    """Create Cython extension modules."""
    
    # Get common settings
    cpp_sources = get_cpp_sources()
    include_dirs = get_include_dirs()
    compile_args = get_compile_args()
    link_args = get_link_args()
    
    print(f"Found {len(cpp_sources)} C++ source files")
    print(f"Include directories: {include_dirs}")
    
    extensions = []
    
    # Core mesh extension
    mesh_ext = Extension(
        "qdsim_cython.core.mesh",
        sources=["qdsim_cython/core/mesh.pyx"] + cpp_sources,
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++"
    )
    extensions.append(mesh_ext)
    
    # Core physics extension
    physics_ext = Extension(
        "qdsim_cython.core.physics",
        sources=["qdsim_cython/core/physics.pyx"] + cpp_sources,
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++"
    )
    extensions.append(physics_ext)
    
    # Core materials extension
    materials_ext = Extension(
        "qdsim_cython.core.materials",
        sources=["qdsim_cython/core/materials.pyx"] + cpp_sources,
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++"
    )
    extensions.append(materials_ext)
    
    # Poisson solver extension
    poisson_ext = Extension(
        "qdsim_cython.solvers.poisson",
        sources=["qdsim_cython/solvers/poisson.pyx"] + cpp_sources,
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++"
    )
    extensions.append(poisson_ext)
    
    return extensions

def main():
    """Main setup function."""
    
    # Check dependencies
    print("Checking dependencies...")
    deps = check_dependencies()
    
    missing_deps = [lib for lib, found in deps.items() if not found]
    if missing_deps:
        print(f"Warning: Missing dependencies: {missing_deps}")
        print("Some features may not be available.")
    
    # Create extensions
    extensions = create_extensions()
    
    # Cythonize extensions
    cythonized_extensions = cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'embedsignature': True,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False
        },
        annotate=True  # Generate HTML annotation files for debugging
    )
    
    # Setup configuration
    setup(
        name="qdsim-cython",
        version="2.0.0",
        description="High-performance quantum dot simulation library with Cython bindings",
        long_description=open("README.md").read() if os.path.exists("README.md") else "",
        long_description_content_type="text/markdown",
        author="Dr. Mazharuddin Mohammed",
        author_email="mazhar@example.com",
        url="https://github.com/example/qdsim",
        packages=find_packages(include=['qdsim_cython*']),
        ext_modules=cythonized_extensions,
        cmdclass={'build_ext': build_ext},
        install_requires=[
            'numpy>=1.18.0',
            'scipy>=1.5.0',
            'matplotlib>=3.0.0',
            'cython>=0.29.0'
        ],
        extras_require={
            'dev': [
                'pytest>=6.0.0',
                'pytest-cov>=2.10.0',
                'sphinx>=3.0.0',
                'sphinx-rtd-theme>=0.5.0'
            ],
            'gpu': [
                'cupy>=8.0.0'  # For GPU acceleration
            ]
        },
        python_requires='>=3.8',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Cython',
            'Programming Language :: C++',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Mathematics'
        ],
        zip_safe=False,
        include_package_data=True
    )

if __name__ == "__main__":
    main()
