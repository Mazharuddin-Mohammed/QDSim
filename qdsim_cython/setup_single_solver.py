#!/usr/bin/env python3
"""
Single Solver Build Script

Build one solver at a time to debug compilation issues.
"""

import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def build_single_solver(solver_name):
    """Build a single solver"""
    
    # Basic include directories
    include_dirs = [
        np.get_include(),
        ".",
        "..",
    ]
    
    # Basic libraries
    libraries = []
    library_dirs = []
    
    # Compiler flags
    extra_compile_args = ["-std=c++17", "-O2"]
    extra_link_args = []
    
    # Check for Eigen
    eigen_found = False
    eigen_paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3", 
        "/opt/homebrew/include/eigen3",
        "/usr/include",
        "/usr/local/include"
    ]
    
    for path in eigen_paths:
        if os.path.exists(os.path.join(path, "Eigen", "Dense")):
            include_dirs.append(path)
            eigen_found = True
            print(f"✅ Found Eigen at: {path}")
            break
    
    if not eigen_found:
        print("⚠️  Eigen not found, some features may not work")
    
    # Define solver modules
    solver_files = {
        'poisson': 'solvers/poisson_solver.pyx',
        'schrodinger': 'solvers/schrodinger_solver.pyx',
        'fem': 'solvers/fem_solver.pyx',
        'eigen': 'solvers/eigen_solver.pyx',
        'self_consistent': 'solvers/self_consistent_solver.pyx'
    }
    
    if solver_name not in solver_files:
        print(f"❌ Unknown solver: {solver_name}")
        print(f"Available solvers: {list(solver_files.keys())}")
        return False
    
    pyx_file = solver_files[solver_name]
    module_name = f"qdsim_cython.solvers.{solver_name}_solver"
    
    print(f"🔧 Building {solver_name} solver...")
    print(f"   Module: {module_name}")
    print(f"   Source: {pyx_file}")
    
    # Create extension
    extension = Extension(
        module_name,
        [pyx_file],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    
    # Build
    try:
        setup(
            name=f"qdsim_cython_{solver_name}_solver",
            ext_modules=cythonize([extension], compiler_directives={'language_level': 3}),
            zip_safe=False,
            script_name='setup_single_solver.py',
            script_args=['build_ext', '--inplace']
        )
        print(f"✅ {solver_name} solver built successfully!")
        return True
        
    except Exception as e:
        print(f"❌ {solver_name} solver build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup_single_solver.py <solver_name>")
        print("Available solvers: poisson, schrodinger, fem, eigen, self_consistent")
        sys.exit(1)
    
    solver_name = sys.argv[1]
    success = build_single_solver(solver_name)
    sys.exit(0 if success else 1)
