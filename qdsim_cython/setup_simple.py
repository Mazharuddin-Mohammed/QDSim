#!/usr/bin/env python3
"""
Simplified Setup Script for QDSim Cython modules

This script builds Cython extensions with minimal dependencies
for testing and development purposes.
"""

import os
import sys
from setuptools import setup, Extension

# Check for required dependencies
def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import numpy as np
        print(f"✅ NumPy found: {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
        print("❌ NumPy not found")
    
    try:
        from Cython.Build import cythonize
        import Cython
        print(f"✅ Cython found: {Cython.__version__}")
    except ImportError:
        missing_deps.append("cython")
        print("❌ Cython not found")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

if not check_dependencies():
    sys.exit(1)

# Now import after checking
import numpy as np
from Cython.Build import cythonize

# Simplified compiler flags (no MKL, no advanced optimizations)
extra_compile_args = [
    '-std=c++17',
    '-O2',  # Less aggressive optimization
    '-fPIC',
]

extra_link_args = [
    '-lpthread',
    '-lm',
]

# Simplified include directories
include_dirs = [
    np.get_include(),
]

# Check for Eigen headers
eigen_paths = [
    '/usr/include/eigen3',
    '/usr/local/include/eigen3',
    '../backend/external/eigen',
    '/opt/homebrew/include/eigen3',  # macOS with Homebrew
]

eigen_found = False
for eigen_path in eigen_paths:
    if os.path.exists(eigen_path):
        include_dirs.append(eigen_path)
        eigen_found = True
        print(f"✅ Eigen found: {eigen_path}")
        break

if not eigen_found:
    print("⚠️  Eigen not found - some modules may not compile")

# No external libraries for simplified build
library_dirs = []
libraries = []

# Define minimal Cython extensions (only core modules that don't need backend)
extensions = [
    # Core modules that can work standalone
    Extension(
        "qdsim_cython.core.materials_minimal",
        ["core/materials_minimal.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),

    # Add mesh module
    Extension(
        "qdsim_cython.core.mesh_minimal",
        ["core/mesh_minimal.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

# Add analysis module with fixed dependencies
if eigen_found:
    extensions.append(
        Extension(
            "qdsim_cython.analysis.quantum_analysis",
            ["analysis/quantum_analysis.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++",
        )
    )

# Simplified compiler directives
compiler_directives = {
    'language_level': 3,
    'boundscheck': True,  # Enable for debugging
    'wraparound': True,   # Enable for debugging
    'embedsignature': True,
}

def main():
    """Main build function"""
    print("🔧 QDSim Simplified Cython Build")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Extensions to build: {len(extensions)}")
    print("=" * 50)
    
    # List extensions
    for ext in extensions:
        print(f"📦 {ext.name}")
    
    setup(
        name="qdsim_cython_simple",
        version="0.1.0",
        description="Simplified Cython extensions for QDSim testing",
        
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=True,  # Generate HTML annotation files
        ),
        
        zip_safe=False,
        
        install_requires=[
            "numpy>=1.20.0",
            "cython>=0.29.0",
        ],
        
        python_requires=">=3.8",
    )

if __name__ == "__main__":
    main()
