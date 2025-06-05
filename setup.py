#!/usr/bin/env python3
"""
QDSim Setup Configuration

Advanced Quantum Dot Simulator with Cython backend, GPU acceleration,
and comprehensive open quantum system support.
"""

from setuptools import setup, find_packages, Extension
import os

# Try to import Cython, but make it optional for documentation builds
try:
    from Cython.Build import cythonize
    import numpy
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    cythonize = None
    numpy = None

# Read version from __init__.py
def get_version():
    """Extract version from package __init__.py"""
    version_file = os.path.join('qdsim_cython', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return '2.0.0'

# Read long description from README
def get_long_description():
    """Read long description from README.md"""
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return "Advanced Quantum Dot Simulator"

# Define Cython extensions (only if Cython is available)
extensions = []
if CYTHON_AVAILABLE:
    extensions = [
        Extension(
            "qdsim_cython.solvers.fixed_open_system_solver",
            ["qdsim_cython/solvers/fixed_open_system_solver.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math", "-march=native"],
            extra_link_args=["-O3"],
            language="c++"
        ),
        Extension(
            "qdsim_cython.memory.advanced_memory_manager",
            ["qdsim_cython/memory/advanced_memory_manager.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
            language="c++"
        ),
        Extension(
            "qdsim_cython.materials",
            ["qdsim_cython/materials.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3"],
            language="c++"
        )
    ]

setup(
    name="qdsim",
    version=get_version(),
    author="Dr. Mazharuddin Mohammed",
    author_email="mazharuddin.mohammed.official@gmail.com",
    description="Advanced Quantum Dot Simulator with Open System Support",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/QDSim",
    project_urls={
        "Documentation": "https://qdsimx.readthedocs.io",
        "Source": "https://github.com/your-username/QDSim",
        "Tracker": "https://github.com/your-username/QDSim/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Cython",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "cython>=0.29.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-copybutton>=0.5.0",
            "sphinx-tabs>=3.4.0",
            "myst-parser>=0.18.0",
            "nbsphinx>=0.8.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "sphinx-gallery>=0.11.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
            "pycuda>=2021.1",
        ],
        "all": [
            # Include all optional dependencies
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "pre-commit>=2.15.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-copybutton>=0.5.0",
            "sphinx-tabs>=3.4.0",
            "myst-parser>=0.18.0",
            "nbsphinx>=0.8.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "sphinx-gallery>=0.11.0",
            "cupy>=9.0.0",
            "pycuda>=2021.1",
        ],
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
        }
    ) if CYTHON_AVAILABLE and extensions else [],
    include_package_data=True,
    package_data={
        "qdsim_cython": ["*.pyx", "*.pxd", "*.h"],
    },
    entry_points={
        "console_scripts": [
            "qdsim-validate=qdsim_cython.validation:main",
            "qdsim-benchmark=qdsim_cython.benchmark:main",
        ],
    },
    keywords=[
        "quantum mechanics",
        "quantum dots",
        "semiconductor physics",
        "finite element method",
        "eigenvalue problems",
        "open quantum systems",
        "GPU acceleration",
        "scientific computing",
    ],
    zip_safe=False,  # Required for Cython extensions
)
