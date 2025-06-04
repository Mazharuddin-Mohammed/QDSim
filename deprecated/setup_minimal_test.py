#!/usr/bin/env python3
"""
Minimal setup script for testing
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("minimal_cython_test.pyx", language_level=3),
    zip_safe=False,
)
