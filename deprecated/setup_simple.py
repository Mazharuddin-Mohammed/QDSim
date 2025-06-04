#!/usr/bin/env python3
"""
Setup script for simple Cython test
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("test_simple.pyx", language_level=3)
)
