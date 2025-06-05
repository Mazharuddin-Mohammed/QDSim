# distutils: language = c++
# cython: language_level = 3

"""
Simplified Mesh declarations for QDSim Cython modules

This provides minimal mesh type declarations to fix compilation issues.
"""

from libcpp.vector cimport vector
from libcpp cimport bool as bint

# Simple mesh class declaration
cdef extern from "simple_mesh.h":
    cdef cppclass SimpleMesh:
        SimpleMesh() except +
        SimpleMesh(int nx, int ny) except +
        int getNumNodes() const
        int getNumElements() const
        double getNodeX(int i) const
        double getNodeY(int i) const

# Alias for compatibility
ctypedef SimpleMesh Mesh
