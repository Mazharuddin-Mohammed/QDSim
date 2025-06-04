# distutils: language = c++
# cython: language_level = 3

"""
Cython declaration file for Mesh and related classes

This file declares the C++ classes and functions from the backend
that will be wrapped by Cython.
"""

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# Simplified mesh struct for Cython compilation
cdef struct CppMesh:
    # Basic properties
    double Lx                    # Domain length in x
    double Ly                    # Domain length in y
    int nx                       # Number of elements in x
    int ny                       # Number of elements in y
    int element_order            # Element order (1, 2, 3)
    int num_nodes               # Total number of nodes
    int num_elements            # Total number of elements

cdef extern from "adaptive_mesh.h":
    cdef cppclass AdaptiveMesh:
        # Static methods for adaptive refinement
        @staticmethod
        void refineMesh(Mesh& mesh, const vector[bool]& refine_flags) except +
        
        @staticmethod
        vector[bool] computeRefinementFlags(const Mesh& mesh, 
                                          const vector[double]& solution,
                                          double threshold) except +

cdef extern from "simple_mesh.h":
    cdef cppclass SimpleMesh:
        # Constructors
        SimpleMesh(const vector[Vector2d]& nodes, 
                  const vector[array[int, 3]]& elements) except +
        
        # Access methods
        const vector[Vector2d]& getNodes() const
        const vector[array[int, 3]]& getElements() const

# Note: Eigen declarations are in ../eigen.pxd
