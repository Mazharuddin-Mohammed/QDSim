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
from libcpp.array cimport array
from ..eigen cimport Vector2d, VectorXd, SparseMatrixXd

# Forward declarations
cdef extern from "mesh.h":
    cdef cppclass Mesh:
        # Constructors
        Mesh(double Lx, double Ly, int nx, int ny, int element_order) except +
        
        # Node and element access
        const vector[Vector2d]& getNodes() const
        const vector[array[int, 3]]& getElements() const
        const vector[array[int, 6]]& getQuadraticElements() const
        const vector[array[int, 10]]& getCubicElements() const
        
        # Mesh properties
        int getNumNodes() const
        int getNumElements() const
        int getElementOrder() const
        double get_lx() const
        double get_ly() const
        int get_nx() const
        int get_ny() const
        
        # Mesh refinement
        void refine(const vector[bool]& refine_flags) except +
        bool refine(const vector[int]& element_indices, int max_refinement_level) except +
        
        # I/O operations
        void save(const string& filename) const except +
        
        # Static methods
        @staticmethod
        Mesh load(const string& filename) except +

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
