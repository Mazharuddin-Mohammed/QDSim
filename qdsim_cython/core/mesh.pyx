# distutils: language = c++
# cython: language_level = 3

"""
Cython wrapper for Mesh and related classes

This module provides Python access to the C++ Mesh class and related
functionality for finite element mesh generation and manipulation.
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.array cimport array

# Import C++ declarations
from .mesh cimport Mesh as CppMesh
from .mesh cimport AdaptiveMesh as CppAdaptiveMesh
from .mesh cimport SimpleMesh as CppSimpleMesh
from ..eigen cimport Vector2d, VectorXd

# Initialize NumPy
cnp.import_array()

cdef class Mesh:
    """
    Python wrapper for the C++ Mesh class.
    
    Represents a 2D triangular mesh for finite element simulations.
    Supports linear (P1), quadratic (P2), and cubic (P3) elements.
    
    Parameters
    ----------
    Lx : float
        Width of the domain in nanometers
    Ly : float  
        Height of the domain in nanometers
    nx : int
        Number of elements in x-direction
    ny : int
        Number of elements in y-direction
    element_order : int, optional
        Element order (1=P1, 2=P2, 3=P3), default=1
    """
    
    cdef CppMesh* _mesh
    cdef bool _owns_mesh
    
    def __cinit__(self, double Lx, double Ly, int nx, int ny, int element_order=1):
        """Initialize the mesh with given parameters."""
        self._mesh = new CppMesh(Lx, Ly, nx, ny, element_order)
        self._owns_mesh = True
        
    def __dealloc__(self):
        """Clean up the C++ mesh object."""
        if self._owns_mesh and self._mesh is not NULL:
            del self._mesh
            
    @staticmethod
    cdef Mesh from_cpp_mesh(CppMesh* cpp_mesh, bool owns_mesh=False):
        """Create a Python Mesh from a C++ Mesh pointer."""
        cdef Mesh mesh = Mesh.__new__(Mesh)
        mesh._mesh = cpp_mesh
        mesh._owns_mesh = owns_mesh
        return mesh
        
    @property
    def nodes(self):
        """Get mesh nodes as a NumPy array."""
        cdef const vector[Vector2d]& cpp_nodes = self._mesh.getNodes()
        cdef int num_nodes = cpp_nodes.size()
        
        # Create NumPy array
        cdef cnp.ndarray[double, ndim=2] nodes_array = np.empty((num_nodes, 2), dtype=np.float64)
        
        # Copy data from C++ to NumPy
        cdef int i
        for i in range(num_nodes):
            nodes_array[i, 0] = cpp_nodes[i].x()
            nodes_array[i, 1] = cpp_nodes[i].y()
            
        return nodes_array
        
    @property
    def elements(self):
        """Get mesh elements as a NumPy array."""
        cdef const vector[array[int, 3]]& cpp_elements = self._mesh.getElements()
        cdef int num_elements = cpp_elements.size()
        
        # Create NumPy array
        cdef cnp.ndarray[int, ndim=2] elements_array = np.empty((num_elements, 3), dtype=np.int32)
        
        # Copy data from C++ to NumPy
        cdef int i, j
        for i in range(num_elements):
            for j in range(3):
                elements_array[i, j] = cpp_elements[i][j]
                
        return elements_array
        
    @property
    def quadratic_elements(self):
        """Get quadratic mesh elements as a NumPy array (P2 elements only)."""
        if self.element_order < 2:
            return None
            
        cdef const vector[array[int, 6]]& cpp_elements = self._mesh.getQuadraticElements()
        cdef int num_elements = cpp_elements.size()
        
        # Create NumPy array
        cdef cnp.ndarray[int, ndim=2] elements_array = np.empty((num_elements, 6), dtype=np.int32)
        
        # Copy data from C++ to NumPy
        cdef int i, j
        for i in range(num_elements):
            for j in range(6):
                elements_array[i, j] = cpp_elements[i][j]
                
        return elements_array
        
    @property
    def cubic_elements(self):
        """Get cubic mesh elements as a NumPy array (P3 elements only)."""
        if self.element_order < 3:
            return None
            
        cdef const vector[array[int, 10]]& cpp_elements = self._mesh.getCubicElements()
        cdef int num_elements = cpp_elements.size()
        
        # Create NumPy array
        cdef cnp.ndarray[int, ndim=2] elements_array = np.empty((num_elements, 10), dtype=np.int32)
        
        # Copy data from C++ to NumPy
        cdef int i, j
        for i in range(num_elements):
            for j in range(10):
                elements_array[i, j] = cpp_elements[i][j]
                
        return elements_array
        
    @property
    def num_nodes(self):
        """Get the number of nodes in the mesh."""
        return self._mesh.getNumNodes()
        
    @property
    def num_elements(self):
        """Get the number of elements in the mesh."""
        return self._mesh.getNumElements()
        
    @property
    def element_order(self):
        """Get the element order (1=P1, 2=P2, 3=P3)."""
        return self._mesh.getElementOrder()
        
    @property
    def Lx(self):
        """Get the domain width in nanometers."""
        return self._mesh.get_lx()
        
    @property
    def Ly(self):
        """Get the domain height in nanometers."""
        return self._mesh.get_ly()
        
    @property
    def nx(self):
        """Get the number of elements in x-direction."""
        return self._mesh.get_nx()
        
    @property
    def ny(self):
        """Get the number of elements in y-direction."""
        return self._mesh.get_ny()
        
    def refine(self, refine_flags):
        """
        Refine the mesh based on refinement flags.
        
        Parameters
        ----------
        refine_flags : array-like of bool
            Boolean flags indicating which elements to refine
        """
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] flags_array = np.asarray(refine_flags, dtype=np.uint8)
        cdef vector[bool] cpp_flags
        
        # Convert NumPy array to C++ vector
        cdef int i
        for i in range(flags_array.shape[0]):
            cpp_flags.push_back(flags_array[i] != 0)
            
        # Call C++ refine method
        self._mesh.refine(cpp_flags)
        
    def refine_elements(self, element_indices, int max_refinement_level=3):
        """
        Refine specific elements by index.
        
        Parameters
        ----------
        element_indices : array-like of int
            Indices of elements to refine
        max_refinement_level : int, optional
            Maximum refinement level, default=3
            
        Returns
        -------
        bool
            True if refinement was successful
        """
        cdef cnp.ndarray[int, ndim=1] indices_array = np.asarray(element_indices, dtype=np.int32)
        cdef vector[int] cpp_indices
        
        # Convert NumPy array to C++ vector
        cdef int i
        for i in range(indices_array.shape[0]):
            cpp_indices.push_back(indices_array[i])
            
        # Call C++ refine method
        return self._mesh.refine(cpp_indices, max_refinement_level)
        
    def save(self, filename):
        """
        Save the mesh to a file.
        
        Parameters
        ----------
        filename : str
            Name of the file to save the mesh to
        """
        cdef string cpp_filename = filename.encode('utf-8')
        self._mesh.save(cpp_filename)
        
    @staticmethod
    def load(filename):
        """
        Load a mesh from a file.
        
        Parameters
        ----------
        filename : str
            Name of the file to load the mesh from
            
        Returns
        -------
        Mesh
            Loaded mesh object
        """
        cdef string cpp_filename = filename.encode('utf-8')
        cdef CppMesh cpp_mesh = CppMesh.load(cpp_filename)
        
        # Create a new Python mesh object
        # Note: This creates a copy of the C++ mesh
        cdef Mesh mesh = Mesh.__new__(Mesh)
        mesh._mesh = new CppMesh(cpp_mesh)
        mesh._owns_mesh = True
        return mesh

# Utility functions for adaptive mesh refinement
def compute_refinement_flags(Mesh mesh, solution, double threshold):
    """
    Compute refinement flags based on solution gradients.
    
    Parameters
    ----------
    mesh : Mesh
        The mesh to compute refinement flags for
    solution : array-like
        Solution values at mesh nodes
    threshold : float
        Refinement threshold
        
    Returns
    -------
    numpy.ndarray
        Boolean array indicating which elements to refine
    """
    cdef cnp.ndarray[double, ndim=1] sol_array = np.asarray(solution, dtype=np.float64)
    cdef vector[double] cpp_solution
    
    # Convert NumPy array to C++ vector
    cdef int i
    for i in range(sol_array.shape[0]):
        cpp_solution.push_back(sol_array[i])
        
    # Call C++ function
    cdef vector[bool] cpp_flags = CppAdaptiveMesh.computeRefinementFlags(
        mesh._mesh[0], cpp_solution, threshold)
        
    # Convert result back to NumPy
    cdef int num_flags = cpp_flags.size()
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] flags_array = np.empty(num_flags, dtype=np.uint8)
    
    for i in range(num_flags):
        flags_array[i] = 1 if cpp_flags[i] else 0
        
    return flags_array.astype(bool)
