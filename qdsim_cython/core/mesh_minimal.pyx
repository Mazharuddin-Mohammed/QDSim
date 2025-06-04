# distutils: language = c++
# cython: language_level = 3

"""
Minimal Mesh Module for QDSim Cython

This module provides a simplified mesh implementation that can be compiled
without complex dependencies.
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector

# Initialize NumPy
cnp.import_array()

cdef class SimpleMesh:
    """
    Simplified mesh class for testing and development.
    
    This provides basic mesh functionality without requiring
    complex C++ backend dependencies.
    """
    
    cdef public int nx, ny
    cdef public double Lx, Ly
    cdef public int num_nodes, num_elements
    cdef vector[double] nodes_x, nodes_y
    cdef vector[int] elements
    
    def __cinit__(self, int nx=10, int ny=10, double Lx=100e-9, double Ly=100e-9):
        """
        Initialize simple mesh.
        
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        Lx, Ly : float
            Domain size in x and y directions (meters)
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.num_nodes = nx * ny
        self.num_elements = (nx - 1) * (ny - 1) * 2  # Triangular elements
        
        self._generate_nodes()
        self._generate_elements()
    
    cdef void _generate_nodes(self):
        """Generate node coordinates"""
        cdef int i, j, node_id
        cdef double dx = self.Lx / (self.nx - 1)
        cdef double dy = self.Ly / (self.ny - 1)
        
        self.nodes_x.clear()
        self.nodes_y.clear()
        
        for j in range(self.ny):
            for i in range(self.nx):
                self.nodes_x.push_back(i * dx)
                self.nodes_y.push_back(j * dy)
    
    cdef void _generate_elements(self):
        """Generate triangular elements"""
        cdef int i, j, n0, n1, n2, n3
        
        self.elements.clear()
        
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                # Node indices for current quad
                n0 = j * self.nx + i
                n1 = j * self.nx + (i + 1)
                n2 = (j + 1) * self.nx + i
                n3 = (j + 1) * self.nx + (i + 1)
                
                # First triangle
                self.elements.push_back(n0)
                self.elements.push_back(n1)
                self.elements.push_back(n2)
                
                # Second triangle
                self.elements.push_back(n1)
                self.elements.push_back(n3)
                self.elements.push_back(n2)
    
    def get_nodes(self):
        """
        Get node coordinates as NumPy arrays.
        
        Returns:
        --------
        tuple
            (x_coords, y_coords) as NumPy arrays
        """
        cdef cnp.ndarray[double, ndim=1] x_coords = np.empty(self.num_nodes, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] y_coords = np.empty(self.num_nodes, dtype=np.float64)
        
        for i in range(self.num_nodes):
            x_coords[i] = self.nodes_x[i]
            y_coords[i] = self.nodes_y[i]
        
        return x_coords, y_coords
    
    def get_elements(self):
        """
        Get element connectivity as NumPy array.
        
        Returns:
        --------
        numpy.ndarray
            Element connectivity array (num_elements x 3)
        """
        cdef int num_triangles = self.num_elements
        cdef cnp.ndarray[int, ndim=2] elem_array = np.empty((num_triangles, 3), dtype=np.int32)
        
        for i in range(num_triangles):
            elem_array[i, 0] = self.elements[3*i]
            elem_array[i, 1] = self.elements[3*i + 1]
            elem_array[i, 2] = self.elements[3*i + 2]
        
        return elem_array
    
    def get_node_at(self, int node_id):
        """
        Get coordinates of specific node.
        
        Parameters:
        -----------
        node_id : int
            Node index
        
        Returns:
        --------
        tuple
            (x, y) coordinates
        """
        if node_id < 0 or node_id >= self.num_nodes:
            raise IndexError(f"Node ID {node_id} out of range [0, {self.num_nodes-1}]")
        
        return (self.nodes_x[node_id], self.nodes_y[node_id])
    
    def find_nearest_node(self, double x, double y):
        """
        Find nearest node to given coordinates.
        
        Parameters:
        -----------
        x, y : float
            Target coordinates
        
        Returns:
        --------
        int
            Index of nearest node
        """
        cdef int nearest_id = 0
        cdef double min_dist = float('inf')
        cdef double dist, dx, dy
        
        for i in range(self.num_nodes):
            dx = self.nodes_x[i] - x
            dy = self.nodes_y[i] - y
            dist = dx*dx + dy*dy
            
            if dist < min_dist:
                min_dist = dist
                nearest_id = i
        
        return nearest_id
    
    def is_inside_domain(self, double x, double y):
        """
        Check if point is inside the mesh domain.
        
        Parameters:
        -----------
        x, y : float
            Point coordinates
        
        Returns:
        --------
        bool
            True if point is inside domain
        """
        return (0 <= x <= self.Lx) and (0 <= y <= self.Ly)
    
    def get_mesh_info(self):
        """
        Get mesh information dictionary.
        
        Returns:
        --------
        dict
            Mesh information
        """
        return {
            'nx': self.nx,
            'ny': self.ny,
            'Lx': self.Lx,
            'Ly': self.Ly,
            'num_nodes': self.num_nodes,
            'num_elements': self.num_elements,
            'dx': self.Lx / (self.nx - 1),
            'dy': self.Ly / (self.ny - 1)
        }

def create_simple_mesh(nx=20, ny=10, Lx=100e-9, Ly=50e-9):
    """
    Create a simple mesh with specified parameters.
    
    Parameters:
    -----------
    nx, ny : int
        Number of grid points
    Lx, Ly : float
        Domain dimensions
    
    Returns:
    --------
    SimpleMesh
        Created mesh object
    """
    return SimpleMesh(nx, ny, Lx, Ly)

def test_mesh_functionality():
    """Test basic mesh functionality"""
    try:
        # Create test mesh
        mesh = SimpleMesh(5, 4, 10e-9, 8e-9)
        
        # Test basic properties
        assert mesh.nx == 5
        assert mesh.ny == 4
        assert mesh.num_nodes == 20
        assert mesh.num_elements == 24  # (5-1)*(4-1)*2
        
        # Test node access
        x_coords, y_coords = mesh.get_nodes()
        assert len(x_coords) == 20
        assert len(y_coords) == 20
        
        # Test element access
        elements = mesh.get_elements()
        assert elements.shape == (24, 3)
        
        # Test specific node
        x, y = mesh.get_node_at(0)
        assert x == 0.0 and y == 0.0
        
        # Test domain check
        assert mesh.is_inside_domain(5e-9, 4e-9)
        assert not mesh.is_inside_domain(-1e-9, 4e-9)
        
        # Test nearest node
        nearest = mesh.find_nearest_node(0.0, 0.0)
        assert nearest == 0
        
        return True
        
    except Exception as e:
        print(f"Mesh test failed: {e}")
        return False
