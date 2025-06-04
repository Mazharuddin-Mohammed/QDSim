# distutils: language = c++
# cython: language_level = 3

"""
Cython-based FEMSolver Implementation

This module provides a complete Cython implementation of the finite element
method solver, replacing the C++ backend implementation with high-performance
Cython code.
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp cimport bool as bint
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from libc.math cimport sqrt, abs as c_abs, pi
import time

# Initialize NumPy
cnp.import_array()

cdef class CythonFEMSolver:
    """
    High-performance Cython implementation of finite element method solver.
    
    Provides general FEM capabilities for assembling matrices and solving
    partial differential equations on triangular meshes.
    """
    
    cdef public int num_nodes, num_elements
    cdef public double Lx, Ly
    cdef public int nx, ny
    cdef cnp.ndarray nodes_x, nodes_y
    cdef cnp.ndarray elements
    cdef object stiffness_matrix
    cdef object mass_matrix
    cdef object load_vector
    cdef bint is_assembled
    cdef double last_assembly_time
    cdef int element_order
    
    def __cinit__(self, mesh, int element_order=1):
        """
        Initialize the FEM solver.
        
        Parameters:
        -----------
        mesh : SimpleMesh
            The mesh object
        element_order : int
            Order of finite elements (1=linear, 2=quadratic)
        """
        # Store mesh information
        self.num_nodes = mesh.num_nodes
        self.num_elements = mesh.num_elements
        self.Lx = mesh.Lx
        self.Ly = mesh.Ly
        self.nx = mesh.nx
        self.ny = mesh.ny
        self.element_order = element_order
        
        # Get mesh data
        self.nodes_x, self.nodes_y = mesh.get_nodes()
        self.elements = mesh.get_elements()
        
        # Initialize matrices
        self.stiffness_matrix = None
        self.mass_matrix = None
        self.load_vector = None
        
        # Assembly flag
        self.is_assembled = False
        self.last_assembly_time = 0.0
    
    def assemble_stiffness_matrix(self, coefficient_func=None):
        """
        Assemble the stiffness matrix for Laplacian operator.
        
        Parameters:
        -----------
        coefficient_func : callable, optional
            Function returning coefficient at (x, y). Default is 1.0.
        
        Returns:
        --------
        scipy.sparse matrix
            The assembled stiffness matrix
        """
        cdef double start_time = time.time()
        
        if coefficient_func is None:
            coefficient_func = lambda x, y: 1.0
        
        cdef int i, j, elem_idx
        cdef int n0, n1, n2
        cdef double x0, y0, x1, y1, x2, y2
        cdef double area, coeff_avg
        cdef cnp.ndarray[double, ndim=2] K_elem = np.zeros((3, 3), dtype=np.float64)
        
        # Initialize sparse matrix builders
        row_indices = []
        col_indices = []
        stiffness_data = []
        
        # Loop over all elements
        for elem_idx in range(self.num_elements):
            # Get element nodes
            n0 = self.elements[elem_idx, 0]
            n1 = self.elements[elem_idx, 1]
            n2 = self.elements[elem_idx, 2]
            
            # Get node coordinates
            x0, y0 = self.nodes_x[n0], self.nodes_y[n0]
            x1, y1 = self.nodes_x[n1], self.nodes_y[n1]
            x2, y2 = self.nodes_x[n2], self.nodes_y[n2]
            
            # Calculate element area
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            
            if area < 1e-15:
                continue  # Skip degenerate elements
            
            # Calculate average coefficient for element
            x_center = (x0 + x1 + x2) / 3.0
            y_center = (y0 + y1 + y2) / 3.0
            coeff_avg = coefficient_func(x_center, y_center)
            
            # Assemble element stiffness matrix
            self._assemble_element_stiffness(K_elem, x0, y0, x1, y1, x2, y2, coeff_avg)
            
            # Add to global matrix
            nodes = [n0, n1, n2]
            for i in range(3):
                for j in range(3):
                    row_indices.append(nodes[i])
                    col_indices.append(nodes[j])
                    stiffness_data.append(K_elem[i, j])
        
        # Create sparse matrix
        self.stiffness_matrix = sp.csr_matrix(
            (stiffness_data, (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        self.last_assembly_time = time.time() - start_time
        return self.stiffness_matrix
    
    def assemble_mass_matrix(self, density_func=None):
        """
        Assemble the mass matrix.
        
        Parameters:
        -----------
        density_func : callable, optional
            Function returning density at (x, y). Default is 1.0.
        
        Returns:
        --------
        scipy.sparse matrix
            The assembled mass matrix
        """
        cdef double start_time = time.time()
        
        if density_func is None:
            density_func = lambda x, y: 1.0
        
        cdef int i, j, elem_idx
        cdef int n0, n1, n2
        cdef double x0, y0, x1, y1, x2, y2
        cdef double area, density_avg
        cdef cnp.ndarray[double, ndim=2] M_elem = np.zeros((3, 3), dtype=np.float64)
        
        # Initialize sparse matrix builders
        row_indices = []
        col_indices = []
        mass_data = []
        
        # Loop over all elements
        for elem_idx in range(self.num_elements):
            # Get element nodes
            n0 = self.elements[elem_idx, 0]
            n1 = self.elements[elem_idx, 1]
            n2 = self.elements[elem_idx, 2]
            
            # Get node coordinates
            x0, y0 = self.nodes_x[n0], self.nodes_y[n0]
            x1, y1 = self.nodes_x[n1], self.nodes_y[n1]
            x2, y2 = self.nodes_x[n2], self.nodes_y[n2]
            
            # Calculate element area
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            
            if area < 1e-15:
                continue  # Skip degenerate elements
            
            # Calculate average density for element
            x_center = (x0 + x1 + x2) / 3.0
            y_center = (y0 + y1 + y2) / 3.0
            density_avg = density_func(x_center, y_center)
            
            # Assemble element mass matrix
            self._assemble_element_mass(M_elem, area, density_avg)
            
            # Add to global matrix
            nodes = [n0, n1, n2]
            for i in range(3):
                for j in range(3):
                    row_indices.append(nodes[i])
                    col_indices.append(nodes[j])
                    mass_data.append(M_elem[i, j])
        
        # Create sparse matrix
        self.mass_matrix = sp.csr_matrix(
            (mass_data, (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        self.last_assembly_time += time.time() - start_time
        return self.mass_matrix
    
    def assemble_load_vector(self, source_func):
        """
        Assemble the load vector.
        
        Parameters:
        -----------
        source_func : callable
            Function returning source term at (x, y)
        
        Returns:
        --------
        numpy.ndarray
            The assembled load vector
        """
        cdef cnp.ndarray[double, ndim=1] load_vec = np.zeros(self.num_nodes, dtype=np.float64)
        cdef int i, elem_idx
        cdef int n0, n1, n2
        cdef double x0, y0, x1, y1, x2, y2
        cdef double area, source_avg
        cdef double contrib
        
        # Loop over all elements
        for elem_idx in range(self.num_elements):
            # Get element nodes
            n0 = self.elements[elem_idx, 0]
            n1 = self.elements[elem_idx, 1]
            n2 = self.elements[elem_idx, 2]
            
            # Get node coordinates
            x0, y0 = self.nodes_x[n0], self.nodes_y[n0]
            x1, y1 = self.nodes_x[n1], self.nodes_y[n1]
            x2, y2 = self.nodes_x[n2], self.nodes_y[n2]
            
            # Calculate element area
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            
            if area < 1e-15:
                continue  # Skip degenerate elements
            
            # Calculate average source for element
            x_center = (x0 + x1 + x2) / 3.0
            y_center = (y0 + y1 + y2) / 3.0
            source_avg = source_func(x_center, y_center)
            
            # Contribution to each node (equal distribution for linear elements)
            contrib = source_avg * area / 3.0
            
            load_vec[n0] += contrib
            load_vec[n1] += contrib
            load_vec[n2] += contrib
        
        self.load_vector = load_vec
        return load_vec
    
    cdef void _assemble_element_stiffness(self, cnp.ndarray[double, ndim=2] K_elem,
                                         double x0, double y0, double x1, double y1,
                                         double x2, double y2, double coefficient):
        """Assemble element stiffness matrix for Laplacian operator"""
        cdef double det_J = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        cdef double inv_det_J = 1.0 / det_J
        
        # Shape function gradients in reference element
        cdef double dN0_dx = (y1 - y2) * inv_det_J
        cdef double dN0_dy = (x2 - x1) * inv_det_J
        cdef double dN1_dx = (y2 - y0) * inv_det_J
        cdef double dN1_dy = (x0 - x2) * inv_det_J
        cdef double dN2_dx = (y0 - y1) * inv_det_J
        cdef double dN2_dy = (x1 - x0) * inv_det_J
        
        cdef double area = 0.5 * abs(det_J)
        cdef double factor = coefficient * area
        
        # Assemble stiffness matrix: K_ij = ∫ c ∇N_i · ∇N_j dΩ
        K_elem[0, 0] = factor * (dN0_dx * dN0_dx + dN0_dy * dN0_dy)
        K_elem[0, 1] = factor * (dN0_dx * dN1_dx + dN0_dy * dN1_dy)
        K_elem[0, 2] = factor * (dN0_dx * dN2_dx + dN0_dy * dN2_dy)
        K_elem[1, 0] = K_elem[0, 1]
        K_elem[1, 1] = factor * (dN1_dx * dN1_dx + dN1_dy * dN1_dy)
        K_elem[1, 2] = factor * (dN1_dx * dN2_dx + dN1_dy * dN2_dy)
        K_elem[2, 0] = K_elem[0, 2]
        K_elem[2, 1] = K_elem[1, 2]
        K_elem[2, 2] = factor * (dN2_dx * dN2_dx + dN2_dy * dN2_dy)
    
    cdef void _assemble_element_mass(self, cnp.ndarray[double, ndim=2] M_elem, 
                                    double area, double density):
        """Assemble element mass matrix"""
        cdef double factor = density * area / 12.0
        
        # Mass matrix for linear triangular elements
        M_elem[0, 0] = 2.0 * factor
        M_elem[0, 1] = factor
        M_elem[0, 2] = factor
        M_elem[1, 0] = factor
        M_elem[1, 1] = 2.0 * factor
        M_elem[1, 2] = factor
        M_elem[2, 0] = factor
        M_elem[2, 1] = factor
        M_elem[2, 2] = 2.0 * factor
    
    def apply_dirichlet_bc(self, boundary_nodes, boundary_values, matrix, rhs=None):
        """
        Apply Dirichlet boundary conditions.
        
        Parameters:
        -----------
        boundary_nodes : list
            List of boundary node indices
        boundary_values : list
            List of boundary values
        matrix : scipy.sparse matrix
            Matrix to modify
        rhs : numpy.ndarray, optional
            Right-hand side vector to modify
        
        Returns:
        --------
        tuple
            (modified_matrix, modified_rhs)
        """
        A_bc = matrix.tolil()
        rhs_bc = rhs.copy() if rhs is not None else np.zeros(self.num_nodes)
        
        for i, val in zip(boundary_nodes, boundary_values):
            # Set row to identity
            A_bc[i, :] = 0
            A_bc[i, i] = 1
            rhs_bc[i] = val
        
        return A_bc.tocsr(), rhs_bc
    
    def solve_linear_system(self, matrix, rhs, method='direct'):
        """
        Solve linear system Ax = b.
        
        Parameters:
        -----------
        matrix : scipy.sparse matrix
            System matrix
        rhs : numpy.ndarray
            Right-hand side vector
        method : str
            Solver method ('direct', 'cg', 'gmres')
        
        Returns:
        --------
        numpy.ndarray
            Solution vector
        """
        if method == 'direct':
            try:
                return spla.spsolve(matrix, rhs)
            except Exception as e:
                print(f"Direct solver failed: {e}, falling back to iterative")
                method = 'cg'
        
        if method == 'cg':
            solution, info = spla.cg(matrix, rhs, tol=1e-8)
            if info != 0:
                print(f"CG solver convergence issue (info={info})")
            return solution
        
        elif method == 'gmres':
            solution, info = spla.gmres(matrix, rhs, tol=1e-8)
            if info != 0:
                print(f"GMRES solver convergence issue (info={info})")
            return solution
        
        else:
            raise ValueError(f"Unknown solver method: {method}")
    
    def get_assembly_time(self):
        """Get the last assembly time"""
        return self.last_assembly_time
    
    def get_matrix_info(self):
        """Get information about assembled matrices"""
        info = {
            'num_nodes': self.num_nodes,
            'num_elements': self.num_elements,
            'element_order': self.element_order,
            'assembly_time': self.last_assembly_time
        }
        
        if self.stiffness_matrix is not None:
            info['stiffness_nnz'] = self.stiffness_matrix.nnz
        if self.mass_matrix is not None:
            info['mass_nnz'] = self.mass_matrix.nnz
        
        return info
    
    def compute_element_quality(self):
        """Compute mesh quality metrics"""
        cdef cnp.ndarray[double, ndim=1] quality = np.zeros(self.num_elements, dtype=np.float64)
        cdef int elem_idx
        cdef int n0, n1, n2
        cdef double x0, y0, x1, y1, x2, y2
        cdef double area, perimeter
        cdef double side1, side2, side3
        
        for elem_idx in range(self.num_elements):
            # Get element nodes
            n0 = self.elements[elem_idx, 0]
            n1 = self.elements[elem_idx, 1]
            n2 = self.elements[elem_idx, 2]
            
            # Get node coordinates
            x0, y0 = self.nodes_x[n0], self.nodes_y[n0]
            x1, y1 = self.nodes_x[n1], self.nodes_y[n1]
            x2, y2 = self.nodes_x[n2], self.nodes_y[n2]
            
            # Calculate area
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            
            # Calculate side lengths
            side1 = sqrt((x1 - x0)**2 + (y1 - y0)**2)
            side2 = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            side3 = sqrt((x0 - x2)**2 + (y0 - y2)**2)
            
            perimeter = side1 + side2 + side3
            
            # Quality metric: 4*sqrt(3)*area / perimeter^2
            # Perfect equilateral triangle has quality = 1
            if perimeter > 1e-15:
                quality[elem_idx] = 4.0 * sqrt(3.0) * area / (perimeter * perimeter)
            else:
                quality[elem_idx] = 0.0
        
        return quality

def create_fem_solver(mesh, element_order=1):
    """
    Create a Cython-based FEM solver.
    
    Parameters:
    -----------
    mesh : SimpleMesh
        The mesh object
    element_order : int
        Order of finite elements
    
    Returns:
    --------
    CythonFEMSolver
        The created solver
    """
    return CythonFEMSolver(mesh, element_order)

def test_fem_solver():
    """Test the Cython FEM solver"""
    try:
        # Import mesh module
        import sys
        sys.path.insert(0, '..')
        from core.mesh_minimal import SimpleMesh
        
        # Create test mesh
        mesh = SimpleMesh(10, 8, 20e-9, 16e-9)
        
        # Create FEM solver
        fem_solver = CythonFEMSolver(mesh)
        
        # Test matrix assembly
        start_time = time.time()
        K = fem_solver.assemble_stiffness_matrix()
        M = fem_solver.assemble_mass_matrix()
        assembly_time = time.time() - start_time
        
        # Test load vector
        def source_func(x, y):
            return 1.0  # Constant source
        
        f = fem_solver.assemble_load_vector(source_func)
        
        # Test boundary conditions
        boundary_nodes = [0, mesh.nx-1]  # Left and right boundaries
        boundary_values = [0.0, 1.0]
        K_bc, f_bc = fem_solver.apply_dirichlet_bc(boundary_nodes, boundary_values, K, f)
        
        # Test solver
        solution = fem_solver.solve_linear_system(K_bc, f_bc)
        
        # Test quality
        quality = fem_solver.compute_element_quality()
        
        print(f"✅ FEM solver test successful")
        print(f"   Matrix size: {K.shape[0]} × {K.shape[1]}")
        print(f"   Stiffness nnz: {K.nnz}")
        print(f"   Mass nnz: {M.nnz}")
        print(f"   Assembly time: {assembly_time:.3f} s")
        print(f"   Solution range: {np.min(solution):.6f} to {np.max(solution):.6f}")
        print(f"   Average element quality: {np.mean(quality):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FEM solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
