# distutils: language = c++
# cython: language_level = 3

"""
Cython-based PoissonSolver Implementation

This module provides a complete Cython implementation of the Poisson solver
for electrostatic calculations, replacing the C++ backend implementation
with high-performance Cython code.
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp cimport bool as bint
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from libc.math cimport sqrt, abs as c_abs
import time

# Import Eigen types
from ..eigen cimport VectorXd, MatrixXd, SparseMatrixXd

# Initialize NumPy
cnp.import_array()

# Physical constants
cdef double EPSILON_0 = 8.854187817e-12  # F/m - vacuum permittivity
cdef double Q_E = 1.602176634e-19        # C - elementary charge

cdef class CythonPoissonSolver:
    """
    High-performance Cython implementation of Poisson equation solver.
    
    Solves the Poisson equation: ∇·(ε∇φ) = -ρ
    where φ is the electrostatic potential, ε is permittivity, and ρ is charge density.
    """
    
    cdef public int num_nodes, num_elements
    cdef public double Lx, Ly
    cdef public int nx, ny
    cdef cnp.ndarray nodes_x, nodes_y
    cdef cnp.ndarray elements
    cdef cnp.ndarray potential
    cdef cnp.ndarray electric_field_x, electric_field_y
    cdef public object stiffness_matrix
    cdef public object mass_matrix
    cdef object epsilon_r_func
    cdef object rho_func
    cdef bint is_assembled
    cdef double last_solve_time
    
    def __cinit__(self, mesh, epsilon_r_func, rho_func):
        """
        Initialize the Poisson solver.
        
        Parameters:
        -----------
        mesh : SimpleMesh
            The mesh object
        epsilon_r_func : callable
            Function returning relative permittivity at (x, y)
        rho_func : callable
            Function returning charge density at (x, y, n, p)
        """
        # Store mesh information
        self.num_nodes = mesh.num_nodes
        self.num_elements = mesh.num_elements
        self.Lx = mesh.Lx
        self.Ly = mesh.Ly
        self.nx = mesh.nx
        self.ny = mesh.ny
        
        # Get mesh data
        self.nodes_x, self.nodes_y = mesh.get_nodes()
        self.elements = mesh.get_elements()
        
        # Store material functions
        self.epsilon_r_func = epsilon_r_func
        self.rho_func = rho_func
        
        # Initialize solution arrays
        self.potential = np.zeros(self.num_nodes, dtype=np.float64)
        self.electric_field_x = np.zeros(self.num_nodes, dtype=np.float64)
        self.electric_field_y = np.zeros(self.num_nodes, dtype=np.float64)
        
        # Assembly flag
        self.is_assembled = False
        self.last_solve_time = 0.0
        
        # Assemble matrices
        self._assemble_matrices()
    
    cdef void _assemble_matrices(self):
        """Assemble the stiffness and mass matrices using finite elements"""
        cdef int i, j, k, elem_idx
        cdef int n0, n1, n2
        cdef double x0, y0, x1, y1, x2, y2
        cdef double area, det_J
        cdef double epsilon_avg
        cdef cnp.ndarray[double, ndim=2] K_elem = np.zeros((3, 3), dtype=np.float64)
        cdef cnp.ndarray[double, ndim=2] M_elem = np.zeros((3, 3), dtype=np.float64)
        
        # Initialize sparse matrix builders
        row_indices = []
        col_indices = []
        stiffness_data = []
        mass_data = []
        
        # Loop over all elements
        cdef int valid_elements = 0
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

            valid_elements += 1
            
            # Calculate average permittivity for element
            x_center = (x0 + x1 + x2) / 3.0
            y_center = (y0 + y1 + y2) / 3.0
            epsilon_avg = self.epsilon_r_func(x_center, y_center) * EPSILON_0
            
            # Assemble element stiffness matrix
            self._assemble_element_stiffness(K_elem, x0, y0, x1, y1, x2, y2, epsilon_avg)
            
            # Assemble element mass matrix
            self._assemble_element_mass(M_elem, area)
            
            # Add to global matrix
            nodes = [n0, n1, n2]
            for i in range(3):
                for j in range(3):
                    row_indices.append(nodes[i])
                    col_indices.append(nodes[j])
                    stiffness_data.append(K_elem[i, j])
                    mass_data.append(M_elem[i, j])
        
        # Create sparse matrices
        self.stiffness_matrix = sp.csr_matrix(
            (stiffness_data, (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        self.mass_matrix = sp.csr_matrix(
            (mass_data, (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        self.is_assembled = True

        # Debug output
        print(f"Matrix assembly: {valid_elements} valid elements processed")
        print(f"Stiffness matrix: {self.stiffness_matrix.nnz} non-zeros")
        print(f"Mass matrix: {self.mass_matrix.nnz} non-zeros")
    
    cdef void _assemble_element_stiffness(self, cnp.ndarray[double, ndim=2] K_elem,
                                         double x0, double y0, double x1, double y1,
                                         double x2, double y2, double epsilon):
        """Assemble element stiffness matrix for Poisson equation"""
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
        cdef double factor = epsilon * area
        
        # Assemble stiffness matrix: K_ij = ∫ ε ∇N_i · ∇N_j dΩ
        K_elem[0, 0] = factor * (dN0_dx * dN0_dx + dN0_dy * dN0_dy)
        K_elem[0, 1] = factor * (dN0_dx * dN1_dx + dN0_dy * dN1_dy)
        K_elem[0, 2] = factor * (dN0_dx * dN2_dx + dN0_dy * dN2_dy)
        K_elem[1, 0] = K_elem[0, 1]
        K_elem[1, 1] = factor * (dN1_dx * dN1_dx + dN1_dy * dN1_dy)
        K_elem[1, 2] = factor * (dN1_dx * dN2_dx + dN1_dy * dN2_dy)
        K_elem[2, 0] = K_elem[0, 2]
        K_elem[2, 1] = K_elem[1, 2]
        K_elem[2, 2] = factor * (dN2_dx * dN2_dx + dN2_dy * dN2_dy)
    
    cdef void _assemble_element_mass(self, cnp.ndarray[double, ndim=2] M_elem, double area):
        """Assemble element mass matrix"""
        cdef double factor = area / 12.0
        
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
    
    def solve(self, double V_p, double V_n, n=None, p=None):
        """
        Solve the Poisson equation.
        
        Parameters:
        -----------
        V_p : float
            Potential at p-type boundary (V)
        V_n : float
            Potential at n-type boundary (V)
        n : array_like, optional
            Electron concentration
        p : array_like, optional
            Hole concentration
        """
        cdef double start_time = time.time()
        
        if not self.is_assembled:
            self._assemble_matrices()
        
        # Create right-hand side vector
        rhs = self._assemble_rhs(n, p)
        
        # Apply boundary conditions
        A_bc, rhs_bc = self._apply_boundary_conditions(self.stiffness_matrix.copy(), rhs, V_p, V_n)
        
        # Solve linear system
        try:
            self.potential = spla.spsolve(A_bc, rhs_bc)
        except Exception as e:
            print(f"Solver failed: {e}")
            # Fallback to iterative solver
            self.potential, info = spla.cg(A_bc, rhs_bc, tol=1e-8)
            if info != 0:
                print(f"Warning: Iterative solver convergence issue (info={info})")
        
        # Compute electric field
        self._compute_electric_field()
        
        self.last_solve_time = time.time() - start_time
    
    def _assemble_rhs(self, n, p):
        """Assemble right-hand side vector from charge density"""
        cdef cnp.ndarray[double, ndim=1] rhs = np.zeros(self.num_nodes, dtype=np.float64)
        cdef int i
        cdef double x, y, rho_val
        
        # Default carrier concentrations if not provided
        if n is None:
            n = np.zeros(self.num_nodes)
        if p is None:
            p = np.zeros(self.num_nodes)
        
        # Assemble RHS: ∫ ρ N_i dΩ
        for i in range(self.num_nodes):
            x = self.nodes_x[i]
            y = self.nodes_y[i]
            rho_val = self.rho_func(x, y, n, p)
            rhs[i] = -rho_val  # Negative because of Poisson equation sign convention
        
        # Apply mass matrix to get proper RHS
        return self.mass_matrix.dot(rhs)
    
    def _apply_boundary_conditions(self, A, rhs, V_p, V_n):
        """Apply Dirichlet boundary conditions"""
        cdef int i
        cdef double x, y
        
        # Identify boundary nodes
        boundary_nodes = []
        boundary_values = []
        
        for i in range(self.num_nodes):
            x = self.nodes_x[i]
            y = self.nodes_y[i]
            
            # Left boundary (p-type)
            if abs(x) < 1e-12:
                boundary_nodes.append(i)
                boundary_values.append(V_p)
            # Right boundary (n-type)
            elif abs(x - self.Lx) < 1e-12:
                boundary_nodes.append(i)
                boundary_values.append(V_n)
        
        # Apply boundary conditions
        A_bc = A.tolil()
        rhs_bc = rhs.copy()
        
        for i, val in zip(boundary_nodes, boundary_values):
            # Set row to identity
            A_bc[i, :] = 0
            A_bc[i, i] = 1
            rhs_bc[i] = val
        
        return A_bc.tocsr(), rhs_bc
    
    def _compute_electric_field(self):
        """Compute electric field from potential gradient"""
        cdef int i, j
        cdef double dx = self.Lx / (self.nx - 1)
        cdef double dy = self.Ly / (self.ny - 1)
        
        # Compute electric field using finite differences
        for i in range(self.num_nodes):
            # Get grid indices
            ix = i % self.nx
            iy = i // self.nx
            
            # Compute gradients with boundary handling
            if ix > 0 and ix < self.nx - 1:
                self.electric_field_x[i] = -(self.potential[i + 1] - self.potential[i - 1]) / (2 * dx)
            elif ix == 0:
                self.electric_field_x[i] = -(self.potential[i + 1] - self.potential[i]) / dx
            else:
                self.electric_field_x[i] = -(self.potential[i] - self.potential[i - 1]) / dx
            
            if iy > 0 and iy < self.ny - 1:
                self.electric_field_y[i] = -(self.potential[i + self.nx] - self.potential[i - self.nx]) / (2 * dy)
            elif iy == 0:
                self.electric_field_y[i] = -(self.potential[i + self.nx] - self.potential[i]) / dy
            else:
                self.electric_field_y[i] = -(self.potential[i] - self.potential[i - self.nx]) / dy
    
    def get_potential(self):
        """Get the computed potential"""
        return self.potential.copy()
    
    def get_electric_field(self):
        """Get the computed electric field"""
        return self.electric_field_x.copy(), self.electric_field_y.copy()
    
    def get_solve_time(self):
        """Get the last solve time"""
        return self.last_solve_time
    
    def get_matrix_info(self):
        """Get information about the assembled matrices"""
        if not self.is_assembled:
            return None
        
        return {
            'stiffness_nnz': self.stiffness_matrix.nnz,
            'mass_nnz': self.mass_matrix.nnz,
            'matrix_size': self.num_nodes,
            'condition_number': 'not_computed'  # Could add condition number estimation
        }

def create_poisson_solver(mesh, epsilon_r_func, rho_func):
    """
    Create a Cython-based Poisson solver.
    
    Parameters:
    -----------
    mesh : SimpleMesh
        The mesh object
    epsilon_r_func : callable
        Function returning relative permittivity
    rho_func : callable
        Function returning charge density
    
    Returns:
    --------
    CythonPoissonSolver
        The created solver
    """
    return CythonPoissonSolver(mesh, epsilon_r_func, rho_func)

def test_poisson_solver():
    """Test the Cython Poisson solver"""
    try:
        # Import mesh module
        import sys
        sys.path.insert(0, '..')
        from core.mesh_minimal import SimpleMesh
        
        # Create test mesh
        mesh = SimpleMesh(20, 10, 100e-9, 50e-9)
        
        # Define material functions
        def epsilon_r(x, y):
            return 12.9  # GaAs
        
        def rho(x, y, n, p):
            return 0.0  # No free charges for test
        
        # Create solver
        solver = CythonPoissonSolver(mesh, epsilon_r, rho)
        
        # Solve
        solver.solve(0.0, 1.0)  # 1V bias
        
        potential = solver.get_potential()
        Ex, Ey = solver.get_electric_field()
        
        print(f"✅ Poisson solver test successful")
        print(f"   Potential range: {np.min(potential):.6f} to {np.max(potential):.6f} V")
        print(f"   Electric field range: {np.min(Ex):.2e} to {np.max(Ex):.2e} V/m")
        print(f"   Solve time: {solver.get_solve_time():.3f} s")
        
        return True
        
    except Exception as e:
        print(f"❌ Poisson solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
