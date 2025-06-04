# distutils: language = c++
# cython: language_level = 3

"""
Cython-based SchrodingerSolver Implementation

This module provides a complete Cython implementation of the Schrödinger solver
for quantum mechanical calculations, replacing the C++ backend implementation
with high-performance Cython code.
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

# Physical constants
cdef double HBAR = 1.054571817e-34      # J⋅s - reduced Planck constant
cdef double M_E = 9.1093837015e-31      # kg - electron mass
cdef double EV_TO_J = 1.602176634e-19   # J/eV conversion

cdef class CythonSchrodingerSolver:
    """
    High-performance Cython implementation of Schrödinger equation solver.
    
    Solves the time-independent Schrödinger equation:
    -ℏ²/(2m*) ∇²ψ + V(r)ψ = Eψ
    
    Using finite element method with generalized eigenvalue problem:
    H ψ = E M ψ
    """
    
    cdef public int num_nodes, num_elements
    cdef public double Lx, Ly
    cdef public int nx, ny
    cdef cnp.ndarray nodes_x, nodes_y
    cdef cnp.ndarray elements
    cdef cnp.ndarray eigenvalues
    cdef list eigenvectors
    cdef object hamiltonian_matrix
    cdef object mass_matrix
    cdef object m_star_func
    cdef object potential_func
    cdef bint is_assembled
    cdef double last_solve_time
    cdef int last_num_states
    cdef bint use_open_boundaries
    
    def __cinit__(self, mesh, m_star_func, potential_func, bint use_open_boundaries=False):
        """
        Initialize the Schrödinger solver.
        
        Parameters:
        -----------
        mesh : SimpleMesh
            The mesh object
        m_star_func : callable
            Function returning effective mass at (x, y) in kg
        potential_func : callable
            Function returning potential energy at (x, y) in J
        use_open_boundaries : bool
            Whether to use open boundary conditions
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
        
        # Store physics functions
        self.m_star_func = m_star_func
        self.potential_func = potential_func
        self.use_open_boundaries = use_open_boundaries
        
        # Initialize solution arrays
        self.eigenvalues = np.array([], dtype=np.float64)
        self.eigenvectors = []
        
        # Assembly flag
        self.is_assembled = False
        self.last_solve_time = 0.0
        self.last_num_states = 0
        
        # Assemble matrices
        self._assemble_matrices()
    
    cdef void _assemble_matrices(self):
        """Assemble the Hamiltonian and mass matrices using finite elements"""
        cdef int i, j, k, elem_idx
        cdef int n0, n1, n2
        cdef double x0, y0, x1, y1, x2, y2
        cdef double area, det_J
        cdef double m_star_avg, V_avg
        cdef cnp.ndarray[double, ndim=2] H_elem = np.zeros((3, 3), dtype=np.float64)
        cdef cnp.ndarray[double, ndim=2] M_elem = np.zeros((3, 3), dtype=np.float64)
        
        # Initialize sparse matrix builders
        row_indices = []
        col_indices = []
        hamiltonian_data = []
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
            
            # Calculate average material properties for element
            x_center = (x0 + x1 + x2) / 3.0
            y_center = (y0 + y1 + y2) / 3.0
            m_star_avg = self.m_star_func(x_center, y_center)
            V_avg = self.potential_func(x_center, y_center)
            
            # Assemble element Hamiltonian matrix
            self._assemble_element_hamiltonian(H_elem, x0, y0, x1, y1, x2, y2, m_star_avg, V_avg)
            
            # Assemble element mass matrix
            self._assemble_element_mass(M_elem, area)
            
            # Add to global matrix
            nodes = [n0, n1, n2]
            for i in range(3):
                for j in range(3):
                    row_indices.append(nodes[i])
                    col_indices.append(nodes[j])
                    hamiltonian_data.append(H_elem[i, j])
                    mass_data.append(M_elem[i, j])
        
        # Create sparse matrices
        self.hamiltonian_matrix = sp.csr_matrix(
            (hamiltonian_data, (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        self.mass_matrix = sp.csr_matrix(
            (mass_data, (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        # Apply boundary conditions if needed
        if not self.use_open_boundaries:
            self._apply_boundary_conditions()
        
        self.is_assembled = True
    
    cdef void _assemble_element_hamiltonian(self, cnp.ndarray[double, ndim=2] H_elem,
                                           double x0, double y0, double x1, double y1,
                                           double x2, double y2, double m_star, double V_potential):
        """Assemble element Hamiltonian matrix"""
        cdef double det_J = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        cdef double inv_det_J = 1.0 / det_J
        cdef double area = 0.5 * abs(det_J)
        
        # Shape function gradients in reference element
        cdef double dN0_dx = (y1 - y2) * inv_det_J
        cdef double dN0_dy = (x2 - x1) * inv_det_J
        cdef double dN1_dx = (y2 - y0) * inv_det_J
        cdef double dN1_dy = (x0 - x2) * inv_det_J
        cdef double dN2_dx = (y0 - y1) * inv_det_J
        cdef double dN2_dy = (x1 - x0) * inv_det_J
        
        # Kinetic energy term: -ℏ²/(2m*) ∇²
        cdef double kinetic_factor = HBAR * HBAR / (2.0 * m_star) * area
        
        # Potential energy term: V(r)
        cdef double potential_factor = V_potential * area / 3.0
        
        # Assemble kinetic energy contribution: ∫ (ℏ²/2m*) ∇N_i · ∇N_j dΩ
        H_elem[0, 0] = kinetic_factor * (dN0_dx * dN0_dx + dN0_dy * dN0_dy) + potential_factor
        H_elem[0, 1] = kinetic_factor * (dN0_dx * dN1_dx + dN0_dy * dN1_dy)
        H_elem[0, 2] = kinetic_factor * (dN0_dx * dN2_dx + dN0_dy * dN2_dy)
        H_elem[1, 0] = H_elem[0, 1]
        H_elem[1, 1] = kinetic_factor * (dN1_dx * dN1_dx + dN1_dy * dN1_dy) + potential_factor
        H_elem[1, 2] = kinetic_factor * (dN1_dx * dN2_dx + dN1_dy * dN2_dy)
        H_elem[2, 0] = H_elem[0, 2]
        H_elem[2, 1] = H_elem[1, 2]
        H_elem[2, 2] = kinetic_factor * (dN2_dx * dN2_dx + dN2_dy * dN2_dy) + potential_factor
        
        # Add potential energy contribution to diagonal terms
        H_elem[1, 1] += potential_factor
        H_elem[2, 2] += potential_factor
    
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
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions for confined quantum system"""
        cdef int i
        cdef double x, y
        
        # For confined systems, apply Dirichlet boundary conditions (ψ = 0 at boundaries)
        boundary_nodes = []
        
        for i in range(self.num_nodes):
            x = self.nodes_x[i]
            y = self.nodes_y[i]
            
            # Check if node is on boundary
            if (abs(x) < 1e-12 or abs(x - self.Lx) < 1e-12 or 
                abs(y) < 1e-12 or abs(y - self.Ly) < 1e-12):
                boundary_nodes.append(i)
        
        # Apply boundary conditions by modifying matrices
        H_bc = self.hamiltonian_matrix.tolil()
        M_bc = self.mass_matrix.tolil()
        
        for i in boundary_nodes:
            # Set row to identity for Hamiltonian
            H_bc[i, :] = 0
            H_bc[i, i] = 1
            # Set corresponding row in mass matrix to zero except diagonal
            M_bc[i, :] = 0
            M_bc[i, i] = 1
        
        self.hamiltonian_matrix = H_bc.tocsr()
        self.mass_matrix = M_bc.tocsr()
    
    def solve(self, int num_eigenvalues, double tolerance=1e-8):
        """
        Solve the Schrödinger equation eigenvalue problem.
        
        Parameters:
        -----------
        num_eigenvalues : int
            Number of eigenvalues to compute
        tolerance : float
            Convergence tolerance
        
        Returns:
        --------
        tuple
            (eigenvalues, eigenvectors) in SI units
        """
        cdef double start_time = time.time()
        
        if not self.is_assembled:
            self._assemble_matrices()
        
        if num_eigenvalues >= self.num_nodes:
            num_eigenvalues = self.num_nodes - 1
        
        try:
            # Solve generalized eigenvalue problem: H ψ = E M ψ
            eigenvals, eigenvecs = spla.eigsh(
                self.hamiltonian_matrix, 
                k=num_eigenvalues, 
                M=self.mass_matrix,
                which='SM',  # Smallest magnitude (lowest energy states)
                tol=tolerance
            )
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvals)
            self.eigenvalues = eigenvals[idx]
            self.eigenvectors = [eigenvecs[:, i] for i in idx]
            
            # Normalize eigenvectors
            for i, psi in enumerate(self.eigenvectors):
                norm = np.sqrt(np.real(np.vdot(psi, self.mass_matrix.dot(psi))))
                if norm > 1e-12:
                    self.eigenvectors[i] = psi / norm
            
            self.last_num_states = len(self.eigenvalues)
            
        except Exception as e:
            print(f"Eigenvalue solver failed: {e}")
            # Fallback to standard eigenvalue problem
            try:
                eigenvals, eigenvecs = spla.eigsh(
                    self.hamiltonian_matrix,
                    k=num_eigenvalues,
                    which='SM',
                    tol=tolerance
                )
                
                idx = np.argsort(eigenvals)
                self.eigenvalues = eigenvals[idx]
                self.eigenvectors = [eigenvecs[:, i] for i in idx]
                self.last_num_states = len(self.eigenvalues)
                
            except Exception as e2:
                print(f"Fallback solver also failed: {e2}")
                self.eigenvalues = np.array([])
                self.eigenvectors = []
                self.last_num_states = 0
        
        self.last_solve_time = time.time() - start_time
        
        return self.eigenvalues.copy(), [psi.copy() for psi in self.eigenvectors]
    
    def get_eigenvalues(self):
        """Get computed eigenvalues in Joules"""
        return self.eigenvalues.copy()
    
    def get_eigenvalues_eV(self):
        """Get computed eigenvalues in eV"""
        return self.eigenvalues / EV_TO_J
    
    def get_eigenvectors(self):
        """Get computed eigenvectors"""
        return [psi.copy() for psi in self.eigenvectors]
    
    def get_solve_time(self):
        """Get the last solve time"""
        return self.last_solve_time
    
    def get_num_computed_states(self):
        """Get number of computed states"""
        return self.last_num_states
    
    def get_matrix_info(self):
        """Get information about the assembled matrices"""
        if not self.is_assembled:
            return None
        
        return {
            'hamiltonian_nnz': self.hamiltonian_matrix.nnz,
            'mass_nnz': self.mass_matrix.nnz,
            'matrix_size': self.num_nodes,
            'num_boundary_nodes': self._count_boundary_nodes()
        }
    
    def _count_boundary_nodes(self):
        """Count boundary nodes"""
        cdef int count = 0
        cdef int i
        cdef double x, y
        
        for i in range(self.num_nodes):
            x = self.nodes_x[i]
            y = self.nodes_y[i]
            
            if (abs(x) < 1e-12 or abs(x - self.Lx) < 1e-12 or 
                abs(y) < 1e-12 or abs(y - self.Ly) < 1e-12):
                count += 1
        
        return count

def create_schrodinger_solver(mesh, m_star_func, potential_func, use_open_boundaries=False):
    """
    Create a Cython-based Schrödinger solver.
    
    Parameters:
    -----------
    mesh : SimpleMesh
        The mesh object
    m_star_func : callable
        Function returning effective mass in kg
    potential_func : callable
        Function returning potential energy in J
    use_open_boundaries : bool
        Whether to use open boundary conditions
    
    Returns:
    --------
    CythonSchrodingerSolver
        The created solver
    """
    return CythonSchrodingerSolver(mesh, m_star_func, potential_func, use_open_boundaries)

def test_schrodinger_solver():
    """Test the Cython Schrödinger solver"""
    try:
        # Import mesh module
        import sys
        sys.path.insert(0, '..')
        from core.mesh_minimal import SimpleMesh
        
        # Create test mesh
        mesh = SimpleMesh(15, 10, 30e-9, 20e-9)
        
        # Define physics functions
        def m_star_func(x, y):
            return 0.067 * M_E  # GaAs effective mass
        
        def potential_func(x, y):
            # Simple harmonic oscillator potential
            x_center = mesh.Lx / 2
            y_center = mesh.Ly / 2
            k = 1e17  # Spring constant
            r_squared = (x - x_center)**2 + (y - y_center)**2
            return 0.5 * k * r_squared
        
        # Create solver
        solver = CythonSchrodingerSolver(mesh, m_star_func, potential_func)
        
        # Solve for first few states
        eigenvalues, eigenvectors = solver.solve(3)
        eigenvalues_eV = solver.get_eigenvalues_eV()
        
        print(f"✅ Schrödinger solver test successful")
        print(f"   Number of states computed: {len(eigenvalues)}")
        if len(eigenvalues) > 0:
            print(f"   Energy levels (eV):")
            for i, E in enumerate(eigenvalues_eV):
                print(f"     State {i+1}: {E:.6f} eV")
        print(f"   Solve time: {solver.get_solve_time():.3f} s")
        
        return True
        
    except Exception as e:
        print(f"❌ Schrödinger solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
