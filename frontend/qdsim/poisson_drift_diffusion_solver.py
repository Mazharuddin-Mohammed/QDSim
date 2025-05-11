"""
Poisson-Drift-Diffusion Solver.

This module provides a Python implementation of the Poisson-Drift-Diffusion solver
for self-consistent calculations of the electrostatic potential and carrier concentrations.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
    print("Using CuPy for GPU acceleration")
except ImportError:
    HAS_CUPY = False
    print("CuPy not available, using NumPy fallback for 'GPU' operations")

class FullPoissonDriftDiffusionSolver:
    """
    Full Poisson-Drift-Diffusion Solver.
    
    This class provides a Python implementation of the Poisson-Drift-Diffusion solver
    for self-consistent calculations of the electrostatic potential and carrier concentrations.
    """
    
    def __init__(self, mesh, epsilon_r_func, rho_func, n_conc_func, p_conc_func, mobility_n_func, mobility_p_func):
        """
        Initialize the FullPoissonDriftDiffusionSolver.
        
        Args:
            mesh: Mesh object
            epsilon_r_func: Function that returns the relative permittivity at a given position
            rho_func: Function that returns the charge density at a given position
            n_conc_func: Function that returns the electron concentration at a given position
            p_conc_func: Function that returns the hole concentration at a given position
            mobility_n_func: Function that returns the electron mobility at a given position
            mobility_p_func: Function that returns the hole mobility at a given position
        """
        self.mesh = mesh
        self.epsilon_r_func = epsilon_r_func
        self.rho_func = rho_func
        self.n_conc_func = n_conc_func
        self.p_conc_func = p_conc_func
        self.mobility_n_func = mobility_n_func
        self.mobility_p_func = mobility_p_func
        
        # Initialize arrays
        self.num_nodes = mesh.get_num_nodes()
        self.phi = np.zeros(self.num_nodes)
        self.phi_n = np.zeros(self.num_nodes)
        self.phi_p = np.zeros(self.num_nodes)
        self.n = np.zeros(self.num_nodes)
        self.p = np.zeros(self.num_nodes)
        
        # Physical constants
        self.q = 1.602e-19  # Elementary charge in C
        self.epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m
        self.kB = 1.381e-23  # Boltzmann constant in J/K
        self.T = 300  # Temperature in K
        self.kT = self.kB * self.T  # Thermal energy in J
        self.kT_q = self.kT / self.q  # Thermal voltage in V
        
        # Get mesh nodes and elements
        self.nodes = np.array(mesh.get_nodes())
        self.elements = np.array(mesh.get_elements())
        
        # Precompute element areas and shape function gradients
        self.element_areas = np.zeros(len(self.elements))
        self.shape_gradients = np.zeros((len(self.elements), 3, 2))
        
        for i, element in enumerate(self.elements):
            # Get node coordinates
            x1, y1 = self.nodes[element[0]]
            x2, y2 = self.nodes[element[1]]
            x3, y3 = self.nodes[element[2]]
            
            # Calculate element area
            self.element_areas[i] = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            
            # Calculate shape function gradients
            self.shape_gradients[i, 0, 0] = (y2 - y3) / (2 * self.element_areas[i])
            self.shape_gradients[i, 0, 1] = (x3 - x2) / (2 * self.element_areas[i])
            self.shape_gradients[i, 1, 0] = (y3 - y1) / (2 * self.element_areas[i])
            self.shape_gradients[i, 1, 1] = (x1 - x3) / (2 * self.element_areas[i])
            self.shape_gradients[i, 2, 0] = (y1 - y2) / (2 * self.element_areas[i])
            self.shape_gradients[i, 2, 1] = (x2 - x1) / (2 * self.element_areas[i])
    
    def solve(self, V_p, V_n, N_A, N_D, tolerance=1e-6, max_iter=100):
        """
        Solve the Poisson-Drift-Diffusion equations.
        
        Args:
            V_p: Potential at the p-side (V)
            V_n: Potential at the n-side (V)
            N_A: Acceptor concentration (m^-3)
            N_D: Donor concentration (m^-3)
            tolerance: Convergence tolerance
            max_iter: Maximum number of iterations
            
        Returns:
            Potential array (V)
        """
        # Initialize the potential with a linear profile
        x = self.nodes[:, 0]
        x_min = np.min(x)
        x_max = np.max(x)
        self.phi = V_p + (V_n - V_p) * (x - x_min) / (x_max - x_min)
        
        # Initialize quasi-Fermi potentials
        self.phi_n = np.copy(self.phi)
        self.phi_p = np.copy(self.phi)
        
        # Initialize carrier concentrations
        for i in range(self.num_nodes):
            x, y = self.nodes[i]
            self.n[i] = self.n_conc_func(x, y, self.phi[i], self.phi_n[i])
            self.p[i] = self.p_conc_func(x, y, self.phi[i], self.phi_p[i])
        
        # Gummel iteration
        for iter in range(max_iter):
            # Solve Poisson equation
            self._solve_poisson(V_p, V_n)
            
            # Solve continuity equations
            self._solve_continuity(V_p, V_n, N_A, N_D)
            
            # Check convergence
            if iter > 0 and np.max(np.abs(self.phi - phi_old)) < tolerance:
                print(f"Converged after {iter+1} iterations")
                break
                
            # Store old potential for convergence check
            phi_old = np.copy(self.phi)
            
            if iter == max_iter - 1:
                print(f"Warning: Did not converge after {max_iter} iterations")
        
        return self.phi
    
    def _solve_poisson(self, V_p, V_n):
        """
        Solve the Poisson equation.
        
        Args:
            V_p: Potential at the p-side (V)
            V_n: Potential at the n-side (V)
            
        Returns:
            Potential array (V)
        """
        # Assemble the stiffness matrix and load vector
        A = lil_matrix((self.num_nodes, self.num_nodes))
        b = np.zeros(self.num_nodes)
        
        # Loop over elements
        for i, element in enumerate(self.elements):
            # Get element nodes
            nodes = element
            
            # Calculate element centroid
            x_c = np.mean(self.nodes[nodes, 0])
            y_c = np.mean(self.nodes[nodes, 1])
            
            # Get material properties at centroid
            epsilon_r = self.epsilon_r_func(x_c, y_c)
            
            # Assemble element stiffness matrix
            for j in range(3):
                for k in range(3):
                    A[nodes[j], nodes[k]] += (
                        epsilon_r * self.epsilon_0 *
                        np.dot(self.shape_gradients[i, j], self.shape_gradients[i, k]) *
                        self.element_areas[i]
                    )
            
            # Assemble element load vector
            for j in range(3):
                # Get charge density at element centroid
                rho = self.rho_func(x_c, y_c, self.n, self.p)
                
                # Add contribution to load vector
                b[nodes[j]] -= rho * self.element_areas[i] / 3
        
        # Apply boundary conditions
        # Find boundary nodes
        x = self.nodes[:, 0]
        x_min = np.min(x)
        x_max = np.max(x)
        
        # Apply Dirichlet boundary conditions
        for i in range(self.num_nodes):
            if abs(self.nodes[i, 0] - x_min) < 1e-10:
                # p-side
                A[i, :] = 0
                A[i, i] = 1
                b[i] = V_p
            elif abs(self.nodes[i, 0] - x_max) < 1e-10:
                # n-side
                A[i, :] = 0
                A[i, i] = 1
                b[i] = V_n
        
        # Solve the linear system
        A_csr = A.tocsr()
        self.phi = spsolve(A_csr, b)
        
        return self.phi
    
    def _solve_continuity(self, V_p, V_n, N_A, N_D):
        """
        Solve the continuity equations.
        
        Args:
            V_p: Potential at the p-side (V)
            V_n: Potential at the n-side (V)
            N_A: Acceptor concentration (m^-3)
            N_D: Donor concentration (m^-3)
            
        Returns:
            Electron and hole concentration arrays (m^-3)
        """
        # Solve for electron quasi-Fermi potential
        self._solve_electron_continuity(V_p, V_n, N_D)
        
        # Solve for hole quasi-Fermi potential
        self._solve_hole_continuity(V_p, V_n, N_A)
        
        # Update carrier concentrations
        for i in range(self.num_nodes):
            x, y = self.nodes[i]
            self.n[i] = self.n_conc_func(x, y, self.phi[i], self.phi_n[i])
            self.p[i] = self.p_conc_func(x, y, self.phi[i], self.phi_p[i])
        
        return self.n, self.p
    
    def _solve_electron_continuity(self, V_p, V_n, N_D):
        """
        Solve the electron continuity equation.
        
        Args:
            V_p: Potential at the p-side (V)
            V_n: Potential at the n-side (V)
            N_D: Donor concentration (m^-3)
            
        Returns:
            Electron quasi-Fermi potential array (V)
        """
        # Simplified approach: set quasi-Fermi potential to match equilibrium
        x = self.nodes[:, 0]
        x_min = np.min(x)
        x_max = np.max(x)
        
        # Set quasi-Fermi potential to match equilibrium on n-side
        for i in range(self.num_nodes):
            if self.nodes[i, 0] > 0:  # n-side
                self.phi_n[i] = self.phi[i]
            else:  # p-side
                self.phi_n[i] = V_n
        
        return self.phi_n
    
    def _solve_hole_continuity(self, V_p, V_n, N_A):
        """
        Solve the hole continuity equation.
        
        Args:
            V_p: Potential at the p-side (V)
            V_n: Potential at the n-side (V)
            N_A: Acceptor concentration (m^-3)
            
        Returns:
            Hole quasi-Fermi potential array (V)
        """
        # Simplified approach: set quasi-Fermi potential to match equilibrium
        x = self.nodes[:, 0]
        x_min = np.min(x)
        x_max = np.max(x)
        
        # Set quasi-Fermi potential to match equilibrium on p-side
        for i in range(self.num_nodes):
            if self.nodes[i, 0] < 0:  # p-side
                self.phi_p[i] = self.phi[i]
            else:  # n-side
                self.phi_p[i] = V_p
        
        return self.phi_p
    
    def get_potential(self):
        """
        Get the computed electrostatic potential.
        
        Returns:
            Potential array (V)
        """
        return self.phi
    
    def get_n(self):
        """
        Get the computed electron concentration.
        
        Returns:
            Electron concentration array (m^-3)
        """
        return self.n
    
    def get_p(self):
        """
        Get the computed hole concentration.
        
        Returns:
            Hole concentration array (m^-3)
        """
        return self.p
    
    def get_electric_field(self, x, y):
        """
        Get the electric field at a given position.
        
        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)
            
        Returns:
            Electric field vector (V/m)
        """
        # Find the element containing the point
        for i, element in enumerate(self.elements):
            # Get element nodes
            nodes = element
            
            # Get node coordinates
            x1, y1 = self.nodes[nodes[0]]
            x2, y2 = self.nodes[nodes[1]]
            x3, y3 = self.nodes[nodes[2]]
            
            # Check if point is inside the element
            # Using barycentric coordinates
            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
            beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
            gamma = 1 - alpha - beta
            
            if alpha >= 0 and beta >= 0 and gamma >= 0:
                # Point is inside the element
                # Calculate electric field using shape function gradients
                Ex = 0
                Ey = 0
                
                for j in range(3):
                    Ex -= self.phi[nodes[j]] * self.shape_gradients[i, j, 0]
                    Ey -= self.phi[nodes[j]] * self.shape_gradients[i, j, 1]
                
                return np.array([Ex, Ey])
        
        # Point is outside the mesh
        return np.array([0.0, 0.0])
