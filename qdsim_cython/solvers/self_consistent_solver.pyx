# distutils: language = c++
# cython: language_level = 3

"""
Cython-based Self-Consistent Solver Implementation

This module provides a complete Cython implementation of the self-consistent
Poisson-Schrödinger solver for quantum device simulations.
"""

import numpy as np
cimport numpy as cnp
from libcpp cimport bool as bint
import time

# Import our Cython solvers
from .poisson_solver import CythonPoissonSolver
from .schrodinger_solver import CythonSchrodingerSolver

# Initialize NumPy
cnp.import_array()

# Physical constants
cdef double K_B = 1.380649e-23          # J/K - Boltzmann constant
cdef double Q_E = 1.602176634e-19       # C - elementary charge
cdef double EV_TO_J = 1.602176634e-19   # J/eV conversion
cdef double M_E = 9.1093837015e-31      # kg - electron mass

cdef class CythonSelfConsistentSolver:
    """
    High-performance Cython implementation of self-consistent Poisson-Schrödinger solver.
    
    Solves the coupled Poisson and Schrödinger equations self-consistently
    for quantum device simulations.
    """
    
    cdef object poisson_solver
    cdef object schrodinger_solver
    cdef object mesh
    cdef cnp.ndarray potential
    cdef cnp.ndarray electron_density
    cdef cnp.ndarray hole_density
    cdef cnp.ndarray eigenvalues
    cdef list eigenvectors
    cdef double temperature
    cdef double fermi_level
    cdef int max_iterations
    cdef double tolerance
    cdef double last_solve_time
    cdef int last_iterations
    cdef bint converged
    cdef list convergence_history
    
    def __cinit__(self, mesh, epsilon_r_func, m_star_func, potential_func, 
                  double temperature=300.0):
        """
        Initialize the self-consistent solver.
        
        Parameters:
        -----------
        mesh : SimpleMesh
            The mesh object
        epsilon_r_func : callable
            Function returning relative permittivity at (x, y)
        m_star_func : callable
            Function returning effective mass at (x, y) in kg
        potential_func : callable
            Function returning potential energy at (x, y) in J
        temperature : float
            Temperature in Kelvin
        """
        self.mesh = mesh
        self.temperature = temperature
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.fermi_level = 0.0
        
        # Initialize solution arrays
        self.potential = np.zeros(mesh.num_nodes, dtype=np.float64)
        self.electron_density = np.zeros(mesh.num_nodes, dtype=np.float64)
        self.hole_density = np.zeros(mesh.num_nodes, dtype=np.float64)
        self.eigenvalues = np.array([], dtype=np.float64)
        self.eigenvectors = []
        
        # Initialize convergence tracking
        self.last_solve_time = 0.0
        self.last_iterations = 0
        self.converged = False
        self.convergence_history = []
        
        # Create charge density function for Poisson solver
        def rho_func(x, y, n, p):
            return self._compute_charge_density(x, y, n, p)
        
        # Create solvers
        self.poisson_solver = CythonPoissonSolver(mesh, epsilon_r_func, rho_func)
        self.schrodinger_solver = CythonSchrodingerSolver(mesh, m_star_func, potential_func)
    
    def _compute_charge_density(self, double x, double y, n, p):
        """Compute charge density from carrier concentrations"""
        cdef double n_val = 0.0
        cdef double p_val = 0.0
        
        # Get carrier concentrations at this point
        if n is not None and len(n) > 0:
            # Find nearest node (simplified)
            node_idx = self._find_nearest_node(x, y)
            if 0 <= node_idx < len(n):
                n_val = n[node_idx]
        
        if p is not None and len(p) > 0:
            node_idx = self._find_nearest_node(x, y)
            if 0 <= node_idx < len(p):
                p_val = p[node_idx]
        
        # Charge density: ρ = q(p - n)
        return Q_E * (p_val - n_val)
    
    def _find_nearest_node(self, double x, double y):
        """Find nearest mesh node to given coordinates"""
        cdef int ix = int(x / self.mesh.Lx * (self.mesh.nx - 1))
        cdef int iy = int(y / self.mesh.Ly * (self.mesh.ny - 1))
        
        # Clamp to valid range
        ix = max(0, min(ix, self.mesh.nx - 1))
        iy = max(0, min(iy, self.mesh.ny - 1))
        
        return iy * self.mesh.nx + ix
    
    def solve_self_consistent(self, double V_p, double V_n, int num_quantum_states=5,
                             double N_A=1e24, double N_D=1e24):
        """
        Solve the self-consistent Poisson-Schrödinger equations.
        
        Parameters:
        -----------
        V_p : float
            Potential at p-type boundary (V)
        V_n : float
            Potential at n-type boundary (V)
        num_quantum_states : int
            Number of quantum states to compute
        N_A : float
            Acceptor concentration (m^-3)
        N_D : float
            Donor concentration (m^-3)
        
        Returns:
        --------
        dict
            Solution results
        """
        cdef double start_time = time.time()
        cdef int iteration = 0
        cdef double residual = float('inf')
        cdef cnp.ndarray old_potential
        
        self.convergence_history = []
        
        # Initial guess for potential (linear interpolation)
        self._initialize_potential(V_p, V_n)
        
        print(f"Starting self-consistent iteration...")
        
        while iteration < self.max_iterations and residual > self.tolerance:
            iteration += 1
            
            # Store old potential for convergence check
            old_potential = self.potential.copy()
            
            # Step 1: Solve Schrödinger equation with current potential
            self._update_schrodinger_potential()
            eigenvalues, eigenvectors = self.schrodinger_solver.solve(num_quantum_states)
            
            if len(eigenvalues) > 0:
                self.eigenvalues = eigenvalues
                self.eigenvectors = eigenvectors
                
                # Step 2: Compute carrier densities from quantum states
                self._compute_carrier_densities(N_A, N_D)
            else:
                print(f"Warning: No quantum states computed in iteration {iteration}")
                # Use classical carrier densities
                self._compute_classical_carrier_densities(N_A, N_D)
            
            # Step 3: Solve Poisson equation with updated carrier densities
            self.poisson_solver.solve(V_p, V_n, self.electron_density, self.hole_density)
            self.potential = self.poisson_solver.get_potential()
            
            # Check convergence
            residual = np.max(np.abs(self.potential - old_potential))
            self.convergence_history.append(residual)
            
            print(f"  Iteration {iteration}: residual = {residual:.2e}")
            
            if residual < self.tolerance:
                self.converged = True
                break
        
        self.last_solve_time = time.time() - start_time
        self.last_iterations = iteration
        
        if self.converged:
            print(f"✅ Self-consistent solution converged in {iteration} iterations")
        else:
            print(f"⚠️  Self-consistent solution did not converge after {iteration} iterations")
        
        return self._create_solution_dict()
    
    def _initialize_potential(self, double V_p, double V_n):
        """Initialize potential with linear interpolation"""
        cdef int i
        cdef double x
        
        nodes_x, _ = self.mesh.get_nodes()
        
        for i in range(self.mesh.num_nodes):
            x = nodes_x[i]
            # Linear interpolation between boundaries
            self.potential[i] = V_p + (V_n - V_p) * x / self.mesh.Lx
    
    def _update_schrodinger_potential(self):
        """Update Schrödinger solver with current electrostatic potential"""
        # Create new potential function that includes electrostatic potential
        def updated_potential_func(x, y):
            # Get electrostatic potential at this point
            node_idx = self._find_nearest_node(x, y)
            V_electrostatic = 0.0
            if 0 <= node_idx < len(self.potential):
                V_electrostatic = self.potential[node_idx]
            
            # Add original potential (e.g., quantum well structure)
            V_original = self.schrodinger_solver.potential_func(x, y)
            
            # Total potential energy
            return V_original - Q_E * V_electrostatic  # Electron sees negative of electrostatic potential
        
        # Update the potential function in Schrödinger solver
        self.schrodinger_solver.potential_func = updated_potential_func
        
        # Force reassembly of matrices
        self.schrodinger_solver.is_assembled = False
    
    def _compute_carrier_densities(self, double N_A, double N_D):
        """Compute carrier densities from quantum states"""
        cdef int i, state_idx
        cdef double kT = K_B * self.temperature
        cdef double n_total, p_total
        cdef cnp.ndarray psi
        cdef double E_state, occupation
        
        # Initialize densities
        self.electron_density.fill(0.0)
        self.hole_density.fill(0.0)
        
        # Estimate Fermi level (simplified)
        if len(self.eigenvalues) > 0:
            self.fermi_level = self.eigenvalues[0] + 0.5 * kT
        
        # Compute electron density from quantum states
        for state_idx in range(len(self.eigenvalues)):
            E_state = self.eigenvalues[state_idx]
            psi = self.eigenvectors[state_idx]
            
            # Fermi-Dirac occupation
            if (E_state - self.fermi_level) / kT < 50:  # Avoid overflow
                occupation = 1.0 / (1.0 + np.exp((E_state - self.fermi_level) / kT))
            else:
                occupation = 0.0
            
            # Add contribution to electron density
            for i in range(self.mesh.num_nodes):
                self.electron_density[i] += occupation * abs(psi[i])**2
        
        # Scale electron density (simplified)
        if np.sum(self.electron_density) > 0:
            self.electron_density *= N_D / np.mean(self.electron_density)
        
        # Compute hole density (simplified classical approximation)
        for i in range(self.mesh.num_nodes):
            V_node = self.potential[i]
            # Simplified hole density calculation
            if V_node > 0:
                self.hole_density[i] = N_A * np.exp(-Q_E * V_node / kT)
            else:
                self.hole_density[i] = N_A
    
    def _compute_classical_carrier_densities(self, double N_A, double N_D):
        """Compute classical carrier densities (fallback)"""
        cdef int i
        cdef double kT = K_B * self.temperature
        cdef double V_node
        
        for i in range(self.mesh.num_nodes):
            V_node = self.potential[i]
            
            # Classical electron density
            self.electron_density[i] = N_D * np.exp(-Q_E * V_node / kT)
            
            # Classical hole density
            self.hole_density[i] = N_A * np.exp(Q_E * V_node / kT)
    
    def _create_solution_dict(self):
        """Create solution dictionary"""
        return {
            'potential': self.potential.copy(),
            'electron_density': self.electron_density.copy(),
            'hole_density': self.hole_density.copy(),
            'eigenvalues': self.eigenvalues.copy(),
            'eigenvalues_eV': self.eigenvalues / EV_TO_J,
            'eigenvectors': [psi.copy() for psi in self.eigenvectors],
            'fermi_level': self.fermi_level,
            'fermi_level_eV': self.fermi_level / EV_TO_J,
            'converged': self.converged,
            'iterations': self.last_iterations,
            'solve_time': self.last_solve_time,
            'convergence_history': self.convergence_history.copy()
        }
    
    def get_potential(self):
        """Get the computed potential"""
        return self.potential.copy()
    
    def get_carrier_densities(self):
        """Get the computed carrier densities"""
        return self.electron_density.copy(), self.hole_density.copy()
    
    def get_quantum_states(self):
        """Get the computed quantum states"""
        return self.eigenvalues.copy(), [psi.copy() for psi in self.eigenvectors]
    
    def get_convergence_info(self):
        """Get convergence information"""
        return {
            'converged': self.converged,
            'iterations': self.last_iterations,
            'final_residual': self.convergence_history[-1] if self.convergence_history else float('inf'),
            'solve_time': self.last_solve_time,
            'convergence_history': self.convergence_history.copy()
        }
    
    def set_solver_parameters(self, int max_iterations=100, double tolerance=1e-6):
        """Set solver parameters"""
        self.max_iterations = max_iterations
        self.tolerance = tolerance

def create_self_consistent_solver(mesh, epsilon_r_func, m_star_func, potential_func, temperature=300.0):
    """
    Create a Cython-based self-consistent solver.
    
    Parameters:
    -----------
    mesh : SimpleMesh
        The mesh object
    epsilon_r_func : callable
        Function returning relative permittivity
    m_star_func : callable
        Function returning effective mass in kg
    potential_func : callable
        Function returning potential energy in J
    temperature : float
        Temperature in Kelvin
    
    Returns:
    --------
    CythonSelfConsistentSolver
        The created solver
    """
    return CythonSelfConsistentSolver(mesh, epsilon_r_func, m_star_func, potential_func, temperature)

def test_self_consistent_solver():
    """Test the Cython self-consistent solver"""
    try:
        # Import mesh module
        import sys
        sys.path.insert(0, '..')
        from core.mesh_minimal import SimpleMesh
        
        # Create test mesh
        mesh = SimpleMesh(15, 10, 50e-9, 30e-9)
        
        # Define physics functions
        def epsilon_r_func(x, y):
            return 12.9  # GaAs
        
        def m_star_func(x, y):
            return 0.067 * M_E  # GaAs effective mass
        
        def potential_func(x, y):
            # Simple quantum well
            x_center = mesh.Lx / 2
            well_width = 20e-9
            if abs(x - x_center) < well_width / 2:
                return 0.0  # Inside well
            else:
                return 0.1 * EV_TO_J  # Barrier
        
        # Create solver
        solver = CythonSelfConsistentSolver(mesh, epsilon_r_func, m_star_func, potential_func)
        
        # Set solver parameters
        solver.set_solver_parameters(max_iterations=10, tolerance=1e-4)
        
        # Solve
        result = solver.solve_self_consistent(
            V_p=0.0, V_n=1.0,  # 1V bias
            num_quantum_states=3,
            N_A=1e23, N_D=1e23
        )
        
        print(f"✅ Self-consistent solver test successful")
        print(f"   Converged: {result['converged']}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Solve time: {result['solve_time']:.3f} s")
        print(f"   Potential range: {np.min(result['potential']):.3f} to {np.max(result['potential']):.3f} V")
        print(f"   Number of quantum states: {len(result['eigenvalues'])}")
        if len(result['eigenvalues']) > 0:
            print(f"   Energy levels (eV): {result['eigenvalues_eV']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Self-consistent solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
