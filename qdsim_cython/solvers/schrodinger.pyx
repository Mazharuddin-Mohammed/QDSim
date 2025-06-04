# distutils: language = c++
# cython: language_level = 3

"""
Cython wrapper for SchrodingerSolver class

This module provides high-performance Python access to the C++ SchrodingerSolver class
for quantum mechanical calculations in semiconductor devices.

Features:
- Open quantum system physics with CAP boundaries
- Dirac-delta normalization for scattering states
- Device-specific optimization (nanowire, quantum dot, wide channel)
- Advanced eigenvalue filtering and degeneracy breaking
- Performance monitoring and error handling
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as bint
from ..eigen cimport VectorXd, VectorXcd
from ..core.mesh cimport Mesh

# Import C++ declarations
from .schrodinger cimport (
    SchrodingerSolver as CppSchrodingerSolver,
    mass_func_t, potential_func_t,
    SolverConfig, SolverResults,
    NanowireParams, QuantumDotParams, WideChannelParams
)

# Initialize NumPy
cnp.import_array()

# Global callback storage for function pointers
cdef dict _mass_callbacks = {}
cdef dict _potential_callbacks = {}
cdef int _callback_counter = 0

# Callback wrapper functions for C++ function pointers
cdef double mass_callback_wrapper(double x, double y) with gil:
    """Wrapper function to call Python mass functions from C++"""
    cdef int callback_id = 0  # This will be set by the calling context
    if callback_id in _mass_callbacks:
        return _mass_callbacks[callback_id](x, y)
    return 1.0  # Default mass

cdef double potential_callback_wrapper(double x, double y) with gil:
    """Wrapper function to call Python potential functions from C++"""
    cdef int callback_id = 0  # This will be set by the calling context
    if callback_id in _potential_callbacks:
        return _potential_callbacks[callback_id](x, y)
    return 0.0  # Default potential

cdef class SchrodingerSolver:
    """
    High-performance Cython wrapper for the C++ SchrodingerSolver class.
    
    Solves the time-independent Schrödinger equation for quantum mechanical
    systems in semiconductor devices using advanced numerical methods.
    
    The Schrödinger equation solved is:
    [-ℏ²/(2m(r))∇² + V(r)]ψ = Eψ
    
    Features:
    - Open quantum system physics with Complex Absorbing Potentials (CAP)
    - Dirac-delta normalization for scattering states
    - Device-specific optimization for different geometries
    - Advanced eigenvalue filtering and degeneracy breaking
    - Performance monitoring and comprehensive error handling
    """
    
    cdef CppSchrodingerSolver* _solver
    cdef bint _owns_solver
    cdef object _mass_func
    cdef object _potential_func
    cdef int _callback_id
    cdef object _mesh_ref  # Keep reference to mesh to prevent garbage collection
    
    def __cinit__(self, Mesh mesh, mass_func, potential_func, bint use_dirichlet=False):
        """
        Initialize the Schrödinger solver.
        
        Parameters:
        -----------
        mesh : Mesh
            Finite element mesh for the computational domain
        mass_func : callable
            Function m(x, y) returning effective mass at position (x, y)
        potential_func : callable
            Function V(x, y) returning potential energy at position (x, y)
        use_dirichlet : bool, optional
            Whether to use Dirichlet boundary conditions (default: False for open systems)
        """
        self._solver = NULL
        self._owns_solver = False
        self._mass_func = mass_func
        self._potential_func = potential_func
        self._mesh_ref = mesh
        
        # Get unique callback ID
        global _callback_counter
        self._callback_id = _callback_counter
        _callback_counter += 1
        
        # Store callbacks
        _mass_callbacks[self._callback_id] = mass_func
        _potential_callbacks[self._callback_id] = potential_func
        
        try:
            # Create C++ solver instance
            self._solver = new CppSchrodingerSolver(
                mesh._mesh[0],  # Dereference mesh
                mass_callback_wrapper,
                potential_callback_wrapper,
                use_dirichlet
            )
            self._owns_solver = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to create Schrödinger solver: {e}")
    
    def __dealloc__(self):
        """Clean up C++ resources"""
        if self._owns_solver and self._solver != NULL:
            del self._solver
            self._solver = NULL
        
        # Clean up callbacks
        if self._callback_id in _mass_callbacks:
            del _mass_callbacks[self._callback_id]
        if self._callback_id in _potential_callbacks:
            del _potential_callbacks[self._callback_id]
    
    def solve(self, int num_eigenvalues=5):
        """
        Solve the Schrödinger equation for the specified number of eigenvalues.
        
        Parameters:
        -----------
        num_eigenvalues : int, optional
            Number of eigenvalues to compute (default: 5)
        
        Raises:
        -------
        RuntimeError
            If the solver fails or encounters numerical issues
        """
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        
        try:
            self._solver.solve(num_eigenvalues)
        except Exception as e:
            raise RuntimeError(f"Solver failed: {e}")
    
    def solve_with_tolerance(self, int num_eigenvalues=5, double tolerance=1e-10):
        """
        Solve with specified convergence tolerance.
        
        Parameters:
        -----------
        num_eigenvalues : int, optional
            Number of eigenvalues to compute (default: 5)
        tolerance : float, optional
            Convergence tolerance (default: 1e-10)
        """
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        
        try:
            self._solver.solve_with_tolerance(num_eigenvalues, tolerance)
        except Exception as e:
            raise RuntimeError(f"Solver with tolerance failed: {e}")
    
    def get_eigenvalues(self):
        """
        Get computed eigenvalues.
        
        Returns:
        --------
        numpy.ndarray
            Array of eigenvalues in Joules
        """
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        
        cdef VectorXd eigenvals = self._solver.get_eigenvalues()
        cdef int n = eigenvals.size()
        
        # Convert to NumPy array
        cdef cnp.ndarray[double, ndim=1] result = np.empty(n, dtype=np.float64)
        for i in range(n):
            result[i] = eigenvals[i]
        
        return result
    
    def get_eigenvectors(self):
        """
        Get computed eigenvectors.
        
        Returns:
        --------
        list of numpy.ndarray
            List of eigenvector arrays
        """
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        
        cdef vector[VectorXd] eigenvecs = self._solver.get_eigenvectors()
        cdef int num_vecs = eigenvecs.size()
        
        result = []
        for i in range(num_vecs):
            cdef VectorXd vec = eigenvecs[i]
            cdef int n = vec.size()
            cdef cnp.ndarray[double, ndim=1] np_vec = np.empty(n, dtype=np.float64)
            
            for j in range(n):
                np_vec[j] = vec[j]
            
            result.append(np_vec)
        
        return result
    
    def get_num_eigenvalues(self):
        """Get number of computed eigenvalues"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        return self._solver.get_num_eigenvalues()
    
    def apply_open_system_boundary_conditions(self):
        """Apply open system boundary conditions with CAP"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.apply_open_system_boundary_conditions()
    
    def apply_dirac_delta_normalization(self):
        """Apply Dirac-delta normalization for scattering states"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.apply_dirac_delta_normalization()
    
    def configure_device_specific_solver(self, str device_type, double bias_voltage=0.0):
        """
        Configure solver for specific device type.
        
        Parameters:
        -----------
        device_type : str
            Device type: 'nanowire', 'quantum_dot', 'wide_channel', 'pn_junction'
        bias_voltage : float, optional
            Applied bias voltage in Volts (default: 0.0)
        """
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        
        cdef string cpp_device_type = device_type.encode('utf-8')
        self._solver.configure_device_specific_solver(cpp_device_type, bias_voltage)
    
    def set_solver_tolerance(self, double tolerance):
        """Set solver convergence tolerance"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.set_solver_tolerance(tolerance)
    
    def set_max_iterations(self, int max_iter):
        """Set maximum number of solver iterations"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.set_max_iterations(max_iter)
    
    def enable_cap_boundaries(self, bint enable):
        """Enable or disable CAP boundaries"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.enable_cap_boundaries(enable)
    
    def set_cap_parameters(self, double strength, double thickness):
        """
        Set CAP parameters.
        
        Parameters:
        -----------
        strength : float
            CAP absorption strength
        thickness : float
            CAP layer thickness as fraction of domain
        """
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.set_cap_parameters(strength, thickness)
    
    def set_energy_filter_range(self, double min_energy, double max_energy):
        """Set energy filtering range for eigenvalues"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.set_energy_filter_range(min_energy, max_energy)
    
    def enable_degeneracy_breaking(self, bint enable):
        """Enable or disable degeneracy breaking"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.enable_degeneracy_breaking(enable)
    
    def get_last_solve_time(self):
        """Get time taken for last solve operation"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        return self._solver.get_last_solve_time()
    
    def get_last_iteration_count(self):
        """Get number of iterations for last solve"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        return self._solver.get_last_iteration_count()
    
    def get_solver_status(self):
        """Get current solver status as string"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        cdef string status = self._solver.get_solver_status()
        return status.decode('utf-8')
    
    def clear_results(self):
        """Clear computed results and free memory"""
        if self._solver == NULL:
            raise RuntimeError("Solver not initialized")
        self._solver.clear_results()

# Utility functions for Cython module
def clear_all_callbacks():
    """Clear all stored callbacks to free memory"""
    global _mass_callbacks, _potential_callbacks
    _mass_callbacks.clear()
    _potential_callbacks.clear()

def get_callback_count():
    """Get number of active callbacks"""
    return len(_mass_callbacks) + len(_potential_callbacks)
