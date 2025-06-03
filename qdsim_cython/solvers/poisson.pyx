# distutils: language = c++
# cython: language_level = 3

"""
Cython wrapper for PoissonSolver class

This module provides Python access to the C++ PoissonSolver class
for solving the Poisson equation in electrostatic calculations.
"""

import numpy as np
cimport numpy as cnp
from ..eigen cimport VectorXd, Vector2d
from ..core.mesh cimport Mesh

# Import C++ declarations
from .poisson cimport PoissonSolver as CppPoissonSolver
from .poisson cimport epsilon_r_func_t, rho_func_t

# Initialize NumPy
cnp.import_array()

# Global callback storage for function pointers
cdef dict _epsilon_r_callbacks = {}
cdef dict _rho_callbacks = {}
cdef int _callback_counter = 0

# C callback functions that call Python functions
cdef double epsilon_r_callback(double x, double y) with gil:
    """C callback for epsilon_r function."""
    try:
        # Get the current callback function
        if 0 in _epsilon_r_callbacks:
            return _epsilon_r_callbacks[0](x, y)
        else:
            return 12.9  # Default GaAs permittivity
    except:
        return 12.9  # Default on error

cdef double rho_callback(double x, double y, const VectorXd& n, const VectorXd& p) with gil:
    """C callback for charge density function."""
    try:
        # Get the current callback function
        if 0 in _rho_callbacks:
            # Convert C++ vectors to NumPy arrays
            cdef cnp.ndarray[double, ndim=1] n_array = np.empty(n.size(), dtype=np.float64)
            cdef cnp.ndarray[double, ndim=1] p_array = np.empty(p.size(), dtype=np.float64)
            
            cdef int i
            for i in range(n.size()):
                n_array[i] = n[i]
            for i in range(p.size()):
                p_array[i] = p[i]
                
            return _rho_callbacks[0](x, y, n_array, p_array)
        else:
            return 0.0  # Default charge density
    except:
        return 0.0  # Default on error

cdef class PoissonSolver:
    """
    Python wrapper for the C++ PoissonSolver class.
    
    Solves the Poisson equation for electrostatic calculations using
    the finite element method.
    
    The Poisson equation is:
    ∇·(εᵣ∇φ) = -ρ/ε₀
    
    where φ is the electrostatic potential, εᵣ is the relative permittivity,
    ρ is the charge density, and ε₀ is the vacuum permittivity.
    """
    
    cdef CppPoissonSolver* _solver
    cdef bool _owns_solver
    cdef object _epsilon_r_func
    cdef object _rho_func
    cdef int _callback_id
    
    def __cinit__(self, Mesh mesh, epsilon_r_func, rho_func):
        """Initialize the Poisson solver."""
        global _callback_counter
        
        # Store Python callback functions
        self._epsilon_r_func = epsilon_r_func
        self._rho_func = rho_func
        self._callback_id = _callback_counter
        _callback_counter += 1
        
        # Register callbacks
        _epsilon_r_callbacks[self._callback_id] = epsilon_r_func
        _rho_callbacks[self._callback_id] = rho_func
        
        # Create C++ solver with C callback functions
        self._solver = new CppPoissonSolver(mesh._mesh[0], epsilon_r_callback, rho_callback)
        self._owns_solver = True
        
    def __dealloc__(self):
        """Clean up the C++ solver object."""
        if self._owns_solver and self._solver is not NULL:
            del self._solver
            
        # Clean up callbacks
        if self._callback_id in _epsilon_r_callbacks:
            del _epsilon_r_callbacks[self._callback_id]
        if self._callback_id in _rho_callbacks:
            del _rho_callbacks[self._callback_id]
            
    def solve(self, double V_p, double V_n, n_array=None, p_array=None):
        """
        Solve the Poisson equation.
        
        Parameters
        ----------
        V_p : float
            Potential at the p-type boundary in volts
        V_n : float
            Potential at the n-type boundary in volts
        n_array : array-like, optional
            Electron concentration at each node
        p_array : array-like, optional
            Hole concentration at each node
        """
        if n_array is not None and p_array is not None:
            # Convert NumPy arrays to C++ VectorXd
            cdef cnp.ndarray[double, ndim=1] n_np = np.asarray(n_array, dtype=np.float64)
            cdef cnp.ndarray[double, ndim=1] p_np = np.asarray(p_array, dtype=np.float64)
            
            cdef VectorXd n_cpp, p_cpp
            n_cpp.resize(n_np.shape[0])
            p_cpp.resize(p_np.shape[0])
            
            cdef int i
            for i in range(n_np.shape[0]):
                n_cpp[i] = n_np[i]
            for i in range(p_np.shape[0]):
                p_cpp[i] = p_np[i]
                
            # Solve with carrier concentrations
            self._solver.solve(V_p, V_n, n_cpp, p_cpp)
        else:
            # Solve without carrier concentrations
            self._solver.solve(V_p, V_n)
            
    def set_potential(self, potential_array):
        """
        Set the potential values directly.
        
        Parameters
        ----------
        potential_array : array-like
            Potential values to set
        """
        cdef cnp.ndarray[double, ndim=1] pot_np = np.asarray(potential_array, dtype=np.float64)
        cdef VectorXd pot_cpp
        pot_cpp.resize(pot_np.shape[0])
        
        cdef int i
        for i in range(pot_np.shape[0]):
            pot_cpp[i] = pot_np[i]
            
        self._solver.set_potential(pot_cpp)
        
    def update_and_solve(self, potential_array, double V_p, double V_n, 
                        n_array, p_array):
        """
        Update potential values and solve the Poisson equation.
        
        Parameters
        ----------
        potential_array : array-like
            Potential values to set
        V_p : float
            Potential at the p-type boundary in volts
        V_n : float
            Potential at the n-type boundary in volts
        n_array : array-like
            Electron concentration at each node
        p_array : array-like
            Hole concentration at each node
        """
        # Convert arrays to C++ vectors
        cdef cnp.ndarray[double, ndim=1] pot_np = np.asarray(potential_array, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] n_np = np.asarray(n_array, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] p_np = np.asarray(p_array, dtype=np.float64)
        
        cdef VectorXd pot_cpp, n_cpp, p_cpp
        pot_cpp.resize(pot_np.shape[0])
        n_cpp.resize(n_np.shape[0])
        p_cpp.resize(p_np.shape[0])
        
        cdef int i
        for i in range(pot_np.shape[0]):
            pot_cpp[i] = pot_np[i]
        for i in range(n_np.shape[0]):
            n_cpp[i] = n_np[i]
        for i in range(p_np.shape[0]):
            p_cpp[i] = p_np[i]
            
        self._solver.update_and_solve(pot_cpp, V_p, V_n, n_cpp, p_cpp)
        
    def initialize(self, Mesh mesh, epsilon_r_func, rho_func):
        """
        Initialize the solver with a new mesh and functions.
        
        Parameters
        ----------
        mesh : Mesh
            The mesh to use
        epsilon_r_func : callable
            Function that returns relative permittivity at (x, y)
        rho_func : callable
            Function that returns charge density at (x, y, n, p)
        """
        # Update callback functions
        self._epsilon_r_func = epsilon_r_func
        self._rho_func = rho_func
        _epsilon_r_callbacks[self._callback_id] = epsilon_r_func
        _rho_callbacks[self._callback_id] = rho_func
        
        # Reinitialize C++ solver
        self._solver.initialize(mesh._mesh[0], epsilon_r_callback, rho_callback)
        
    def set_charge_density(self, charge_density_array):
        """
        Set the charge density values directly.
        
        Parameters
        ----------
        charge_density_array : array-like
            Charge density values to set
        """
        cdef cnp.ndarray[double, ndim=1] rho_np = np.asarray(charge_density_array, dtype=np.float64)
        cdef VectorXd rho_cpp
        rho_cpp.resize(rho_np.shape[0])
        
        cdef int i
        for i in range(rho_np.shape[0]):
            rho_cpp[i] = rho_np[i]
            
        self._solver.set_charge_density(rho_cpp)
        
    @property
    def potential(self):
        """Get the computed electrostatic potential as a NumPy array."""
        cdef const VectorXd& phi_cpp = self._solver.get_potential()
        cdef int size = phi_cpp.size()
        
        # Create NumPy array
        cdef cnp.ndarray[double, ndim=1] phi_array = np.empty(size, dtype=np.float64)
        
        # Copy data from C++ to NumPy
        cdef int i
        for i in range(size):
            phi_array[i] = phi_cpp[i]
            
        return phi_array
        
    def get_electric_field(self, double x, double y):
        """
        Compute the electric field at a given position.
        
        Parameters
        ----------
        x : float
            x-coordinate in nanometers
        y : float
            y-coordinate in nanometers
            
        Returns
        -------
        numpy.ndarray
            Electric field vector [Ex, Ey] in V/nm
        """
        cdef Vector2d E_cpp = self._solver.get_electric_field(x, y)
        return np.array([E_cpp.x(), E_cpp.y()])
        
    def get_potential(self):
        """
        Get the computed electrostatic potential.
        
        Returns
        -------
        numpy.ndarray
            Potential values at mesh nodes in volts
        """
        return self.potential
