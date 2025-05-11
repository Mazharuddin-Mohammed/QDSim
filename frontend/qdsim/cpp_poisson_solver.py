"""
Python wrapper for the C++ PoissonSolver class.

This module provides a Python wrapper for the C++ PoissonSolver class,
allowing it to be used with the same interface as the Python PoissonSolver classes.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import sys
import os

# Try to import the C++ extension
try:
    from . import qdsim_cpp
except ImportError:
    # Try to import from the build directory
    build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'build')
    if os.path.exists(build_dir):
        sys.path.append(build_dir)
        try:
            import qdsim_cpp
        except ImportError:
            qdsim_cpp = None
    else:
        qdsim_cpp = None

class PoissonSolver:
    """
    Python wrapper for the C++ PoissonSolver class.
    
    This class provides a Python interface to the C++ PoissonSolver class,
    allowing it to be used with the same interface as the Python PoissonSolver classes.
    """
    
    def __init__(self, mesh, epsilon_r_func, rho_func):
        """
        Initialize the PoissonSolver.
        
        Args:
            mesh: Mesh object
            epsilon_r_func: Function that returns the relative permittivity at a given position
            rho_func: Function that returns the charge density at a given position
        """
        if qdsim_cpp is None:
            raise ImportError("C++ extension not available")
        
        self.mesh = mesh
        self.epsilon_r_func = epsilon_r_func
        self.rho_func = rho_func
        
        # Register the callbacks with the C++ code
        try:
            qdsim_cpp.setCallback("epsilon_r", epsilon_r_func)
            qdsim_cpp.setCallback("rho", rho_func)
        except Exception as e:
            print(f"Warning: Could not register callbacks: {e}")
            print("This might be expected if the callback system is not fully implemented")
        
        # Create the C++ PoissonSolver
        try:
            self.solver = qdsim_cpp.PoissonSolver(mesh, epsilon_r_func, rho_func)
        except Exception as e:
            raise RuntimeError(f"Failed to create C++ PoissonSolver: {e}")
        
        # Initialize the potential array
        self.phi = np.zeros(mesh.get_num_nodes())
    
    def solve(self, V_p, V_n, n=None, p=None):
        """
        Solve the Poisson equation.
        
        Args:
            V_p: Potential at the p-side (V)
            V_n: Potential at the n-side (V)
            n: Electron concentration array (optional)
            p: Hole concentration array (optional)
            
        Returns:
            Potential array (V)
        """
        try:
            if n is not None and p is not None:
                # Solve with carrier concentrations
                self.solver.solve(V_p, V_n, n, p)
            else:
                # Solve without carrier concentrations
                self.solver.solve(V_p, V_n)
            
            # Get the potential
            self.phi = np.array(self.solver.get_potential())
            
            return self.phi
        except Exception as e:
            raise RuntimeError(f"Failed to solve Poisson equation: {e}")
    
    def set_potential(self, potential):
        """
        Set the potential values directly.
        
        Args:
            potential: Potential array (V)
        """
        try:
            self.solver.set_potential(potential)
            self.phi = np.array(potential)
        except Exception as e:
            raise RuntimeError(f"Failed to set potential: {e}")
    
    def set_charge_density(self, charge_density):
        """
        Set the charge density values directly.
        
        Args:
            charge_density: Charge density array (C/m^3)
        """
        try:
            self.solver.set_charge_density(charge_density)
        except Exception as e:
            raise RuntimeError(f"Failed to set charge density: {e}")
    
    def get_potential(self):
        """
        Get the computed electrostatic potential.
        
        Returns:
            Potential array (V)
        """
        return self.phi
    
    def get_electric_field(self, x, y):
        """
        Get the electric field at a given position.
        
        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)
            
        Returns:
            Electric field vector (V/m)
        """
        try:
            return self.solver.get_electric_field(x, y)
        except Exception as e:
            raise RuntimeError(f"Failed to get electric field: {e}")
