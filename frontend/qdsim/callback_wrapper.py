"""
Callback wrapper for C++ callbacks.

This module provides wrapper functions for C++ callbacks to handle
the different parameter types and ensure proper conversion between
Python and C++.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np

class CallbackWrapper:
    """
    Wrapper for C++ callbacks.
    
    This class provides wrapper functions for C++ callbacks to handle
    the different parameter types and ensure proper conversion between
    Python and C++.
    """
    
    @staticmethod
    def wrap_epsilon_r(func):
        """
        Wrap the epsilon_r callback function.
        
        Args:
            func: Original epsilon_r function with signature (x, y) -> float
            
        Returns:
            Wrapped function with signature (x, y, p_mat, n_mat) -> float
        """
        def wrapped(x, y, p_mat, n_mat):
            try:
                # Try to use the original function with just x and y
                return func(x, y)
            except Exception as e:
                # If that fails, try to extract epsilon_r from the material
                try:
                    # Use the p-side material's epsilon_r
                    if hasattr(p_mat, 'epsilon_r'):
                        return p_mat.epsilon_r
                    # If p_mat is a tuple, it might be (m_e, m_h, E_g, E_v, epsilon_r, ...)
                    elif isinstance(p_mat, tuple) and len(p_mat) >= 5:
                        return p_mat[4]  # epsilon_r is typically the 5th element
                    else:
                        # Default to GaAs
                        return 12.9
                except Exception as e2:
                    print(f"Warning: Error in epsilon_r wrapper: {e2}")
                    # Default to GaAs
                    return 12.9
        
        return wrapped
    
    @staticmethod
    def wrap_rho(func):
        """
        Wrap the rho callback function.
        
        Args:
            func: Original rho function with signature (x, y, n, p) -> float
            
        Returns:
            Wrapped function with signature (x, y, n, p) -> float
        """
        def wrapped(x, y, n, p):
            try:
                return func(x, y, n, p)
            except Exception as e:
                print(f"Warning: Error in rho wrapper: {e}")
                # Default to zero charge density
                return 0.0
        
        return wrapped
    
    @staticmethod
    def wrap_m_star(func):
        """
        Wrap the m_star callback function.
        
        Args:
            func: Original m_star function with signature (x, y) -> float
            
        Returns:
            Wrapped function with signature (x, y) -> float
        """
        def wrapped(x, y):
            try:
                return func(x, y)
            except Exception as e:
                print(f"Warning: Error in m_star wrapper: {e}")
                # Default to GaAs electron effective mass
                return 0.067
        
        return wrapped
    
    @staticmethod
    def wrap_potential(func):
        """
        Wrap the potential callback function.
        
        Args:
            func: Original potential function with signature (x, y) -> float
            
        Returns:
            Wrapped function with signature (x, y) -> float
        """
        def wrapped(x, y):
            try:
                return func(x, y)
            except Exception as e:
                print(f"Warning: Error in potential wrapper: {e}")
                # Default to zero potential
                return 0.0
        
        return wrapped
    
    @staticmethod
    def wrap_cap(func):
        """
        Wrap the cap callback function.
        
        Args:
            func: Original cap function with signature (x, y) -> float
            
        Returns:
            Wrapped function with signature (x, y) -> float
        """
        def wrapped(x, y):
            try:
                return func(x, y)
            except Exception as e:
                print(f"Warning: Error in cap wrapper: {e}")
                # Default to zero capacitance
                return 0.0
        
        return wrapped
    
    @staticmethod
    def wrap_n_conc(func):
        """
        Wrap the n_conc callback function.
        
        Args:
            func: Original n_conc function with signature (x, y, phi, phi_n) -> float
            
        Returns:
            Wrapped function with signature (x, y, phi, phi_n, material) -> float
        """
        def wrapped(x, y, phi, phi_n, material=None):
            try:
                if material is None:
                    return func(x, y, phi, phi_n)
                else:
                    # Ignore the material parameter
                    return func(x, y, phi, phi_n)
            except Exception as e:
                print(f"Warning: Error in n_conc wrapper: {e}")
                # Default to intrinsic concentration
                return 1.0e10
        
        return wrapped
    
    @staticmethod
    def wrap_p_conc(func):
        """
        Wrap the p_conc callback function.
        
        Args:
            func: Original p_conc function with signature (x, y, phi, phi_p) -> float
            
        Returns:
            Wrapped function with signature (x, y, phi, phi_p, material) -> float
        """
        def wrapped(x, y, phi, phi_p, material=None):
            try:
                if material is None:
                    return func(x, y, phi, phi_p)
                else:
                    # Ignore the material parameter
                    return func(x, y, phi, phi_p)
            except Exception as e:
                print(f"Warning: Error in p_conc wrapper: {e}")
                # Default to intrinsic concentration
                return 1.0e10
        
        return wrapped
