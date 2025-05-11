"""
Callback wrapper for C++ callbacks.

This module provides wrapper functions for C++ callbacks to handle
the different parameter types and ensure proper conversion between
Python and C++.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from typing import Callable, Dict, Any, Optional, Union, Tuple

try:
    import qdsim_cpp
except ImportError:
    qdsim_cpp = None

class CallbackWrapper:
    """
    Wrapper for C++ callbacks.

    This class provides wrapper functions for C++ callbacks to handle
    the different parameter types and ensure proper conversion between
    Python and C++.
    """

    def __init__(self):
        """Initialize the callback wrapper."""
        self.callbacks = {}

    def register_callback(self, name: str, callback: Callable) -> None:
        """
        Register a callback function.

        Args:
            name: The name of the callback.
            callback: The callback function.
        """
        self.callbacks[name] = callback

        # Register the callback with the C++ backend
        if qdsim_cpp is not None:
            try:
                qdsim_cpp.setCallback(name, callback)
            except Exception as e:
                print(f"Warning: Could not register callback {name} with C++ backend: {e}")
                print("This might be expected if the callback system is not fully implemented")

    def unregister_callback(self, name: str) -> None:
        """
        Unregister a callback function.

        Args:
            name: The name of the callback.
        """
        if name in self.callbacks:
            del self.callbacks[name]

            # Unregister the callback from the C++ backend
            if qdsim_cpp is not None:
                try:
                    qdsim_cpp.clearCallback(name)
                except Exception as e:
                    print(f"Warning: Could not unregister callback {name} from C++ backend: {e}")

    def unregister_all_callbacks(self) -> None:
        """Unregister all callback functions."""
        self.callbacks.clear()

        # Unregister all callbacks from the C++ backend
        if qdsim_cpp is not None:
            try:
                qdsim_cpp.clearCallbacks()
            except Exception as e:
                print(f"Warning: Could not unregister all callbacks from C++ backend: {e}")

    def get_callback(self, name: str) -> Optional[Callable]:
        """
        Get a callback function.

        Args:
            name: The name of the callback.

        Returns:
            The callback function, or None if not found.
        """
        return self.callbacks.get(name)

    def has_callback(self, name: str) -> bool:
        """
        Check if a callback function exists.

        Args:
            name: The name of the callback.

        Returns:
            True if the callback exists, False otherwise.
        """
        return name in self.callbacks

    def wrap_epsilon_r(self, func: Callable) -> Callable:
        """
        Wrap the epsilon_r callback function.

        Args:
            func: Original epsilon_r function with signature (x, y) -> float

        Returns:
            Wrapped function with signature (x, y, p_mat, n_mat) -> float
        """
        def wrapped(x, y, p_mat=None, n_mat=None):
            try:
                # Try to use the original function with all parameters
                try:
                    return func(x, y, p_mat, n_mat)
                except TypeError:
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

        # Register the wrapped function
        self.register_callback("epsilon_r", wrapped)

        return wrapped

    def wrap_rho(self, func: Callable) -> Callable:
        """
        Wrap the rho callback function.

        Args:
            func: Original rho function with signature (x, y, n, p) -> float

        Returns:
            Wrapped function with signature (x, y, n, p) -> float
        """
        def wrapped(x, y, n=None, p=None):
            try:
                # Try to call the function with all parameters
                try:
                    return func(x, y, n, p)
                except TypeError:
                    # Try to call the function with just x and y
                    return func(x, y)
            except Exception as e:
                print(f"Warning: Error in rho wrapper: {e}")
                # Default to zero charge density
                return 0.0

        # Register the wrapped function
        self.register_callback("rho", wrapped)

        return wrapped

    def wrap_m_star(self, func: Callable) -> Callable:
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

        # Register the wrapped function
        self.register_callback("m_star", wrapped)

        return wrapped

    def wrap_potential(self, func: Callable) -> Callable:
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

        # Register the wrapped function
        self.register_callback("potential", wrapped)

        return wrapped

    def wrap_cap(self, func: Callable) -> Callable:
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

        # Register the wrapped function
        self.register_callback("cap", wrapped)

        return wrapped

    def wrap_n_conc(self, func: Callable) -> Callable:
        """
        Wrap the n_conc callback function.

        Args:
            func: Original n_conc function with signature (x, y, phi, phi_n) -> float

        Returns:
            Wrapped function with signature (x, y, phi, phi_n, material) -> float
        """
        def wrapped(x, y, phi=0.0, phi_n=0.0, material=None):
            try:
                # Try different call signatures
                try:
                    if material is None:
                        return func(x, y, phi, phi_n)
                    else:
                        # Include the material parameter if available
                        return func(x, y, phi, phi_n, material)
                except TypeError:
                    # Try with fewer parameters
                    try:
                        return func(x, y, phi)
                    except TypeError:
                        # Try with just x and y
                        return func(x, y)
            except Exception as e:
                print(f"Warning: Error in n_conc wrapper: {e}")
                # Default to intrinsic concentration
                return 1.0e10

        # Register the wrapped function
        self.register_callback("n_conc", wrapped)

        return wrapped

    def wrap_p_conc(self, func: Callable) -> Callable:
        """
        Wrap the p_conc callback function.

        Args:
            func: Original p_conc function with signature (x, y, phi, phi_p) -> float

        Returns:
            Wrapped function with signature (x, y, phi, phi_p, material) -> float
        """
        def wrapped(x, y, phi=0.0, phi_p=0.0, material=None):
            try:
                # Try different call signatures
                try:
                    if material is None:
                        return func(x, y, phi, phi_p)
                    else:
                        # Include the material parameter if available
                        return func(x, y, phi, phi_p, material)
                except TypeError:
                    # Try with fewer parameters
                    try:
                        return func(x, y, phi)
                    except TypeError:
                        # Try with just x and y
                        return func(x, y)
            except Exception as e:
                print(f"Warning: Error in p_conc wrapper: {e}")
                # Default to intrinsic concentration
                return 1.0e10

        # Register the wrapped function
        self.register_callback("p_conc", wrapped)

        return wrapped

# Create a global instance of the callback wrapper
callback_wrapper = CallbackWrapper()

# Export the wrapper functions
wrap_epsilon_r = callback_wrapper.wrap_epsilon_r
wrap_rho = callback_wrapper.wrap_rho
wrap_m_star = callback_wrapper.wrap_m_star
wrap_potential = callback_wrapper.wrap_potential
wrap_cap = callback_wrapper.wrap_cap
wrap_n_conc = callback_wrapper.wrap_n_conc
wrap_p_conc = callback_wrapper.wrap_p_conc

# Export the callback management functions
register_callback = callback_wrapper.register_callback
unregister_callback = callback_wrapper.unregister_callback
unregister_all_callbacks = callback_wrapper.unregister_all_callbacks
get_callback = callback_wrapper.get_callback
has_callback = callback_wrapper.has_callback