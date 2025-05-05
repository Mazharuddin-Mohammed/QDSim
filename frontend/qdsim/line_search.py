"""
Line search implementation for QDSim.

This module provides a line search implementation for use in the Poisson-Drift-Diffusion solver.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np

def line_search(f, x, dx, f_x, alpha_min=0.1, alpha_max=1.0, max_iter=5):
    """
    Perform a line search to find the optimal step size.
    
    Args:
        f: Function that returns the residual vector f(x)
        x: Current solution vector
        dx: Search direction
        f_x: Current residual vector f(x)
        alpha_min: Minimum step size
        alpha_max: Maximum step size
        max_iter: Maximum number of line search iterations
        
    Returns:
        alpha: Optimal step size
    """
    # Initial error
    error0 = np.linalg.norm(f_x)
    
    # Best step size and error
    best_alpha = 0.5  # Default damping factor
    best_error = error0
    
    print(f"Performing line search to find optimal step size")
    
    # Try different step sizes
    for i in range(max_iter):
        # Calculate alpha for this iteration
        alpha = alpha_min + (alpha_max - alpha_min) * i / (max_iter - 1)
        
        # Try this step size
        x_new = x + alpha * dx
        f_new = f(x_new)
        error = np.linalg.norm(f_new)
        
        print(f"  Line search iteration {i + 1}: alpha = {alpha:.3f}, error = {error:.6e}")
        
        # Update best step size if this one is better
        if error < best_error:
            best_error = error
            best_alpha = alpha
    
    print(f"Line search complete: using alpha = {best_alpha:.3f}")
    
    return best_alpha
