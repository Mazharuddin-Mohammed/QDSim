"""
Thomas algorithm implementation for QDSim.

This module provides an implementation of the Thomas algorithm for solving tridiagonal systems.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Solve a tridiagonal system using the Thomas algorithm.
    
    Args:
        a: Lower diagonal (first element is not used)
        b: Main diagonal
        c: Upper diagonal (last element is not used)
        d: Right-hand side
        
    Returns:
        x: Solution vector
    """
    n = len(d)
    
    # Create copies of the input arrays
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    
    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        denominator = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / denominator if i < n-1 else 0
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator
    
    # Back substitution
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x
