"""
Non-linear solver module for QDSim.

This module provides non-linear solvers for use in the Poisson-Drift-Diffusion solver.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

# Try to import GPU acceleration
try:
    from .gpu_fallback import is_gpu_available, cp, to_gpu, to_cpu
    has_gpu = is_gpu_available()
except ImportError:
    has_gpu = False
    
    def to_gpu(x):
        return x
        
    def to_cpu(x):
        return x

class NewtonSolver:
    """
    Newton's method for solving non-linear systems of equations.
    """
    
    def __init__(self, max_iter=20, tolerance=1e-6, damping_factor=0.5, line_search=True):
        """
        Initialize the Newton solver.
        
        Args:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            damping_factor: Damping factor for better convergence
            line_search: Whether to use line search
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.damping_factor = damping_factor
        self.line_search = line_search
        self.use_gpu = has_gpu
        
    def solve(self, F, J, x0, args=()):
        """
        Solve the non-linear system F(x) = 0.
        
        Args:
            F: Function that returns the residual vector F(x)
            J: Function that returns the Jacobian matrix J(x)
            x0: Initial guess
            args: Additional arguments to pass to F and J
            
        Returns:
            x: Solution vector
            info: Dictionary with information about the solution process
        """
        # Initialize
        x = np.copy(x0)
        iter_count = 0
        error = float('inf')
        
        # Track convergence history
        error_history = []
        
        # Start timing
        start_time = time.time()
        
        # Line search parameters
        line_search_max_iter = 5
        line_search_alpha_min = 0.1
        line_search_alpha_max = 1.0
        
        # Main iteration loop
        while error > self.tolerance and iter_count < self.max_iter:
            # Evaluate residual and Jacobian
            f = F(x, *args)
            jac = J(x, *args)
            
            # Compute the error
            error = np.linalg.norm(f)
            error_history.append(error)
            
            # Print iteration info
            elapsed_time = time.time() - start_time
            print(f"Newton iteration {iter_count + 1}: error = {error:.6e}, elapsed time = {elapsed_time:.2f}s")
            
            # Check for convergence
            if error < self.tolerance:
                break
                
            # Solve the linear system J(x) * dx = -F(x)
            try:
                # Convert to sparse matrix for better performance
                if isinstance(jac, np.ndarray):
                    jac_sparse = csr_matrix(jac)
                else:
                    jac_sparse = jac
                    
                # Solve the linear system
                dx = spsolve(jac_sparse, -f)
            except Exception as e:
                print(f"Error solving linear system: {e}")
                # Fallback to dense solver
                dx = np.linalg.solve(jac, -f)
            
            # Perform line search if enabled
            if self.line_search:
                alpha = self._line_search(F, x, dx, f, args, line_search_max_iter, line_search_alpha_min, line_search_alpha_max)
            else:
                alpha = self.damping_factor
            
            # Update solution
            x = x + alpha * dx
            
            # Increment iteration counter
            iter_count += 1
        
        # Check if the solution converged
        converged = error <= self.tolerance
        
        # Create info dictionary
        info = {
            'converged': converged,
            'iterations': iter_count,
            'error': error,
            'error_history': error_history,
            'elapsed_time': time.time() - start_time
        }
        
        return x, info
    
    def _line_search(self, F, x, dx, f, args, max_iter, alpha_min, alpha_max):
        """
        Perform line search to find the optimal step size.
        
        Args:
            F: Function that returns the residual vector F(x)
            x: Current solution vector
            dx: Search direction
            f: Current residual vector F(x)
            args: Additional arguments to pass to F
            max_iter: Maximum number of line search iterations
            alpha_min: Minimum step size
            alpha_max: Maximum step size
            
        Returns:
            alpha: Optimal step size
        """
        # Initial error
        error0 = np.linalg.norm(f)
        
        # Best step size and error
        best_alpha = self.damping_factor
        best_error = error0
        
        print(f"Performing line search to find optimal step size")
        
        # Try different step sizes
        for i in range(max_iter):
            # Calculate alpha for this iteration
            alpha = alpha_min + (alpha_max - alpha_min) * i / (max_iter - 1)
            
            # Try this step size
            x_new = x + alpha * dx
            f_new = F(x_new, *args)
            error = np.linalg.norm(f_new)
            
            print(f"  Line search iteration {i + 1}: alpha = {alpha:.3f}, error = {error:.6e}")
            
            # Update best step size if this one is better
            if error < best_error:
                best_error = error
                best_alpha = alpha
        
        print(f"Line search complete: using alpha = {best_alpha:.3f}")
        
        return best_alpha

class AndersonSolver:
    """
    Anderson acceleration for solving non-linear systems of equations.
    """
    
    def __init__(self, max_iter=20, tolerance=1e-6, damping_factor=0.5, m=5):
        """
        Initialize the Anderson solver.
        
        Args:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            damping_factor: Damping factor for better convergence
            m: Number of previous iterations to use for acceleration
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.damping_factor = damping_factor
        self.m = m
        self.use_gpu = has_gpu
        
    def solve(self, G, x0, args=()):
        """
        Solve the fixed-point problem x = G(x).
        
        Args:
            G: Function that returns the next iterate G(x)
            x0: Initial guess
            args: Additional arguments to pass to G
            
        Returns:
            x: Solution vector
            info: Dictionary with information about the solution process
        """
        # Initialize
        x = np.copy(x0)
        n = len(x)
        iter_count = 0
        error = float('inf')
        
        # Track convergence history
        error_history = []
        
        # Start timing
        start_time = time.time()
        
        # Storage for previous iterations
        X = np.zeros((n, self.m))  # Previous iterates
        F = np.zeros((n, self.m))  # Previous residuals
        
        # Main iteration loop
        while error > self.tolerance and iter_count < self.max_iter:
            # Evaluate G(x)
            g = G(x, *args)
            
            # Compute the residual
            f = g - x
            
            # Compute the error
            error = np.linalg.norm(f)
            error_history.append(error)
            
            # Print iteration info
            elapsed_time = time.time() - start_time
            print(f"Anderson iteration {iter_count + 1}: error = {error:.6e}, elapsed time = {elapsed_time:.2f}s")
            
            # Check for convergence
            if error < self.tolerance:
                break
            
            # Determine the number of previous iterations to use
            k = min(iter_count, self.m)
            
            if k > 0:
                # Update storage
                if k == self.m:
                    # Shift columns to the left
                    X[:, 0:self.m-1] = X[:, 1:self.m]
                    F[:, 0:self.m-1] = F[:, 1:self.m]
                
                # Store current iterate and residual
                X[:, k-1] = x
                F[:, k-1] = f
                
                # Compute the weights
                try:
                    # Compute the Gram matrix
                    gram = np.zeros((k, k))
                    for i in range(k):
                        for j in range(k):
                            gram[i, j] = np.dot(F[:, i], F[:, j])
                    
                    # Solve for the weights
                    weights = np.linalg.solve(gram, np.ones(k))
                    weights = weights / np.sum(weights)
                    
                    # Compute the new iterate
                    x_new = np.zeros(n)
                    for i in range(k):
                        x_new += weights[i] * (X[:, i] + self.damping_factor * F[:, i])
                except Exception as e:
                    print(f"Error in Anderson acceleration: {e}")
                    # Fallback to simple damping
                    x_new = x + self.damping_factor * f
            else:
                # First iteration, just use damping
                x_new = x + self.damping_factor * f
            
            # Update solution
            x = x_new
            
            # Increment iteration counter
            iter_count += 1
        
        # Check if the solution converged
        converged = error <= self.tolerance
        
        # Create info dictionary
        info = {
            'converged': converged,
            'iterations': iter_count,
            'error': error,
            'error_history': error_history,
            'elapsed_time': time.time() - start_time
        }
        
        return x, info

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
