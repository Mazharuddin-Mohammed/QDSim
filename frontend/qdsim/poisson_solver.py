"""
Poisson Solver Module for QDSim.

This module provides classes and functions for solving Poisson's equation
in 1D and 2D domains, with support for arbitrary charge distributions
and boundary conditions.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class PoissonSolver1D:
    """
    Class for solving Poisson's equation in 1D.
    
    This class solves the 1D Poisson equation:
    d^2V/dx^2 = -rho/epsilon
    
    with Dirichlet or Neumann boundary conditions.
    """
    
    def __init__(self, x_min, x_max, nx, epsilon_r=1.0, bc_type='dirichlet'):
        """
        Initialize the 1D Poisson solver.
        
        Args:
            x_min: Minimum x-coordinate (m)
            x_max: Maximum x-coordinate (m)
            nx: Number of grid points
            epsilon_r: Relative permittivity
            bc_type: Boundary condition type ('dirichlet' or 'neumann')
        """
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        self.dx = (x_max - x_min) / (nx - 1)
        self.x = np.linspace(x_min, x_max, nx)
        self.epsilon_r = epsilon_r
        self.epsilon_0 = 8.85418782e-12  # Vacuum permittivity (F/m)
        self.epsilon = self.epsilon_r * self.epsilon_0
        self.bc_type = bc_type.lower()
        
        # Initialize the system matrix and right-hand side
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the system matrix for the Poisson equation."""
        # Create the system matrix
        if self.bc_type == 'dirichlet':
            # For Dirichlet boundary conditions, we fix the values at the boundaries
            # and solve for the interior points
            n = self.nx - 2
            diagonals = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
            offsets = [-1, 0, 1]
            self.A = sp.diags(diagonals, offsets, shape=(n, n), format='csr')
            self.A = self.A / self.dx**2
            
            # Initialize the right-hand side
            self.b = np.zeros(n)
            
            # Initialize the solution vector
            self.V = np.zeros(self.nx)
        elif self.bc_type == 'neumann':
            # For Neumann boundary conditions, we include the boundary points
            # in the system and apply the boundary conditions as constraints
            n = self.nx
            diagonals = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
            offsets = [-1, 0, 1]
            self.A = sp.diags(diagonals, offsets, shape=(n, n), format='csr')
            
            # Apply Neumann boundary conditions
            # Left boundary: dV/dx = 0 => V[0] = V[1]
            self.A[0, 0] = -1
            self.A[0, 1] = 1
            
            # Right boundary: dV/dx = 0 => V[n-1] = V[n-2]
            self.A[n-1, n-1] = -1
            self.A[n-1, n-2] = 1
            
            # Initialize the right-hand side
            self.b = np.zeros(n)
            
            # Initialize the solution vector
            self.V = np.zeros(self.nx)
        else:
            raise ValueError(f"Unsupported boundary condition type: {self.bc_type}")
    
    def set_boundary_conditions(self, left_value, right_value):
        """
        Set the boundary conditions.
        
        Args:
            left_value: Value at the left boundary (V)
            right_value: Value at the right boundary (V)
        """
        if self.bc_type == 'dirichlet':
            # Set the boundary values
            self.V[0] = left_value
            self.V[-1] = right_value
            
            # Update the right-hand side to account for the boundary values
            self.b[0] = -left_value / self.dx**2
            self.b[-1] = -right_value / self.dx**2
        elif self.bc_type == 'neumann':
            # For Neumann boundary conditions, the boundary values are the derivatives
            # We need to modify the right-hand side
            self.b[0] = left_value
            self.b[-1] = right_value
    
    def set_charge_density(self, rho):
        """
        Set the charge density.
        
        Args:
            rho: Charge density array (C/m^3)
        """
        if len(rho) != self.nx:
            raise ValueError(f"Charge density array length ({len(rho)}) does not match grid size ({self.nx})")
        
        if self.bc_type == 'dirichlet':
            # Update the right-hand side for interior points
            self.b = -rho[1:-1] / self.epsilon
        elif self.bc_type == 'neumann':
            # Update the right-hand side for all points except boundaries
            self.b[1:-1] = -rho[1:-1] / self.epsilon
    
    def solve(self):
        """
        Solve the Poisson equation.
        
        Returns:
            Potential array (V)
        """
        if self.bc_type == 'dirichlet':
            # Solve the system for interior points
            V_interior = spla.spsolve(self.A, self.b)
            
            # Update the solution vector
            self.V[1:-1] = V_interior
        elif self.bc_type == 'neumann':
            # Solve the system for all points
            self.V = spla.spsolve(self.A, self.b)
        
        return self.V
    
    def get_electric_field(self):
        """
        Calculate the electric field from the potential.
        
        Returns:
            Electric field array (V/m)
        """
        # Calculate the electric field as the negative gradient of the potential
        E = np.zeros(self.nx)
        
        # Use central differences for interior points
        E[1:-1] = -(self.V[2:] - self.V[:-2]) / (2 * self.dx)
        
        # Use forward/backward differences for boundary points
        E[0] = -(self.V[1] - self.V[0]) / self.dx
        E[-1] = -(self.V[-1] - self.V[-2]) / self.dx
        
        return E


class PoissonSolver2D:
    """
    Class for solving Poisson's equation in 2D.
    
    This class solves the 2D Poisson equation:
    d^2V/dx^2 + d^2V/dy^2 = -rho/epsilon
    
    with Dirichlet or Neumann boundary conditions.
    """
    
    def __init__(self, x_min, x_max, nx, y_min, y_max, ny, epsilon_r=1.0, bc_type='dirichlet'):
        """
        Initialize the 2D Poisson solver.
        
        Args:
            x_min: Minimum x-coordinate (m)
            x_max: Maximum x-coordinate (m)
            nx: Number of grid points in x-direction
            y_min: Minimum y-coordinate (m)
            y_max: Maximum y-coordinate (m)
            ny: Number of grid points in y-direction
            epsilon_r: Relative permittivity
            bc_type: Boundary condition type ('dirichlet' or 'neumann')
        """
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        self.dx = (x_max - x_min) / (nx - 1)
        self.x = np.linspace(x_min, x_max, nx)
        
        self.y_min = y_min
        self.y_max = y_max
        self.ny = ny
        self.dy = (y_max - y_min) / (ny - 1)
        self.y = np.linspace(y_min, y_max, ny)
        
        self.epsilon_r = epsilon_r
        self.epsilon_0 = 8.85418782e-12  # Vacuum permittivity (F/m)
        self.epsilon = self.epsilon_r * self.epsilon_0
        self.bc_type = bc_type.lower()
        
        # Create 2D grid
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize the system matrix and right-hand side
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the system matrix for the Poisson equation."""
        if self.bc_type == 'dirichlet':
            # For Dirichlet boundary conditions, we fix the values at the boundaries
            # and solve for the interior points
            nx_interior = self.nx - 2
            ny_interior = self.ny - 2
            n = nx_interior * ny_interior
            
            # Create the system matrix using the 5-point stencil
            main_diag = -2 * (1/self.dx**2 + 1/self.dy**2) * np.ones(n)
            x_diag = np.ones(n) / self.dx**2
            y_diag = np.ones(n) / self.dy**2
            
            # Adjust the diagonals for the boundary conditions
            diagonals = [main_diag, x_diag, x_diag, y_diag, y_diag]
            offsets = [0, -1, 1, -nx_interior, nx_interior]
            
            # Create the sparse matrix
            self.A = sp.diags(diagonals, offsets, shape=(n, n), format='csr')
            
            # Initialize the right-hand side
            self.b = np.zeros(n)
            
            # Initialize the solution vector
            self.V = np.zeros((self.ny, self.nx))
        elif self.bc_type == 'neumann':
            # For Neumann boundary conditions, we include the boundary points
            # in the system and apply the boundary conditions as constraints
            n = self.nx * self.ny
            
            # Create the system matrix using the 5-point stencil
            main_diag = -2 * (1/self.dx**2 + 1/self.dy**2) * np.ones(n)
            x_diag = np.ones(n) / self.dx**2
            y_diag = np.ones(n) / self.dy**2
            
            # Create the sparse matrix
            self.A = sp.diags([main_diag, x_diag, x_diag, y_diag, y_diag],
                             [0, -1, 1, -self.nx, self.nx],
                             shape=(n, n), format='lil')
            
            # Apply Neumann boundary conditions
            # Left boundary: dV/dx = 0 => V[i,0] = V[i,1]
            for i in range(self.ny):
                idx = i * self.nx
                self.A[idx, idx] = -1
                self.A[idx, idx+1] = 1
            
            # Right boundary: dV/dx = 0 => V[i,nx-1] = V[i,nx-2]
            for i in range(self.ny):
                idx = i * self.nx + self.nx - 1
                self.A[idx, idx] = -1
                self.A[idx, idx-1] = 1
            
            # Bottom boundary: dV/dy = 0 => V[0,j] = V[1,j]
            for j in range(self.nx):
                idx = j
                self.A[idx, idx] = -1
                self.A[idx, idx+self.nx] = 1
            
            # Top boundary: dV/dy = 0 => V[ny-1,j] = V[ny-2,j]
            for j in range(self.nx):
                idx = (self.ny - 1) * self.nx + j
                self.A[idx, idx] = -1
                self.A[idx, idx-self.nx] = 1
            
            # Convert to CSR format for efficient solving
            self.A = self.A.tocsr()
            
            # Initialize the right-hand side
            self.b = np.zeros(n)
            
            # Initialize the solution vector
            self.V = np.zeros((self.ny, self.nx))
        else:
            raise ValueError(f"Unsupported boundary condition type: {self.bc_type}")
    
    def set_boundary_conditions(self, boundary_values):
        """
        Set the boundary conditions.
        
        Args:
            boundary_values: Dictionary with keys 'left', 'right', 'bottom', 'top'
                             and values as arrays or constants
        """
        if self.bc_type == 'dirichlet':
            # Set the boundary values
            if 'left' in boundary_values:
                self.V[:, 0] = boundary_values['left']
            if 'right' in boundary_values:
                self.V[:, -1] = boundary_values['right']
            if 'bottom' in boundary_values:
                self.V[0, :] = boundary_values['bottom']
            if 'top' in boundary_values:
                self.V[-1, :] = boundary_values['top']
            
            # Update the right-hand side to account for the boundary values
            nx_interior = self.nx - 2
            ny_interior = self.ny - 2
            
            # Reshape the right-hand side to match the interior grid
            b_reshaped = self.b.reshape(ny_interior, nx_interior)
            
            # Left boundary
            b_reshaped[:, 0] -= self.V[1:-1, 0] / self.dx**2
            
            # Right boundary
            b_reshaped[:, -1] -= self.V[1:-1, -1] / self.dx**2
            
            # Bottom boundary
            b_reshaped[0, :] -= self.V[0, 1:-1] / self.dy**2
            
            # Top boundary
            b_reshaped[-1, :] -= self.V[-1, 1:-1] / self.dy**2
            
            # Reshape back to 1D
            self.b = b_reshaped.flatten()
        elif self.bc_type == 'neumann':
            # For Neumann boundary conditions, the boundary values are the derivatives
            # We need to modify the right-hand side
            if 'left' in boundary_values:
                for i in range(self.ny):
                    idx = i * self.nx
                    self.b[idx] = boundary_values['left']
            
            if 'right' in boundary_values:
                for i in range(self.ny):
                    idx = i * self.nx + self.nx - 1
                    self.b[idx] = boundary_values['right']
            
            if 'bottom' in boundary_values:
                for j in range(self.nx):
                    idx = j
                    self.b[idx] = boundary_values['bottom']
            
            if 'top' in boundary_values:
                for j in range(self.nx):
                    idx = (self.ny - 1) * self.nx + j
                    self.b[idx] = boundary_values['top']
    
    def set_charge_density(self, rho):
        """
        Set the charge density.
        
        Args:
            rho: Charge density array (C/m^3) with shape (ny, nx)
        """
        if rho.shape != (self.ny, self.nx):
            raise ValueError(f"Charge density array shape {rho.shape} does not match grid shape {(self.ny, self.nx)}")
        
        if self.bc_type == 'dirichlet':
            # Update the right-hand side for interior points
            self.b = -rho[1:-1, 1:-1].flatten() / self.epsilon
        elif self.bc_type == 'neumann':
            # Update the right-hand side for all points except boundaries
            rho_flat = rho.flatten()
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    idx = i * self.nx + j
                    self.b[idx] = -rho_flat[idx] / self.epsilon
    
    def solve(self):
        """
        Solve the Poisson equation.
        
        Returns:
            Potential array (V) with shape (ny, nx)
        """
        if self.bc_type == 'dirichlet':
            # Solve the system for interior points
            V_interior = spla.spsolve(self.A, self.b)
            
            # Reshape the solution to match the interior grid
            V_interior_reshaped = V_interior.reshape(self.ny-2, self.nx-2)
            
            # Update the solution vector
            self.V[1:-1, 1:-1] = V_interior_reshaped
        elif self.bc_type == 'neumann':
            # Solve the system for all points
            V_flat = spla.spsolve(self.A, self.b)
            
            # Reshape the solution to match the grid
            self.V = V_flat.reshape(self.ny, self.nx)
            
            # Normalize the potential (Neumann problem is only determined up to a constant)
            self.V = self.V - np.mean(self.V)
        
        return self.V
    
    def get_electric_field(self):
        """
        Calculate the electric field from the potential.
        
        Returns:
            Electric field components (Ex, Ey) with shape (ny, nx)
        """
        # Calculate the electric field as the negative gradient of the potential
        Ex = np.zeros((self.ny, self.nx))
        Ey = np.zeros((self.ny, self.nx))
        
        # Use central differences for interior points
        Ex[:, 1:-1] = -(self.V[:, 2:] - self.V[:, :-2]) / (2 * self.dx)
        Ey[1:-1, :] = -(self.V[2:, :] - self.V[:-2, :]) / (2 * self.dy)
        
        # Use forward/backward differences for boundary points
        Ex[:, 0] = -(self.V[:, 1] - self.V[:, 0]) / self.dx
        Ex[:, -1] = -(self.V[:, -1] - self.V[:, -2]) / self.dx
        Ey[0, :] = -(self.V[1, :] - self.V[0, :]) / self.dy
        Ey[-1, :] = -(self.V[-1, :] - self.V[-2, :]) / self.dy
        
        return Ex, Ey


class NonlinearPoissonSolver2D:
    """
    Class for solving the nonlinear Poisson equation in 2D.
    
    This class solves the nonlinear Poisson equation:
    d^2V/dx^2 + d^2V/dy^2 = -rho(V)/epsilon
    
    where rho(V) is a nonlinear function of the potential V,
    with Dirichlet or Neumann boundary conditions.
    """
    
    def __init__(self, x_min, x_max, nx, y_min, y_max, ny, epsilon_r=1.0, bc_type='dirichlet'):
        """
        Initialize the nonlinear Poisson solver.
        
        Args:
            x_min: Minimum x-coordinate (m)
            x_max: Maximum x-coordinate (m)
            nx: Number of grid points in x-direction
            y_min: Minimum y-coordinate (m)
            y_max: Maximum y-coordinate (m)
            ny: Number of grid points in y-direction
            epsilon_r: Relative permittivity
            bc_type: Boundary condition type ('dirichlet' or 'neumann')
        """
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        self.dx = (x_max - x_min) / (nx - 1)
        self.x = np.linspace(x_min, x_max, nx)
        
        self.y_min = y_min
        self.y_max = y_max
        self.ny = ny
        self.dy = (y_max - y_min) / (ny - 1)
        self.y = np.linspace(y_min, y_max, ny)
        
        self.epsilon_r = epsilon_r
        self.epsilon_0 = 8.85418782e-12  # Vacuum permittivity (F/m)
        self.epsilon = self.epsilon_r * self.epsilon_0
        self.bc_type = bc_type.lower()
        
        # Create 2D grid
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize the linear Poisson solver
        self.linear_solver = PoissonSolver2D(x_min, x_max, nx, y_min, y_max, ny, epsilon_r, bc_type)
        
        # Initialize the solution vector
        self.V = np.zeros((self.ny, self.nx))
        
        # Initialize the charge density function
        self.rho_func = None
        
        # Solver parameters
        self.max_iter = 100
        self.tolerance = 1e-6
        self.damping = 0.5  # Damping factor for Newton iterations
    
    def set_boundary_conditions(self, boundary_values):
        """
        Set the boundary conditions.
        
        Args:
            boundary_values: Dictionary with keys 'left', 'right', 'bottom', 'top'
                             and values as arrays or constants
        """
        self.linear_solver.set_boundary_conditions(boundary_values)
        
        # Set the boundary values in the solution vector
        if self.bc_type == 'dirichlet':
            if 'left' in boundary_values:
                self.V[:, 0] = boundary_values['left']
            if 'right' in boundary_values:
                self.V[:, -1] = boundary_values['right']
            if 'bottom' in boundary_values:
                self.V[0, :] = boundary_values['bottom']
            if 'top' in boundary_values:
                self.V[-1, :] = boundary_values['top']
    
    def set_charge_density_function(self, rho_func):
        """
        Set the charge density function.
        
        Args:
            rho_func: Function that takes potential V and returns charge density rho
        """
        self.rho_func = rho_func
    
    def solve(self, initial_guess=None):
        """
        Solve the nonlinear Poisson equation using Newton's method.
        
        Args:
            initial_guess: Initial guess for the potential (optional)
            
        Returns:
            Potential array (V) with shape (ny, nx)
        """
        if self.rho_func is None:
            raise ValueError("Charge density function not set")
        
        # Set initial guess
        if initial_guess is not None:
            if initial_guess.shape != (self.ny, self.nx):
                raise ValueError(f"Initial guess shape {initial_guess.shape} does not match grid shape {(self.ny, self.nx)}")
            self.V = initial_guess.copy()
        
        # Newton iteration
        for iter in range(self.max_iter):
            # Calculate charge density from current potential
            rho = self.rho_func(self.V)
            
            # Set charge density in linear solver
            self.linear_solver.set_charge_density(rho)
            
            # Solve linear Poisson equation
            V_new = self.linear_solver.solve()
            
            # Calculate residual
            residual = np.max(np.abs(V_new - self.V))
            
            # Update potential with damping
            self.V = self.V + self.damping * (V_new - self.V)
            
            # Check convergence
            if residual < self.tolerance:
                print(f"Nonlinear Poisson solver converged in {iter+1} iterations")
                break
        else:
            print(f"Nonlinear Poisson solver did not converge in {self.max_iter} iterations")
        
        return self.V
    
    def get_electric_field(self):
        """
        Calculate the electric field from the potential.
        
        Returns:
            Electric field components (Ex, Ey) with shape (ny, nx)
        """
        return self.linear_solver.get_electric_field()
