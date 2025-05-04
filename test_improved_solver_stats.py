#!/usr/bin/env python3
"""
Test script that creates a mesh, an improved solver, runs the solver, gets the results, converts them to NumPy arrays, and prints some statistics.
"""

import sys
import os
import numpy as np

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

# Import the C++ module directly
import qdsim_cpp as cpp

def main():
    """
    Main function to test creating a mesh, an improved solver, running the solver, getting the results, converting them to NumPy arrays, and printing some statistics.
    """
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)
    
    # Define callback functions
    def epsilon_r(x, y):
        """Relative permittivity function."""
        return 12.9  # GaAs
    
    def rho(x, y, n, p):
        """Charge density function."""
        return 0.0  # Simplified: no charge density
    
    # Create an improved solver
    solver = cpp.ImprovedSelfConsistentSolver(mesh, epsilon_r, rho)
    
    # Solve the self-consistent Poisson-drift-diffusion equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 0.7  # Voltage at the n-contact (forward bias)
    N_A = 1e18  # Acceptor doping concentration
    N_D = 1e18  # Donor doping concentration
    tolerance = 1e-6
    max_iter = 100
    
    print("Solving the self-consistent Poisson-drift-diffusion equations...")
    solver.solve(V_p, V_n, N_A, N_D, tolerance, max_iter)
    
    # Get the results
    potential = solver.get_potential()
    n = solver.get_n()
    p = solver.get_p()
    
    print(f"Potential shape: {len(potential)}")
    print(f"Electron concentration shape: {len(n)}")
    print(f"Hole concentration shape: {len(p)}")
    
    # Convert to NumPy arrays
    potential_np = np.array(potential)
    n_np = np.array(n)
    p_np = np.array(p)
    
    print(f"NumPy potential shape: {potential_np.shape}")
    print(f"NumPy electron concentration shape: {n_np.shape}")
    print(f"NumPy hole concentration shape: {p_np.shape}")
    
    # Print some statistics
    print(f"Potential min: {potential_np.min()}, max: {potential_np.max()}")
    print(f"Electron concentration min: {n_np.min()}, max: {n_np.max()}")
    print(f"Hole concentration min: {p_np.min()}, max: {p_np.max()}")
    
    # Check for NaN values
    nan_count = np.isnan(potential_np).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in potential.")
    
    nan_count = np.isnan(n_np).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in n.")
    
    nan_count = np.isnan(p_np).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in p.")
    
    print("Successfully printed statistics.")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
