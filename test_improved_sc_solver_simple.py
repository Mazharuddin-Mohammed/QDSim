#!/usr/bin/env python3
"""
Simple test script for the ImprovedSelfConsistentSolver implementation.
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
    Main function to test the ImprovedSelfConsistentSolver implementation.
    """
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 100
    ny = 50
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)
    
    # Define callback functions
    def epsilon_r(x, y):
        """Relative permittivity function."""
        return 12.9  # GaAs
    
    def rho(x, y, n, p):
        """Charge density function."""
        q = 1.602e-19  # Elementary charge in C
        if len(n) == 0 or len(p) == 0:
            return 0.0
        
        # Find the nearest node
        nodes = np.array(mesh.get_nodes())
        distances = np.sqrt((nodes[:, 0] - x)**2 + (nodes[:, 1] - y)**2)
        idx = np.argmin(distances)
        
        # Return the charge density at the nearest node
        return q * (p[idx] - n[idx])
    
    # Create the ImprovedSelfConsistentSolver using the helper function
    sc_solver = cpp.create_improved_self_consistent_solver(mesh, epsilon_r, rho)
    
    # Solve the self-consistent Poisson-drift-diffusion equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 0.7  # Voltage at the n-contact (forward bias)
    N_A = 1e18  # Acceptor doping concentration
    N_D = 1e18  # Donor doping concentration
    tolerance = 1e-6
    max_iter = 100
    
    print("Solving the self-consistent Poisson-drift-diffusion equations...")
    sc_solver.solve(V_p, V_n, N_A, N_D, tolerance, max_iter)
    
    # Get the results
    potential = np.array(sc_solver.get_potential())
    n = np.array(sc_solver.get_n())
    p = np.array(sc_solver.get_p())
    
    print(f"Potential shape: {len(potential)}")
    print(f"Electron concentration shape: {len(n)}")
    print(f"Hole concentration shape: {len(p)}")
    
    # Check for NaN values
    nan_count = np.isnan(potential).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in potential.")
    
    nan_count = np.isnan(n).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in n.")
    
    nan_count = np.isnan(p).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in p.")
    
    # Print some statistics
    print(f"Potential min: {potential.min()}, max: {potential.max()}")
    print(f"Electron concentration min: {n.min()}, max: {n.max()}")
    print(f"Hole concentration min: {p.min()}, max: {p.max()}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
