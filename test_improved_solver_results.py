#!/usr/bin/env python3
"""
Test script that creates a mesh, an improved solver, runs the solver, and gets the results.
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
    Main function to test creating a mesh, an improved solver, running the solver, and getting the results.
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
    
    print("Successfully got the results.")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
