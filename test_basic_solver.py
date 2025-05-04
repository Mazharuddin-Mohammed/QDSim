#!/usr/bin/env python3
"""
Test script for the BasicSolver implementation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

# Import the C++ module directly
import qdsim_cpp as cpp

def main():
    """
    Main function to test the BasicSolver implementation.
    """
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)
    
    # Create the BasicSolver
    solver = cpp.BasicSolver(mesh)
    
    # Solve the equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 1.0  # Voltage at the n-contact
    N_A = 1e18  # Acceptor doping concentration
    N_D = 1e18  # Donor doping concentration
    
    print("Setting up simple fields...")
    solver.solve(V_p, V_n, N_A, N_D)
    
    # Get the results
    potential = np.array(solver.get_potential())
    n = np.array(solver.get_n())
    p = np.array(solver.get_p())
    
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
    
    # Plot the results
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the potential
    sc1 = ax1.tricontourf(x, y, potential, 20, cmap='viridis')
    ax1.set_title('Electrostatic Potential (V)')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(sc1, ax=ax1)
    
    # Plot the electron concentration
    sc2 = ax2.tricontourf(x, y, n, 20, cmap='plasma')
    ax2.set_title('Electron Concentration (cm^-3)')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(sc2, ax=ax2)
    
    # Plot the hole concentration
    sc3 = ax3.tricontourf(x, y, p, 20, cmap='inferno')
    ax3.set_title('Hole Concentration (cm^-3)')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(sc3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('basic_solver_results.png')
    plt.show()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
