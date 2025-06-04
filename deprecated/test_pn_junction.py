#!/usr/bin/env python3
"""
Test script for simulating a p-n junction using the BasicSolver.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

# Import the C++ module directly
import qdsim_cpp as cpp

def main():
    """
    Main function to simulate a p-n junction using the BasicSolver.
    """
    # Create a mesh with higher resolution
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 100
    ny = 50
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)
    
    # Create the BasicSolver
    solver = cpp.BasicSolver(mesh)
    
    # Solve the equations for a p-n junction
    V_p = 0.0    # Voltage at the p-contact
    V_n = 0.7    # Voltage at the n-contact (forward bias)
    N_A = 1e18   # Acceptor doping concentration (cm^-3)
    N_D = 1e18   # Donor doping concentration (cm^-3)
    
    print("Setting up p-n junction simulation...")
    solver.solve(V_p, V_n, N_A, N_D)
    
    # Get the results
    potential = np.array(solver.get_potential())
    n = np.array(solver.get_n())
    p = np.array(solver.get_p())
    
    print(f"Potential shape: {len(potential)}")
    print(f"Electron concentration shape: {len(n)}")
    print(f"Hole concentration shape: {len(p)}")
    
    # Calculate the electric field (negative gradient of potential)
    # For simplicity, we'll just calculate it along the x-axis at y=Ly/2
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # Find nodes along the middle of the device (y ≈ Ly/2)
    middle_nodes = np.where(np.abs(y - Ly/2) < 1.0)[0]
    x_middle = x[middle_nodes]
    potential_middle = potential[middle_nodes]
    
    # Sort by x-coordinate
    sort_idx = np.argsort(x_middle)
    x_middle = x_middle[sort_idx]
    potential_middle = potential_middle[sort_idx]
    
    # Calculate electric field (negative gradient of potential)
    E_field = -np.gradient(potential_middle, x_middle)
    
    # Create a figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot the potential
    sc1 = ax1.tricontourf(x, y, potential, 20, cmap='viridis')
    ax1.set_title('Electrostatic Potential (V)')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(sc1, ax=ax1)
    
    # Plot the electron concentration (log scale)
    sc2 = ax2.tricontourf(x, y, n, 20, cmap='plasma', norm=LogNorm(vmin=1e10, vmax=1e19))
    ax2.set_title('Electron Concentration (cm^-3)')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(sc2, ax=ax2)
    
    # Plot the hole concentration (log scale)
    sc3 = ax3.tricontourf(x, y, p, 20, cmap='inferno', norm=LogNorm(vmin=1e10, vmax=1e19))
    ax3.set_title('Hole Concentration (cm^-3)')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(sc3, ax=ax3)
    
    # Plot the electric field along the middle of the device
    ax4.plot(x_middle, E_field, 'r-', linewidth=2)
    ax4.set_title('Electric Field along y=Ly/2')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('Electric Field (V/nm)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('pn_junction_results.png')
    plt.show()
    
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
