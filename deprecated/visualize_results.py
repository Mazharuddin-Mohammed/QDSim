#!/usr/bin/env python3
"""
Script to visualize the results of the p-n junction with quantum dot simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    """
    Main function to visualize the results of the p-n junction with quantum dot simulation.
    """
    # Check if the results directory exists
    if not os.path.exists("results"):
        print("Error: Results directory not found.")
        return
    
    # Load the mesh
    x = np.loadtxt("results/mesh_x.txt")
    y = np.loadtxt("results/mesh_y.txt")
    
    # Load the potentials
    pn_potential = np.loadtxt("results/pn_potential.txt")
    qd_potential = np.loadtxt("results/qd_potential.txt")
    combined_potential = np.loadtxt("results/combined_potential.txt")
    
    # Load the carrier concentrations
    n = np.loadtxt("results/electron_concentration.txt")
    p = np.loadtxt("results/hole_concentration.txt")
    
    # Print some statistics
    print(f"Mesh shape: ({len(x)}, {len(y)})")
    print(f"P-N potential shape: {pn_potential.shape}")
    print(f"QD potential shape: {qd_potential.shape}")
    print(f"Combined potential shape: {combined_potential.shape}")
    print(f"Electron concentration shape: {n.shape}")
    print(f"Hole concentration shape: {p.shape}")
    
    # Create a regular grid for plotting
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 100
    ny = 50
    
    # Create a regular grid
    x_grid = np.linspace(-Lx/2, Lx/2, nx+1)
    y_grid = np.linspace(0, Ly, ny+1)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Reshape the data to match the grid
    pn_potential_2d = np.zeros((ny+1, nx+1))
    qd_potential_2d = np.zeros((ny+1, nx+1))
    combined_potential_2d = np.zeros((ny+1, nx+1))
    n_2d = np.zeros((ny+1, nx+1))
    p_2d = np.zeros((ny+1, nx+1))
    
    # Fill in the data (simplified approach)
    for i in range(len(x)):
        ix = int((x[i] + Lx/2) / Lx * nx)
        iy = int(y[i] / Ly * ny)
        if 0 <= ix < nx+1 and 0 <= iy < ny+1:
            pn_potential_2d[iy, ix] = pn_potential[i]
            qd_potential_2d[iy, ix] = qd_potential[i]
            combined_potential_2d[iy, ix] = combined_potential[i]
            n_2d[iy, ix] = n[i]
            p_2d[iy, ix] = p[i]
    
    # Create a figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Plot the p-n junction potential
    sc1 = axes[0, 0].pcolormesh(X, Y, pn_potential_2d, cmap='viridis')
    axes[0, 0].set_title('P-N Junction Potential (V)')
    axes[0, 0].set_xlabel('x (nm)')
    axes[0, 0].set_ylabel('y (nm)')
    plt.colorbar(sc1, ax=axes[0, 0])
    
    # Plot the quantum dot potential
    sc2 = axes[0, 1].pcolormesh(X, Y, qd_potential_2d, cmap='plasma')
    axes[0, 1].set_title('Quantum Dot Potential (eV)')
    axes[0, 1].set_xlabel('x (nm)')
    axes[0, 1].set_ylabel('y (nm)')
    plt.colorbar(sc2, ax=axes[0, 1])
    
    # Plot the combined potential
    sc3 = axes[1, 0].pcolormesh(X, Y, combined_potential_2d, cmap='viridis')
    axes[1, 0].set_title('Combined Potential (V)')
    axes[1, 0].set_xlabel('x (nm)')
    axes[1, 0].set_ylabel('y (nm)')
    plt.colorbar(sc3, ax=axes[1, 0])
    
    # Plot the electron concentration (log scale)
    n_2d_masked = np.ma.masked_less_equal(n_2d, 0)
    sc4 = axes[1, 1].pcolormesh(X, Y, n_2d_masked, cmap='plasma', norm=LogNorm(vmin=1e10, vmax=1e19))
    axes[1, 1].set_title('Electron Concentration (cm^-3)')
    axes[1, 1].set_xlabel('x (nm)')
    axes[1, 1].set_ylabel('y (nm)')
    plt.colorbar(sc4, ax=axes[1, 1])
    
    # Plot the hole concentration (log scale)
    p_2d_masked = np.ma.masked_less_equal(p_2d, 0)
    sc5 = axes[2, 0].pcolormesh(X, Y, p_2d_masked, cmap='inferno', norm=LogNorm(vmin=1e10, vmax=1e19))
    axes[2, 0].set_title('Hole Concentration (cm^-3)')
    axes[2, 0].set_xlabel('x (nm)')
    axes[2, 0].set_ylabel('y (nm)')
    plt.colorbar(sc5, ax=axes[2, 0])
    
    # Plot the potential along the x-axis at y=Ly/2
    middle_y = int(ny/2)
    axes[2, 1].plot(x_grid, pn_potential_2d[middle_y, :], 'b-', label='P-N Junction')
    axes[2, 1].plot(x_grid, qd_potential_2d[middle_y, :], 'r-', label='Quantum Dot')
    axes[2, 1].plot(x_grid, combined_potential_2d[middle_y, :], 'g-', label='Combined')
    axes[2, 1].set_title('Potential along y=Ly/2')
    axes[2, 1].set_xlabel('x (nm)')
    axes[2, 1].set_ylabel('Potential (V)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('pn_junction_qd_results.png')
    plt.show()
    
    print("Visualization completed successfully!")

if __name__ == "__main__":
    main()
