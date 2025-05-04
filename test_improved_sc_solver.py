#!/usr/bin/env python3
"""
Test script for the ImprovedSelfConsistentSolver implementation.
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
        print(f"Warning: {nan_count} NaN values found in potential. Replacing with zeros.")
        potential = np.nan_to_num(potential, nan=0.0)

    nan_count = np.isnan(n).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in n. Replacing with zeros.")
        n = np.nan_to_num(n, nan=0.0)

    nan_count = np.isnan(p).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in p. Replacing with zeros.")
        p = np.nan_to_num(p, nan=0.0)

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

    # Reshape the data into a 2D grid for safer plotting
    nx_plot = nx + 1
    ny_plot = ny + 1

    # Create a regular grid for plotting
    x_grid = np.linspace(-Lx/2, Lx/2, nx_plot)
    y_grid = np.linspace(0, Ly, ny_plot)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Reshape the data to match the grid
    potential_2d = np.zeros((ny_plot, nx_plot))
    n_2d = np.zeros((ny_plot, nx_plot))
    p_2d = np.zeros((ny_plot, nx_plot))

    # Fill in the data (simplified approach)
    for i in range(len(potential)):
        if i < len(nodes):
            ix = int((nodes[i][0] + Lx/2) / Lx * nx)
            iy = int(nodes[i][1] / Ly * ny)
            if 0 <= ix < nx_plot and 0 <= iy < ny_plot:
                potential_2d[iy, ix] = potential[i]
                n_2d[iy, ix] = n[i]
                p_2d[iy, ix] = p[i]

    # Plot the potential
    sc1 = ax1.pcolormesh(X, Y, potential_2d, cmap='viridis')
    ax1.set_title('Electrostatic Potential (V)')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(sc1, ax=ax1)

    # Plot the electron concentration (log scale)
    n_2d_masked = np.ma.masked_less_equal(n_2d, 0)
    sc2 = ax2.pcolormesh(X, Y, n_2d_masked, cmap='plasma', norm=LogNorm(vmin=1e10, vmax=1e19))
    ax2.set_title('Electron Concentration (cm^-3)')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(sc2, ax=ax2)

    # Plot the hole concentration (log scale)
    p_2d_masked = np.ma.masked_less_equal(p_2d, 0)
    sc3 = ax3.pcolormesh(X, Y, p_2d_masked, cmap='inferno', norm=LogNorm(vmin=1e10, vmax=1e19))
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
    plt.savefig('improved_sc_solver_results.png')
    plt.show()

    print("Test completed successfully!")

if __name__ == "__main__":
    main()
