#!/usr/bin/env python3
"""
Test script for the SimpleSelfConsistentSolver implementation.
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
    Main function to test the SimpleSelfConsistentSolver implementation.
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
        q = 1.602e-19  # Elementary charge in C
        if len(n) == 0 or len(p) == 0:
            return 0.0

        # Find the nearest node
        nodes = np.array(mesh.get_nodes())
        distances = np.sqrt((nodes[:, 0] - x)**2 + (nodes[:, 1] - y)**2)
        idx = np.argmin(distances)

        # Return the charge density at the nearest node
        return q * (p[idx] - n[idx])

    # Create the SimpleSelfConsistentSolver using the helper function
    sc_solver = cpp.create_simple_self_consistent_solver(mesh, epsilon_r, rho)

    # Solve the self-consistent Poisson-drift-diffusion equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 1.0  # Voltage at the n-contact
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

    # Plot the results using a simpler approach
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Reshape the data into a 2D grid
    nx = mesh.get_nx()
    ny = mesh.get_ny()

    # Create a regular grid for plotting
    x = np.linspace(0, mesh.get_lx(), nx+1)
    y = np.linspace(0, mesh.get_ly(), ny+1)
    X, Y = np.meshgrid(x, y)

    # Reshape the data to match the grid
    potential_2d = np.zeros((ny+1, nx+1))
    n_2d = np.zeros((ny+1, nx+1))
    p_2d = np.zeros((ny+1, nx+1))

    # Fill in the data (simplified approach)
    for i in range(len(potential)):
        ix = i % (nx+1)
        iy = i // (nx+1)
        if iy < ny+1 and ix < nx+1:
            potential_2d[iy, ix] = potential[i]
            n_2d[iy, ix] = n[i]
            p_2d[iy, ix] = p[i]

    # Plot the potential
    sc1 = ax1.pcolormesh(X, Y, potential_2d, cmap='viridis')
    ax1.set_title('Electrostatic Potential (V)')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(sc1, ax=ax1)

    # Plot the electron concentration
    sc2 = ax2.pcolormesh(X, Y, n_2d, cmap='plasma')
    ax2.set_title('Electron Concentration (cm^-3)')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(sc2, ax=ax2)

    # Plot the hole concentration
    sc3 = ax3.pcolormesh(X, Y, p_2d, cmap='inferno')
    ax3.set_title('Hole Concentration (cm^-3)')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(sc3, ax=ax3)

    plt.tight_layout()
    plt.savefig('simple_self_consistent_results.png')
    plt.show()

if __name__ == "__main__":
    main()
