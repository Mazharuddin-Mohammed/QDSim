#!/usr/bin/env python3
"""
Test script to identify the cause of the segmentation fault with a more complex scenario.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import time

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

# Import the C++ module directly
import qdsim_cpp as cpp

def main():
    """
    Main function to test a more complex scenario.
    """
    # Create a mesh with higher resolution
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 100
    ny = 50
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)

    # Define quantum dot parameters
    qd_x = 0.0   # QD position (at the junction)
    qd_y = Ly/2  # QD position (at the center of the device)
    qd_radius = 5.0  # QD radius (nm)
    qd_depth = 0.3   # QD potential depth (eV)

    # Define callback functions
    def epsilon_r(x, y):
        """Relative permittivity function."""
        # Check if the point is inside the quantum dot
        r = np.sqrt((x - qd_x)**2 + (y - qd_y)**2)
        if r < qd_radius:
            return 12.9  # GaAs
        else:
            return 10.0  # AlGaAs

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

    print(f"Potential shape: {potential.shape}")
    print(f"Electron concentration shape: {n.shape}")
    print(f"Hole concentration shape: {p.shape}")

    # Get the mesh nodes
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Add the quantum dot potential
    qd_potential = np.zeros_like(potential)
    for i in range(len(nodes)):
        # Distance from the quantum dot center
        r = np.sqrt((x[i] - qd_x)**2 + (y[i] - qd_y)**2)

        # Gaussian potential well
        if r < 3*qd_radius:  # Truncate at 3*radius for efficiency
            qd_potential[i] = -qd_depth * np.exp(-r**2 / (2*qd_radius**2))

    # Combine the p-n junction potential with the quantum dot potential
    combined_potential = potential + qd_potential

    # Reshape the data into a 2D grid for plotting
    nx_plot = nx + 1
    ny_plot = ny + 1

    # Create a regular grid for plotting
    x_grid = np.linspace(-Lx/2, Lx/2, nx_plot)
    y_grid = np.linspace(0, Ly, ny_plot)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Reshape the data to match the grid
    pn_potential_2d = np.zeros((ny_plot, nx_plot))
    qd_potential_2d = np.zeros((ny_plot, nx_plot))
    combined_potential_2d = np.zeros((ny_plot, nx_plot))
    n_2d = np.zeros((ny_plot, nx_plot))
    p_2d = np.zeros((ny_plot, nx_plot))

    # Fill in the data (simplified approach)
    for i in range(len(potential)):
        ix = int((x[i] + Lx/2) / Lx * nx)
        iy = int(y[i] / Ly * ny)
        if 0 <= ix < nx_plot and 0 <= iy < ny_plot:
            pn_potential_2d[iy, ix] = potential[i]
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
    sc4 = axes[1, 1].pcolormesh(X, Y, n_2d_masked, cmap='plasma')
    axes[1, 1].set_title('Electron Concentration (cm^-3)')
    axes[1, 1].set_xlabel('x (nm)')
    axes[1, 1].set_ylabel('y (nm)')
    plt.colorbar(sc4, ax=axes[1, 1])

    # Plot the hole concentration (log scale)
    p_2d_masked = np.ma.masked_less_equal(p_2d, 0)
    sc5 = axes[2, 0].pcolormesh(X, Y, p_2d_masked, cmap='inferno')
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
    plt.savefig('complex_test_results.png')
    plt.close()

    print("Test completed successfully!")

    # Clear the callbacks to avoid memory leaks
    print("Clearing callbacks...")
    cpp.clear_callbacks()

    # Sleep for a few seconds to see if the segmentation fault occurs after a delay
    print("Sleeping for 5 seconds...")
    time.sleep(5)
    print("Done sleeping.")

if __name__ == "__main__":
    main()
