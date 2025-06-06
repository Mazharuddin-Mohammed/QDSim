#!/usr/bin/env python3
"""
Test script for the SelfConsistentSolver implementation using the new create_self_consistent_solver function.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend'))

# Directly import the C++ module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))
import qdsim_cpp

def main():
    """
    Main function to test the SelfConsistentSolver implementation.
    """
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = qdsim_cpp.Mesh(Lx, Ly, nx, ny, element_order)

    # Get material properties
    mat_db = qdsim_cpp.MaterialDatabase()
    p_mat_props = mat_db.get_material("GaAs")
    n_mat_props = mat_db.get_material("GaAs")

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

    def n_conc(x, y, phi, mat):
        """Electron concentration function."""
        kT = 0.0259  # eV at 300K
        q = 1.602e-19  # Elementary charge in C
        E_F = 0.0  # Simplified; assume constant Fermi level
        # Use fixed values instead of mat properties
        N_c = 2.1e23  # Example value for GaAs
        return N_c * np.exp((-q * phi - E_F) / kT)

    def p_conc(x, y, phi, mat):
        """Hole concentration function."""
        kT = 0.0259  # eV at 300K
        q = 1.602e-19  # Elementary charge in C
        E_F = 0.0  # Simplified; assume constant Fermi level
        # Use fixed values instead of mat properties
        N_v = 8.0e24  # Example value for GaAs
        E_g = 1.42    # Example value for GaAs
        return N_v * np.exp((q * phi + E_g - E_F) / kT)

    def mu_n(x, y, mat):
        """Electron mobility function."""
        # Use fixed value instead of mat.mu_n
        return 0.85  # Example value for GaAs in m^2/(V*s)

    def mu_p(x, y, mat):
        """Hole mobility function."""
        # Use fixed value instead of mat.mu_p
        return 0.04  # Example value for GaAs in m^2/(V*s)

    # Create the SelfConsistentSolver using the new helper function
    sc_solver = qdsim_cpp.create_self_consistent_solver(mesh, epsilon_r, rho, n_conc, p_conc, mu_n, mu_p)

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
    potential = sc_solver.get_potential()
    n = sc_solver.get_n()
    p = sc_solver.get_p()

    print(f"Potential shape: {potential.shape}")
    print(f"Electron concentration shape: {n.shape}")
    print(f"Hole concentration shape: {p.shape}")

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
    plt.savefig('self_consistent_results.png')
    plt.show()

if __name__ == "__main__":
    main()
