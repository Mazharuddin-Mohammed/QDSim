#!/usr/bin/env python3
"""
Test script for the SelfConsistentSolver implementation.
This script directly imports the C++ module without going through the Python wrapper.
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

    # Define callback functions with the exact signatures expected by C++
    def epsilon_r(x, y):
        """Relative permittivity function."""
        return 12.9  # GaAs

    def rho(x, y, n, p):
        """Charge density function."""
        q = 1.602e-19  # Elementary charge in C
        if len(n) == 0 or len(p) == 0:
            return 0.0

        # Find the nearest node
        nodes = np.array(mesh.getNodes())
        distances = np.sqrt((nodes[:, 0] - x)**2 + (nodes[:, 1] - y)**2)
        idx = np.argmin(distances)

        # Return the charge density at the nearest node
        return q * (p[idx] - n[idx])

    def n_conc(x, y, phi, mat):
        """Electron concentration function."""
        kT = 0.0259  # eV at 300K
        q = 1.602e-19  # Elementary charge in C
        E_F = 0.0  # Simplified; assume constant Fermi level
        return mat.N_c * np.exp((-q * phi - E_F) / kT)

    def p_conc(x, y, phi, mat):
        """Hole concentration function."""
        kT = 0.0259  # eV at 300K
        q = 1.602e-19  # Elementary charge in C
        E_F = 0.0  # Simplified; assume constant Fermi level
        return mat.N_v * np.exp((q * phi + mat.E_g - E_F) / kT)

    def mu_n(x, y, mat):
        """Electron mobility function."""
        return mat.mu_n

    def mu_p(x, y, mat):
        """Hole mobility function."""
        return mat.mu_p

    # Create C function pointers from Python functions
    # This is necessary because pybind11 expects C function pointers, not Python functions
    epsilon_r_ptr = qdsim_cpp.create_epsilon_r_callback(epsilon_r)
    rho_ptr = qdsim_cpp.create_rho_callback(rho)
    n_conc_ptr = qdsim_cpp.create_n_conc_callback(n_conc)
    p_conc_ptr = qdsim_cpp.create_p_conc_callback(p_conc)
    mu_n_ptr = qdsim_cpp.create_mu_n_callback(mu_n)
    mu_p_ptr = qdsim_cpp.create_mu_p_callback(mu_p)

    # Create the SelfConsistentSolver
    sc_solver = qdsim_cpp.SelfConsistentSolver(mesh, epsilon_r, rho, n_conc, p_conc, mu_n, mu_p)

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
    nodes = np.array(mesh.getNodes())
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
