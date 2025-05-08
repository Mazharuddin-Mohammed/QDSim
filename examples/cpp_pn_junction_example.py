#!/usr/bin/env python3
"""
Example script demonstrating the use of the C++ PNJunction class.

This script creates a P-N junction using the C++ implementation and
visualizes the potential, carrier concentrations, and band diagram.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qdsim_cpp as qdc
import sys
import os

def main():
    # Create a mesh with original size
    Lx = 200.0  # nm
    Ly = 100.0  # nm
    nx = 201
    ny = 101
    mesh = qdc.Mesh(Lx, Ly, nx, ny, 1)  # P1 elements

    # P-N junction parameters
    epsilon_r = 12.9  # Relative permittivity of GaAs
    N_A = 1e16  # Acceptor concentration (cm^-3)
    N_D = 1e16  # Donor concentration (cm^-3)
    T = 300.0  # Temperature (K)
    junction_position = 0.0  # Junction at x = 0
    V_r = 1.0  # Reverse bias voltage (V)

    # Convert doping concentrations from cm^-3 to nm^-3
    N_A_nm3 = N_A * 1e-21
    N_D_nm3 = N_D * 1e-21

    # Create the P-N junction
    print("Creating P-N junction...")
    pn_junction = qdc.PNJunction(mesh, epsilon_r, N_A_nm3, N_D_nm3, T, junction_position, V_r)

    # Print junction parameters
    print(f"Built-in potential: {pn_junction.V_bi:.3f} V")
    print(f"Depletion width: {pn_junction.W:.3f} nm")
    print(f"P-side depletion width: {pn_junction.W_p:.3f} nm")
    print(f"N-side depletion width: {pn_junction.W_n:.3f} nm")
    print(f"Intrinsic carrier concentration: {pn_junction.n_i:.3e} nm^-3")

    # Solve the Poisson equation
    print("Solving Poisson equation...")
    pn_junction.solve()

    # Create a grid for visualization
    vis_nx, vis_ny = 100, 50  # Higher resolution for visualization
    x = np.linspace(-Lx/2, Lx/2, vis_nx)
    y = np.linspace(-Ly/2, Ly/2, vis_ny)
    X, Y = np.meshgrid(x, y)

    # Calculate potential, carrier concentrations, and band edges on the grid
    print("Calculating quantities on grid...")
    potential = np.zeros((vis_ny, vis_nx))
    electron_conc = np.zeros((vis_ny, vis_nx))
    hole_conc = np.zeros((vis_ny, vis_nx))
    conduction_band = np.zeros((vis_ny, vis_nx))
    valence_band = np.zeros((vis_ny, vis_nx))

    # Use a progress indicator
    total_points = vis_nx * vis_ny
    point_count = 0

    for i in range(vis_ny):
        for j in range(vis_nx):
            point_count += 1
            if point_count % 500 == 0 or point_count == total_points:
                print(f"Processing point {point_count}/{total_points}...")

            potential[i, j] = pn_junction.get_potential(X[i, j], Y[i, j])
            electron_conc[i, j] = pn_junction.get_electron_concentration(X[i, j], Y[i, j])
            hole_conc[i, j] = pn_junction.get_hole_concentration(X[i, j], Y[i, j])
            conduction_band[i, j] = pn_junction.get_conduction_band_edge(X[i, j], Y[i, j])
            valence_band[i, j] = pn_junction.get_valence_band_edge(X[i, j], Y[i, j])

    # Plot the potential
    print("Plotting potential...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a custom plot_potential_3d function
    def plot_potential_3d(ax, X, Y, potential, title=None):
        surf = ax.plot_surface(X, Y, potential, cmap='viridis', edgecolor='none')
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Potential (V)')
        if title:
            ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Potential (V)')
        ax.view_init(30, 45)

    plot_potential_3d(ax, X, Y, potential, title="P-N Junction Potential (C++ Implementation)")
    plt.savefig("cpp_pn_junction_potential.png", dpi=300, bbox_inches='tight')

    # Plot carrier concentrations
    print("Plotting carrier concentrations...")
    plt.figure(figsize=(10, 6))

    # Plot along the x-axis at y = 0
    mid_y = vis_ny // 2
    plt.semilogy(x, electron_conc[mid_y, :], 'b-', label='Electron concentration')
    plt.semilogy(x, hole_conc[mid_y, :], 'r-', label='Hole concentration')
    plt.semilogy(x, np.ones_like(x) * N_A_nm3, 'r--', label='N_A')
    plt.semilogy(x, np.ones_like(x) * N_D_nm3, 'b--', label='N_D')
    plt.semilogy(x, np.ones_like(x) * pn_junction.n_i, 'g--', label='n_i')

    plt.xlabel('Position (nm)')
    plt.ylabel('Carrier concentration (nm$^{-3}$)')
    plt.title('Carrier Concentrations in P-N Junction (C++ Implementation)')
    plt.grid(True)
    plt.legend()
    plt.savefig("cpp_pn_junction_carriers.png", dpi=300, bbox_inches='tight')

    # Plot band diagram
    print("Plotting band diagram...")
    plt.figure(figsize=(10, 6))

    # Create a custom plot_band_diagram function
    def plot_band_diagram(x, conduction_band, valence_band, E_F_electrons=None, E_F_holes=None, title=None):
        plt.plot(x, conduction_band, 'b-', label='Conduction band')
        plt.plot(x, valence_band, 'r-', label='Valence band')

        if E_F_electrons is not None:
            plt.plot(x, E_F_electrons, 'b--', label='Electron quasi-Fermi level')
        if E_F_holes is not None:
            plt.plot(x, E_F_holes, 'r--', label='Hole quasi-Fermi level')

        plt.xlabel('Position (nm)')
        plt.ylabel('Energy (eV)')
        if title:
            plt.title(title)
        plt.grid(True)
        plt.legend()

    # Plot along the x-axis at y = 0
    # Add quasi-Fermi levels
    E_F_electrons = np.array([pn_junction.get_quasi_fermi_level_electrons(xi, 0) for xi in x])
    E_F_holes = np.array([pn_junction.get_quasi_fermi_level_holes(xi, 0) for xi in x])

    plot_band_diagram(x, conduction_band[mid_y, :], valence_band[mid_y, :],
                     E_F_electrons, E_F_holes,
                     title='Band Diagram of P-N Junction (C++ Implementation)')

    plt.savefig("cpp_pn_junction_bands.png", dpi=300, bbox_inches='tight')

    # Plot electric field
    print("Plotting electric field...")
    plt.figure(figsize=(10, 6))

    # Calculate electric field along the x-axis at y = 0
    E_field = np.array([pn_junction.get_electric_field(xi, 0)[0] for xi in x])

    plt.plot(x, E_field, 'k-')
    plt.xlabel('Position (nm)')
    plt.ylabel('Electric field (V/nm)')
    plt.title('Electric Field in P-N Junction (C++ Implementation)')
    plt.grid(True)
    plt.savefig("cpp_pn_junction_field.png", dpi=300, bbox_inches='tight')

    print("Done! Plots saved as png files.")

if __name__ == "__main__":
    main()
