#!/usr/bin/env python3
"""
Direct P-N Junction Example

This example demonstrates a P-N junction model using the C++ PNJunction class directly.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import qdsim_cpp as qdc

def main():
    """Main function."""
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 101
    ny = 51
    mesh = qdc.Mesh(Lx, Ly, nx, ny, 1)  # P1 elements

    # P-N junction parameters
    epsilon_r = 12.9  # Relative permittivity of GaAs
    N_A = 1e-9  # Acceptor concentration (nm^-3)
    N_D = 1e-9  # Donor concentration (nm^-3)
    T = 300.0   # Temperature (K)
    junction_position = Lx / 2  # Junction at the center of the domain
    V_r = 0.5  # Reverse bias (V)

    # Create the PNJunction
    print("Creating PNJunction...")
    pn = qdc.PNJunction(mesh, epsilon_r, N_A, N_D, T, junction_position, V_r)

    # Calculate built-in potential and depletion width
    V_bi = pn.calculate_built_in_potential()
    W = pn.calculate_depletion_width()
    n_i = pn.calculate_intrinsic_carrier_concentration()

    print(f"Built-in potential: {V_bi:.4f} V")
    print(f"Depletion width: {W:.4f} nm")
    print(f"Intrinsic carrier concentration: {n_i:.4e} nm^-3")

    # Solve the Poisson equation
    print("Solving Poisson equation...")
    pn.solve()

    # Create a grid for visualization
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Calculate quantities on grid
    print("Calculating quantities on grid...")
    potential = np.zeros(nx)
    electron_conc = np.zeros(nx)
    hole_conc = np.zeros(nx)
    electric_field = np.zeros((nx, 2))
    conduction_band = np.zeros(nx)
    valence_band = np.zeros(nx)

    # Calculate at the middle of the domain
    mid_y = Ly / 2
    for i in range(nx):
        xi = x[i]
        potential[i] = pn.get_potential(xi, mid_y) / 1.602e-19  # Convert from J to eV
        electron_conc[i] = pn.get_electron_concentration(xi, mid_y)
        hole_conc[i] = pn.get_hole_concentration(xi, mid_y)
        electric_field[i] = pn.get_electric_field(xi, mid_y)
        conduction_band[i] = pn.get_conduction_band_edge(xi, mid_y) / 1.602e-19  # Convert from J to eV
        valence_band[i] = pn.get_valence_band_edge(xi, mid_y) / 1.602e-19  # Convert from J to eV

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot potential
    axs[0, 0].plot(x, potential, 'b-', linewidth=2)
    axs[0, 0].axvline(x=junction_position, color='k', linestyle='--', label='Junction')
    axs[0, 0].set_xlabel('x (nm)')
    axs[0, 0].set_ylabel('Potential (eV)')
    axs[0, 0].set_title('Electrostatic Potential')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot carrier concentrations
    axs[0, 1].semilogy(x, electron_conc, 'b-', linewidth=2, label='n')
    axs[0, 1].semilogy(x, hole_conc, 'r-', linewidth=2, label='p')
    axs[0, 1].semilogy(x, np.ones_like(x) * N_A, 'r--', linewidth=1, label='N_A')
    axs[0, 1].semilogy(x, np.ones_like(x) * N_D, 'b--', linewidth=1, label='N_D')
    axs[0, 1].axvline(x=junction_position, color='k', linestyle='--')
    axs[0, 1].set_xlabel('x (nm)')
    axs[0, 1].set_ylabel('Carrier Concentration (nm^-3)')
    axs[0, 1].set_title('Carrier Concentrations')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Plot band diagram
    axs[1, 0].plot(x, conduction_band, 'b-', linewidth=2, label='E_c')
    axs[1, 0].plot(x, valence_band, 'r-', linewidth=2, label='E_v')
    axs[1, 0].axvline(x=junction_position, color='k', linestyle='--')
    axs[1, 0].set_xlabel('x (nm)')
    axs[1, 0].set_ylabel('Energy (eV)')
    axs[1, 0].set_title('Band Diagram')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Plot electric field
    axs[1, 1].plot(x, electric_field[:, 0], 'b-', linewidth=2, label='E_x')
    axs[1, 1].axvline(x=junction_position, color='k', linestyle='--')
    axs[1, 1].set_xlabel('x (nm)')
    axs[1, 1].set_ylabel('Electric Field (V/nm)')
    axs[1, 1].set_title('Electric Field')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('pn_junction_direct.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
