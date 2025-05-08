#!/usr/bin/env python3
"""
Heterojunction Example

This example demonstrates a heterojunction between GaAs and AlGaAs
using the C++ SelfConsistentSolver implementation.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import qdsim_cpp as qdc

def main():
    """Main function."""
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 101
    ny = 51
    mesh = qdc.Mesh(Lx, Ly, nx, ny, 1)  # P1 elements

    # Physical constants
    e_charge = 1.602e-19  # Elementary charge (C)
    m_e = 9.109e-31  # Electron mass (kg)
    k_B = 1.381e-23  # Boltzmann constant (J/K)
    epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
    h_bar = 1.055e-34  # Reduced Planck constant (J·s)
    T = 300  # Temperature (K)

    # Define materials
    # GaAs
    gaas = qdc.Material()
    gaas.N_c = 4.7e17  # Effective density of states in conduction band (nm^-3)
    gaas.N_v = 7.0e18  # Effective density of states in valence band (nm^-3)
    gaas.E_g = 1.424   # Band gap (eV)
    gaas.mu_n = 8500   # Electron mobility (cm^2/V·s)
    gaas.mu_p = 400    # Hole mobility (cm^2/V·s)
    gaas.epsilon_r = 12.9  # Relative permittivity

    # AlGaAs (Al_0.3Ga_0.7As)
    algaas = qdc.Material()
    algaas.N_c = 5.9e17  # Effective density of states in conduction band (nm^-3)
    algaas.N_v = 5.0e18  # Effective density of states in valence band (nm^-3)
    algaas.E_g = 1.798   # Band gap (eV)
    algaas.mu_n = 4000   # Electron mobility (cm^2/V·s)
    algaas.mu_p = 200    # Hole mobility (cm^2/V·s)
    algaas.epsilon_r = 12.2  # Relative permittivity

    # Doping parameters
    N_A = 5e22 * 1e-27  # Acceptor concentration (m^-3 to nm^-3)
    N_D = 5e22 * 1e-27  # Donor concentration (m^-3 to nm^-3)

    # P-N junction parameters
    junction_position = 0.0  # Junction at the center of the domain
    V_r = 0.5  # Reverse bias (V)

    # Define callback functions for the SelfConsistentSolver
    def epsilon_r_callback(x, y):
        # Return the relative permittivity at (x, y)
        if x < 0:
            return gaas.epsilon_r
        else:
            return algaas.epsilon_r

    def rho_callback(x, y, n, p):
        # Charge density in C/nm^3
        q = e_charge  # Elementary charge in C
        
        # Doping contribution
        rho_doping = q * (N_D if x > junction_position else -N_A)
        
        # Carrier contribution
        # We need to interpolate n and p at (x, y)
        # For simplicity, we'll use the first values
        n_val = n[0] if len(n) > 0 else 0.0
        p_val = p[0] if len(p) > 0 else 0.0
        
        rho_carriers = q * (p_val - n_val)
        
        return rho_doping + rho_carriers

    def n_conc_callback(x, y, phi, mat):
        # Constants
        kT = k_B * T / e_charge  # eV
        q = e_charge  # Elementary charge in C
        
        # Calculate Fermi levels
        E_F_p = -mat.E_g + kT * np.log(mat.N_v / N_A)
        E_F_n = -kT * np.log(mat.N_c / N_D)
        
        # Calculate band edges
        E_c = -q * phi - (E_F_p if x < junction_position else E_F_n)
        
        # Calculate electron concentration using Boltzmann statistics
        n = mat.N_c * np.exp(-E_c / kT)
        
        # Apply limits for numerical stability
        n_min = 1e-15  # Minimum concentration (nm^-3)
        n_max = 1e-3   # Maximum concentration (nm^-3)
        return max(min(n, n_max), n_min)

    def p_conc_callback(x, y, phi, mat):
        # Constants
        kT = k_B * T / e_charge  # eV
        q = e_charge  # Elementary charge in C
        
        # Calculate Fermi levels
        E_F_p = -mat.E_g + kT * np.log(mat.N_v / N_A)
        E_F_n = -kT * np.log(mat.N_c / N_D)
        
        # Calculate band edges
        E_c = -q * phi - (E_F_p if x < junction_position else E_F_n)
        E_v = E_c - mat.E_g
        
        # Calculate hole concentration using Boltzmann statistics
        p = mat.N_v * np.exp(E_v / kT)
        
        # Apply limits for numerical stability
        p_min = 1e-15  # Minimum concentration (nm^-3)
        p_max = 1e-3   # Maximum concentration (nm^-3)
        return max(min(p, p_max), p_min)

    def mu_n_callback(x, y, mat):
        return mat.mu_n

    def mu_p_callback(x, y, mat):
        return mat.mu_p

    # Create the SelfConsistentSolver
    print("Creating SelfConsistentSolver...")
    solver = qdc.create_self_consistent_solver(
        mesh, epsilon_r_callback, rho_callback,
        n_conc_callback, p_conc_callback,
        mu_n_callback, mu_p_callback
    )

    # Define heterojunction regions
    def gaas_region(x, y):
        return x < 0

    def algaas_region(x, y):
        return x >= 0

    # Set heterojunction
    print("Setting heterojunction...")
    solver.set_heterojunction([gaas, algaas], [gaas_region, algaas_region])

    # Set convergence acceleration parameters
    solver.damping_factor = 0.3
    solver.anderson_history_size = 3

    # Solve the self-consistent Poisson-drift-diffusion equations
    print("Solving self-consistent equations...")
    solver.solve(0.0, -V_r, N_A, N_D, 1e-6, 100)

    # Create a grid for visualization
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)

    # Get the potential, electron concentration, and hole concentration
    print("Getting results...")
    potential = solver.get_potential()
    n_conc = solver.get_n()
    p_conc = solver.get_p()

    # Create interpolators
    print("Creating interpolators...")
    simple_mesh = qdc.create_simple_mesh(mesh)
    interpolator = qdc.SimpleInterpolator(simple_mesh)

    # Calculate quantities on grid
    print("Calculating quantities on grid...")
    potential_grid = np.zeros((ny, nx))
    electron_conc = np.zeros((ny, nx))
    hole_conc = np.zeros((ny, nx))
    conduction_band = np.zeros((ny, nx))
    valence_band = np.zeros((ny, nx))

    # Constants
    kT = k_B * T / e_charge  # eV

    # Calculate Fermi levels
    E_F_p = -gaas.E_g + kT * np.log(gaas.N_v / N_A)
    E_F_n = -kT * np.log(algaas.N_c / N_D)

    # Use a progress indicator
    total_points = nx * ny
    point_count = 0

    for i in range(ny):
        for j in range(nx):
            point_count += 1
            if point_count % 500 == 0 or point_count == total_points:
                print(f"Processing point {point_count}/{total_points}...")

            # Get position
            xi, yi = X[i, j], Y[i, j]

            # Get material at this position
            mat = gaas if xi < 0 else algaas

            # Interpolate potential
            try:
                potential_grid[i, j] = interpolator.interpolate(xi, yi, potential)
            except:
                potential_grid[i, j] = 0.0

            # Calculate band edges
            V = potential_grid[i, j] / e_charge  # Convert from V to eV
            E_c = -e_charge * V - (E_F_p if xi < 0 else E_F_n)
            E_v = E_c - mat.E_g

            conduction_band[i, j] = E_c
            valence_band[i, j] = E_v

            # Interpolate carrier concentrations
            try:
                electron_conc[i, j] = interpolator.interpolate(xi, yi, n_conc)
                hole_conc[i, j] = interpolator.interpolate(xi, yi, p_conc)
            except:
                # Use Boltzmann approximation as fallback
                electron_conc[i, j] = n_conc_callback(xi, yi, V, mat)
                hole_conc[i, j] = p_conc_callback(xi, yi, V, mat)

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Plot potential
    ax1 = fig.add_subplot(gs[0, 0])
    mid_y = ny // 2
    ax1.plot(x, potential_grid[mid_y, :], 'b-', linewidth=2)
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('Potential (V)')
    ax1.set_title('Electrostatic Potential')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add junction position annotation
    ax1.axvline(x=junction_position, color='k', linestyle='-')

    # Plot carrier concentrations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(x, electron_conc[mid_y, :], 'b-', linewidth=2, label='n')
    ax2.semilogy(x, hole_conc[mid_y, :], 'r-', linewidth=2, label='p')
    ax2.semilogy(x, np.ones_like(x) * N_A, 'r--', linewidth=1, label='N_A')
    ax2.semilogy(x, np.ones_like(x) * N_D, 'b--', linewidth=1, label='N_D')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('Carrier Concentration (nm^-3)')
    ax2.set_title('Carrier Concentrations')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add junction position annotation
    ax2.axvline(x=junction_position, color='k', linestyle='-')

    # Plot band diagram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, conduction_band[mid_y, :], 'b-', linewidth=2, label='E_c')
    ax3.plot(x, valence_band[mid_y, :], 'r-', linewidth=2, label='E_v')
    
    # Plot Fermi levels
    E_F = np.zeros_like(x)
    for j in range(nx):
        E_F[j] = E_F_p if x[j] < junction_position else E_F_n
    ax3.plot(x, E_F, 'g--', linewidth=1, label='E_F')
    
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('Energy (eV)')
    ax3.set_title('Band Diagram')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Add junction position annotation
    ax3.axvline(x=junction_position, color='k', linestyle='-')

    # Plot electric field
    ax4 = fig.add_subplot(gs[1, 1])
    E_field = np.zeros_like(x)
    for j in range(1, nx-1):
        E_field[j] = -(potential_grid[mid_y, j+1] - potential_grid[mid_y, j-1]) / (x[j+1] - x[j-1])
    
    # Set boundary values
    E_field[0] = E_field[1]
    E_field[-1] = E_field[-2]
    
    ax4.plot(x, E_field, 'k-', linewidth=2)
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('Electric Field (V/nm)')
    ax4.set_title('Electric Field')
    ax4.grid(True, linestyle='--', alpha=0.7)

    # Add junction position annotation
    ax4.axvline(x=junction_position, color='k', linestyle='-')

    # Add heterojunction parameters annotation
    plt.figtext(0.5, 0.01,
                f'GaAs-AlGaAs Heterojunction, V_r = {V_r:.2f} V, '
                f'N_A = {N_A*1e27:.2e} m^-3, N_D = {N_D*1e27:.2e} m^-3',
                ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig('heterojunction_example.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
