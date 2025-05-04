#!/usr/bin/env python3
"""
Self-Consistent P-N Junction Example

This example demonstrates a physically accurate P-N junction model
with proper calculation of the potential from charge distributions
using a self-consistent Poisson solver.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend'))

# Import the QDSim modules
from qdsim.config import Config
from qdsim.pn_junction import PNJunction, SelfConsistentPNJunction

def create_config():
    """Create a configuration with realistic physical parameters."""
    config = Config()

    # Physical constants
    config.e_charge = 1.602e-19  # Elementary charge (C)
    config.m_e = 9.109e-31  # Electron mass (kg)
    config.k_B = 1.381e-23  # Boltzmann constant (J/K)
    config.epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
    config.h_bar = 1.055e-34  # Reduced Planck constant (J·s)
    config.T = 300  # Temperature (K)

    # Domain size - 100 nm x 50 nm
    config.Lx = 100e-9  # 100 nm
    config.Ly = 50e-9   # 50 nm

    # Material parameters for GaAs
    config.epsilon_r = 12.9  # Relative permittivity of GaAs
    config.m_star = 0.067 * config.m_e  # Effective mass in GaAs
    config.E_g = 1.42 * config.e_charge  # Band gap of GaAs (J)
    config.chi = 4.07 * config.e_charge  # Electron affinity of GaAs (J)

    # Doping parameters - realistic values for GaAs
    config.N_A = 5e22  # Acceptor concentration (m^-3)
    config.N_D = 5e22  # Donor concentration (m^-3)

    # P-N junction parameters
    config.junction_position = config.Lx / 2  # Junction at the center of the domain
    config.V_r = 0.5  # Reverse bias (V)

    return config

def compare_pn_junction_models(config):
    """Compare analytical and self-consistent P-N junction models."""
    # Create analytical P-N junction
    print("\nCreating analytical P-N junction model...")
    analytical_pn = PNJunction(config)

    # Create self-consistent P-N junction
    print("\nCreating self-consistent P-N junction model...")
    self_consistent_pn = SelfConsistentPNJunction(config)

    # Print junction parameters
    print("\nP-N Junction Parameters:")
    print(f"  Built-in potential: {analytical_pn.V_bi:.3f} V")
    print(f"  Reverse bias: {analytical_pn.V_r:.3f} V")
    print(f"  Total potential: {analytical_pn.V_total:.3f} V")
    print(f"  Depletion width: {analytical_pn.W*1e9:.3f} nm (W_p = {analytical_pn.W_p*1e9:.3f} nm, W_n = {analytical_pn.W_n*1e9:.3f} nm)")
    print(f"  Intrinsic carrier concentration: {analytical_pn.n_i:.3e} m^-3")

    # Create a grid for plotting
    x = np.linspace(0, config.Lx, 1000)
    y = config.Ly / 2  # Middle of the domain

    # Calculate properties for analytical model
    V_analytical = np.zeros_like(x)
    n_analytical = np.zeros_like(x)
    p_analytical = np.zeros_like(x)
    E_c_analytical = np.zeros_like(x)
    E_v_analytical = np.zeros_like(x)
    E_field_analytical = np.zeros_like(x)

    for i in range(len(x)):
        V_analytical[i] = analytical_pn.potential(x[i], y) / config.e_charge  # Convert to eV
        n_analytical[i] = analytical_pn.electron_concentration(x[i], y)
        p_analytical[i] = analytical_pn.hole_concentration(x[i], y)
        E_c_analytical[i] = analytical_pn.conduction_band_edge(x[i], y) / config.e_charge  # Convert to eV
        E_v_analytical[i] = analytical_pn.valence_band_edge(x[i], y) / config.e_charge  # Convert to eV
        E_field_analytical[i] = analytical_pn.electric_field(x[i], y)[0]

    # Calculate properties for self-consistent model
    V_self_consistent = np.zeros_like(x)
    n_self_consistent = np.zeros_like(x)
    p_self_consistent = np.zeros_like(x)
    E_field_self_consistent = np.zeros_like(x)

    for i in range(len(x)):
        V_self_consistent[i] = self_consistent_pn.potential_interpolated(x[i], y) / config.e_charge  # Convert to eV
        n_self_consistent[i] = self_consistent_pn.electron_concentration(x[i], y)
        p_self_consistent[i] = self_consistent_pn.hole_concentration(x[i], y)
        E_field_self_consistent[i] = self_consistent_pn.electric_field_interpolated(x[i], y)[0]

    # Calculate band edges for self-consistent model
    E_c_self_consistent = -config.chi / config.e_charge - V_self_consistent
    E_v_self_consistent = E_c_self_consistent - config.E_g / config.e_charge

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Plot potential
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x*1e9, V_analytical, 'b-', linewidth=2, label='Analytical')
    ax1.plot(x*1e9, V_self_consistent, 'r--', linewidth=2, label='Self-Consistent')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('Potential (eV)')
    ax1.set_title('Electrostatic Potential')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add depletion width annotation
    ax1.axvline(x=(analytical_pn.junction_position - analytical_pn.W_p)*1e9, color='b', linestyle='--')
    ax1.axvline(x=analytical_pn.junction_position*1e9, color='k', linestyle='-')
    ax1.axvline(x=(analytical_pn.junction_position + analytical_pn.W_n)*1e9, color='b', linestyle='--')

    # Plot carrier concentrations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(x*1e9, n_analytical, 'b-', linewidth=2, label='n (Analytical)')
    ax2.semilogy(x*1e9, p_analytical, 'g-', linewidth=2, label='p (Analytical)')
    ax2.semilogy(x*1e9, n_self_consistent, 'r--', linewidth=2, label='n (Self-Consistent)')
    ax2.semilogy(x*1e9, p_self_consistent, 'm--', linewidth=2, label='p (Self-Consistent)')
    ax2.semilogy(x*1e9, np.ones_like(x) * analytical_pn.n_i, 'k--', linewidth=1, label='n_i')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('Carrier Concentration (m^-3)')
    ax2.set_title('Carrier Concentrations')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add depletion width annotation
    ax2.axvline(x=(analytical_pn.junction_position - analytical_pn.W_p)*1e9, color='b', linestyle='--')
    ax2.axvline(x=analytical_pn.junction_position*1e9, color='k', linestyle='-')
    ax2.axvline(x=(analytical_pn.junction_position + analytical_pn.W_n)*1e9, color='b', linestyle='--')

    # Plot band diagram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x*1e9, E_c_analytical, 'b-', linewidth=2, label='E_c (Analytical)')
    ax3.plot(x*1e9, E_v_analytical, 'g-', linewidth=2, label='E_v (Analytical)')
    ax3.plot(x*1e9, E_c_self_consistent, 'r--', linewidth=2, label='E_c (Self-Consistent)')
    ax3.plot(x*1e9, E_v_self_consistent, 'm--', linewidth=2, label='E_v (Self-Consistent)')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('Energy (eV)')
    ax3.set_title('Band Diagram')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Add depletion width annotation
    ax3.axvline(x=(analytical_pn.junction_position - analytical_pn.W_p)*1e9, color='b', linestyle='--')
    ax3.axvline(x=analytical_pn.junction_position*1e9, color='k', linestyle='-')
    ax3.axvline(x=(analytical_pn.junction_position + analytical_pn.W_n)*1e9, color='b', linestyle='--')

    # Plot electric field
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x*1e9, E_field_analytical/1e6, 'b-', linewidth=2, label='Analytical')  # Convert to MV/m
    ax4.plot(x*1e9, E_field_self_consistent/1e6, 'r--', linewidth=2, label='Self-Consistent')  # Convert to MV/m
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('Electric Field (MV/m)')
    ax4.set_title('Electric Field')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)

    # Add depletion width annotation
    ax4.axvline(x=(analytical_pn.junction_position - analytical_pn.W_p)*1e9, color='b', linestyle='--')
    ax4.axvline(x=analytical_pn.junction_position*1e9, color='k', linestyle='-')
    ax4.axvline(x=(analytical_pn.junction_position + analytical_pn.W_n)*1e9, color='b', linestyle='--')

    # Add junction parameters annotation
    plt.figtext(0.5, 0.01,
                f'V_bi = {analytical_pn.V_bi:.2f} V, V_r = {analytical_pn.V_r:.2f} V, '
                f'N_A = {analytical_pn.N_A:.2e} m^-3, N_D = {analytical_pn.N_D:.2e} m^-3, '
                f'W = {analytical_pn.W*1e9:.2f} nm',
                ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig('self_consistent_pn_junction.png', dpi=300, bbox_inches='tight')

    return fig

def main():
    """Main function."""
    # Create a configuration with realistic parameters
    config = create_config()

    # Compare analytical and self-consistent P-N junction models
    print("\nComparing analytical and self-consistent P-N junction models...")
    compare_pn_junction_models(config)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
