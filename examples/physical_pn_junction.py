#!/usr/bin/env python3
"""
Physical P-N Junction Example

This example demonstrates a physically accurate P-N junction model
with proper calculation of the potential from doping concentrations
and quasi-Fermi levels.

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
from qdsim.pn_junction import PNJunction

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

def plot_pn_junction(pn_junction, config):
    """Plot the P-N junction properties."""
    # Create a grid for plotting
    x = np.linspace(0, config.Lx, 1000)
    y = config.Ly / 2  # Middle of the domain
    
    # Calculate properties
    V = np.zeros_like(x)
    n = np.zeros_like(x)
    p = np.zeros_like(x)
    E_c = np.zeros_like(x)
    E_v = np.zeros_like(x)
    E_Fn = np.zeros_like(x)
    E_Fp = np.zeros_like(x)
    
    for i in range(len(x)):
        V[i] = pn_junction.potential(x[i], y) / config.e_charge  # Convert to eV
        n[i] = pn_junction.electron_concentration(x[i], y)
        p[i] = pn_junction.hole_concentration(x[i], y)
        E_c[i] = pn_junction.conduction_band_edge(x[i], y) / config.e_charge  # Convert to eV
        E_v[i] = pn_junction.valence_band_edge(x[i], y) / config.e_charge  # Convert to eV
        E_Fn[i] = pn_junction.quasi_fermi_level_electrons(x[i], y) / config.e_charge  # Convert to eV
        E_Fp[i] = pn_junction.quasi_fermi_level_holes(x[i], y) / config.e_charge  # Convert to eV
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot potential
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x*1e9, V, 'b-', linewidth=2)
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('Potential (eV)')
    ax1.set_title('Electrostatic Potential')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add depletion width annotation
    ax1.axvline(x=(pn_junction.junction_position - pn_junction.W_p)*1e9, color='r', linestyle='--')
    ax1.axvline(x=pn_junction.junction_position*1e9, color='k', linestyle='-')
    ax1.axvline(x=(pn_junction.junction_position + pn_junction.W_n)*1e9, color='r', linestyle='--')
    
    ax1.text((pn_junction.junction_position - pn_junction.W_p/2)*1e9, 0.1, f'W_p = {pn_junction.W_p*1e9:.1f} nm', 
            ha='center', va='bottom')
    ax1.text((pn_junction.junction_position + pn_junction.W_n/2)*1e9, 0.1, f'W_n = {pn_junction.W_n*1e9:.1f} nm', 
            ha='center', va='bottom')
    
    # Plot carrier concentrations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(x*1e9, n, 'b-', linewidth=2, label='Electrons')
    ax2.semilogy(x*1e9, p, 'r-', linewidth=2, label='Holes')
    ax2.semilogy(x*1e9, np.ones_like(x) * pn_junction.n_i, 'k--', linewidth=1, label='n_i')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('Carrier Concentration (m^-3)')
    ax2.set_title('Carrier Concentrations')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add depletion width annotation
    ax2.axvline(x=(pn_junction.junction_position - pn_junction.W_p)*1e9, color='r', linestyle='--')
    ax2.axvline(x=pn_junction.junction_position*1e9, color='k', linestyle='-')
    ax2.axvline(x=(pn_junction.junction_position + pn_junction.W_n)*1e9, color='r', linestyle='--')
    
    # Plot band diagram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x*1e9, E_c, 'b-', linewidth=2, label='E_c')
    ax3.plot(x*1e9, E_v, 'r-', linewidth=2, label='E_v')
    ax3.plot(x*1e9, E_Fn, 'b--', linewidth=1, label='E_Fn')
    ax3.plot(x*1e9, E_Fp, 'r--', linewidth=1, label='E_Fp')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('Energy (eV)')
    ax3.set_title('Band Diagram')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Add depletion width annotation
    ax3.axvline(x=(pn_junction.junction_position - pn_junction.W_p)*1e9, color='r', linestyle='--')
    ax3.axvline(x=pn_junction.junction_position*1e9, color='k', linestyle='-')
    ax3.axvline(x=(pn_junction.junction_position + pn_junction.W_n)*1e9, color='r', linestyle='--')
    
    # Plot electric field
    ax4 = fig.add_subplot(gs[1, 1])
    E_field = np.zeros_like(x)
    for i in range(len(x)):
        E_field[i] = pn_junction.electric_field(x[i], y)[0]
    
    ax4.plot(x*1e9, E_field/1e6, 'g-', linewidth=2)  # Convert to MV/m
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('Electric Field (MV/m)')
    ax4.set_title('Electric Field')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add depletion width annotation
    ax4.axvline(x=(pn_junction.junction_position - pn_junction.W_p)*1e9, color='r', linestyle='--')
    ax4.axvline(x=pn_junction.junction_position*1e9, color='k', linestyle='-')
    ax4.axvline(x=(pn_junction.junction_position + pn_junction.W_n)*1e9, color='r', linestyle='--')
    
    # Add junction parameters annotation
    plt.figtext(0.5, 0.01, 
                f'V_bi = {pn_junction.V_bi:.2f} V, V_r = {pn_junction.V_r:.2f} V, '
                f'N_A = {pn_junction.N_A:.2e} m^-3, N_D = {pn_junction.N_D:.2e} m^-3, '
                f'W = {pn_junction.W*1e9:.2f} nm',
                ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('physical_pn_junction.png', dpi=300, bbox_inches='tight')
    
    return fig

def main():
    """Main function."""
    # Create a configuration with realistic parameters
    config = create_config()
    
    # Create a P-N junction
    pn_junction = PNJunction(config)
    
    # Print junction parameters
    print("\nP-N Junction Parameters:")
    print(f"  Built-in potential: {pn_junction.V_bi:.3f} V")
    print(f"  Reverse bias: {pn_junction.V_r:.3f} V")
    print(f"  Total potential: {pn_junction.V_total:.3f} V")
    print(f"  Depletion width: {pn_junction.W*1e9:.3f} nm (W_p = {pn_junction.W_p*1e9:.3f} nm, W_n = {pn_junction.W_n*1e9:.3f} nm)")
    print(f"  Intrinsic carrier concentration: {pn_junction.n_i:.3e} m^-3")
    print(f"  Quasi-Fermi level (p-side): {pn_junction.E_F_p/config.e_charge:.3f} eV")
    print(f"  Quasi-Fermi level (n-side): {pn_junction.E_F_n/config.e_charge:.3f} eV")
    
    # Plot the P-N junction
    print("\nPlotting P-N junction properties...")
    plot_pn_junction(pn_junction, config)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
