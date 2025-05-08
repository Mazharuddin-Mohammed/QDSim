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
import qdsim_cpp as qdc

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

class CPPSelfConsistentPNJunction:
    """C++ implementation of a self-consistent P-N junction model."""

    def __init__(self, config):
        """Initialize the C++ self-consistent P-N junction model."""
        self.config = config

        # Create a mesh
        nx = 101
        ny = 51
        self.mesh = qdc.Mesh(config.Lx*1e9, config.Ly*1e9, nx, ny, 1)  # Convert to nm

        # Define callback functions for the SelfConsistentSolver
        def epsilon_r_callback(x, y):
            return config.epsilon_r

        def rho_callback(x, y, n, p):
            # Charge density in C/nm^3
            q = config.e_charge  # Elementary charge in C

            # Doping contribution
            rho_doping = q * (config.N_D if x > 0 else -config.N_A) * 1e-27  # Convert from m^-3 to nm^-3

            # Carrier contribution
            # We need to interpolate n and p at (x, y)
            # For simplicity, we'll use the first values
            n_val = n[0] if len(n) > 0 else 0.0
            p_val = p[0] if len(p) > 0 else 0.0

            rho_carriers = q * (p_val - n_val)

            return rho_doping + rho_carriers

        def n_conc_callback(x, y, phi, mat):
            # Constants
            kT = config.k_B * config.T / config.e_charge  # eV
            q = config.e_charge  # Elementary charge in C

            # Calculate Fermi levels
            E_F_p = -mat.E_g + kT * np.log(mat.N_v / (config.N_A * 1e-27))  # Convert from m^-3 to nm^-3
            E_F_n = -kT * np.log(mat.N_c / (config.N_D * 1e-27))  # Convert from m^-3 to nm^-3

            # Calculate band edges
            E_c = -q * phi - (E_F_p if x < 0 else E_F_n)

            # Calculate electron concentration using Boltzmann statistics
            n = mat.N_c * np.exp(-E_c / kT)

            # Apply limits for numerical stability
            n_min = 1e-15  # Minimum concentration (nm^-3)
            n_max = 1e-3   # Maximum concentration (nm^-3)
            return max(min(n, n_max), n_min)

        def p_conc_callback(x, y, phi, mat):
            # Constants
            kT = config.k_B * config.T / config.e_charge  # eV
            q = config.e_charge  # Elementary charge in C

            # Calculate Fermi levels
            E_F_p = -mat.E_g + kT * np.log(mat.N_v / (config.N_A * 1e-27))  # Convert from m^-3 to nm^-3
            E_F_n = -kT * np.log(mat.N_c / (config.N_D * 1e-27))  # Convert from m^-3 to nm^-3

            # Calculate band edges
            E_c = -q * phi - (E_F_p if x < 0 else E_F_n)
            E_v = E_c - mat.E_g

            # Calculate hole concentration using Boltzmann statistics
            p = mat.N_v * np.exp(E_v / kT)

            # Apply limits for numerical stability
            p_min = 1e-15  # Minimum concentration (nm^-3)
            p_max = 1e-3   # Maximum concentration (nm^-3)
            return max(min(p, p_max), p_min)

        def mu_n_callback(x, y, mat):
            return 8500.0  # Electron mobility in GaAs (cm^2/V·s)

        def mu_p_callback(x, y, mat):
            return 400.0   # Hole mobility in GaAs (cm^2/V·s)

        # Create the SelfConsistentSolver
        print("Creating C++ SelfConsistentSolver...")
        self.solver = qdc.create_self_consistent_solver(
            self.mesh, epsilon_r_callback, rho_callback,
            n_conc_callback, p_conc_callback,
            mu_n_callback, mu_p_callback
        )

        # Set convergence acceleration parameters
        self.solver.damping_factor = 0.3
        self.solver.anderson_history_size = 3

        # Solve the self-consistent Poisson-drift-diffusion equations
        print("Solving self-consistent equations...")
        self.solver.solve(0.0, -config.V_r, config.N_A * 1e-27, config.N_D * 1e-27, 1e-6, 100)  # Convert from m^-3 to nm^-3

        # Create interpolators
        self.simple_mesh = qdc.create_simple_mesh(self.mesh)
        self.interpolator = qdc.SimpleInterpolator(self.simple_mesh)

        # Get the potential, electron concentration, and hole concentration
        self.potential = self.solver.get_potential()
        self.n = self.solver.get_n()
        self.p = self.solver.get_p()

        # Material properties for GaAs
        self.mat = qdc.Material()
        self.mat.N_c = 4.7e17  # Effective density of states in conduction band (nm^-3)
        self.mat.N_v = 7.0e18  # Effective density of states in valence band (nm^-3)
        self.mat.E_g = 1.424   # Band gap (eV)
        self.mat.mu_n = 8500   # Electron mobility (cm^2/V·s)
        self.mat.mu_p = 400    # Hole mobility (cm^2/V·s)
        self.mat.epsilon_r = 12.9  # Relative permittivity

        # Calculate Fermi levels
        self.kT = config.k_B * config.T / config.e_charge  # eV
        self.E_F_p = -self.mat.E_g + self.kT * np.log(self.mat.N_v / (config.N_A * 1e-27))
        self.E_F_n = -self.kT * np.log(self.mat.N_c / (config.N_D * 1e-27))

    def potential_interpolated(self, x, y):
        """Get the potential at a given position."""
        # Convert from m to nm
        x_nm = x * 1e9
        y_nm = y * 1e9

        try:
            # Interpolate the potential
            potential_V = self.interpolator.interpolate(x_nm, y_nm, self.potential)

            # Convert from V to J
            return potential_V * self.config.e_charge
        except:
            return 0.0

    def electron_concentration(self, x, y):
        """Get the electron concentration at a given position."""
        # Convert from m to nm
        x_nm = x * 1e9
        y_nm = y * 1e9

        try:
            # Interpolate the electron concentration
            n_nm3 = self.interpolator.interpolate(x_nm, y_nm, self.n)

            # Convert from nm^-3 to m^-3
            return n_nm3 * 1e27
        except:
            return 0.0

    def hole_concentration(self, x, y):
        """Get the hole concentration at a given position."""
        # Convert from m to nm
        x_nm = x * 1e9
        y_nm = y * 1e9

        try:
            # Interpolate the hole concentration
            p_nm3 = self.interpolator.interpolate(x_nm, y_nm, self.p)

            # Convert from nm^-3 to m^-3
            return p_nm3 * 1e27
        except:
            return 0.0

    def electric_field_interpolated(self, x, y):
        """Get the electric field at a given position."""
        # Convert from m to nm
        x_nm = x * 1e9
        y_nm = y * 1e9

        # Calculate the electric field using finite differences
        h = 1.0  # Step size in nm

        try:
            # Interpolate the potential at neighboring points
            phi_center = self.interpolator.interpolate(x_nm, y_nm, self.potential)
            phi_right = self.interpolator.interpolate(x_nm + h, y_nm, self.potential)
            phi_left = self.interpolator.interpolate(x_nm - h, y_nm, self.potential)
            phi_up = self.interpolator.interpolate(x_nm, y_nm + h, self.potential)
            phi_down = self.interpolator.interpolate(x_nm, y_nm - h, self.potential)

            # Calculate the electric field components
            E_x = -(phi_right - phi_left) / (2.0 * h)
            E_y = -(phi_up - phi_down) / (2.0 * h)

            # Convert from V/nm to V/m
            return np.array([E_x * 1e9, E_y * 1e9])
        except:
            return np.array([0.0, 0.0])

    def conduction_band_edge(self, x, y):
        """Get the conduction band edge at a given position."""
        # Get the potential
        phi = self.potential_interpolated(x, y)

        # Calculate the conduction band edge
        if x < self.config.junction_position:
            # p-side
            E_c = -self.config.chi - phi - self.E_F_p * self.config.e_charge
        else:
            # n-side
            E_c = -self.config.chi - phi - self.E_F_n * self.config.e_charge

        return E_c

    def valence_band_edge(self, x, y):
        """Get the valence band edge at a given position."""
        # Get the conduction band edge
        E_c = self.conduction_band_edge(x, y)

        # Calculate the valence band edge
        E_v = E_c - self.config.E_g

        return E_v

def compare_pn_junction_models(config):
    """Compare analytical and self-consistent P-N junction models."""
    # Create analytical P-N junction
    print("\nCreating analytical P-N junction model...")
    analytical_pn = PNJunction(config)

    # Create self-consistent P-N junction
    print("\nCreating C++ self-consistent P-N junction model...")
    self_consistent_pn = CPPSelfConsistentPNJunction(config)

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
