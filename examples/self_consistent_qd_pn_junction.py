#!/usr/bin/env python3
"""
Self-Consistent Quantum Dot at P-N Junction Example

This example demonstrates a quantum dot positioned at the interface of a P-N junction,
using a physically accurate P-N junction model with proper calculation of the potential
from charge distributions using a self-consistent Poisson solver.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend'))

# Import the QDSim modules
from qdsim.config import Config
from qdsim.pn_junction import PNJunction, SelfConsistentPNJunction
from qdsim.simulator import Simulator
from qdsim.analysis import calculate_transition_energies, find_bound_states

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
    
    # Mesh parameters - fine mesh for accurate results
    config.nx = 201  # Number of elements in x direction
    config.ny = 101  # Number of elements in y direction
    config.element_order = 1  # Linear elements
    
    # Material parameters for GaAs
    config.diode_p_material = "GaAs"
    config.diode_n_material = "GaAs"
    config.matrix_material = "GaAs"
    config.qd_material = "InAs"
    
    # GaAs parameters
    config.epsilon_r = 12.9  # Relative permittivity of GaAs
    config.m_star = 0.067 * config.m_e  # Effective mass in GaAs
    config.E_g = 1.42 * config.e_charge  # Band gap of GaAs (J)
    config.chi = 4.07 * config.e_charge  # Electron affinity of GaAs (J)
    
    # InAs parameters (for the quantum dot)
    config.m_star_qd = 0.023 * config.m_e  # Effective mass in InAs
    config.E_g_qd = 0.354 * config.e_charge  # Band gap of InAs (J)
    config.chi_qd = 4.9 * config.e_charge  # Electron affinity of InAs (J)
    
    # Doping parameters - realistic values for GaAs
    config.N_A = 5e22  # Acceptor concentration (m^-3)
    config.N_D = 5e22  # Donor concentration (m^-3)
    
    # P-N junction parameters
    config.junction_position = config.Lx / 2  # Junction at the center of the domain
    config.V_r = 0.5  # Reverse bias (V)
    
    # Quantum dot parameters
    config.R = 5e-9  # 5 nm radius - smaller for more realistic QD
    config.V_0 = 0.3 * config.e_charge  # 0.3 eV depth - more realistic potential depth
    config.potential_type = "gaussian"  # Gaussian potential
    
    # Position the quantum dot at the junction interface
    config.qd_position_x = config.junction_position
    config.qd_position_y = config.Ly / 2
    
    return config

def create_combined_potential(config, pn_junction):
    """Create a combined potential function for the P-N junction and quantum dot."""
    def combined_potential(x, y):
        # P-N junction potential
        V_pn = pn_junction.potential_interpolated(x, y)
        
        # Quantum dot potential (Gaussian)
        r = np.sqrt((x - config.qd_position_x)**2 + (y - config.qd_position_y)**2)
        if config.potential_type == "gaussian":
            V_qd = -config.V_0 * np.exp(-r**2 / (2 * config.R**2))
        else:  # square well
            V_qd = -config.V_0 if r <= config.R else 0
        
        # Combined potential
        return V_pn + V_qd
    
    return combined_potential

def create_effective_mass_function(config):
    """Create an effective mass function for the quantum dot."""
    def effective_mass_function(x, y):
        # Check if the point is inside the quantum dot
        r = np.sqrt((x - config.qd_position_x)**2 + (y - config.qd_position_y)**2)
        if r <= 2*config.R:  # Use 2*R as the effective QD boundary for smooth transition
            # Smoothly transition from QD to matrix material
            alpha = np.exp(-r**2 / (2 * config.R**2))
            return alpha * config.m_star_qd + (1 - alpha) * config.m_star
        else:
            return config.m_star
    
    return effective_mass_function

def plot_combined_potential(config, combined_potential):
    """Plot the combined potential of the P-N junction and quantum dot."""
    # Create a grid for plotting
    x = np.linspace(0, config.Lx, 500)
    y = np.linspace(0, config.Ly, 250)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the combined potential
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = combined_potential(X[i, j], Y[i, j])
    
    # Convert to eV for plotting
    Z_eV = Z / config.e_charge
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot the combined potential in 3D
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf1 = ax1.plot_surface(X*1e9, Y*1e9, Z_eV, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_zlabel('Potential (eV)')
    ax1.set_title('Combined Potential (3D)')
    ax1.view_init(elev=30, azim=45)
    
    # Plot the combined potential in 2D
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(X*1e9, Y*1e9, Z_eV, cmap='viridis', shading='auto')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_title('Combined Potential (2D)')
    plt.colorbar(im2, ax=ax2, label='Potential (eV)')
    
    # Add annotations
    pn_junction = PNJunction(config)
    ax2.axvline(x=(pn_junction.junction_position - pn_junction.W_p)*1e9, color='r', linestyle='--')
    ax2.axvline(x=pn_junction.junction_position*1e9, color='k', linestyle='-')
    ax2.axvline(x=(pn_junction.junction_position + pn_junction.W_n)*1e9, color='r', linestyle='--')
    circle = plt.Circle((config.qd_position_x*1e9, config.qd_position_y*1e9), config.R*1e9, 
                        fill=False, color='w', linestyle='--')
    ax2.add_patch(circle)
    
    # Plot a 1D slice of the potential along y = Ly/2
    ax3 = fig.add_subplot(gs[1, 0])
    y_mid = config.Ly / 2
    Z_slice = np.zeros_like(x)
    for i in range(len(x)):
        Z_slice[i] = combined_potential(x[i], y_mid)
    
    # Convert to eV for plotting
    Z_slice_eV = Z_slice / config.e_charge
    
    ax3.plot(x*1e9, Z_slice_eV, 'b-', linewidth=2)
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('Potential (eV)')
    ax3.set_title('Potential Slice along y = Ly/2')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations
    ax3.axvline(x=(pn_junction.junction_position - pn_junction.W_p)*1e9, color='r', linestyle='--')
    ax3.axvline(x=pn_junction.junction_position*1e9, color='k', linestyle='-')
    ax3.axvline(x=(pn_junction.junction_position + pn_junction.W_n)*1e9, color='r', linestyle='--')
    
    # Plot a 1D slice of the potential along x = junction_position
    ax4 = fig.add_subplot(gs[1, 1])
    x_mid = config.junction_position
    Z_slice = np.zeros_like(y)
    for i in range(len(y)):
        Z_slice[i] = combined_potential(x_mid, y[i])
    
    # Convert to eV for plotting
    Z_slice_eV = Z_slice / config.e_charge
    
    ax4.plot(y*1e9, Z_slice_eV, 'r-', linewidth=2)
    ax4.set_xlabel('y (nm)')
    ax4.set_ylabel('Potential (eV)')
    ax4.set_title('Potential Slice along x = junction_position')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations
    ax4.axvline(x=config.qd_position_y*1e9, color='k', linestyle='-')
    ax4.axvline(x=(config.qd_position_y - config.R)*1e9, color='r', linestyle='--')
    ax4.axvline(x=(config.qd_position_y + config.R)*1e9, color='r', linestyle='--')
    
    # Add potential values annotation
    plt.figtext(0.5, 0.01, 
                f'V_bi = {pn_junction.V_bi:.2f} V, V_r = {pn_junction.V_r:.2f} V, '
                f'QD depth = {config.V_0/config.e_charge:.2f} eV, QD radius = {config.R*1e9:.1f} nm',
                ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('self_consistent_qd_pn_combined_potential.png', dpi=300, bbox_inches='tight')
    
    return fig

def main():
    """Main function."""
    # Create a configuration with realistic parameters
    config = create_config()
    
    # Create a self-consistent P-N junction
    print("\nCreating self-consistent P-N junction model...")
    self_consistent_pn = SelfConsistentPNJunction(config)
    
    # Print junction parameters
    print("\nP-N Junction Parameters:")
    print(f"  Built-in potential: {self_consistent_pn.V_bi:.3f} V")
    print(f"  Reverse bias: {self_consistent_pn.V_r:.3f} V")
    print(f"  Total potential: {self_consistent_pn.V_total:.3f} V")
    print(f"  Depletion width: {self_consistent_pn.W*1e9:.3f} nm (W_p = {self_consistent_pn.W_p*1e9:.3f} nm, W_n = {self_consistent_pn.W_n*1e9:.3f} nm)")
    print(f"  Quantum dot: {config.qd_material} in {config.matrix_material}, R = {config.R*1e9:.1f} nm")
    print(f"  QD potential depth: {config.V_0/config.e_charge:.3f} eV")
    print(f"  QD position: ({config.qd_position_x*1e9:.1f}, {config.qd_position_y*1e9:.1f}) nm")
    
    # Create combined potential function
    combined_potential = create_combined_potential(config, self_consistent_pn)
    
    # Create effective mass function
    effective_mass_function = create_effective_mass_function(config)
    
    # Set the potential and effective mass functions in the config
    config.potential_function = combined_potential
    config.m_star_function = effective_mass_function
    
    # Plot the combined potential
    print("\nPlotting combined potential...")
    plot_combined_potential(config, combined_potential)
    
    # Create the simulator
    print("\nCreating simulator...")
    sim = Simulator(config)
    
    # Solve for eigenvalues
    print("\nSolving for eigenvalues...")
    num_states = 10
    eigenvalues, eigenvectors = sim.solve(num_states)
    
    # Print the eigenvalues
    print("\nEigenvalues (eV):")
    for i, ev in enumerate(eigenvalues):
        print(f"  {i}: {ev/config.e_charge:.6f}")
    
    # Find bound states
    bound_states = find_bound_states(eigenvalues, 0.0, config.e_charge)
    print("\nBound states:")
    if bound_states:
        for i in bound_states:
            energy = eigenvalues[i] / config.e_charge
            print(f"  State {i}: {energy:.6f} eV")
    else:
        print("  No bound states found")
    
    # Calculate transition energies
    transitions = calculate_transition_energies(eigenvalues, config.e_charge)
    print("\nTransition energies (eV):")
    for i in range(min(3, len(eigenvalues))):
        for j in range(i+1, min(3, len(eigenvalues))):
            print(f"  {i} -> {j}: {transitions[i, j]:.6f}")
    
    # Create a dashboard of results
    print("\nCreating results dashboard...")
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot the potential in 3D
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Create a grid for plotting
    x = np.linspace(0, config.Lx, 100)
    y = np.linspace(0, config.Ly, 50)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the potential on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = combined_potential(X[i, j], Y[i, j])
    
    # Convert to eV for plotting
    Z_eV = Z / config.e_charge
    
    # Plot the potential
    surf1 = ax1.plot_surface(X*1e9, Y*1e9, Z_eV, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_zlabel('Potential (eV)')
    ax1.set_title('Combined Potential')
    ax1.view_init(elev=30, azim=45)
    
    # Plot the potential in 2D
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(X*1e9, Y*1e9, Z_eV, cmap='viridis', shading='auto')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_title('Combined Potential')
    plt.colorbar(im2, ax=ax2, label='Potential (eV)')
    
    # Add annotations
    ax2.axvline(x=(self_consistent_pn.junction_position - self_consistent_pn.W_p)*1e9, color='r', linestyle='--')
    ax2.axvline(x=self_consistent_pn.junction_position*1e9, color='k', linestyle='-')
    ax2.axvline(x=(self_consistent_pn.junction_position + self_consistent_pn.W_n)*1e9, color='r', linestyle='--')
    circle = plt.Circle((config.qd_position_x*1e9, config.qd_position_y*1e9), config.R*1e9, 
                        fill=False, color='w', linestyle='--')
    ax2.add_patch(circle)
    
    # Plot energy levels
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Convert eigenvalues to eV
    energies = [ev.real / config.e_charge for ev in eigenvalues[:num_states]]
    
    # Plot the energy levels
    for i, energy in enumerate(energies):
        if i in bound_states:
            ax3.axhline(y=energy, color='r', linestyle='-', linewidth=2)
            ax3.text(1.02, energy, f'{energy:.6f} eV (bound)', va='center', color='r')
        else:
            ax3.axhline(y=energy, color='b', linestyle='-', linewidth=1)
            ax3.text(1.02, energy, f'{energy:.6f} eV', va='center')
    
    # Set the labels and title
    ax3.set_xlabel('State')
    ax3.set_ylabel('Energy (eV)')
    ax3.set_title('Energy Levels')
    
    # Set the x-axis limits and ticks
    ax3.set_xlim(0, 1)
    ax3.set_xticks([])
    
    # Set the y-axis limits with some padding
    if len(energies) > 0:
        y_min = min(energies) - 0.1 * (max(energies) - min(energies))
        y_max = max(energies) + 0.1 * (max(energies) - min(energies))
        ax3.set_ylim(y_min, y_max)
    
    # Plot the wavefunctions
    for i in range(min(3, len(bound_states))):
        state_idx = bound_states[i]
        ax = fig.add_subplot(gs[1, i], projection='3d')
        
        # Interpolate the wavefunction onto the grid
        Z_wf = np.zeros_like(X)
        try:
            for j in range(X.shape[0]):
                for k in range(X.shape[1]):
                    # Get the wavefunction value at this point
                    wf_value = sim.interpolator.interpolate(X[j, k], Y[j, k], eigenvectors[:, state_idx])
                    # Calculate the probability density
                    Z_wf[j, k] = abs(wf_value)**2
        except Exception as e:
            print(f"Warning: Failed to interpolate wavefunction: {e}")
        
        # Normalize the wavefunction for better visualization
        if np.max(Z_wf) > 0:
            Z_wf = Z_wf / np.max(Z_wf)
        
        # Plot the wavefunction
        surf = ax.plot_surface(X*1e9, Y*1e9, Z_wf, cmap='plasma', alpha=0.8)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('Probability density')
        
        # Set the title
        energy = eigenvalues[state_idx] / config.e_charge
        ax.set_title(f'State {state_idx} (E = {energy:.6f} eV)')
        
        # Set the view angle
        ax.view_init(elev=30, azim=45)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('self_consistent_qd_pn_results.png', dpi=300, bbox_inches='tight')
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
