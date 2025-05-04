#!/usr/bin/env python3
"""
Quantum Dot at P-N Junction Example (Fixed Version)

This example demonstrates a quantum dot positioned at the interface of a P-N junction.
It shows the combined potential of the quantum dot and the P-N junction depletion region,
and calculates the bound states in this potential.

This version fixes the P-N junction potential shape to be step-like in reverse bias,
and ensures proper physical units throughout the simulation.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend'))

# Import the QDSim modules
from qdsim.config import Config
from qdsim.simulator import Simulator
from qdsim.visualization import plot_potential, plot_wavefunction
from qdsim.analysis import calculate_transition_energies, find_bound_states

def create_realistic_config():
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
    config.N_A = 5e22  # Acceptor concentration (m^-3) - reduced for smaller depletion width
    config.N_D = 5e22  # Donor concentration (m^-3) - reduced for smaller depletion width
    
    # P-N junction parameters
    config.junction_position = config.Lx / 2  # Junction at the center of the domain
    config.V_bi = (config.k_B * config.T / config.e_charge) * np.log(config.N_A * config.N_D / (1e12 * 1e12))  # Built-in potential (V)
    config.V_r = 0.5  # Reverse bias (V)
    config.V_total = config.V_bi + config.V_r  # Total potential across the junction (V)
    
    # Calculate depletion width using the depletion approximation
    config.W = np.sqrt(2 * config.epsilon_0 * config.epsilon_r * config.V_total / config.e_charge * 
                      (1/config.N_A + 1/config.N_D))  # Depletion width (m)
    config.W_p = config.W * config.N_D / (config.N_A + config.N_D)  # P-side depletion width (m)
    config.W_n = config.W * config.N_A / (config.N_A + config.N_D)  # N-side depletion width (m)
    
    # Quantum dot parameters
    config.R = 5e-9  # 5 nm radius - smaller for more realistic QD
    config.V_0 = 0.3 * config.e_charge  # 0.3 eV depth - more realistic potential depth
    config.potential_type = "gaussian"  # Gaussian potential
    
    # Position the quantum dot at the junction interface
    config.qd_position_x = config.junction_position
    config.qd_position_y = config.Ly / 2
    
    # Define custom potential function to combine P-N junction and QD potentials
    def custom_potential(x, y):
        # P-N junction potential (step-like in reverse bias)
        if x < config.junction_position - config.W_p:
            # P-side outside depletion region
            V_pn = 0
        elif x > config.junction_position + config.W_n:
            # N-side outside depletion region
            V_pn = -config.V_total * config.e_charge
        else:
            # Inside depletion region - step function at junction
            V_pn = -config.V_total * config.e_charge * (x >= config.junction_position)
        
        # Quantum dot potential (Gaussian)
        r = np.sqrt((x - config.qd_position_x)**2 + (y - config.qd_position_y)**2)
        if config.potential_type == "gaussian":
            V_qd = -config.V_0 * np.exp(-r**2 / (2 * config.R**2))
        else:  # square well
            V_qd = -config.V_0 if r <= config.R else 0
        
        # Combined potential
        return V_pn + V_qd
    
    # Set the custom potential function
    config.potential_function = custom_potential
    
    # Define effective mass function (position-dependent)
    def m_star_function(x, y):
        # Check if the point is inside the quantum dot
        r = np.sqrt((x - config.qd_position_x)**2 + (y - config.qd_position_y)**2)
        if r <= 2*config.R:  # Use 2*R as the effective QD boundary for smooth transition
            # Smoothly transition from QD to matrix material
            alpha = np.exp(-r**2 / (2 * config.R**2))
            return alpha * config.m_star_qd + (1 - alpha) * config.m_star
        else:
            return config.m_star
    
    # Set the effective mass function
    config.m_star_function = m_star_function
    
    # Solver parameters
    config.tolerance = 1e-6
    config.max_iter = 100
    config.use_mpi = False
    
    return config

def plot_separate_potentials(config):
    """Plot the P-N junction and QD potentials separately."""
    # Create a grid for plotting
    x = np.linspace(0, config.Lx, 500)
    y = np.linspace(0, config.Ly, 250)
    X, Y = np.meshgrid(x, y)
    
    # Calculate P-N junction potential
    V_pn = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] < config.junction_position - config.W_p:
                # P-side outside depletion region
                V_pn[i, j] = 0
            elif X[i, j] > config.junction_position + config.W_n:
                # N-side outside depletion region
                V_pn[i, j] = -config.V_total * config.e_charge
            else:
                # Inside depletion region - step function at junction
                V_pn[i, j] = -config.V_total * config.e_charge * (X[i, j] >= config.junction_position)
    
    # Calculate QD potential
    V_qd = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r = np.sqrt((X[i, j] - config.qd_position_x)**2 + (Y[i, j] - config.qd_position_y)**2)
            if config.potential_type == "gaussian":
                V_qd[i, j] = -config.V_0 * np.exp(-r**2 / (2 * config.R**2))
            else:  # square well
                V_qd[i, j] = -config.V_0 if r <= config.R else 0
    
    # Calculate combined potential
    V_combined = V_pn + V_qd
    
    # Convert to eV for plotting
    V_pn_eV = V_pn / config.e_charge
    V_qd_eV = V_qd / config.e_charge
    V_combined_eV = V_combined / config.e_charge
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Plot P-N junction potential in 3D
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf1 = ax1.plot_surface(X*1e9, Y*1e9, V_pn_eV, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_zlabel('Potential (eV)')
    ax1.set_title('P-N Junction Potential')
    ax1.view_init(elev=30, azim=45)
    
    # Plot QD potential in 3D
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    surf2 = ax2.plot_surface(X*1e9, Y*1e9, V_qd_eV, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_zlabel('Potential (eV)')
    ax2.set_title('Quantum Dot Potential')
    ax2.view_init(elev=30, azim=45)
    
    # Plot combined potential in 3D
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    surf3 = ax3.plot_surface(X*1e9, Y*1e9, V_combined_eV, cmap='inferno', alpha=0.8)
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    ax3.set_zlabel('Potential (eV)')
    ax3.set_title('Combined Potential')
    ax3.view_init(elev=30, azim=45)
    
    # Plot P-N junction potential in 2D
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.pcolormesh(X*1e9, Y*1e9, V_pn_eV, cmap='viridis', shading='auto')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')
    ax4.set_title('P-N Junction Potential')
    plt.colorbar(im4, ax=ax4, label='Potential (eV)')
    
    # Plot QD potential in 2D
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.pcolormesh(X*1e9, Y*1e9, V_qd_eV, cmap='plasma', shading='auto')
    ax5.set_xlabel('x (nm)')
    ax5.set_ylabel('y (nm)')
    ax5.set_title('Quantum Dot Potential')
    plt.colorbar(im5, ax=ax5, label='Potential (eV)')
    
    # Plot combined potential in 2D
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.pcolormesh(X*1e9, Y*1e9, V_combined_eV, cmap='inferno', shading='auto')
    ax6.set_xlabel('x (nm)')
    ax6.set_ylabel('y (nm)')
    ax6.set_title('Combined Potential')
    plt.colorbar(im6, ax=ax6, label='Potential (eV)')
    
    # Add depletion width annotation
    ax4.axvline(x=(config.junction_position - config.W_p)*1e9, color='r', linestyle='--')
    ax4.axvline(x=config.junction_position*1e9, color='k', linestyle='-')
    ax4.axvline(x=(config.junction_position + config.W_n)*1e9, color='r', linestyle='--')
    ax4.text((config.junction_position - config.W_p/2)*1e9, 5, f'W_p = {config.W_p*1e9:.1f} nm', 
             ha='center', va='bottom', color='r')
    ax4.text((config.junction_position + config.W_n/2)*1e9, 5, f'W_n = {config.W_n*1e9:.1f} nm', 
             ha='center', va='bottom', color='r')
    
    # Add QD annotation
    circle = plt.Circle((config.qd_position_x*1e9, config.qd_position_y*1e9), config.R*1e9, 
                        fill=False, color='w', linestyle='--')
    ax5.add_patch(circle)
    ax5.text(config.qd_position_x*1e9, (config.qd_position_y + 1.5*config.R)*1e9, 
             f'QD: R = {config.R*1e9:.1f} nm', ha='center', va='bottom', color='w')
    
    # Add combined annotations
    ax6.axvline(x=(config.junction_position - config.W_p)*1e9, color='r', linestyle='--')
    ax6.axvline(x=config.junction_position*1e9, color='k', linestyle='-')
    ax6.axvline(x=(config.junction_position + config.W_n)*1e9, color='r', linestyle='--')
    circle = plt.Circle((config.qd_position_x*1e9, config.qd_position_y*1e9), config.R*1e9, 
                        fill=False, color='w', linestyle='--')
    ax6.add_patch(circle)
    
    # Add potential values annotation
    ax6.text(5, 5, f'V_bi = {config.V_bi:.2f} V\nV_r = {config.V_r:.2f} V\nV_0 = {config.V_0/config.e_charge:.2f} eV', 
             ha='left', va='bottom', color='w', bbox=dict(facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('fixed_qd_pn_potentials.png', dpi=300, bbox_inches='tight')
    
    return fig

def plot_1d_potential_slice(config):
    """Plot a 1D slice of the potential along the x-axis."""
    # Create a grid for plotting
    x = np.linspace(0, config.Lx, 1000)
    y = config.Ly / 2  # Middle of the domain
    
    # Calculate potentials
    V_pn = np.zeros_like(x)
    V_qd = np.zeros_like(x)
    V_combined = np.zeros_like(x)
    
    for i in range(len(x)):
        # P-N junction potential
        if x[i] < config.junction_position - config.W_p:
            # P-side outside depletion region
            V_pn[i] = 0
        elif x[i] > config.junction_position + config.W_n:
            # N-side outside depletion region
            V_pn[i] = -config.V_total * config.e_charge
        else:
            # Inside depletion region - step function at junction
            V_pn[i] = -config.V_total * config.e_charge * (x[i] >= config.junction_position)
        
        # Quantum dot potential
        r = np.sqrt((x[i] - config.qd_position_x)**2 + (y - config.qd_position_y)**2)
        if config.potential_type == "gaussian":
            V_qd[i] = -config.V_0 * np.exp(-r**2 / (2 * config.R**2))
        else:  # square well
            V_qd[i] = -config.V_0 if r <= config.R else 0
        
        # Combined potential
        V_combined[i] = V_pn[i] + V_qd[i]
    
    # Convert to eV for plotting
    V_pn_eV = V_pn / config.e_charge
    V_qd_eV = V_qd / config.e_charge
    V_combined_eV = V_combined / config.e_charge
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot potentials
    ax.plot(x*1e9, V_pn_eV, 'b-', label='P-N Junction')
    ax.plot(x*1e9, V_qd_eV, 'r-', label='Quantum Dot')
    ax.plot(x*1e9, V_combined_eV, 'g-', label='Combined')
    
    # Add annotations
    ax.axvline(x=(config.junction_position - config.W_p)*1e9, color='k', linestyle='--')
    ax.axvline(x=config.junction_position*1e9, color='k', linestyle='-')
    ax.axvline(x=(config.junction_position + config.W_n)*1e9, color='k', linestyle='--')
    
    ax.text((config.junction_position - config.W_p/2)*1e9, 0.1, f'W_p = {config.W_p*1e9:.1f} nm', 
            ha='center', va='bottom')
    ax.text((config.junction_position + config.W_n/2)*1e9, 0.1, f'W_n = {config.W_n*1e9:.1f} nm', 
            ha='center', va='bottom')
    
    # Set labels and title
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('Potential (eV)')
    ax.set_title('Potential Profile along y = Ly/2')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig('fixed_qd_pn_potential_slice.png', dpi=300, bbox_inches='tight')
    
    return fig

def main():
    """Main function."""
    # Create a configuration with realistic parameters
    config = create_realistic_config()
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Domain size: {config.Lx*1e9:.1f} x {config.Ly*1e9:.1f} nm")
    print(f"  Mesh: {config.nx} x {config.ny} elements, order {config.element_order}")
    print(f"  P-N junction: {config.diode_p_material}/{config.diode_n_material}")
    print(f"  Built-in potential: {config.V_bi:.3f} V")
    print(f"  Reverse bias: {config.V_r:.3f} V")
    print(f"  Total potential: {config.V_total:.3f} V")
    print(f"  Depletion width: {config.W*1e9:.3f} nm (W_p = {config.W_p*1e9:.3f} nm, W_n = {config.W_n*1e9:.3f} nm)")
    print(f"  Quantum dot: {config.qd_material} in {config.matrix_material}, R = {config.R*1e9:.1f} nm")
    print(f"  QD potential depth: {config.V_0/config.e_charge:.3f} eV")
    print(f"  QD position: ({config.qd_position_x*1e9:.1f}, {config.qd_position_y*1e9:.1f}) nm")
    
    # Plot the separate potentials
    print("\nPlotting potentials...")
    plot_separate_potentials(config)
    
    # Plot a 1D slice of the potential
    print("Plotting 1D potential slice...")
    plot_1d_potential_slice(config)
    
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
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
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
            Z[i, j] = config.potential_function(X[i, j], Y[i, j])
    
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
    ax2.axvline(x=(config.junction_position - config.W_p)*1e9, color='r', linestyle='--')
    ax2.axvline(x=config.junction_position*1e9, color='k', linestyle='-')
    ax2.axvline(x=(config.junction_position + config.W_n)*1e9, color='r', linestyle='--')
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
    plt.savefig('fixed_qd_pn_results.png', dpi=300, bbox_inches='tight')
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
