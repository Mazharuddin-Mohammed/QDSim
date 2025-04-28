#!/usr/bin/env python3
"""
Simulation of a 2D quantum dot with deeper potential at the junction of a 2D pn diode.
This script uses a finer mesh and deeper QD potential to observe bound or quasi-bound states.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction, plot_potential, plot_electric_field

# Create output directory
os.makedirs('results_deep_qd', exist_ok=True)

# Configuration for a high-resolution 2D simulation with deeper QD
config = Config()
config.Lx = 200e-9  # 200 nm domain width
config.Ly = 200e-9  # 200 nm domain height
config.nx = 50      # Increased mesh points for better resolution
config.ny = 50
config.element_order = 2  # Use quadratic elements for better accuracy
config.max_refinements = 4  # More refinements for convergence
config.adaptive_threshold = 0.01  # Lower threshold for more aggressive refinement
config.use_mpi = False
config.potential_type = "square"  # Can be "square" or "gaussian"
config.R = 15e-9    # QD radius: 15 nm
config.V_0 = 1.0 * 1.602e-19  # Deeper potential: 1.0 eV (increased from 0.3 eV)
config.V_bi = 1.0 * 1.602e-19  # Built-in potential: 1.0 eV
config.N_A = 5e23   # Acceptor concentration: 5e23 m^-3
config.N_D = 5e23   # Donor concentration: 5e23 m^-3
config.eta = 0.1 * 1.602e-19  # CAP strength: 0.1 eV
config.hbar = 1.054e-34  # Reduced Planck constant
config.matrix_material = "GaAs"
config.qd_material = "InAs"

# Define a more realistic pn junction depletion region
config.depletion_width = 50e-9  # 50 nm depletion width
config.junction_position = 0.0  # Junction at the center of the domain

# Voltage shifts to simulate
V_r_values = np.linspace(-0.5, 0.5, 5)  # 5 voltage values from -0.5V to 0.5V

# Number of energy levels to compute
num_eigenvalues = 5  # Increased to see more bound states

# Create a figure for the combined results
fig_combined = plt.figure(figsize=(15, 12))
gs = GridSpec(3, len(V_r_values), figure=fig_combined)

# Plot the pn diode potential profile (1D slice)
ax_pn = fig_combined.add_subplot(gs[0, :])
ax_pn.set_title('pn Diode Potential Profile with Deep QD (1D slice at y=0)')
ax_pn.set_xlabel('Position (nm)')
ax_pn.set_ylabel('Potential (eV)')

# Create x-axis for 1D plots
x = np.linspace(-config.Lx/2, config.Lx/2, 200)
y = np.zeros_like(x)

# More realistic pn junction potential model
def pn_potential_1d(x, V_bi, V_r, depletion_width, junction_position):
    """
    Calculate the potential profile of a pn junction.
    
    Args:
        x: Position array
        V_bi: Built-in potential
        V_r: Reverse bias voltage
        depletion_width: Width of the depletion region
        junction_position: Position of the junction
    
    Returns:
        Potential profile
    """
    V = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < junction_position - depletion_width/2:
            V[i] = 0  # p-side
        elif xi > junction_position + depletion_width/2:
            V[i] = V_bi - V_r  # n-side
        else:
            # Quadratic potential in depletion region for more realism
            # Normalized position in depletion region (-1 to 1)
            pos = 2 * (xi - junction_position) / depletion_width
            # Quadratic profile: V = a*x^2 + b*x + c
            # with boundary conditions: V(-1) = 0, V(1) = V_bi - V_r, dV/dx(0) = 0
            V[i] = (V_bi - V_r) * (pos**2 + pos + 1) / 4
    return V

# QD potential models
def square_qd_potential(x, y, R, V_0):
    r = np.sqrt(x**2 + y**2)
    return np.where(r <= R, -V_0, 0)

def gaussian_qd_potential(x, y, R, V_0):
    r2 = x**2 + y**2
    return -V_0 * np.exp(-r2/(2*R**2))

# Plot pn junction potential for different bias voltages
for V_r in V_r_values:
    V = pn_potential_1d(x, config.V_bi/1.602e-19, V_r, 
                      config.depletion_width, config.junction_position)
    
    # Add QD potential
    if config.potential_type == "square":
        V_qd = square_qd_potential(x, y, config.R, config.V_0/1.602e-19)
    else:
        V_qd = gaussian_qd_potential(x, y, config.R, config.V_0/1.602e-19)
    
    V_combined = V + V_qd
    ax_pn.plot(x*1e9, V_combined, label=f'V_r = {V_r:.2f}V')
    
ax_pn.legend()
ax_pn.grid(True)

# Dictionary to store results
results = {}

# Run simulations for each voltage shift
for i, V_r in enumerate(V_r_values):
    print(f"Simulating V_r = {V_r:.2f}V...")
    config.V_r = V_r * 1.602e-19  # Convert to Joules
    
    # Create a new simulator with updated config
    sim = Simulator(config)
    
    # Run the simulation
    eigenvalues, eigenvectors = sim.run(num_eigenvalues)
    
    # Store results
    results[V_r] = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'simulator': sim
    }
    
    # Plot 2D potential for this voltage
    ax_pot = fig_combined.add_subplot(gs[1, i])
    
    # Since we don't have direct access to the poisson solver, we'll plot the potential manually
    nodes = np.array(sim.mesh.get_nodes())
    elements = np.array(sim.mesh.get_elements())
    
    # Create a tricontourf plot of the potential
    im = ax_pot.tricontourf(nodes[:, 0]*1e9, nodes[:, 1]*1e9, elements, sim.phi, cmap='viridis')
    plt.colorbar(im, ax=ax_pot, label='Potential (V)')
    ax_pot.set_title(f'Potential (V_r = {V_r:.2f}V)')
    ax_pot.set_xlabel('x (nm)')
    ax_pot.set_ylabel('y (nm)')
    
    # Plot wavefunction for ground state
    ax_wf = fig_combined.add_subplot(gs[2, i])
    plot_wavefunction(ax_wf, sim.mesh, eigenvectors[:, 0])
    ax_wf.set_title(f'Ground State (E = {np.real(eigenvalues[0])/1.602e-19:.4f} eV)')
    
    # Save individual plots for each energy level
    for j in range(min(num_eigenvalues, len(eigenvalues))):
        fig_level = plt.figure(figsize=(10, 8))
        ax_level = fig_level.add_subplot(111)
        plot_wavefunction(ax_level, sim.mesh, eigenvectors[:, j])
        ax_level.set_title(f'Energy Level {j+1} (V_r = {V_r:.2f}V, E = {np.real(eigenvalues[j])/1.602e-19:.4f} eV)')
        fig_level.savefig(f'results_deep_qd/wavefunction_Vr{V_r:.2f}_E{j}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_level)

# Save the combined figure
fig_combined.tight_layout()
fig_combined.savefig('results_deep_qd/combined_results.png', dpi=300, bbox_inches='tight')

# Create a figure for energy levels vs. voltage
fig_energy = plt.figure(figsize=(10, 6))
ax_energy = fig_energy.add_subplot(111)
ax_energy.set_title('Energy Levels vs. Voltage')
ax_energy.set_xlabel('Voltage (V)')
ax_energy.set_ylabel('Energy (eV)')

# Plot energy levels vs. voltage
for j in range(num_eigenvalues):
    energies = []
    linewidths = []
    for V_r in V_r_values:
        if V_r in results and j < len(results[V_r]['eigenvalues']):
            energies.append(np.real(results[V_r]['eigenvalues'][j]) / 1.602e-19)
            linewidths.append(-2 * np.imag(results[V_r]['eigenvalues'][j]) / 1.602e-19)
        else:
            energies.append(np.nan)
            linewidths.append(np.nan)
    ax_energy.plot(V_r_values, energies, 'o-', label=f'Level {j+1}')
    
    # Create a separate figure for each energy level
    fig_level_energy = plt.figure(figsize=(10, 6))
    ax_level = fig_level_energy.add_subplot(111)
    ax_level.set_title(f'Energy Level {j+1} vs. Voltage')
    ax_level.set_xlabel('Voltage (V)')
    ax_level.set_ylabel('Energy (eV)')
    ax_level.plot(V_r_values, energies, 'o-', color='blue', label='Energy')
    
    # Add linewidth on secondary y-axis
    ax_linewidth = ax_level.twinx()
    ax_linewidth.set_ylabel('Linewidth (eV)')
    ax_linewidth.plot(V_r_values, linewidths, 's-', color='red', label='Linewidth')
    
    # Add legends
    ax_level.legend(loc='upper left')
    ax_linewidth.legend(loc='upper right')
    
    fig_level_energy.tight_layout()
    fig_level_energy.savefig(f'results_deep_qd/energy_level{j+1}_vs_voltage.png', dpi=300, bbox_inches='tight')
    plt.close(fig_level_energy)

ax_energy.legend()
ax_energy.grid(True)
fig_energy.tight_layout()
fig_energy.savefig('results_deep_qd/energy_levels_vs_voltage.png', dpi=300, bbox_inches='tight')

# Create a figure to analyze bound states
fig_bound = plt.figure(figsize=(12, 10))
fig_bound.suptitle('Analysis of Bound States', fontsize=16)

# Create a grid of subplots
gs_bound = GridSpec(2, 2, figure=fig_bound)

# Plot energy levels
ax_energy_bound = fig_bound.add_subplot(gs_bound[0, 0])
ax_energy_bound.set_title('Energy Levels')
ax_energy_bound.set_xlabel('Level Number')
ax_energy_bound.set_ylabel('Energy (eV)')

# Plot linewidths
ax_linewidth_bound = fig_bound.add_subplot(gs_bound[0, 1])
ax_linewidth_bound.set_title('Linewidths')
ax_linewidth_bound.set_xlabel('Level Number')
ax_linewidth_bound.set_ylabel('Linewidth (eV)')

# Plot localization
ax_localization = fig_bound.add_subplot(gs_bound[1, 0])
ax_localization.set_title('Wavefunction Localization')
ax_localization.set_xlabel('Level Number')
ax_localization.set_ylabel('Localization (%)')

# Plot probability in QD
ax_probability = fig_bound.add_subplot(gs_bound[1, 1])
ax_probability.set_title('Probability in QD')
ax_probability.set_xlabel('Level Number')
ax_probability.set_ylabel('Probability (%)')

# Analyze bound states for middle voltage
middle_idx = len(V_r_values) // 2
V_r_middle = V_r_values[middle_idx]

if V_r_middle in results:
    # Get eigenvalues and eigenvectors
    eigenvalues = results[V_r_middle]['eigenvalues']
    eigenvectors = results[V_r_middle]['eigenvectors']
    sim = results[V_r_middle]['simulator']
    
    # Calculate energy levels and linewidths
    energy_levels = [np.real(e) / 1.602e-19 for e in eigenvalues]
    linewidths = [-2 * np.imag(e) / 1.602e-19 for e in eigenvalues]
    
    # Plot energy levels and linewidths
    ax_energy_bound.bar(range(1, len(energy_levels) + 1), energy_levels)
    ax_linewidth_bound.bar(range(1, len(linewidths) + 1), linewidths)
    
    # Calculate localization and probability in QD
    nodes = np.array(sim.mesh.get_nodes())
    elements = np.array(sim.mesh.get_elements())
    
    localization = []
    probability_in_qd = []
    
    for j in range(len(eigenvalues)):
        # Calculate wavefunction probability density
        psi_squared = np.abs(eigenvectors[:, j]) ** 2
        
        # Calculate localization (inverse participation ratio)
        # Higher value means more localized
        localization_value = np.sum(psi_squared ** 2) / (np.sum(psi_squared) ** 2) * 100
        localization.append(localization_value)
        
        # Calculate probability in QD
        in_qd = []
        for i, node in enumerate(nodes):
            x, y = node
            r = np.sqrt(x**2 + y**2)
            if r <= config.R:
                in_qd.append(i)
        
        probability = np.sum(psi_squared[in_qd]) / np.sum(psi_squared) * 100
        probability_in_qd.append(probability)
    
    # Plot localization and probability in QD
    ax_localization.bar(range(1, len(localization) + 1), localization)
    ax_probability.bar(range(1, len(probability_in_qd) + 1), probability_in_qd)
    
    # Add threshold line for bound states (probability > 50%)
    ax_probability.axhline(y=50, color='r', linestyle='--', label='Bound State Threshold')
    ax_probability.legend()

fig_bound.tight_layout(rect=[0, 0, 1, 0.95])
fig_bound.savefig('results_deep_qd/bound_states_analysis.png', dpi=300, bbox_inches='tight')

print("All plots saved to the 'results_deep_qd' directory.")
