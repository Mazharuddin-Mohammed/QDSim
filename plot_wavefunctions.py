#!/usr/bin/env python3
"""
Script to plot wavefunction probability densities for each energy level and voltage shift,
along with the pn diode potential and QD potential.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction, plot_potential, plot_electric_field

# Create output directory
os.makedirs('plots', exist_ok=True)

# Configuration
config = Config()
config.Lx = 100e-9  # 100 nm
config.Ly = 100e-9  # 100 nm
config.nx = 50
config.ny = 50
config.element_order = 1
config.max_refinements = 1
config.adaptive_threshold = 0.1
config.use_mpi = False
config.potential_type = "square"  # or "gaussian"
config.R = 10e-9  # QD radius: 10 nm
config.V_0 = 0.5 * 1.602e-19  # Potential height: 0.5 eV
config.V_bi = 1.5 * 1.602e-19  # Built-in potential: 1.5 eV
config.N_A = 1e24  # Acceptor concentration: 1e24 m^-3
config.N_D = 1e24  # Donor concentration: 1e24 m^-3
config.eta = 0.1 * 1.602e-19  # CAP strength: 0.1 eV
config.hbar = 1.054e-34  # Reduced Planck constant
config.matrix_material = "GaAs"
config.qd_material = "InAs"

# Voltage shifts to simulate
V_r_values = np.linspace(-0.5, 0.5, 5)  # 5 voltage values from -0.5V to 0.5V

# Number of energy levels to compute
num_eigenvalues = 3

# Create a figure for the report
report_fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=report_fig)

# Plot the pn diode potential
ax_pn = report_fig.add_subplot(gs[0, 0])
ax_pn.set_title('pn Diode Potential')
ax_pn.set_xlabel('Position (nm)')
ax_pn.set_ylabel('Potential (eV)')

# Create x-axis for 1D plots
x = np.linspace(-config.Lx/2, config.Lx/2, 100)
y = np.zeros_like(x)

# Simple model for pn junction potential
def pn_potential(x, V_bi, V_r):
    # Depletion width
    W = 50e-9  # 50 nm
    # Position of the junction
    x_j = 0
    # Potential profile
    V = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < x_j - W/2:
            V[i] = 0  # p-side
        elif xi > x_j + W/2:
            V[i] = V_bi - V_r  # n-side
        else:
            # Linear transition in depletion region
            V[i] = (V_bi - V_r) * (xi - (x_j - W/2)) / W
    return V

# Plot pn junction potential for different bias voltages
for V_r in V_r_values:
    V = pn_potential(x, config.V_bi/1.602e-19, V_r)
    ax_pn.plot(x*1e9, V, label=f'V_r = {V_r:.2f}V')
ax_pn.legend()
ax_pn.grid(True)

# Plot the QD potential
ax_qd = report_fig.add_subplot(gs[0, 1])
ax_qd.set_title('Quantum Dot Potential')
ax_qd.set_xlabel('Position (nm)')
ax_qd.set_ylabel('Potential (eV)')

# QD potential models
def square_qd_potential(x, y, R, V_0):
    r = np.sqrt(x**2 + y**2)
    return np.where(r <= R, 0, V_0)

def gaussian_qd_potential(x, y, R, V_0):
    r2 = x**2 + y**2
    return V_0 * np.exp(-r2/(2*R**2))

# Plot QD potential along x-axis
if config.potential_type == "square":
    V_qd = square_qd_potential(x, y, config.R, config.V_0/1.602e-19)
    ax_qd.plot(x*1e9, V_qd, label='Square QD')
else:
    V_qd = gaussian_qd_potential(x, y, config.R, config.V_0/1.602e-19)
    ax_qd.plot(x*1e9, V_qd, label='Gaussian QD')
ax_qd.legend()
ax_qd.grid(True)

# Plot the combined potential (pn + QD)
ax_combined = report_fig.add_subplot(gs[0, 2])
ax_combined.set_title('Combined Potential (pn + QD)')
ax_combined.set_xlabel('Position (nm)')
ax_combined.set_ylabel('Potential (eV)')

# Plot combined potential for middle voltage value
middle_idx = len(V_r_values) // 2
V_r_middle = V_r_values[middle_idx]
V_pn = pn_potential(x, config.V_bi/1.602e-19, V_r_middle)
V_combined = V_pn + V_qd
ax_combined.plot(x*1e9, V_combined, label=f'V_r = {V_r_middle:.2f}V')
ax_combined.legend()
ax_combined.grid(True)

# Create a simulator
sim = Simulator(config)

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
        'eigenvectors': eigenvectors
    }

    # Plot wavefunction probability densities for each energy level
    for j in range(min(num_eigenvalues, len(eigenvalues))):
        # Create figure for wavefunction
        fig_wf = plt.figure(figsize=(10, 8))
        ax_wf = fig_wf.add_subplot(111)

        # Plot wavefunction probability density
        plot_wavefunction(ax_wf, sim.mesh, eigenvectors[:, j])
        ax_wf.set_title(f'Wavefunction Probability Density (V_r = {V_r:.2f}V, E = {np.real(eigenvalues[j])/1.602e-19:.4f} eV)')

        # Save figure
        fig_wf.savefig(f'plots/wavefunction_Vr{V_r:.2f}_E{j}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_wf)

        # Add to report (only first energy level)
        if j == 0:
            ax_report = report_fig.add_subplot(gs[1, i % 3])
            plot_wavefunction(ax_report, sim.mesh, eigenvectors[:, j])
            ax_report.set_title(f'Wavefunction (V_r = {V_r:.2f}V)')

# Save the report
report_fig.tight_layout()
report_fig.savefig('plots/wavefunction_report.png', dpi=300, bbox_inches='tight')

print("All plots saved to the 'plots' directory.")
print("Report saved as 'plots/wavefunction_report.png'.")
