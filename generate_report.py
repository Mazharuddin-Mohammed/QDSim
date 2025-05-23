#!/usr/bin/env python3
"""
Script to generate a comprehensive PDF report with all the plots.
"""

import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

# Create a PDF file for the report
with PdfPages('qdsim_report.pdf') as pdf:

    # Title page
    fig_title = plt.figure(figsize=(8.5, 11))
    fig_title.suptitle('Quantum Dot Simulator Report', fontsize=16)
    plt.figtext(0.5, 0.5, 'Simulation of Quantum Dot in pn Junction',
                ha='center', fontsize=14)
    plt.figtext(0.5, 0.45, f'Quantum Dot Radius: {config.R*1e9:.1f} nm',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.4, f'Potential Type: {config.potential_type.capitalize()}',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.35, f'Voltage Range: {V_r_values[0]:.2f}V to {V_r_values[-1]:.2f}V',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.3, f'Number of Energy Levels: {num_eigenvalues}',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.2, 'Generated by QDSim', ha='center', fontsize=10)
    plt.figtext(0.5, 0.15, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d")}',
                ha='center', fontsize=10)
    pdf.savefig(fig_title)
    plt.close(fig_title)

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

    # QD potential models
    def square_qd_potential(x, y, R, V_0):
        r = np.sqrt(x**2 + y**2)
        return np.where(r <= R, 0, V_0)

    def gaussian_qd_potential(x, y, R, V_0):
        r2 = x**2 + y**2
        return V_0 * np.exp(-r2/(2*R**2))

    # Potentials page
    fig_potentials = plt.figure(figsize=(8.5, 11))
    fig_potentials.suptitle('Potential Profiles', fontsize=16)

    # pn junction potential
    ax_pn = fig_potentials.add_subplot(311)
    ax_pn.set_title('pn Diode Potential')
    ax_pn.set_xlabel('Position (nm)')
    ax_pn.set_ylabel('Potential (eV)')

    # Plot pn junction potential for different bias voltages
    for V_r in V_r_values:
        V = pn_potential(x, config.V_bi/1.602e-19, V_r)
        ax_pn.plot(x*1e9, V, label=f'V_r = {V_r:.2f}V')
    ax_pn.legend()
    ax_pn.grid(True)

    # QD potential
    ax_qd = fig_potentials.add_subplot(312)
    ax_qd.set_title('Quantum Dot Potential')
    ax_qd.set_xlabel('Position (nm)')
    ax_qd.set_ylabel('Potential (eV)')

    # Plot QD potential along x-axis
    if config.potential_type == "square":
        V_qd = square_qd_potential(x, y, config.R, config.V_0/1.602e-19)
        ax_qd.plot(x*1e9, V_qd, label='Square QD')
    else:
        V_qd = gaussian_qd_potential(x, y, config.R, config.V_0/1.602e-19)
        ax_qd.plot(x*1e9, V_qd, label='Gaussian QD')
    ax_qd.legend()
    ax_qd.grid(True)

    # Combined potential
    ax_combined = fig_potentials.add_subplot(313)
    ax_combined.set_title('Combined Potential (pn + QD)')
    ax_combined.set_xlabel('Position (nm)')
    ax_combined.set_ylabel('Potential (eV)')

    # Plot combined potential for each voltage
    for V_r in V_r_values:
        V_pn = pn_potential(x, config.V_bi/1.602e-19, V_r)
        V_combined = V_pn + V_qd
        ax_combined.plot(x*1e9, V_combined, label=f'V_r = {V_r:.2f}V')
    ax_combined.legend()
    ax_combined.grid(True)

    fig_potentials.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig_potentials)
    plt.close(fig_potentials)

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

        # Create a page for this voltage
        fig_voltage = plt.figure(figsize=(8.5, 11))
        fig_voltage.suptitle(f'Wavefunction Probability Densities (V_r = {V_r:.2f}V)', fontsize=16)

        # Plot wavefunction probability densities for each energy level
        for j in range(min(num_eigenvalues, len(eigenvalues))):
            ax_wf = fig_voltage.add_subplot(num_eigenvalues, 1, j+1)

            # Plot wavefunction probability density
            plot_wavefunction(ax_wf, sim.mesh, eigenvectors[:, j])
            ax_wf.set_title(f'Energy Level {j+1}: E = {np.real(eigenvalues[j])/1.602e-19:.4f} eV')

        fig_voltage.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig_voltage)
        plt.close(fig_voltage)

    # Energy levels vs. voltage page
    fig_energy = plt.figure(figsize=(8.5, 11))
    fig_energy.suptitle('Energy Levels vs. Voltage', fontsize=16)

    ax_energy = fig_energy.add_subplot(111)
    ax_energy.set_title('Energy Levels')
    ax_energy.set_xlabel('Voltage (V)')
    ax_energy.set_ylabel('Energy (eV)')

    # Plot energy levels vs. voltage
    for j in range(num_eigenvalues):
        energies = []
        for V_r in V_r_values:
            if V_r in results and j < len(results[V_r]['eigenvalues']):
                energies.append(np.real(results[V_r]['eigenvalues'][j]) / 1.602e-19)
            else:
                energies.append(np.nan)
        ax_energy.plot(V_r_values, energies, 'o-', label=f'Level {j+1}')

    ax_energy.legend()
    ax_energy.grid(True)

    fig_energy.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig_energy)
    plt.close(fig_energy)

    # 2D potential map page
    fig_potential_map = plt.figure(figsize=(8.5, 11))
    fig_potential_map.suptitle('2D Potential Map', fontsize=16)

    # Create 2D grid for potential map
    X, Y = np.meshgrid(np.linspace(-config.Lx/2, config.Lx/2, 100),
                       np.linspace(-config.Ly/2, config.Ly/2, 100))

    # Calculate QD potential on 2D grid
    if config.potential_type == "square":
        V_qd_2d = square_qd_potential(X, Y, config.R, config.V_0/1.602e-19)
    else:
        V_qd_2d = gaussian_qd_potential(X, Y, config.R, config.V_0/1.602e-19)

    # Plot 2D potential map for middle voltage
    middle_idx = len(V_r_values) // 2
    V_r_middle = V_r_values[middle_idx]

    ax_potential_map = fig_potential_map.add_subplot(111)
    im = ax_potential_map.contourf(X*1e9, Y*1e9, V_qd_2d, 50, cmap='viridis')
    ax_potential_map.set_title(f'Quantum Dot Potential (V_r = {V_r_middle:.2f}V)')
    ax_potential_map.set_xlabel('x (nm)')
    ax_potential_map.set_ylabel('y (nm)')
    plt.colorbar(im, ax=ax_potential_map, label='Potential (eV)')

    fig_potential_map.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig_potential_map)
    plt.close(fig_potential_map)

print("Report generated: qdsim_report.pdf")
