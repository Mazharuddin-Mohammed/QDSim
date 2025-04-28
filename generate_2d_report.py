#!/usr/bin/env python3
"""
Generate a comprehensive PDF report for the 2D quantum dot at the junction of a 2D pn diode.
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
os.makedirs('results_2d', exist_ok=True)

# Configuration for a realistic 2D simulation
config = Config()
config.Lx = 200e-9  # 200 nm domain width
config.Ly = 200e-9  # 200 nm domain height
config.nx = 30      # Reduced mesh points to avoid memory issues
config.ny = 30
config.element_order = 1  # Use linear elements to reduce memory usage
config.max_refinements = 3  # More refinements for convergence
config.adaptive_threshold = 0.05  # Lower threshold for more aggressive refinement
config.use_mpi = False
config.potential_type = "square"  # Can be "square" or "gaussian"
config.R = 15e-9    # QD radius: 15 nm
config.V_0 = 0.3 * 1.602e-19  # Potential height: 0.3 eV
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
num_eigenvalues = 3

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

# 2D potential model for pn junction with embedded QD
def pn_qd_potential_2d(X, Y, V_bi, V_r, depletion_width, junction_position, R, V_0, potential_type):
    """
    Calculate the 2D potential of a pn junction with embedded QD.

    Args:
        X, Y: Meshgrid of positions
        V_bi: Built-in potential
        V_r: Reverse bias voltage
        depletion_width: Width of the depletion region
        junction_position: Position of the junction
        R: QD radius
        V_0: QD potential height
        potential_type: "square" or "gaussian"

    Returns:
        2D potential
    """
    # Initialize potential array
    V = np.zeros_like(X)

    # Calculate pn junction potential
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            xi = X[i, j]
            if xi < junction_position - depletion_width/2:
                V[i, j] = 0  # p-side
            elif xi > junction_position + depletion_width/2:
                V[i, j] = V_bi - V_r  # n-side
            else:
                # Quadratic potential in depletion region
                pos = 2 * (xi - junction_position) / depletion_width
                V[i, j] = (V_bi - V_r) * (pos**2 + pos + 1) / 4

    # Add QD potential
    r2 = X**2 + Y**2
    if potential_type == "square":
        r = np.sqrt(r2)
        QD_potential = np.where(r <= R, -V_0, 0)
    else:  # gaussian
        QD_potential = -V_0 * np.exp(-r2/(2*R**2))

    # Combine potentials
    V += QD_potential

    return V

# Create a PDF file for the report
with PdfPages('qdsim_2d_report.pdf') as pdf:

    # Title page
    fig_title = plt.figure(figsize=(8.5, 11))
    fig_title.suptitle('2D Quantum Dot in pn Junction', fontsize=16)
    plt.figtext(0.5, 0.5, 'Simulation of a 2D Quantum Dot at the Junction of a 2D pn Diode',
                ha='center', fontsize=14)
    plt.figtext(0.5, 0.45, f'Quantum Dot Radius: {config.R*1e9:.1f} nm',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.4, f'Potential Type: {config.potential_type.capitalize()}',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.35, f'Voltage Range: {V_r_values[0]:.2f}V to {V_r_values[-1]:.2f}V',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.3, f'Number of Energy Levels: {num_eigenvalues}',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.25, f'Mesh: {config.nx}x{config.ny} elements, order {config.element_order}',
                ha='center', fontsize=12)
    plt.figtext(0.5, 0.2, 'Generated by QDSim', ha='center', fontsize=10)
    plt.figtext(0.5, 0.15, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d")}',
                ha='center', fontsize=10)
    pdf.savefig(fig_title)
    plt.close(fig_title)

    # Create x-axis for 1D plots
    x = np.linspace(-config.Lx/2, config.Lx/2, 200)
    y = np.zeros_like(x)

    # Create 2D grid for potential maps
    X, Y = np.meshgrid(np.linspace(-config.Lx/2, config.Lx/2, 100),
                       np.linspace(-config.Ly/2, config.Ly/2, 100))

    # Potential profiles page
    fig_potentials = plt.figure(figsize=(8.5, 11))
    fig_potentials.suptitle('Potential Profiles', fontsize=16)

    # pn junction potential (1D)
    ax_pn = fig_potentials.add_subplot(311)
    ax_pn.set_title('pn Diode Potential (1D slice at y=0)')
    ax_pn.set_xlabel('Position (nm)')
    ax_pn.set_ylabel('Potential (eV)')

    # Plot pn junction potential for different bias voltages
    for V_r in V_r_values:
        V = pn_potential_1d(x, config.V_bi/1.602e-19, V_r,
                          config.depletion_width, config.junction_position)
        ax_pn.plot(x*1e9, V, label=f'V_r = {V_r:.2f}V')
    ax_pn.legend()
    ax_pn.grid(True)

    # QD potential (1D)
    ax_qd = fig_potentials.add_subplot(312)
    ax_qd.set_title('Quantum Dot Potential (1D slice at y=0)')
    ax_qd.set_xlabel('Position (nm)')
    ax_qd.set_ylabel('Potential (eV)')

    # Plot QD potential along x-axis
    if config.potential_type == "square":
        r = np.sqrt(x**2 + y**2)
        V_qd = np.where(r <= config.R, -config.V_0/1.602e-19, 0)
    else:  # gaussian
        r2 = x**2 + y**2
        V_qd = -config.V_0/1.602e-19 * np.exp(-r2/(2*config.R**2))

    ax_qd.plot(x*1e9, V_qd, label=f'{config.potential_type.capitalize()} QD')
    ax_qd.legend()
    ax_qd.grid(True)

    # Combined potential (1D)
    ax_combined = fig_potentials.add_subplot(313)
    ax_combined.set_title('Combined Potential: pn + QD (1D slice at y=0)')
    ax_combined.set_xlabel('Position (nm)')
    ax_combined.set_ylabel('Potential (eV)')

    # Plot combined potential for each voltage
    for V_r in V_r_values:
        V_pn = pn_potential_1d(x, config.V_bi/1.602e-19, V_r,
                             config.depletion_width, config.junction_position)
        V_combined = V_pn + V_qd
        ax_combined.plot(x*1e9, V_combined, label=f'V_r = {V_r:.2f}V')
    ax_combined.legend()
    ax_combined.grid(True)

    fig_potentials.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig_potentials)
    plt.close(fig_potentials)

    # 2D potential maps page
    fig_potential_maps = plt.figure(figsize=(8.5, 11))
    fig_potential_maps.suptitle('2D Potential Maps', fontsize=16)

    # Create a grid of subplots for different voltages
    gs = GridSpec(3, 2, figure=fig_potential_maps)

    # Plot 2D potential maps for each voltage
    for i, V_r in enumerate(V_r_values):
        if i >= 6:  # Only show first 6 voltages if there are more
            break

        ax = fig_potential_maps.add_subplot(gs[i//2, i%2])

        # Calculate 2D potential
        V_2d = pn_qd_potential_2d(X, Y, config.V_bi/1.602e-19, V_r,
                                config.depletion_width, config.junction_position,
                                config.R, config.V_0/1.602e-19, config.potential_type)

        # Plot 2D potential map
        im = ax.contourf(X*1e9, Y*1e9, V_2d, 50, cmap='viridis')
        ax.set_title(f'V_r = {V_r:.2f}V')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Potential (eV)')

    fig_potential_maps.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig_potential_maps)
    plt.close(fig_potential_maps)

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

    # Energy levels plot
    ax_energy = fig_energy.add_subplot(211)
    ax_energy.set_title('Energy Levels')
    ax_energy.set_xlabel('Voltage (V)')
    ax_energy.set_ylabel('Energy (eV)')

    # Linewidths plot
    ax_linewidth = fig_energy.add_subplot(212)
    ax_linewidth.set_title('Linewidths')
    ax_linewidth.set_xlabel('Voltage (V)')
    ax_linewidth.set_ylabel('Linewidth (eV)')

    # Plot energy levels and linewidths vs. voltage
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
        ax_linewidth.plot(V_r_values, linewidths, 'o-', label=f'Level {j+1}')

    ax_energy.legend()
    ax_energy.grid(True)
    ax_linewidth.legend()
    ax_linewidth.grid(True)

    fig_energy.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig_energy)
    plt.close(fig_energy)

    # Wavefunction comparison page
    fig_wf_compare = plt.figure(figsize=(8.5, 11))
    fig_wf_compare.suptitle('Wavefunction Comparison (Ground State)', fontsize=16)

    # Create a grid of subplots for different voltages
    gs = GridSpec(3, 2, figure=fig_wf_compare)

    # Plot ground state wavefunction for each voltage
    for i, V_r in enumerate(V_r_values):
        if i >= 6:  # Only show first 6 voltages if there are more
            break

        ax = fig_wf_compare.add_subplot(gs[i//2, i%2])

        # Plot wavefunction
        if V_r in results:
            plot_wavefunction(ax, results[V_r]['simulator'].mesh, results[V_r]['eigenvectors'][:, 0])
            ax.set_title(f'V_r = {V_r:.2f}V, E = {np.real(results[V_r]["eigenvalues"][0])/1.602e-19:.4f} eV')

    fig_wf_compare.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig_wf_compare)
    plt.close(fig_wf_compare)

print("Report generated: qdsim_2d_report.pdf")
