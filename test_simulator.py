#!/usr/bin/env python3
"""
Test script for the QDSim simulator.

This script creates a simple quantum dot simulation and runs it to verify
that the C++ implementation is working correctly. It also demonstrates the
visualization capabilities of the QDSim package.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend'))

# Import the QDSim modules
from qdsim.config import Config
from qdsim.simulator import Simulator
from qdsim.visualization import (
    plot_potential_3d, plot_wavefunction_3d,
    create_simulation_dashboard, save_simulation_results
)
from qdsim.analysis import (
    calculate_transition_energies, calculate_transition_probabilities,
    calculate_energy_level_statistics, calculate_wavefunction_localization,
    find_bound_states
)

def main():
    """Main function to test the simulator."""
    # Create a configuration
    config = Config()

    # Domain size
    config.Lx = 200e-9  # 200 nm
    config.Ly = 100e-9  # 100 nm

    # Mesh parameters
    config.nx = 101  # Number of elements in x direction
    config.ny = 51   # Number of elements in y direction
    config.element_order = 1  # Linear elements

    # Quantum dot parameters
    config.R = 10e-9  # 10 nm radius
    config.V_0 = 0.5 * 1.602e-19  # 0.5 eV depth
    config.potential_type = "gaussian"  # Gaussian potential

    # Diode parameters
    config.diode_p_material = "GaAs"
    config.diode_n_material = "GaAs"
    config.qd_material = "InAs"
    config.matrix_material = "GaAs"
    config.N_A = 1e24  # Acceptor concentration (m^-3)
    config.N_D = 1e24  # Donor concentration (m^-3)
    config.V_r = 0.0   # Reverse bias (V)

    # Physical constants
    config.e_charge = 1.602e-19  # Elementary charge (C)
    config.m_e = 9.109e-31  # Electron mass (kg)

    # Solver parameters
    config.tolerance = 1e-6
    config.max_iter = 100
    config.use_mpi = False

    # Create the simulator
    print("Creating simulator...")
    sim = Simulator(config)

    # Solve for the first 5 eigenvalues
    print("\nSolving for eigenvalues...")
    eigenvalues, eigenvectors = sim.solve(5)

    # Print the eigenvalues
    print("\nEigenvalues (eV):")
    for i, ev in enumerate(eigenvalues):
        print(f"  {i}: {ev/config.e_charge:.6f}")

    # Analyze the results
    analyze_results(sim, eigenvalues, eigenvectors)

    # Plot the results using the visualization module
    plot_results(sim)

def analyze_results(sim, eigenvalues, eigenvectors):
    """Analyze the simulation results using the analysis module."""
    print("\nAnalyzing results...")

    # Calculate transition energies
    transitions = calculate_transition_energies(eigenvalues, sim.config.e_charge)
    print("\nTransition energies (eV):")
    for i in range(min(3, len(eigenvalues))):
        for j in range(i+1, min(3, len(eigenvalues))):
            print(f"  {i} -> {j}: {transitions[i, j]:.6f}")

    # Calculate transition probabilities
    if eigenvectors is not None and eigenvectors.shape[1] > 1:
        probs = calculate_transition_probabilities(eigenvectors)
        print("\nTransition probabilities:")
        for i in range(min(3, eigenvectors.shape[1])):
            for j in range(i+1, min(3, eigenvectors.shape[1])):
                print(f"  {i} -> {j}: {probs[i, j]:.6f}")

    # Calculate energy level statistics
    stats = calculate_energy_level_statistics(eigenvalues, sim.config.e_charge)
    print("\nEnergy level statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")

    # Calculate wavefunction localization
    if eigenvectors is not None and eigenvectors.shape[1] > 0:
        ipr = calculate_wavefunction_localization(eigenvectors)
        print("\nWavefunction localization (IPR):")
        for i in range(min(5, len(ipr))):
            print(f"  State {i}: {ipr[i]:.6f}")

    # Find bound states
    bound_states = find_bound_states(eigenvalues, sim.config.V_0 / sim.config.e_charge, sim.config.e_charge)
    print("\nBound states:")
    if bound_states:
        for i in bound_states:
            energy = eigenvalues[i] / sim.config.e_charge
            print(f"  State {i}: {energy:.6f} eV")
    else:
        print("  No bound states found")

    # Save analysis results to file
    with open("qdsim_test_analysis.txt", "w") as f:
        f.write("QDSim Test Analysis Results\n")
        f.write("==========================\n\n")

        f.write("Eigenvalues (eV):\n")
        for i, ev in enumerate(eigenvalues):
            f.write(f"  {i}: {ev/sim.config.e_charge:.6f}\n")

        f.write("\nTransition energies (eV):\n")
        for i in range(min(3, len(eigenvalues))):
            for j in range(i+1, min(3, len(eigenvalues))):
                f.write(f"  {i} -> {j}: {transitions[i, j]:.6f}\n")

        if eigenvectors is not None and eigenvectors.shape[1] > 1:
            f.write("\nTransition probabilities:\n")
            for i in range(min(3, eigenvectors.shape[1])):
                for j in range(i+1, min(3, eigenvectors.shape[1])):
                    f.write(f"  {i} -> {j}: {probs[i, j]:.6f}\n")

        f.write("\nEnergy level statistics:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value:.6f}\n")

        if eigenvectors is not None and eigenvectors.shape[1] > 0:
            f.write("\nWavefunction localization (IPR):\n")
            for i in range(min(5, len(ipr))):
                f.write(f"  State {i}: {ipr[i]:.6f}\n")

        f.write("\nBound states:\n")
        if bound_states:
            for i in bound_states:
                energy = eigenvalues[i] / sim.config.e_charge
                f.write(f"  State {i}: {energy:.6f} eV\n")
        else:
            f.write("  No bound states found\n")

    print("\nAnalysis results saved to qdsim_test_analysis.txt")

def plot_results(sim):
    """Plot the simulation results using the visualization module."""
    # Create a dashboard of all results
    print("\nCreating simulation dashboard...")
    fig = create_simulation_dashboard(sim, num_states=3, resolution=50)

    # Save the dashboard
    dashboard_filename = "qdsim_test_dashboard.png"
    fig.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to {dashboard_filename}")

    # Create individual plots
    print("\nCreating individual plots...")

    # Potential plot
    fig_pot = plt.figure(figsize=(8, 6))
    ax_pot = fig_pot.add_subplot(111, projection='3d')
    plot_potential_3d(sim, ax=ax_pot, resolution=50)
    pot_filename = "qdsim_test_potential.png"
    fig_pot.savefig(pot_filename, dpi=300, bbox_inches='tight')
    print(f"Potential plot saved to {pot_filename}")

    # Wavefunction plots for the first 3 states
    num_states = min(3, len(sim.eigenvalues) if sim.eigenvalues is not None else 0)
    for i in range(num_states):
        fig_wf = plt.figure(figsize=(8, 6))
        ax_wf = fig_wf.add_subplot(111, projection='3d')
        plot_wavefunction_3d(sim, state_idx=i, ax=ax_wf, resolution=50)
        wf_filename = f"qdsim_test_wavefunction_{i}.png"
        fig_wf.savefig(wf_filename, dpi=300, bbox_inches='tight')
        print(f"Wavefunction {i} plot saved to {wf_filename}")

    # Save all results using the save_simulation_results function
    print("\nSaving all results...")
    filenames = save_simulation_results(sim, filename_prefix="qdsim_test_all", format="png", dpi=300)

    # Show the dashboard
    plt.figure(fig.number)
    plt.show()

if __name__ == "__main__":
    main()
