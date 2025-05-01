#!/usr/bin/env python3
"""
Chromium Quantum Dot in AlGaAs P-N Diode Simulation

This script simulates a 2D chromium quantum dot in an AlGaAs P-N diode under reverse bias.
It calculates the bound or quasi-bound states and their corresponding wavefunction
probability densities.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the parent directory to the path so we can import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the patched simulator
import patched_simulator
import qdsim

# Import the fixed visualization functions directly
from qdsim.visualization_fix import plot_wavefunction, plot_potential, plot_electric_field

def main():
    """
    Main function to run the simulation.
    """
    print("Chromium Quantum Dot in AlGaAs P-N Diode Simulation")
    print("==================================================")

    # Create a configuration
    config = qdsim.Config()

    # Mesh configuration
    config.Lx = 200e-9  # 200 nm
    config.Ly = 200e-9  # 200 nm
    config.nx = 100     # More elements for better resolution
    config.ny = 100
    config.element_order = 1  # Linear elements

    # Material configuration
    # AlGaAs properties
    config.matrix_material = "AlGaAs"  # Matrix material

    # P-N junction configuration
    config.N_A = 1e24  # 1e18 cm^-3 = 1e24 m^-3
    config.N_D = 1e24  # 1e18 cm^-3 = 1e24 m^-3
    config.depletion_width = 100e-9  # 100 nm
    config.V_r = 1.0  # 1.0 V reverse bias

    # Quantum dot configuration
    config.qd_material = "Chromium"  # QD material (will use effective mass from this)
    config.R = 10e-9  # 10 nm
    config.V_0 = 0.5  # 0.5 eV - deep well

    # Try both square and Gaussian wells
    well_types = ["square", "gaussian"]

    for well_type in well_types:
        print(f"\nRunning simulation with {well_type} well...")
        config.potential_type = well_type

        # Simulation configuration
        config.num_eigenvalues = 10  # Look for 10 states
        config.max_refinements = 2  # Use adaptive mesh refinement
        config.adaptive_threshold = 0.1  # Refinement threshold

        # Create a simulator
        simulator = qdsim.Simulator(config)

        # Run the simulation
        start_time = time.time()
        results = simulator.run()
        end_time = time.time()

        print(f"Simulation completed in {end_time - start_time:.2f} seconds")

        # Extract results for convenience
        mesh = results["mesh"]
        eigenvectors = results["eigenvectors"]
        eigenvalues = results["eigenvalues"]
        potential = results["potential"]

        # Print eigenvalues (energies)
        print("\nEigenvalues (energies) in eV:")
        for i, energy in enumerate(eigenvalues):
            # Convert from J to eV
            energy_eV = energy / 1.602e-19
            # For complex eigenvalues, print both real and imaginary parts
            if np.iscomplex(energy):
                print(f"State {i}: {energy_eV.real:.6f} eV (real part), {energy_eV.imag:.6f} eV (imaginary part)")
            else:
                print(f"State {i}: {energy_eV:.6f} eV")

        # Identify bound states (negative real part of energy)
        bound_states = [i for i, energy in enumerate(eigenvalues) if np.real(energy) < 0]
        print(f"\nFound {len(bound_states)} bound states: {bound_states}")

        # Identify quasi-bound states (positive real part but small)
        quasi_bound_threshold = 0.1 * 1.602e-19  # 0.1 eV in J
        quasi_bound_states = [i for i, energy in enumerate(eigenvalues)
                             if 0 <= np.real(energy) < quasi_bound_threshold]
        print(f"Found {len(quasi_bound_states)} quasi-bound states: {quasi_bound_states}")

        # Create a figure for visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot potential
        plot_potential(
            ax=axes[0, 0],
            mesh=mesh,
            potential_values=potential,
            use_nm=True,
            center_coords=True,
            title=f"Potential ({well_type} well)",
            convert_to_eV=True
        )

        # Plot ground state wavefunction
        if len(eigenvectors) > 0:
            energy_str = f"{np.real(eigenvalues[0])/1.602e-19:.6f} eV"
            if np.iscomplex(eigenvalues[0]):
                energy_str += f" + {np.imag(eigenvalues[0])/1.602e-19:.6f}i eV"

            plot_wavefunction(
                ax=axes[0, 1],
                mesh=mesh,
                eigenvector=eigenvectors[:, 0],
                use_nm=True,
                center_coords=True,
                title=f"Ground State (E = {energy_str})"
            )

        # Plot first excited state wavefunction (if available)
        if len(eigenvectors) > 1:
            energy_str = f"{np.real(eigenvalues[1])/1.602e-19:.6f} eV"
            if np.iscomplex(eigenvalues[1]):
                energy_str += f" + {np.imag(eigenvalues[1])/1.602e-19:.6f}i eV"

            plot_wavefunction(
                ax=axes[1, 0],
                mesh=mesh,
                eigenvector=eigenvectors[:, 1],
                use_nm=True,
                center_coords=True,
                title=f"First Excited State (E = {energy_str})"
            )

        # Plot second excited state wavefunction (if available)
        if len(eigenvectors) > 2:
            energy_str = f"{np.real(eigenvalues[2])/1.602e-19:.6f} eV"
            if np.iscomplex(eigenvalues[2]):
                energy_str += f" + {np.imag(eigenvalues[2])/1.602e-19:.6f}i eV"

            plot_wavefunction(
                ax=axes[1, 1],
                mesh=mesh,
                eigenvector=eigenvectors[:, 2],
                use_nm=True,
                center_coords=True,
                title=f"Second Excited State (E = {energy_str})"
            )

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"chromium_qd_{well_type}_well.png", dpi=300)

        # Show the figure
        plt.show()

        # Show interactive visualization
        print("\nLaunching interactive visualization...")
        try:
            # For now, skip interactive visualization as it needs more fixes
            print("Interactive visualization is disabled for now.")
            # qdsim.show_interactive_visualization(
            #     mesh=mesh,
            #     eigenvectors=eigenvectors,
            #     eigenvalues=eigenvalues,
            #     potential=potential,
            #     poisson_solver=None  # We don't have a poisson solver
            # )
        except Exception as e:
            print(f"Error launching interactive visualization: {e}")
            print("Continuing with static plots...")

if __name__ == "__main__":
    main()
