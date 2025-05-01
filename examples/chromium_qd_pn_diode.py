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

# Import 3D visualization functions
from qdsim.visualization_3d import plot_3d_wavefunction, plot_3d_potential, create_interactive_3d_plot

def main():
    """
    Main function to run the simulation.
    """
    print("Chromium Quantum Dot in AlGaAs P-N Diode Simulation")
    print("==================================================")

    # Enable debug output
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Create a configuration
    config = qdsim.Config()

    # Mesh configuration
    config.Lx = 200e-9  # 200 nm
    config.Ly = 200e-9  # 200 nm
    config.nx = 201     # More elements for better resolution (odd number for center alignment)
    config.ny = 101     # Odd number for center alignment
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
    config.R = 20e-9  # 20 nm - larger radius for better visualization
    config.V_0 = 0.8  # 0.8 eV - deeper well for more bound states
    config.junction_position = 0.0  # Center the QD at the junction

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

        # Create separate potentials for visualization
        # Get node coordinates
        nodes = np.array(mesh.get_nodes())
        num_nodes = mesh.get_num_nodes()

        # Initialize potential arrays
        pn_potential = np.zeros(num_nodes)
        qd_potential = np.zeros(num_nodes)

        # Calculate separate potentials
        for i in range(num_nodes):
            x = nodes[i, 0]
            y = nodes[i, 1]

            # P-N junction potential
            junction_x = config.junction_position
            depletion_width = config.depletion_width
            V_p = 0.0
            V_n = simulator.built_in_potential() + config.V_r

            if x < junction_x - depletion_width/2:
                # p-side
                pn_potential[i] = V_p
            elif x > junction_x + depletion_width/2:
                # n-side
                pn_potential[i] = V_n
            else:
                # Depletion region - quadratic profile
                pos = 2 * (x - junction_x) / depletion_width
                pn_potential[i] = V_p + (V_n - V_p) * (pos**2 + pos + 1) / 4

            # Quantum dot potential
            r = np.sqrt((x - junction_x)**2 + y**2)  # Distance from junction center
            if well_type == "square":
                qd_potential[i] = -config.V_0 if r <= config.R else 0.0
            else:  # gaussian
                qd_potential[i] = -config.V_0 * np.exp(-r**2 / (2 * config.R**2))

        # Create a figure for visualization
        fig = plt.figure(figsize=(15, 12))

        # Create a 2x3 grid for the plots
        gs = fig.add_gridspec(2, 3)

        # Plot P-N junction potential
        ax1 = fig.add_subplot(gs[0, 0])
        plot_potential(
            ax=ax1,
            mesh=mesh,
            potential_values=pn_potential,
            use_nm=True,
            center_coords=True,
            title=f"P-N Junction Potential",
            convert_to_eV=True
        )

        # Plot quantum dot potential
        ax2 = fig.add_subplot(gs[0, 1])
        plot_potential(
            ax=ax2,
            mesh=mesh,
            potential_values=qd_potential,
            use_nm=True,
            center_coords=True,
            title=f"Quantum Dot Potential ({well_type})",
            convert_to_eV=True
        )

        # Plot combined potential
        ax3 = fig.add_subplot(gs[0, 2])
        plot_potential(
            ax=ax3,
            mesh=mesh,
            potential_values=potential,
            use_nm=True,
            center_coords=True,
            title=f"Combined Potential",
            convert_to_eV=True
        )

        # Plot ground state wavefunction
        if len(eigenvectors) > 0:
            energy_str = f"{np.real(eigenvalues[0])/1.602e-19:.6f} eV"
            if np.iscomplex(eigenvalues[0]):
                energy_str += f" + {np.imag(eigenvalues[0])/1.602e-19:.6f}i eV"

            ax4 = fig.add_subplot(gs[1, 0])
            plot_wavefunction(
                ax=ax4,
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

            ax5 = fig.add_subplot(gs[1, 1])
            plot_wavefunction(
                ax=ax5,
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

            ax6 = fig.add_subplot(gs[1, 2])
            plot_wavefunction(
                ax=ax6,
                mesh=mesh,
                eigenvector=eigenvectors[:, 2],
                use_nm=True,
                center_coords=True,
                title=f"Second Excited State (E = {energy_str})"
            )

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"chromium_qd_{well_type}_well_2d.png", dpi=300)

        # Show the figure
        plt.show()

        # Skip 3D visualizations for now
        print("\nSkipping 3D visualizations for now...")

        # Skip interactive visualization for now
        print("\nSkipping interactive visualization for now...")

if __name__ == "__main__":
    main()
