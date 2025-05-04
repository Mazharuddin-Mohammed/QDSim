#!/usr/bin/env python3
"""
Chromium Quantum Dot in InGaAs P-N Diode Simulation with Self-Consistent Solver

This script simulates a 2D chromium quantum dot at the interface of an InGaAs P-N diode
using the self-consistent Poisson-drift-diffusion solver integrated with the QD Schrödinger solver.
It calculates the bound or quasi-bound states and their corresponding wavefunction
probability densities.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Add the parent directory to the path so we can import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import qdsim
# Import qdsim
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction, plot_potential, plot_electric_field

# Create a simplified SelfConsistentSolver class for this example
class SelfConsistentSolver:
    def __init__(self, mesh, epsilon_r, charge_density, n_conc, p_conc, mu_n, mu_p):
        self.mesh = mesh
        self.epsilon_r = epsilon_r
        self.charge_density = charge_density
        self.n_conc = n_conc
        self.p_conc = p_conc
        self.mu_n = mu_n
        self.mu_p = mu_p
        self.phi = None
        self.n = None
        self.p = None

    def solve(self, V_p, V_n, N_A, N_D, tolerance=1e-6, max_iter=100):
        # Create a simplified potential profile for a pn junction
        nodes = self.mesh.get_nodes()
        num_nodes = self.mesh.get_num_nodes()
        self.phi = np.zeros(num_nodes)
        self.n = np.zeros(num_nodes)
        self.p = np.zeros(num_nodes)

        # Calculate the potential and carrier concentrations
        for i in range(num_nodes):
            x = nodes[i][0]
            y = nodes[i][1]

            # Simple pn junction potential
            if x < 0:
                self.phi[i] = V_p
                self.p[i] = N_A
                self.n[i] = 1e16  # Intrinsic concentration
            else:
                self.phi[i] = V_n
                self.n[i] = N_D
                self.p[i] = 1e16  # Intrinsic concentration

    def get_potential(self):
        return self.phi

    def get_n(self):
        return self.n

    def get_p(self):
        return self.p

    def get_electric_field(self, x, y):
        # Simplified electric field calculation
        return np.array([0.0, 0.0])

def main():
    """
    Main function to run the simulation.
    """
    print("Chromium Quantum Dot in InGaAs P-N Diode Simulation with Self-Consistent Solver")
    print("=============================================================================")

    # Enable debug output
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Create a configuration
    config = Config()

    # Physical constants
    config.e_charge = 1.602e-19  # Elementary charge in C

    # Mesh configuration
    config.Lx = 200e-9  # 200 nm (100nm for P-region, 100nm for N-region)
    config.Ly = 200e-9  # 200 nm
    config.nx = 201     # More elements for better resolution (odd number for center alignment)
    config.ny = 101     # Odd number for center alignment
    config.element_order = 1  # Linear elements

    # Set the junction position at the center (x=0)
    config.junction_position = 0.0

    # Material configuration
    # InGaAs properties
    config.matrix_material = "InGaAs"  # Matrix material
    config.diode_p_material = "InGaAs"  # P-type material
    config.diode_n_material = "InGaAs"  # N-type material

    # P-N junction configuration
    config.N_A = 1e24  # 1e18 cm^-3 = 1e24 m^-3
    config.N_D = 1e24  # 1e18 cm^-3 = 1e24 m^-3
    config.depletion_width = 50e-9  # 50 nm (as specified)
    config.V_r = 1.0  # 1.0 V reverse bias

    # Print junction parameters
    print(f"P-N junction position: x={config.junction_position*1e9:.1f}nm")
    print(f"Depletion width: {config.depletion_width*1e9:.1f}nm")

    # Quantum dot configuration
    config.qd_material = "Chromium"  # QD material (will use effective mass from this)
    config.R = 20e-9  # 20 nm radius
    config.V_0 = 0.8  # 0.8 eV well depth
    # QD is positioned at (0,0) which coincides with the P-N interface
    print(f"Quantum dot position: x=0.0nm, y=0.0nm")
    print(f"Quantum dot radius: {config.R*1e9:.1f}nm")
    print(f"Quantum dot potential depth: {config.V_0:.2f}eV")

    # Try both square and Gaussian wells
    well_types = ["square", "gaussian"]

    for well_type in well_types:
        print(f"\nRunning simulation with {well_type} well...")
        config.potential_type = well_type

        # Simulation configuration
        config.num_eigenvalues = 10  # Look for 10 states
        config.max_refinements = 2  # Use adaptive mesh refinement
        config.adaptive_threshold = 0.1  # Refinement threshold
        config.use_mpi = False  # Disable MPI for this example

        # Create a simulator
        simulator = Simulator(config)

        # Run the simulation
        start_time = time.time()
        eigenvalues, eigenvectors = simulator.run(num_eigenvalues=config.num_eigenvalues)
        end_time = time.time()

        print(f"Simulation completed in {end_time - start_time:.2f} seconds")

        # Convert eigenvalues to eV for display
        eigenvalues_eV = np.real(eigenvalues) / config.e_charge
        print("\nEigenvalues (eV):")
        for i, e in enumerate(eigenvalues_eV):
            print(f"  State {i}: {e:.3f} eV")

        # Identify bound states (negative real part of energy)
        bound_states = [i for i, energy in enumerate(eigenvalues_eV) if energy < 0]
        print(f"\nFound {len(bound_states)} bound states: {bound_states}")

        # Identify quasi-bound states (positive real part but small)
        quasi_bound_threshold = 0.1  # 0.1 eV
        quasi_bound_states = [i for i, energy in enumerate(eigenvalues_eV)
                             if 0 <= energy < quasi_bound_threshold]
        print(f"Found {len(quasi_bound_states)} quasi-bound states: {quasi_bound_states}")

        # Create separate potentials for visualization
        # Get node coordinates
        nodes = np.array(simulator.mesh.get_nodes())
        num_nodes = simulator.mesh.get_num_nodes()

        # Initialize potential arrays
        pn_potential = np.zeros(num_nodes)
        qd_potential = np.zeros(num_nodes)
        combined_potential = np.zeros(num_nodes)

        # Get the self-consistent potential
        sc_potential = simulator.sc_solver.get_potential()

        # Make sure sc_potential has the right size
        if len(sc_potential) != num_nodes:
            print(f"Warning: sc_potential size ({len(sc_potential)}) doesn't match num_nodes ({num_nodes})")
            # Resize sc_potential if needed
            if len(sc_potential) < num_nodes:
                sc_potential = np.pad(sc_potential, (0, num_nodes - len(sc_potential)))
            else:
                sc_potential = sc_potential[:num_nodes]

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
                # Square well with sharp boundaries
                qd_potential[i] = -config.V_0 if r <= config.R else 0.0
            else:  # gaussian
                # Gaussian well with smooth boundaries
                qd_potential[i] = -config.V_0 * np.exp(-r**2 / (2 * config.R**2))

            # Combined potential (from self-consistent solver or calculated)
            if i < len(sc_potential):
                combined_potential[i] = sc_potential[i] / config.e_charge  # Convert to eV
            else:
                # Fallback to calculated potential
                combined_potential[i] = pn_potential[i] + qd_potential[i]

        # Create a figure for visualization
        fig = plt.figure(figsize=(15, 12))

        # Create a 2x3 grid for the plots
        gs = fig.add_gridspec(2, 3)

        # Plot P-N junction potential
        ax1 = fig.add_subplot(gs[0, 0])
        plot_potential(
            ax=ax1,
            mesh=simulator.mesh,
            potential_values=pn_potential,
            use_nm=True,
            center_coords=True,
            title=f"P-N Junction Potential",
            convert_to_eV=False  # Already in eV
        )

        # Plot quantum dot potential
        ax2 = fig.add_subplot(gs[0, 1])
        plot_potential(
            ax=ax2,
            mesh=simulator.mesh,
            potential_values=qd_potential,
            use_nm=True,
            center_coords=True,
            title=f"Quantum Dot Potential ({well_type})",
            convert_to_eV=False  # Already in eV
        )

        # Plot combined potential
        ax3 = fig.add_subplot(gs[0, 2])
        plot_potential(
            ax=ax3,
            mesh=simulator.mesh,
            potential_values=combined_potential,
            use_nm=True,
            center_coords=True,
            title=f"Combined Potential",
            convert_to_eV=False  # Already in eV
        )

        # Plot ground state wavefunction
        if len(bound_states) > 0:
            ax4 = fig.add_subplot(gs[1, 0])
            plot_wavefunction(
                ax=ax4,
                mesh=simulator.mesh,
                eigenvector=eigenvectors[:, bound_states[0]],
                use_nm=True,
                center_coords=True,
                title=f"Ground State (E = {eigenvalues_eV[bound_states[0]]:.3f} eV)"
            )

            # Plot first excited state wavefunction if available
            if len(bound_states) > 1:
                ax5 = fig.add_subplot(gs[1, 1])
                plot_wavefunction(
                    ax=ax5,
                    mesh=simulator.mesh,
                    eigenvector=eigenvectors[:, bound_states[1]],
                    use_nm=True,
                    center_coords=True,
                    title=f"First Excited State (E = {eigenvalues_eV[bound_states[1]]:.3f} eV)"
                )

            # Plot second excited state wavefunction if available
            if len(bound_states) > 2:
                ax6 = fig.add_subplot(gs[1, 2])
                plot_wavefunction(
                    ax=ax6,
                    mesh=simulator.mesh,
                    eigenvector=eigenvectors[:, bound_states[2]],
                    use_nm=True,
                    center_coords=True,
                    title=f"Second Excited State (E = {eigenvalues_eV[bound_states[2]]:.3f} eV)"
                )
        else:
            # If no bound states, plot quasi-bound states if available
            if len(quasi_bound_states) > 0:
                ax4 = fig.add_subplot(gs[1, 0])
                plot_wavefunction(
                    ax=ax4,
                    mesh=simulator.mesh,
                    eigenvector=eigenvectors[:, quasi_bound_states[0]],
                    use_nm=True,
                    center_coords=True,
                    title=f"Quasi-Bound State (E = {eigenvalues_eV[quasi_bound_states[0]]:.3f} eV)"
                )

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"chromium_qd_ingaas_{well_type}_well_2d.png", dpi=300)

        # Create 3D visualization
        fig3d = plt.figure(figsize=(15, 12))
        gs3d = fig3d.add_gridspec(2, 2)

        # 3D plot of combined potential
        ax3d1 = fig3d.add_subplot(gs3d[0, 0], projection='3d')
        plot_potential_3d(
            ax=ax3d1,
            mesh=simulator.mesh,
            potential_values=combined_potential,
            title=f"Combined Potential ({well_type})",
            azimuth=45,
            elevation=30
        )

        # 3D plot of ground state wavefunction if available
        if len(bound_states) > 0:
            ax3d2 = fig3d.add_subplot(gs3d[0, 1], projection='3d')
            plot_wavefunction_3d(
                ax=ax3d2,
                mesh=simulator.mesh,
                eigenvector=eigenvectors[:, bound_states[0]],
                title=f"Ground State (E = {eigenvalues_eV[bound_states[0]]:.3f} eV)",
                azimuth=45,
                elevation=30
            )

            # 3D plot of first excited state if available
            if len(bound_states) > 1:
                ax3d3 = fig3d.add_subplot(gs3d[1, 0], projection='3d')
                plot_wavefunction_3d(
                    ax=ax3d3,
                    mesh=simulator.mesh,
                    eigenvector=eigenvectors[:, bound_states[1]],
                    title=f"First Excited State (E = {eigenvalues_eV[bound_states[1]]:.3f} eV)",
                    azimuth=45,
                    elevation=30
                )

            # 3D plot of second excited state if available
            if len(bound_states) > 2:
                ax3d4 = fig3d.add_subplot(gs3d[1, 1], projection='3d')
                plot_wavefunction_3d(
                    ax=ax3d4,
                    mesh=simulator.mesh,
                    eigenvector=eigenvectors[:, bound_states[2]],
                    title=f"Second Excited State (E = {eigenvalues_eV[bound_states[2]]:.3f} eV)",
                    azimuth=45,
                    elevation=30
                )

        # Adjust layout
        plt.tight_layout()

        # Save the 3D figure
        plt.savefig(f"chromium_qd_ingaas_{well_type}_well_3d.png", dpi=300)

        # Show the figures
        plt.show()

def plot_potential_3d(ax, mesh, potential_values, title, azimuth=30, elevation=30):
    """
    Plot a 3D visualization of the potential.

    Args:
        ax: Matplotlib 3D axis
        mesh: Mesh object
        potential_values: Potential values at mesh nodes
        title: Plot title
        azimuth: Azimuth angle for 3D view
        elevation: Elevation angle for 3D view
    """
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0] * 1e9  # Convert to nm
    y = nodes[:, 1] * 1e9  # Convert to nm
    z = potential_values

    # Create a triangulation for the 3D plot
    elements = np.array(mesh.get_elements())

    # Plot the triangulation
    ax.plot_trisurf(x, y, z, triangles=elements, cmap=cm.viridis, linewidth=0.2, alpha=0.8)

    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Potential (eV)')
    ax.set_title(title)

    # Set the view angle
    ax.view_init(elevation, azimuth)

def plot_wavefunction_3d(ax, mesh, wavefunction, title, azimuth=30, elevation=30):
    """
    Plot a 3D visualization of the wavefunction probability density.

    Args:
        ax: Matplotlib 3D axis
        mesh: Mesh object
        wavefunction: Wavefunction values at mesh nodes
        title: Plot title
        azimuth: Azimuth angle for 3D view
        elevation: Elevation angle for 3D view
    """
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0] * 1e9  # Convert to nm
    y = nodes[:, 1] * 1e9  # Convert to nm

    # Calculate probability density
    z = np.abs(wavefunction)**2

    # Normalize for better visualization
    z = z / np.max(z)

    # Create a triangulation for the 3D plot
    elements = np.array(mesh.get_elements())

    # Plot the triangulation
    ax.plot_trisurf(x, y, z, triangles=elements, cmap=cm.plasma, linewidth=0.2, alpha=0.8)

    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Probability Density (normalized)')
    ax.set_title(title)

    # Set the view angle
    ax.view_init(elevation, azimuth)

if __name__ == "__main__":
    main()
