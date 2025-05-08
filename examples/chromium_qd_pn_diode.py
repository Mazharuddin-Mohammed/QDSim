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

# We won't use the interactive visualization as it requires Vulkan rendering

# Define additional visualization functions
def plot_mesh_grid(mesh, use_nm=True, center_coords=True, title=None):
    """
    Plot the mesh grid in a separate window.

    Args:
        mesh: The mesh object
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        title: Custom title for the plot
    """
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get mesh data
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())

    # Center coordinates if requested
    if center_coords:
        x_center = np.mean(nodes[:, 0])
        y_center = np.mean(nodes[:, 1])
        nodes_plot = nodes.copy()
        nodes_plot[:, 0] -= x_center
        nodes_plot[:, 1] -= y_center
    else:
        nodes_plot = nodes

    # Convert to nm if requested
    scale = 1e9 if use_nm else 1.0
    nodes_plot = nodes_plot * scale

    # Plot the mesh grid
    for element in elements:
        # Get the vertices of the element
        vertices = nodes_plot[element]

        # Add the first vertex again to close the triangle
        vertices = np.vstack([vertices, vertices[0]])

        # Plot the element
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=0.5, alpha=0.5)

    # Set labels
    ax.set_xlabel('x (nm)' if use_nm else 'x (m)')
    ax.set_ylabel('y (nm)' if use_nm else 'y (m)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Mesh Grid')

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)

    return fig, ax

def plot_3d_surface(ax, mesh, values, use_nm=True, center_coords=True,
                   title=None, convert_to_eV=False, azimuth=30, elevation=30):
    """
    Plot a 3D surface of values on the mesh.

    Args:
        ax: Matplotlib 3D axis
        mesh: Mesh object
        values: Values to plot
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        title: Custom title for the plot
        convert_to_eV: If True, convert values from J to eV
        azimuth: Azimuth angle for 3D view in degrees
        elevation: Elevation angle for 3D view in degrees
    """
    # Get mesh data
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())

    # Center coordinates if requested
    if center_coords:
        x_center = np.mean(nodes[:, 0])
        y_center = np.mean(nodes[:, 1])
        nodes_plot = nodes.copy()
        nodes_plot[:, 0] -= x_center
        nodes_plot[:, 1] -= y_center
    else:
        nodes_plot = nodes

    # Convert to nm if requested
    scale = 1e9 if use_nm else 1.0
    nodes_plot = nodes_plot * scale

    # Convert values to eV if requested
    if convert_to_eV:
        values_plot = values / 1.602e-19  # Convert J to eV
        z_label = 'Potential (eV)'
    else:
        values_plot = values
        z_label = 'Value'

    # Create a triangulation for 3D plotting
    from matplotlib.tri import Triangulation
    triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)

    # Plot the surface
    surf = ax.plot_trisurf(triang, values_plot, cmap='viridis', edgecolor='none', alpha=0.8)

    # Add colorbar
    plt.colorbar(surf, ax=ax, label=z_label)

    # Set labels
    ax.set_xlabel('x (nm)' if use_nm else 'x (m)')
    ax.set_ylabel('y (nm)' if use_nm else 'y (m)')
    ax.set_zlabel(z_label)

    # Set title
    if title:
        ax.set_title(title)

    # Set view angle
    ax.view_init(elevation, azimuth)

    return surf

# Interactive 3D visualization removed as it requires Vulkan rendering

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

    # Physical constants
    config.e_charge = 1.602e-19  # Elementary charge in C

    # Mesh configuration
    config.Lx = 200e-9  # 200 nm (100nm for P-region, 100nm for N-region)
    config.Ly = 200e-9  # 200 nm
    config.nx = 201     # More elements for better resolution (odd number for center alignment)
    config.ny = 101     # Odd number for center alignment
    config.element_order = 1  # Linear elements

    # Set the junction position at the center (x=0)
    # This ensures equal P and N regions (100nm each)
    config.junction_position = 0.0

    # Print mesh dimensions
    print(f"Mesh dimensions: {config.Lx*1e9:.1f}nm x {config.Ly*1e9:.1f}nm")
    print(f"P-region length: {config.Lx*1e9/2:.1f}nm")
    print(f"N-region length: {config.Lx*1e9/2:.1f}nm")
    print(f"Junction position: {config.junction_position*1e9:.1f}nm")

    # Material configuration
    # AlGaAs properties
    config.matrix_material = "AlGaAs"  # Matrix material

    # P-N junction configuration
    config.N_A = 1e24  # 1e18 cm^-3 = 1e24 m^-3
    config.N_D = 1e24  # 1e18 cm^-3 = 1e24 m^-3
    config.depletion_width = 50e-9  # 50 nm (as specified)
    config.V_r = 1.0  # 1.0 V reverse bias

    # Calculate built-in potential for display
    kT = 0.026  # Room temperature in eV
    ni = 1e6  # Intrinsic carrier concentration in cm^-3
    built_in = kT * np.log((config.N_A * config.N_D) / (ni * ni))
    print(f"Built-in potential: {built_in:.3f} V")
    print(f"Reverse bias: {config.V_r:.1f} V")
    print(f"Total junction potential: {built_in + config.V_r:.3f} V")

    # Print junction parameters
    print(f"P-N junction position: x={config.junction_position*1e9:.1f}nm")
    print(f"Depletion width: {config.depletion_width*1e9:.1f}nm")

    # Quantum dot configuration
    config.qd_material = "Chromium"  # QD material (will use effective mass from this)
    config.R = 20e-9  # 20 nm radius
    config.V_0 = 0.8  # 0.8 eV well depth

    # QD is positioned at the P-N interface (junction_position, 0)
    # This is explicitly set to ensure proper positioning
    print(f"Quantum dot position: ({config.junction_position*1e9:.1f}nm, 0.0nm)")
    print(f"Quantum dot radius: {config.R*1e9:.1f}nm")
    print(f"Quantum dot potential depth: {config.V_0:.2f} eV")

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

        # Show the mesh grid in a separate window
        print("\nShowing mesh grid...")
        mesh_fig, mesh_ax = plot_mesh_grid(
            mesh=mesh,
            use_nm=True,
            center_coords=True,
            title=f"Mesh Grid ({mesh.get_num_nodes()} nodes, {mesh.get_num_elements()} elements)"
        )
        plt.savefig(f"chromium_qd_{well_type}_mesh.png", dpi=300)
        plt.show()

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
            # Calculate distance from junction center (junction_x, 0)
            # This ensures the QD is properly positioned at the junction
            r = np.sqrt((x - junction_x)**2 + y**2)  # Distance from junction center

            # Convert potential depth from eV to J
            V_0_joules = config.V_0 * config.e_charge

            if well_type == "square":
                # Square well with sharp boundaries
                qd_potential[i] = -V_0_joules if r <= config.R else 0.0
            else:  # gaussian
                # Gaussian well with smooth boundaries
                qd_potential[i] = -V_0_joules * np.exp(-r**2 / (2 * config.R**2))

            # Print some debug info for the first few nodes to verify QD positioning
            if i < 5:
                print(f"Node {i}: x={x*1e9:.1f}nm, y={y*1e9:.1f}nm, r={r*1e9:.1f}nm, QD potential={qd_potential[i]/config.e_charge:.3f}eV")

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

        # Create 3D visualizations with fixed viewing angles
        print("\nCreating 3D visualizations...")

        # Create a figure for 3D visualization
        fig_3d = plt.figure(figsize=(15, 10))

        # Create a 2x2 grid for the plots
        gs_3d = fig_3d.add_gridspec(2, 2)

        # Plot combined potential in 3D with 30° viewing angle
        ax3d_1 = fig_3d.add_subplot(gs_3d[0, 0], projection='3d')
        plot_3d_surface(
            ax=ax3d_1,
            mesh=mesh,
            values=potential,
            use_nm=True,
            center_coords=True,
            title=f"Combined Potential ({well_type}) - 30° view",
            convert_to_eV=True,
            azimuth=30,
            elevation=30
        )

        # Plot ground state wavefunction in 3D with 45° viewing angle
        if len(eigenvectors) > 0:
            energy_str = f"{np.real(eigenvalues[0])/1.602e-19:.6f} eV"
            if np.iscomplex(eigenvalues[0]):
                energy_str += f" + {np.imag(eigenvalues[0])/1.602e-19:.6f}i eV"

            ax3d_2 = fig_3d.add_subplot(gs_3d[0, 1], projection='3d')
            plot_3d_surface(
                ax=ax3d_2,
                mesh=mesh,
                values=np.abs(eigenvectors[:, 0])**2,
                use_nm=True,
                center_coords=True,
                title=f"Ground State (E = {energy_str}) - 45° view",
                convert_to_eV=False,
                azimuth=45,
                elevation=30
            )

        # Plot first excited state wavefunction in 3D with 60° viewing angle
        if len(eigenvectors) > 1:
            energy_str = f"{np.real(eigenvalues[1])/1.602e-19:.6f} eV"
            if np.iscomplex(eigenvalues[1]):
                energy_str += f" + {np.imag(eigenvalues[1])/1.602e-19:.6f}i eV"

            ax3d_3 = fig_3d.add_subplot(gs_3d[1, 0], projection='3d')
            plot_3d_surface(
                ax=ax3d_3,
                mesh=mesh,
                values=np.abs(eigenvectors[:, 1])**2,
                use_nm=True,
                center_coords=True,
                title=f"First Excited State (E = {energy_str}) - 60° view",
                convert_to_eV=False,
                azimuth=60,
                elevation=30
            )

        # Plot second excited state wavefunction in 3D with 45° elevation
        if len(eigenvectors) > 2:
            energy_str = f"{np.real(eigenvalues[2])/1.602e-19:.6f} eV"
            if np.iscomplex(eigenvalues[2]):
                energy_str += f" + {np.imag(eigenvalues[2])/1.602e-19:.6f}i eV"

            ax3d_4 = fig_3d.add_subplot(gs_3d[1, 1], projection='3d')
            plot_3d_surface(
                ax=ax3d_4,
                mesh=mesh,
                values=np.abs(eigenvectors[:, 2])**2,
                use_nm=True,
                center_coords=True,
                title=f"Second Excited State (E = {energy_str}) - 45° elevation",
                convert_to_eV=False,
                azimuth=30,
                elevation=45
            )

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"chromium_qd_{well_type}_well_3d.png", dpi=300)

        # Show the figure
        plt.show()

        print("\nNote: Interactive visualization is not available as it requires Vulkan rendering.")

if __name__ == "__main__":
    main()
