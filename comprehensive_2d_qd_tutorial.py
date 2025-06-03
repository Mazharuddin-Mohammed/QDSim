#!/usr/bin/env python3
"""
Comprehensive 2D Quantum Dot Tutorial for QDSim

This script provides a comprehensive tutorial for simulating a 2D quantum dot
embedded in a PN junction using QDSim. It demonstrates the full workflow from
mesh creation to visualization of results, with detailed explanations at each step.

The tutorial covers:
1. Setting up the simulation domain and mesh
2. Defining material properties and quantum dot parameters
3. Solving the Poisson equation for the electrostatic potential
4. Solving the Schrödinger equation for quantum states
5. Visualizing the results

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add the necessary paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build'))

# Import QDSim modules
try:
    import qdsim_cpp
    print("Successfully imported C++ backend")
except ImportError as e:
    print(f"Warning: Could not import C++ backend: {e}")
    print("Make sure the C++ backend is built and in the Python path")
    sys.exit(1)

# Import frontend modules
try:
    from frontend.qdsim.config import Config
    from frontend.qdsim.simulator import Simulator
    from frontend.qdsim.visualization import plot_wavefunction, plot_potential, plot_electric_field
    print("Successfully imported Python frontend")
except ImportError as e:
    print(f"Warning: Could not import Python frontend: {e}")
    print("Make sure the Python frontend is installed")
    sys.exit(1)

def create_custom_colormap():
    """Create a custom colormap for visualizing potentials and wavefunctions."""
    # Define custom colors for the colormap
    colors = [(0.0, 'darkblue'), (0.25, 'blue'), (0.5, 'white'), (0.75, 'red'), (1.0, 'darkred')]

    # Create the colormap
    cmap_name = 'custom_diverging'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    return cm

def main():
    """Main function implementing the comprehensive 2D quantum dot tutorial."""
    print("\n=== Comprehensive 2D Quantum Dot Tutorial ===\n")

    # Step 1: Set up the simulation domain and mesh
    print("Step 1: Setting up the simulation domain and mesh")

    # Create a configuration object
    config = Config()

    # Set domain size (in nm)
    config.Lx = 100.0  # Width of the domain
    config.Ly = 100.0  # Height of the domain

    # Set mesh parameters (use a smaller mesh to avoid memory issues)
    config.nx = 50   # Number of elements in x-direction
    config.ny = 50   # Number of elements in y-direction
    config.element_order = 1  # Use linear elements to reduce memory usage

    # Step 2: Define material properties and quantum dot parameters
    print("Step 2: Defining material properties and quantum dot parameters")

    # Set material properties
    config.diode_p_material = "GaAs"  # P-type material
    config.diode_n_material = "GaAs"  # N-type material
    config.qd_material = "InAs"       # Quantum dot material
    config.matrix_material = "GaAs"   # Matrix material

    # Set quantum dot parameters
    config.R = 5.0  # Quantum dot radius (nm)
    config.V_0 = 0.3 * 1.602e-19  # Quantum dot potential depth (eV converted to J)
    config.potential_type = "gaussian"  # Potential type (gaussian or square)

    # Set PN junction parameters
    config.N_A = 1e24  # Acceptor concentration (m^-3)
    config.N_D = 1e24  # Donor concentration (m^-3)
    config.V_r = 0.0   # Reverse bias voltage (V)

    # Set the junction position
    config.junction_position = 0.0  # Position of the junction (nm)

    # Set quantum dot position (at the junction for maximum effect)
    config.qd_x = 0.0    # x-position of quantum dot center (nm)
    config.qd_y = 0.0    # y-position of quantum dot center (nm)

    # Set solver parameters
    config.tolerance = 1e-6  # Convergence tolerance
    config.max_iter = 100    # Maximum number of iterations
    config.use_gpu = True    # Use GPU acceleration if available

    # Step 3: Create the simulator
    print("Step 3: Creating the simulator")
    try:
        simulator = Simulator(config)
        print("Simulator created successfully")
    except Exception as e:
        print(f"Error creating simulator: {e}")
        sys.exit(1)

    # Step 4: Solve the Poisson equation for the electrostatic potential
    print("Step 4: Solving the Poisson equation for the electrostatic potential")
    try:
        # The Poisson equation is already solved during simulator initialization
        # Get the potential
        potential = simulator.sc_solver.get_potential()
        print(f"Potential shape: {len(potential)}")
        print(f"Potential range: [{min(potential)}, {max(potential)}]")
    except Exception as e:
        print(f"Error solving Poisson equation: {e}")
        sys.exit(1)

    # Step 5: Solve the Schrödinger equation for quantum states
    print("Step 5: Solving the Schrödinger equation for quantum states")
    try:
        # Create a SchrodingerSolver
        schrodinger_solver = qdsim_cpp.create_schrodinger_solver(
            simulator.mesh,
            lambda x, y: simulator.effective_mass(x, y),
            lambda x, y: simulator.potential(x, y),
            config.use_gpu
        )

        # Solve for the lowest 5 eigenstates
        num_states = 5
        eigenvalues, eigenvectors = schrodinger_solver.solve(num_states)

        print(f"Found {len(eigenvalues)} eigenvalues:")
        for i, e in enumerate(eigenvalues):
            print(f"  E_{i} = {e:.6f} eV")
    except Exception as e:
        print(f"Error solving Schrödinger equation: {e}")
        sys.exit(1)

    # Step 6: Visualize the results
    print("Step 6: Visualizing the results")

    # Create a directory for the results
    results_dir = "results_2d_qd_tutorial"
    os.makedirs(results_dir, exist_ok=True)

    # Create a custom colormap
    cmap = create_custom_colormap()

    # Get mesh nodes
    nodes = np.array(simulator.mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Plot the potential
    plt.figure(figsize=(10, 8))
    plot_potential(simulator, cmap=cmap)
    plt.title("Electrostatic Potential")
    plt.savefig(os.path.join(results_dir, "potential.png"), dpi=300)

    # Plot the wavefunctions
    for i in range(min(num_states, len(eigenvectors))):
        plt.figure(figsize=(10, 8))
        plot_wavefunction(simulator, eigenvectors[i], eigenvalues[i], cmap=cmap)
        plt.title(f"Wavefunction {i} (E = {eigenvalues[i]:.6f} eV)")
        plt.savefig(os.path.join(results_dir, f"wavefunction_{i}.png"), dpi=300)

    # Plot the electric field
    plt.figure(figsize=(10, 8))
    plot_electric_field(simulator)
    plt.title("Electric Field")
    plt.savefig(os.path.join(results_dir, "electric_field.png"), dpi=300)

    # Create a combined visualization
    plt.figure(figsize=(15, 10))

    # Plot potential
    plt.subplot(2, 3, 1)
    plot_potential(simulator, cmap=cmap)
    plt.title("Potential")

    # Plot electric field
    plt.subplot(2, 3, 2)
    plot_electric_field(simulator)
    plt.title("Electric Field")

    # Plot wavefunctions
    for i in range(min(4, len(eigenvectors))):
        plt.subplot(2, 3, i+3)
        plot_wavefunction(simulator, eigenvectors[i], eigenvalues[i], cmap=cmap)
        plt.title(f"Wavefunction {i} (E = {eigenvalues[i]:.6f} eV)")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combined_visualization.png"), dpi=300)

    print(f"\nResults saved to {results_dir}")
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()
