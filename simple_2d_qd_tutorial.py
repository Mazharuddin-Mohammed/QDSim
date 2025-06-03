#!/usr/bin/env python3
"""
Simple 2D Quantum Dot Tutorial for QDSim

This script provides a simplified tutorial for simulating a 2D quantum dot
embedded in a PN junction using QDSim. It demonstrates the basic workflow
with a focus on stability and avoiding memory issues.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

def create_custom_colormap():
    """Create a custom colormap for visualizing potentials and wavefunctions."""
    # Define custom colors for the colormap
    colors = [(0.0, 'darkblue'), (0.25, 'blue'), (0.5, 'white'), (0.75, 'red'), (1.0, 'darkred')]

    # Create the colormap
    cmap_name = 'custom_diverging'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    return cm

def main():
    """Main function implementing the simplified 2D quantum dot tutorial."""
    print("\n=== Simple 2D Quantum Dot Tutorial ===\n")

    # Step 1: Create a mesh
    print("Step 1: Creating a mesh")
    Lx = 100.0  # Width of the domain (nm)
    Ly = 100.0  # Height of the domain (nm)
    nx = 30     # Number of elements in x-direction (small for stability)
    ny = 30     # Number of elements in y-direction (small for stability)
    element_order = 1  # Use linear elements for simplicity

    mesh = qdsim_cpp.Mesh(Lx, Ly, nx, ny, element_order)
    print(f"Mesh created with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")

    # Step 2: Define material properties
    print("Step 2: Defining material properties")

    # Create a material database
    db = qdsim_cpp.MaterialDatabase()

    # Get material properties
    gaas = db.get_material("GaAs")
    inas = db.get_material("InAs")

    # Define default values in case attributes are not accessible
    gaas_epsilon_r = 12.9  # Default value for GaAs
    gaas_band_gap = 1.42   # Default value for GaAs in eV
    inas_epsilon_r = 15.15 # Default value for InAs
    inas_band_gap = 0.354  # Default value for InAs in eV

    # Try to access material properties
    try:
        gaas_epsilon_r = gaas.epsilon_r
        gaas_band_gap = gaas.E_g
        inas_epsilon_r = inas.epsilon_r
        inas_band_gap = inas.E_g
    except AttributeError:
        print("Warning: Could not access material properties directly. Using default values.")

    print(f"GaAs properties: epsilon_r = {gaas_epsilon_r}, band gap = {gaas_band_gap} eV")
    print(f"InAs properties: epsilon_r = {inas_epsilon_r}, band gap = {inas_band_gap} eV")

    # Step 3: Define the quantum dot parameters
    print("Step 3: Defining quantum dot parameters")

    qd_radius = 5.0  # Quantum dot radius (nm)
    qd_depth = 0.3   # Quantum dot potential depth (eV)
    qd_x = 0.0       # x-position of quantum dot center (nm)
    qd_y = 0.0       # y-position of quantum dot center (nm)

    # Step 4: Define the PN junction parameters
    print("Step 4: Defining PN junction parameters")

    N_A = 1e24  # Acceptor concentration (m^-3)
    N_D = 1e24  # Donor concentration (m^-3)
    junction_position = 0.0  # Position of the junction (nm)

    # Step 5: Define callback functions
    print("Step 5: Defining callback functions")

    def epsilon_r(x, y):
        """Relative permittivity function."""
        # Check if the point is inside the quantum dot
        r = np.sqrt((x - qd_x)**2 + (y - qd_y)**2)
        if r < qd_radius:
            return inas_epsilon_r  # InAs
        else:
            return gaas_epsilon_r  # GaAs

    def rho(x, y):
        """Charge density function."""
        q = 1.602e-19  # Elementary charge in C

        # Simple step function for doping
        if x < junction_position:
            return q * N_A  # P-type region
        else:
            return -q * N_D  # N-type region

    # Step 6: Create a Poisson solver
    print("Step 6: Creating a Poisson solver")

    # Set boundary conditions
    V_p = 0.0  # Voltage at the p-contact
    V_n = 0.7  # Voltage at the n-contact (forward bias)

    # Get mesh nodes
    nodes = np.array(mesh.get_nodes())
    num_nodes = len(nodes)

    # Create a FEM solver instead (which includes Poisson solver functionality)
    print("Creating a FEM solver")
    fem_solver = qdsim_cpp.FEMSolver(mesh)

    # Set up the Poisson equation
    print("Setting up the Poisson equation")

    # Create arrays for permittivity and charge density
    eps_r_values = np.zeros(num_nodes)
    rho_values = np.zeros(num_nodes)

    # Fill the arrays
    for i in range(num_nodes):
        x, y = nodes[i]
        eps_r_values[i] = epsilon_r(x, y)
        rho_values[i] = rho(x, y)

    # Set up the Poisson equation in the FEM solver
    fem_solver.setup_poisson_equation(eps_r_values, rho_values)

    # Step 7: Solve the Poisson equation
    print("Step 7: Solving the Poisson equation")

    # Set boundary conditions
    # Create boundary condition arrays
    bc_nodes = []
    bc_values = []

    # Left boundary (p-side)
    for i in range(num_nodes):
        x, y = nodes[i]
        if abs(x - (-Lx/2)) < 1e-6:  # Left boundary
            bc_nodes.append(i)
            bc_values.append(V_p)
        elif abs(x - (Lx/2)) < 1e-6:  # Right boundary
            bc_nodes.append(i)
            bc_values.append(V_n)

    # Set the boundary conditions
    fem_solver.set_dirichlet_boundary_conditions(bc_nodes, bc_values)

    # Solve the Poisson equation
    print("Solving the system...")
    fem_solver.solve()

    # Get the potential
    potential = np.array(fem_solver.get_solution())
    print(f"Potential shape: {potential.shape}")
    print(f"Potential range: [{potential.min()}, {potential.max()}]")

    # Step 8: Add the quantum dot potential
    print("Step 8: Adding the quantum dot potential")

    # Create the quantum dot potential
    qd_potential = np.zeros(num_nodes)
    for i in range(num_nodes):
        x, y = nodes[i]
        r = np.sqrt((x - qd_x)**2 + (y - qd_y)**2)
        if r < 3*qd_radius:  # Truncate at 3*radius for efficiency
            qd_potential[i] = -qd_depth * np.exp(-r**2 / (2*qd_radius**2))

    # Combine the potentials
    combined_potential = potential + qd_potential

    # Step 9: Visualize the results
    print("Step 9: Visualizing the results")

    # Create a directory for the results
    results_dir = "results_simple_2d_qd"
    os.makedirs(results_dir, exist_ok=True)

    # Create a custom colormap
    cmap = create_custom_colormap()

    # Extract x and y coordinates
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Create a grid for plotting
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Create an interpolator
    from scipy.interpolate import griddata

    # Plot the PN junction potential
    plt.figure(figsize=(10, 8))
    zi = griddata((x, y), potential, (xi, yi), method='cubic')
    plt.contourf(xi, yi, zi, 50, cmap=cmap)
    plt.colorbar(label='Potential (V)')
    plt.title('PN Junction Potential')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig(os.path.join(results_dir, "pn_potential.png"), dpi=300)

    # Plot the quantum dot potential
    plt.figure(figsize=(10, 8))
    zi = griddata((x, y), qd_potential, (xi, yi), method='cubic')
    plt.contourf(xi, yi, zi, 50, cmap=cmap)
    plt.colorbar(label='Potential (eV)')
    plt.title('Quantum Dot Potential')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig(os.path.join(results_dir, "qd_potential.png"), dpi=300)

    # Plot the combined potential
    plt.figure(figsize=(10, 8))
    zi = griddata((x, y), combined_potential, (xi, yi), method='cubic')
    plt.contourf(xi, yi, zi, 50, cmap=cmap)
    plt.colorbar(label='Potential (V)')
    plt.title('Combined Potential (PN Junction + Quantum Dot)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig(os.path.join(results_dir, "combined_potential.png"), dpi=300)

    # Create a combined visualization
    plt.figure(figsize=(15, 5))

    # Plot PN junction potential
    plt.subplot(1, 3, 1)
    zi = griddata((x, y), potential, (xi, yi), method='cubic')
    plt.contourf(xi, yi, zi, 50, cmap=cmap)
    plt.colorbar(label='Potential (V)')
    plt.title('PN Junction Potential')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')

    # Plot quantum dot potential
    plt.subplot(1, 3, 2)
    zi = griddata((x, y), qd_potential, (xi, yi), method='cubic')
    plt.contourf(xi, yi, zi, 50, cmap=cmap)
    plt.colorbar(label='Potential (eV)')
    plt.title('Quantum Dot Potential')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')

    # Plot combined potential
    plt.subplot(1, 3, 3)
    zi = griddata((x, y), combined_potential, (xi, yi), method='cubic')
    plt.contourf(xi, yi, zi, 50, cmap=cmap)
    plt.colorbar(label='Potential (V)')
    plt.title('Combined Potential')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combined_visualization.png"), dpi=300)

    print(f"\nResults saved to {results_dir}")
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()
