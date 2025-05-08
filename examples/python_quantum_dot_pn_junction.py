"""
Example script for simulating a chromium quantum dot in an AlGaAs P-N junction.

This script demonstrates the use of the Python implementation of the FullPoissonDriftDiffusionSolver
to simulate a P-N junction diode with a chromium quantum dot.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import QDSim
from frontend.qdsim import Mesh, FullPoissonDriftDiffusionSolver, SchrodingerSolver, create_schrodinger_solver

# Try to import GPU interpolator
try:
    from frontend.qdsim.gpu_interpolator import GPUInterpolator
    has_gpu_interpolator = True
    print("GPU interpolator available")
except ImportError:
    has_gpu_interpolator = False
    print("GPU interpolator not available")

def run_simulation(bias_voltage=0.0, mesh_size=31):
    """
    Run a simulation of a P-N junction diode with a quantum dot.

    Args:
        bias_voltage: The bias voltage applied to the diode
        mesh_size: The size of the mesh (number of nodes in each dimension)
                  Note: Using a smaller mesh size for the Python implementation
                  for better performance.

    Returns:
        solver: The FullPoissonDriftDiffusionSolver object
        mesh: The mesh
        schrodinger_solver: The SchrodingerSolver object
    """
    # Create a mesh for the P-N junction diode
    Lx = 200.0  # nm
    Ly = 100.0  # nm
    mesh = Mesh(Lx, Ly, mesh_size, mesh_size // 2, 1)  # Linear elements

    # Define the quantum dot parameters
    qd_x0 = Lx / 2.0  # QD at the P-N junction
    qd_y0 = Ly / 2.0  # QD at the center of the device
    qd_radius = 5.0  # nm
    qd_depth = 0.5  # eV

    # Define the doping profile
    def doping_profile(x, y):
        # P-N junction at x = Lx/2
        if x < Lx / 2.0:
            return -1e17  # P-type (acceptors)
        else:
            return 1e17  # N-type (donors)

    # Define the relative permittivity
    def epsilon_r(x, y):
        return 12.9  # AlGaAs

    # Create the solver
    solver = FullPoissonDriftDiffusionSolver(mesh, epsilon_r, doping_profile)

    # Set the carrier statistics model
    solver.set_carrier_statistics_model(False)  # Use Boltzmann statistics

    # Enable GPU acceleration if available
    try:
        solver.enable_gpu_acceleration(True)
        print("GPU acceleration enabled for Poisson-Drift-Diffusion solver")
    except AttributeError:
        print("GPU acceleration not available for Poisson-Drift-Diffusion solver")

    # Set the non-linear solver type
    try:
        solver.set_nonlinear_solver('newton', use_line_search=True, use_thomas_algorithm=True)
        print("Using Newton solver with line search and Thomas algorithm")
    except AttributeError:
        print("Non-linear solver settings not available")

    # Solve the coupled Poisson-drift-diffusion equations
    print(f"Solving for bias voltage = {bias_voltage} V")
    start_time = time.time()
    solver.solve(0.0, bias_voltage, 1e-6, 100)
    end_time = time.time()
    print(f"Solution time: {end_time - start_time:.2f} seconds")

    # Get the P-N junction potential
    pn_potential = solver.get_potential()

    # Define the quantum dot potential
    def qd_potential(x, y):
        r = np.sqrt((x - qd_x0)**2 + (y - qd_y0)**2)
        return -qd_depth * np.exp(-r**2 / (2.0 * qd_radius**2))

    # Create a combined potential function
    def combined_potential(x, y):
        # Get the nearest node for the P-N junction potential
        min_dist = float('inf')
        nearest_node = 0

        for i in range(mesh.get_num_nodes()):
            node = mesh.get_nodes()[i]
            dist = (node[0] - x)**2 + (node[1] - y)**2

            if dist < min_dist:
                min_dist = dist
                nearest_node = i

        # Combine the potentials
        return pn_potential[nearest_node] + qd_potential(x, y)

    # Create a Schrodinger solver with GPU acceleration
    try:
        schrodinger_solver = create_schrodinger_solver(mesh, combined_potential, use_gpu=True)
        print("Created Schrodinger solver with GPU acceleration enabled")
    except Exception as e:
        print(f"Failed to create Schrodinger solver with GPU acceleration: {e}")
        schrodinger_solver = create_schrodinger_solver(mesh, combined_potential, use_gpu=False)
        print("Created Schrodinger solver without GPU acceleration")

    # Solve the Schrodinger equation
    print("Solving Schrodinger equation")
    start_time = time.time()
    eigenvalues, eigenvectors = schrodinger_solver.solve(10)  # Find 10 lowest energy states
    end_time = time.time()
    print(f"Schrodinger solve time: {end_time - start_time:.2f} seconds")

    return solver, mesh, schrodinger_solver, eigenvalues, eigenvectors, combined_potential, qd_potential

def plot_results(solver, mesh, schrodinger_solver, eigenvalues, eigenvectors, combined_potential, qd_potential, bias_voltage=0.0):
    """
    Plot the results of the simulation.

    Args:
        solver: The FullPoissonDriftDiffusionSolver object
        mesh: The mesh
        schrodinger_solver: The SchrodingerSolver object
        eigenvalues: The eigenvalues (energy levels)
        eigenvectors: The eigenvectors (wavefunctions)
        combined_potential: The combined potential function
        qd_potential: The quantum dot potential function
        bias_voltage: The bias voltage applied to the diode
    """
    # Get the results
    pn_potential = solver.get_potential()
    n = solver.get_electron_concentration()
    p = solver.get_hole_concentration()
    E_field = solver.get_electric_field_all()

    # Create a grid for plotting
    Lx = mesh.get_lx()
    Ly = mesh.get_ly()
    nx, ny = 200, 100
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Interpolate the results onto the grid
    if has_gpu_interpolator:
        # Use GPU interpolator for faster interpolation
        print("Using GPU interpolator for faster interpolation")
        interpolator = GPUInterpolator(mesh, use_gpu=True)

        # Interpolate the potential
        pn_potential_grid = interpolator.interpolate_grid(0, Lx, 0, Ly, nx, ny, pn_potential)

        # Interpolate the electric field
        Ex_grid = np.zeros((ny, nx))
        Ey_grid = np.zeros((ny, nx))

        # Convert E_field to separate arrays for x and y components
        E_field_x = np.array([E_field[i][0] for i in range(len(E_field))])
        E_field_y = np.array([E_field[i][1] for i in range(len(E_field))])

        # Interpolate the electric field components
        Ex_grid = interpolator.interpolate_grid(0, Lx, 0, Ly, nx, ny, E_field_x)
        Ey_grid = interpolator.interpolate_grid(0, Lx, 0, Ly, nx, ny, E_field_y)

        # Interpolate the carrier concentrations
        n_grid = interpolator.interpolate_grid(0, Lx, 0, Ly, nx, ny, n)
        p_grid = interpolator.interpolate_grid(0, Lx, 0, Ly, nx, ny, p)

        # Calculate the quantum dot potential directly
        qd_potential_grid = np.zeros((ny, nx))
        for i in range(nx):
            for j in range(ny):
                qd_potential_grid[j, i] = qd_potential(x[i], y[j])

        # Calculate the combined potential
        combined_potential_grid = pn_potential_grid + qd_potential_grid
    else:
        # Use simple nearest-neighbor interpolation
        print("Using simple nearest-neighbor interpolation")
        pn_potential_grid = np.zeros((ny, nx))
        qd_potential_grid = np.zeros((ny, nx))
        combined_potential_grid = np.zeros((ny, nx))
        n_grid = np.zeros((ny, nx))
        p_grid = np.zeros((ny, nx))
        Ex_grid = np.zeros((ny, nx))
        Ey_grid = np.zeros((ny, nx))

        # Simple nearest-neighbor interpolation
        for i in range(nx):
            for j in range(ny):
                # Find the nearest node
                min_dist = float('inf')
                nearest_node = 0

                for k in range(mesh.get_num_nodes()):
                    node = mesh.get_nodes()[k]
                    dist = (node[0] - x[i])**2 + (node[1] - y[j])**2

                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = k

                # Interpolate the results
                pn_potential_grid[j, i] = pn_potential[nearest_node]
                qd_potential_grid[j, i] = qd_potential(x[i], y[j])
                combined_potential_grid[j, i] = pn_potential_grid[j, i] + qd_potential_grid[j, i]
                n_grid[j, i] = n[nearest_node]
                p_grid[j, i] = p[nearest_node]
                Ex_grid[j, i] = E_field[nearest_node][0]
                Ey_grid[j, i] = E_field[nearest_node][1]

    # Create the figure for potentials
    fig1 = plt.figure(figsize=(15, 10))

    # Plot the P-N junction potential
    ax1 = fig1.add_subplot(231)
    im1 = ax1.imshow(pn_potential_grid, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    ax1.set_title('P-N Junction Potential (V)')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im1, ax=ax1)

    # Plot the quantum dot potential
    ax2 = fig1.add_subplot(232)
    im2 = ax2.imshow(qd_potential_grid, extent=[0, Lx, 0, Ly], origin='lower', cmap='plasma')
    ax2.set_title('Quantum Dot Potential (eV)')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(im2, ax=ax2)

    # Plot the combined potential
    ax3 = fig1.add_subplot(233)
    im3 = ax3.imshow(combined_potential_grid, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    ax3.set_title('Combined Potential (V)')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3)

    # Plot the combined potential in 3D
    ax4 = fig1.add_subplot(234, projection='3d')
    surf1 = ax4.plot_surface(X, Y, combined_potential_grid, cmap='viridis', linewidth=0, antialiased=False)
    ax4.set_title('Combined Potential (3D)')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')
    ax4.set_zlabel('V (V)')
    ax4.view_init(30, 45)

    # Plot the P-N junction potential in 3D
    ax5 = fig1.add_subplot(235, projection='3d')
    surf2 = ax5.plot_surface(X, Y, pn_potential_grid, cmap='viridis', linewidth=0, antialiased=False)
    ax5.set_title('P-N Junction Potential (3D)')
    ax5.set_xlabel('x (nm)')
    ax5.set_ylabel('y (nm)')
    ax5.set_zlabel('V (V)')
    ax5.view_init(30, 45)

    # Plot the quantum dot potential in 3D
    ax6 = fig1.add_subplot(236, projection='3d')
    surf3 = ax6.plot_surface(X, Y, qd_potential_grid, cmap='plasma', linewidth=0, antialiased=False)
    ax6.set_title('Quantum Dot Potential (3D)')
    ax6.set_xlabel('x (nm)')
    ax6.set_ylabel('y (nm)')
    ax6.set_zlabel('V (eV)')
    ax6.view_init(30, 45)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'qd_pn_potentials_bias_{bias_voltage:.1f}.png')
    plt.close()

    # Create the figure for carrier concentrations and electric field
    fig2 = plt.figure(figsize=(15, 10))

    # Plot the electron concentration
    ax1 = fig2.add_subplot(231)
    im1 = ax1.imshow(np.log10(n_grid), extent=[0, Lx, 0, Ly], origin='lower', cmap='plasma')
    ax1.set_title('Electron Concentration (log10(cm^-3))')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im1, ax=ax1)

    # Plot the hole concentration
    ax2 = fig2.add_subplot(232)
    im2 = ax2.imshow(np.log10(p_grid), extent=[0, Lx, 0, Ly], origin='lower', cmap='plasma')
    ax2.set_title('Hole Concentration (log10(cm^-3))')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(im2, ax=ax2)

    # Plot the electric field
    ax3 = fig2.add_subplot(233)
    E_mag = np.sqrt(Ex_grid**2 + Ey_grid**2)
    im3 = ax3.imshow(E_mag, extent=[0, Lx, 0, Ly], origin='lower', cmap='inferno')
    ax3.set_title('Electric Field Magnitude (V/cm)')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3)

    # Plot the electric field vectors
    ax4 = fig2.add_subplot(234)
    skip = 5
    ax4.quiver(X[::skip, ::skip], Y[::skip, ::skip], Ex_grid[::skip, ::skip], Ey_grid[::skip, ::skip],
              scale=50, width=0.002)
    ax4.set_title('Electric Field Vectors')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')

    # Plot the energy levels
    ax5 = fig2.add_subplot(235)
    ax5.bar(range(len(eigenvalues)), eigenvalues)
    ax5.set_title('Energy Levels')
    ax5.set_xlabel('State')
    ax5.set_ylabel('Energy (eV)')

    # Plot the ground state wavefunction
    if len(eigenvectors) > 0:
        wavefunction = eigenvectors[0]
        wavefunction_grid = np.zeros((ny, nx))

        # Interpolate the wavefunction onto the grid
        if has_gpu_interpolator:
            # Use GPU interpolator for faster interpolation
            interpolator = GPUInterpolator(mesh, use_gpu=True)
            wavefunction_grid = interpolator.interpolate_grid(0, Lx, 0, Ly, nx, ny, wavefunction)
        else:
            # Use simple nearest-neighbor interpolation
            for i in range(nx):
                for j in range(ny):
                    # Find the nearest node
                    min_dist = float('inf')
                    nearest_node = 0

                    for k in range(mesh.get_num_nodes()):
                        node = mesh.get_nodes()[k]
                        dist = (node[0] - x[i])**2 + (node[1] - y[j])**2

                        if dist < min_dist:
                            min_dist = dist
                            nearest_node = k

                    # Interpolate the wavefunction
                    wavefunction_grid[j, i] = wavefunction[nearest_node]

        # Plot the wavefunction
        ax6 = fig2.add_subplot(236)
        im6 = ax6.imshow(wavefunction_grid**2, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
        ax6.set_title('Ground State Probability Density')
        ax6.set_xlabel('x (nm)')
        ax6.set_ylabel('y (nm)')
        plt.colorbar(im6, ax=ax6)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'qd_pn_carriers_bias_{bias_voltage:.1f}.png')
    plt.close()

def main():
    """
    Main function.
    """
    # Run simulations for different bias voltages
    # Using only one bias voltage for the Python implementation for better performance
    bias_voltages = [0.0]

    for bias in bias_voltages:
        # Run the simulation with a smaller mesh size
        print(f"Starting simulation for bias voltage = {bias} V")
        print("Note: Using a smaller mesh size for the Python implementation for better performance.")
        print("This will take a few minutes to complete...")

        solver, mesh, schrodinger_solver, eigenvalues, eigenvectors, combined_potential, qd_potential = run_simulation(bias_voltage=bias, mesh_size=21)

        # Plot the results
        print("Plotting results...")
        plot_results(solver, mesh, schrodinger_solver, eigenvalues, eigenvectors, combined_potential, qd_potential, bias_voltage=bias)

        print(f"Results saved to qd_pn_potentials_bias_{bias:.1f}.png and qd_pn_carriers_bias_{bias:.1f}.png")

        # Print the energy levels
        print(f"Energy levels for bias voltage = {bias} V:")
        for i, energy in enumerate(eigenvalues):
            print(f"  State {i}: {energy:.6f} eV")

if __name__ == "__main__":
    main()
