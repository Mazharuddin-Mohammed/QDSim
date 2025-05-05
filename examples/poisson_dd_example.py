"""
Example script for the FullPoissonDriftDiffusionSolver.

This script demonstrates the use of the FullPoissonDriftDiffusionSolver
to simulate a P-N junction diode with a quantum dot.

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
from frontend.qdsim import Mesh, FullPoissonDriftDiffusionSolver

def run_simulation(bias_voltage=0.0, mesh_size=51):
    """
    Run a simulation of a P-N junction diode with a quantum dot.
    
    Args:
        bias_voltage: The bias voltage applied to the diode
        mesh_size: The size of the mesh (number of nodes in each dimension)
        
    Returns:
        solver: The FullPoissonDriftDiffusionSolver object
        mesh: The mesh
    """
    # Create a mesh for the P-N junction diode
    Lx = 200.0  # nm
    Ly = 100.0  # nm
    mesh = Mesh(Lx, Ly, mesh_size, mesh_size // 2, 1)  # Linear elements
    
    # Define the doping profile
    def doping_profile(x, y):
        # P-N junction at x = Lx/2
        if x < Lx / 2.0:
            return -1e17  # P-type (acceptors)
        else:
            return 1e17  # N-type (donors)
    
    # Define the relative permittivity
    def epsilon_r(x, y):
        return 12.9  # GaAs
    
    # Create the solver
    solver = FullPoissonDriftDiffusionSolver(mesh, epsilon_r, doping_profile)
    
    # Set the carrier statistics model
    solver.set_carrier_statistics_model(False)  # Use Boltzmann statistics
    
    # Solve the coupled Poisson-drift-diffusion equations
    print(f"Solving for bias voltage = {bias_voltage} V")
    start_time = time.time()
    solver.solve(0.0, bias_voltage, 1e-6, 100)
    end_time = time.time()
    print(f"Solution time: {end_time - start_time:.2f} seconds")
    
    return solver, mesh

def plot_results(solver, mesh, bias_voltage=0.0):
    """
    Plot the results of the simulation.
    
    Args:
        solver: The FullPoissonDriftDiffusionSolver object
        mesh: The mesh
        bias_voltage: The bias voltage applied to the diode
    """
    # Get the results
    potential = solver.get_potential()
    n = solver.get_electron_concentration()
    p = solver.get_hole_concentration()
    E_field = solver.get_electric_field_all()
    J_n = solver.get_electron_current_density()
    J_p = solver.get_hole_current_density()
    
    # Create a grid for plotting
    Lx = mesh.get_lx()
    Ly = mesh.get_ly()
    nx, ny = 200, 100
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # Interpolate the results onto the grid
    potential_grid = np.zeros((ny, nx))
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
            potential_grid[j, i] = potential[nearest_node]
            n_grid[j, i] = n[nearest_node]
            p_grid[j, i] = p[nearest_node]
            Ex_grid[j, i] = E_field[nearest_node][0]
            Ey_grid[j, i] = E_field[nearest_node][1]
    
    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the potential
    ax1 = fig.add_subplot(231)
    im1 = ax1.imshow(potential_grid, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    ax1.set_title('Electrostatic Potential (V)')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot the electron concentration
    ax2 = fig.add_subplot(232)
    im2 = ax2.imshow(np.log10(n_grid), extent=[0, Lx, 0, Ly], origin='lower', cmap='plasma')
    ax2.set_title('Electron Concentration (log10(cm^-3))')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(im2, ax=ax2)
    
    # Plot the hole concentration
    ax3 = fig.add_subplot(233)
    im3 = ax3.imshow(np.log10(p_grid), extent=[0, Lx, 0, Ly], origin='lower', cmap='plasma')
    ax3.set_title('Hole Concentration (log10(cm^-3))')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3)
    
    # Plot the electric field
    ax4 = fig.add_subplot(234)
    E_mag = np.sqrt(Ex_grid**2 + Ey_grid**2)
    im4 = ax4.imshow(E_mag, extent=[0, Lx, 0, Ly], origin='lower', cmap='inferno')
    ax4.set_title('Electric Field Magnitude (V/cm)')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')
    plt.colorbar(im4, ax=ax4)
    
    # Plot the electric field vectors
    ax5 = fig.add_subplot(235)
    skip = 5
    ax5.quiver(X[::skip, ::skip], Y[::skip, ::skip], Ex_grid[::skip, ::skip], Ey_grid[::skip, ::skip],
              scale=50, width=0.002)
    ax5.set_title('Electric Field Vectors')
    ax5.set_xlabel('x (nm)')
    ax5.set_ylabel('y (nm)')
    
    # Plot the potential in 3D
    ax6 = fig.add_subplot(236, projection='3d')
    surf = ax6.plot_surface(X, Y, potential_grid, cmap='viridis', linewidth=0, antialiased=False)
    ax6.set_title('Electrostatic Potential (3D)')
    ax6.set_xlabel('x (nm)')
    ax6.set_ylabel('y (nm)')
    ax6.set_zlabel('V (V)')
    ax6.view_init(30, 45)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'pn_junction_bias_{bias_voltage:.1f}.png')
    plt.close()

def main():
    """
    Main function.
    """
    # Run simulations for different bias voltages
    bias_voltages = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for bias in bias_voltages:
        # Run the simulation
        solver, mesh = run_simulation(bias_voltage=bias)
        
        # Plot the results
        plot_results(solver, mesh, bias_voltage=bias)
        
        print(f"Results saved to pn_junction_bias_{bias:.1f}.png")

if __name__ == "__main__":
    main()
