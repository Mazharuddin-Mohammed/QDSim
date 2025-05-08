"""
Visualization Test for QDSim

This script demonstrates the visualization capabilities of QDSim by solving
the Schrödinger equation for a quantum dot in a P-N junction and visualizing
the results.

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
from frontend.qdsim import Mesh, SchrodingerSolver
from frontend.qdsim.adaptive_mesh import AdaptiveMesh

def run_simulation(mesh_size=51, order=1):
    """
    Run a simulation of the Schrödinger equation for a quantum dot in a P-N junction.
    
    Args:
        mesh_size: The size of the mesh (number of nodes in each dimension)
        order: The order of the finite elements (1 for P1, 2 for P2, 3 for P3)
        
    Returns:
        eigenvalues: The computed eigenvalues
        eigenvectors: The computed eigenvectors
        mesh: The mesh
        potentials: Dictionary containing the different potentials
    """
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0  # nm
    mesh = Mesh(Lx, Ly, mesh_size, mesh_size, element_order=order)
    
    # Define the effective mass function (GaAs)
    def m_star(x, y):
        return 0.067  # GaAs effective mass
    
    # Define the potential functions
    
    # P-N junction potential
    def V_pn(x, y):
        if x < Lx / 2.0:
            return -0.5  # P-region
        else:
            return 0.5   # N-region
    
    # Quantum dot potential (Gaussian well)
    def V_qd(x, y):
        x0 = Lx / 2.0
        y0 = Ly / 2.0
        r2 = (x - x0)**2 + (y - y0)**2
        return -5.0 * np.exp(-r2 / (2.0 * 10.0**2))
    
    # Combined potential
    def V(x, y):
        return V_pn(x, y) + V_qd(x, y)
    
    # Create the SchrodingerSolver
    print(f"Creating SchrodingerSolver with {mesh_size}x{mesh_size} mesh and order {order}...")
    solver = SchrodingerSolver(mesh, m_star, V, use_gpu=True)
    
    # Solve the Schrödinger equation
    print("Solving Schrödinger equation...")
    start_time = time.time()
    eigenvalues, eigenvectors = solver.solve(3)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Eigenvalues: {eigenvalues}")
    
    # Compute potentials on a grid
    nx, ny = 200, 100
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    V_pn_grid = np.zeros((ny, nx))
    V_qd_grid = np.zeros((ny, nx))
    V_total_grid = np.zeros((ny, nx))
    
    for i in range(nx):
        for j in range(ny):
            V_pn_grid[j, i] = V_pn(x[i], y[j])
            V_qd_grid[j, i] = V_qd(x[i], y[j])
            V_total_grid[j, i] = V(x[i], y[j])
    
    potentials = {
        'pn': V_pn_grid,
        'qd': V_qd_grid,
        'total': V_total_grid,
        'X': X,
        'Y': Y
    }
    
    return eigenvalues, eigenvectors, mesh, potentials

def plot_potentials(potentials, filename):
    """
    Plot the potentials.
    
    Args:
        potentials: Dictionary containing the different potentials
        filename: The filename to save the plot
    """
    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the P-N junction potential
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(potentials['pn'], extent=[0, potentials['X'][0, -1], 0, potentials['Y'][-1, 0]], 
                     origin='lower', cmap='RdBu_r')
    ax1.set_title('P-N Junction Potential')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot the quantum dot potential
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(potentials['qd'], extent=[0, potentials['X'][0, -1], 0, potentials['Y'][-1, 0]], 
                     origin='lower', cmap='viridis')
    ax2.set_title('Quantum Dot Potential')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(im2, ax=ax2)
    
    # Plot the combined potential
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(potentials['total'], extent=[0, potentials['X'][0, -1], 0, potentials['Y'][-1, 0]], 
                     origin='lower', cmap='viridis')
    ax3.set_title('Combined Potential')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3)
    
    # Plot the combined potential in 3D
    ax4 = fig.add_subplot(224, projection='3d')
    surf = ax4.plot_surface(potentials['X'], potentials['Y'], potentials['total'], 
                           cmap='viridis', linewidth=0, antialiased=False)
    ax4.set_title('Combined Potential (3D)')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')
    ax4.set_zlabel('V (meV)')
    ax4.view_init(30, 45)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename)
    plt.close()

def plot_wavefunctions(mesh, eigenvectors, eigenvalues, potentials, filename):
    """
    Plot the wavefunctions.
    
    Args:
        mesh: The mesh
        eigenvectors: The eigenvectors (wavefunctions)
        eigenvalues: The eigenvalues (energies)
        potentials: Dictionary containing the different potentials
        filename: The filename to save the plot
    """
    # Get the mesh parameters
    Lx = mesh.get_lx()
    Ly = mesh.get_ly()
    
    # Create a grid for plotting
    nx, ny = potentials['X'].shape[1], potentials['Y'].shape[0]
    
    # Plot the wavefunctions
    num_eigenvectors = len(eigenvectors)
    fig = plt.figure(figsize=(15, 15))
    
    for i in range(min(3, num_eigenvectors)):
        # Get the eigenvector
        eigenvector = eigenvectors[i]
        
        # Reshape the eigenvector to match the grid
        wavefunction = np.zeros((ny, nx))
        
        # Get the nodes
        nodes = mesh.get_nodes()
        
        # Map the eigenvector to the grid
        for j, node in enumerate(nodes):
            # Get the node coordinates
            x_pos = node[0]
            y_pos = node[1]
            
            # Find the closest grid point
            x_idx = int(np.round(x_pos / Lx * (nx - 1)))
            y_idx = int(np.round(y_pos / Ly * (ny - 1)))
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, nx - 1))
            y_idx = max(0, min(y_idx, ny - 1))
            
            # Set the wavefunction value
            if j < len(eigenvector):
                wavefunction[y_idx, x_idx] = np.abs(eigenvector[j])
        
        # Normalize the wavefunction
        wavefunction = wavefunction / np.sqrt(np.sum(wavefunction**2))
        
        # Plot the wavefunction
        ax1 = fig.add_subplot(3, 3, i + 1)
        im1 = ax1.imshow(wavefunction, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
        ax1.set_title(fr'$E_{i} = {eigenvalues[i]:.2f}$ meV')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')
        plt.colorbar(im1, ax=ax1)
        
        # Plot the wavefunction in 3D
        ax2 = fig.add_subplot(3, 3, i + 4, projection='3d')
        surf = ax2.plot_surface(potentials['X'], potentials['Y'], wavefunction, 
                               cmap='viridis', linewidth=0, antialiased=False)
        ax2.set_title(fr'$E_{i} = {eigenvalues[i]:.2f}$ meV')
        ax2.set_xlabel('x (nm)')
        ax2.set_ylabel('y (nm)')
        ax2.set_zlabel(r'$|\psi|^2$')
        ax2.view_init(30, 45)
        
        # Plot the wavefunction with potential contours
        ax3 = fig.add_subplot(3, 3, i + 7)
        im3 = ax3.imshow(wavefunction, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
        contour = ax3.contour(potentials['X'], potentials['Y'], potentials['total'], 
                             levels=10, colors='white', alpha=0.5)
        ax3.set_title(fr'$E_{i} = {eigenvalues[i]:.2f}$ meV with Potential')
        ax3.set_xlabel('x (nm)')
        ax3.set_ylabel('y (nm)')
        plt.colorbar(im3, ax=ax3)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename)
    plt.close()

def main():
    """
    Main function.
    """
    # Run simulations with different element orders
    print("Running simulations with different element orders...")
    
    # Test with linear elements (P1)
    print("\nTesting with linear elements (P1)...")
    p1_eigenvalues, p1_eigenvectors, p1_mesh, p1_potentials = run_simulation(mesh_size=51, order=1)
    
    # Plot results
    print("\nPlotting results...")
    plot_potentials(p1_potentials, 'potentials.png')
    plot_wavefunctions(p1_mesh, p1_eigenvectors, p1_eigenvalues, p1_potentials, 'wavefunctions_p1.png')
    
    print("\nDone!")

if __name__ == "__main__":
    main()
