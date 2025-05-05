"""
GPU Acceleration Test for QDSim

This script tests the GPU acceleration capabilities of QDSim by solving the
Schrödinger equation for a quantum dot in a harmonic oscillator potential.

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

def run_test(use_gpu=False, mesh_size=101, order=1):
    """
    Run a test of the Schrödinger solver with or without GPU acceleration.

    Args:
        use_gpu: Whether to use GPU acceleration
        mesh_size: The size of the mesh (number of nodes in each dimension)
        order: The order of the finite elements (1 for P1, 2 for P2, 3 for P3)

    Returns:
        eigenvalues: The computed eigenvalues
        eigenvectors: The computed eigenvectors
        elapsed_time: The time taken to solve the Schrödinger equation
    """
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 100.0  # nm
    mesh = Mesh(Lx, Ly, mesh_size, mesh_size, element_order=order)

    # Define the effective mass function (constant for simplicity)
    def m_star(x, y):
        return 0.067  # GaAs effective mass

    # Define the potential function (harmonic oscillator)
    def V(x, y):
        x0 = Lx / 2.0
        y0 = Ly / 2.0
        r2 = (x - x0)**2 + (y - y0)**2
        return 0.1 * r2  # meV

    # Create the SchrodingerSolver
    print(f"Creating SchrodingerSolver with {'GPU' if use_gpu else 'CPU'} acceleration...")
    solver = SchrodingerSolver(mesh, m_star, V, use_gpu=use_gpu)

    # Solve the Schrödinger equation
    print("Solving Schrödinger equation...")
    start_time = time.time()
    eigenvalues, eigenvectors = solver.solve(10)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Eigenvalues: {eigenvalues}")

    return eigenvalues, eigenvectors, elapsed_time

def plot_wavefunctions(mesh, eigenvectors, eigenvalues):
    """
    Plot the wavefunctions.

    Args:
        mesh: The mesh
        eigenvectors: The eigenvectors (wavefunctions)
        eigenvalues: The eigenvalues (energies)
    """
    # Get the mesh parameters
    Lx = mesh.get_lx()
    Ly = mesh.get_ly()
    nx = mesh.get_nx()
    ny = mesh.get_ny()

    # Create the grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Plot the wavefunctions
    num_eigenvectors = len(eigenvectors)
    fig = plt.figure(figsize=(15, 5))

    for i in range(min(3, num_eigenvectors)):
        # Get the eigenvector
        eigenvector = eigenvectors[i]

        # Reshape the eigenvector to match the grid
        wavefunction = np.zeros((ny, nx))

        # Get the nodes
        nodes = mesh.get_nodes()

        # Map the eigenvector to the grid
        # Check if we have enough values in the eigenvector
        if len(eigenvector) < len(nodes):
            print(f"Warning: Eigenvector length ({len(eigenvector)}) is less than number of nodes ({len(nodes)})")
            # Use only the available nodes
            max_nodes = min(len(eigenvector), len(nodes))
        else:
            max_nodes = len(nodes)

        for j in range(max_nodes):
            # Get the node coordinates
            node = nodes[j]
            x_pos = node[0]
            y_pos = node[1]

            # Find the closest grid point
            x_idx = int(np.round(x_pos / Lx * (nx - 1)))
            y_idx = int(np.round(y_pos / Ly * (ny - 1)))

            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, nx - 1))
            y_idx = max(0, min(y_idx, ny - 1))

            # Set the wavefunction value
            wavefunction[y_idx, x_idx] = np.abs(eigenvector[j])

        # Normalize the wavefunction
        wavefunction = wavefunction / np.sqrt(np.sum(wavefunction**2))

        # Plot the wavefunction
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        surf = ax.plot_surface(X, Y, wavefunction, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel(r'$|\psi|^2$')
        ax.set_title(fr'$E_{i} = {eigenvalues[i]:.2f}$ meV')

    plt.tight_layout()
    plt.savefig('wavefunctions.png')
    plt.close()

def main():
    """
    Main function.
    """
    # Run tests with different mesh sizes
    print("Running tests with different mesh sizes...")

    # Test with linear elements (P1) - small mesh
    print("\nTesting with linear elements (P1) - small mesh...")
    cpu_eigenvalues_small, cpu_eigenvectors_small, cpu_time_small = run_test(use_gpu=False, mesh_size=51, order=1)
    gpu_eigenvalues_small, gpu_eigenvectors_small, gpu_time_small = run_test(use_gpu=True, mesh_size=51, order=1)

    # Test with linear elements (P1) - large mesh
    print("\nTesting with linear elements (P1) - large mesh...")
    cpu_eigenvalues_large, cpu_eigenvectors_large, cpu_time_large = run_test(use_gpu=False, mesh_size=101, order=1)
    gpu_eigenvalues_large, gpu_eigenvectors_large, gpu_time_large = run_test(use_gpu=True, mesh_size=101, order=1)

    # Print results
    print("\nResults:")
    print(f"Small mesh (51x51):")
    print(f"  CPU time: {cpu_time_small:.2f} seconds")
    print(f"  GPU time: {gpu_time_small:.2f} seconds")
    print(f"  Speedup: {cpu_time_small / gpu_time_small:.2f}x")

    print(f"\nLarge mesh (101x101):")
    print(f"  CPU time: {cpu_time_large:.2f} seconds")
    print(f"  GPU time: {gpu_time_large:.2f} seconds")
    print(f"  Speedup: {cpu_time_large / gpu_time_large:.2f}x")

    # Plot wavefunctions
    print("\nPlotting wavefunctions...")
    mesh = Mesh(100.0, 100.0, 101, 101, element_order=1)
    plot_wavefunctions(mesh, gpu_eigenvectors_large, gpu_eigenvalues_large)

    print("\nDone!")

if __name__ == "__main__":
    main()
