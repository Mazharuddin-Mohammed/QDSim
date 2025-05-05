"""
Adaptive Mesh Refinement Test for QDSim

This script demonstrates the adaptive mesh refinement capabilities of QDSim
by solving the Schrödinger equation for a quantum dot in a P-N junction.

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

def run_test(use_adaptive_mesh=False, mesh_size=21, order=1):
    """
    Run a test of the Schrödinger solver with or without adaptive mesh refinement.

    Args:
        use_adaptive_mesh: Whether to use adaptive mesh refinement
        mesh_size: The initial size of the mesh (number of nodes in each dimension)
        order: The order of the finite elements (1 for P1, 2 for P2, 3 for P3)

    Returns:
        eigenvalues: The computed eigenvalues
        eigenvectors: The computed eigenvectors
        mesh: The final mesh
        elapsed_time: The time taken to solve the Schrödinger equation
    """
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0  # nm
    mesh = Mesh(Lx, Ly, mesh_size, mesh_size, element_order=order)

    # Define the effective mass function (GaAs)
    def m_star(x, y):
        return 0.067  # GaAs effective mass

    # Define the potential function (P-N junction with quantum dot)
    def V(x, y):
        # P-N junction potential
        V_pn = 0.0
        if x < Lx / 2.0:
            V_pn = -0.5  # P-region
        else:
            V_pn = 0.5   # N-region

        # Quantum dot potential (Gaussian well)
        x0 = Lx / 2.0
        y0 = Ly / 2.0
        r2 = (x - x0)**2 + (y - y0)**2
        V_qd = -5.0 * np.exp(-r2 / (2.0 * 10.0**2))

        return V_pn + V_qd  # meV

    # Create the SchrodingerSolver
    print(f"Creating SchrodingerSolver with {'adaptive' if use_adaptive_mesh else 'uniform'} mesh...")
    solver = SchrodingerSolver(mesh, m_star, V, use_gpu=True)

    # Solve the Schrödinger equation
    print("Solving Schrödinger equation...")
    start_time = time.time()

    if use_adaptive_mesh:
        # First solve with initial mesh
        eigenvalues, eigenvectors = solver.solve(3)

        # Compute refinement flags based on the first eigenvector
        print("Computing refinement flags...")
        psi = np.abs(eigenvectors[0])
        # Convert to Eigen::VectorXd
        psi_eigen = np.array(psi, dtype=np.float64)
        refine_flags = AdaptiveMesh.compute_refinement_flags(mesh, psi_eigen, 0.1)

        # Refine the mesh
        print("Refining mesh...")
        refined_mesh = AdaptiveMesh.refine_mesh(mesh, refine_flags)

        # Update the mesh and solver
        mesh = refined_mesh
        solver = SchrodingerSolver(mesh, m_star, V, use_gpu=True)

        # Solve again with the refined mesh
        print("Solving with refined mesh...")
        eigenvalues, eigenvectors = solver.solve(3)
    else:
        eigenvalues, eigenvectors = solver.solve(3)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Eigenvalues: {eigenvalues}")

    return eigenvalues, eigenvectors, mesh, elapsed_time

def plot_mesh(mesh, filename):
    """
    Plot the mesh.

    Args:
        mesh: The mesh
        filename: The filename to save the plot
    """
    # Get the mesh parameters
    Lx = mesh.get_lx()
    Ly = mesh.get_ly()

    # Get the nodes and elements
    nodes = mesh.get_nodes()
    elements = mesh.get_elements()

    # Create the figure
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # Plot the elements
    for elem in elements:
        x = [nodes[elem[0]][0], nodes[elem[1]][0], nodes[elem[2]][0], nodes[elem[0]][0]]
        y = [nodes[elem[0]][1], nodes[elem[1]][1], nodes[elem[2]][1], nodes[elem[0]][1]]
        ax.plot(x, y, 'k-', linewidth=0.5)

    # Set the axis limits
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    # Set the axis labels
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')

    # Set the title
    ax.set_title(f'Mesh with {len(nodes)} nodes and {len(elements)} elements')

    # Save the figure
    plt.savefig(filename)
    plt.close()

def plot_potential(mesh, filename):
    """
    Plot the potential.

    Args:
        mesh: The mesh
        filename: The filename to save the plot
    """
    # Get the mesh parameters
    Lx = mesh.get_lx()
    Ly = mesh.get_ly()

    # Define the potential function (P-N junction with quantum dot)
    def V(x, y):
        # P-N junction potential
        V_pn = 0.0
        if x < Lx / 2.0:
            V_pn = -0.5  # P-region
        else:
            V_pn = 0.5   # N-region

        # Quantum dot potential (Gaussian well)
        x0 = Lx / 2.0
        y0 = Ly / 2.0
        r2 = (x - x0)**2 + (y - y0)**2
        V_qd = -5.0 * np.exp(-r2 / (2.0 * 10.0**2))

        return V_pn + V_qd  # meV

    # Create a grid for plotting
    x = np.linspace(0, Lx, 200)
    y = np.linspace(0, Ly, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Compute the potential at each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = V(X[i, j], Y[i, j])

    # Create the figure
    fig = plt.figure(figsize=(15, 10))

    # Plot the P-N junction potential
    ax1 = fig.add_subplot(221)
    V_pn = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] < Lx / 2.0:
                V_pn[i, j] = -0.5  # P-region
            else:
                V_pn[i, j] = 0.5   # N-region
    im1 = ax1.imshow(V_pn, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    ax1.set_title('P-N Junction Potential')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im1, ax=ax1)

    # Plot the quantum dot potential
    ax2 = fig.add_subplot(222)
    V_qd = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x0 = Lx / 2.0
            y0 = Ly / 2.0
            r2 = (X[i, j] - x0)**2 + (Y[i, j] - y0)**2
            V_qd[i, j] = -5.0 * np.exp(-r2 / (2.0 * 10.0**2))
    im2 = ax2.imshow(V_qd, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    ax2.set_title('Quantum Dot Potential')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(im2, ax=ax2)

    # Plot the combined potential
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(Z, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    ax3.set_title('Combined Potential')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3)

    # Plot the combined potential in 3D
    ax4 = fig.add_subplot(224, projection='3d')
    surf = ax4.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
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

def plot_wavefunctions(mesh, eigenvectors, eigenvalues, filename):
    """
    Plot the wavefunctions.

    Args:
        mesh: The mesh
        eigenvectors: The eigenvectors (wavefunctions)
        eigenvalues: The eigenvalues (energies)
        filename: The filename to save the plot
    """
    # Get the mesh parameters
    Lx = mesh.get_lx()
    Ly = mesh.get_ly()

    # Create a grid for plotting
    nx, ny = 200, 100
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Plot the wavefunctions
    num_eigenvectors = len(eigenvectors)
    fig = plt.figure(figsize=(15, 10))

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
        ax1 = fig.add_subplot(2, 4, i + 1)
        im1 = ax1.imshow(wavefunction, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
        ax1.set_title(fr'$E_{i} = {eigenvalues[i]:.2f}$ meV')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')
        plt.colorbar(im1, ax=ax1)

        # Plot the wavefunction in 3D
        ax2 = fig.add_subplot(2, 4, i + 5, projection='3d')
        surf = ax2.plot_surface(X, Y, wavefunction, cmap='viridis', linewidth=0, antialiased=False)
        ax2.set_title(fr'$E_{i} = {eigenvalues[i]:.2f}$ meV')
        ax2.set_xlabel('x (nm)')
        ax2.set_ylabel('y (nm)')
        ax2.set_zlabel(r'$|\psi|^2$')
        ax2.view_init(30, 45)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename)
    plt.close()

def main():
    """
    Main function.
    """
    # Run tests with and without adaptive mesh refinement
    print("Running tests with and without adaptive mesh refinement...")

    # Test with uniform mesh
    print("\nTesting with uniform mesh...")
    uniform_eigenvalues, uniform_eigenvectors, uniform_mesh, uniform_time = run_test(use_adaptive_mesh=False, mesh_size=51, order=1)

    # Test with adaptive mesh refinement
    print("\nTesting with adaptive mesh refinement...")
    adaptive_eigenvalues, adaptive_eigenvectors, adaptive_mesh, adaptive_time = run_test(use_adaptive_mesh=True, mesh_size=51, order=1)

    # Print results
    print("\nResults:")
    print(f"Uniform mesh:")
    print(f"  Number of nodes: {len(uniform_mesh.get_nodes())}")
    print(f"  Number of elements: {len(uniform_mesh.get_elements())}")
    print(f"  Time: {uniform_time:.2f} seconds")
    print(f"  Eigenvalues: {uniform_eigenvalues}")

    print(f"\nAdaptive mesh:")
    print(f"  Number of nodes: {len(adaptive_mesh.get_nodes())}")
    print(f"  Number of elements: {len(adaptive_mesh.get_elements())}")
    print(f"  Time: {adaptive_time:.2f} seconds")
    print(f"  Eigenvalues: {adaptive_eigenvalues}")

    # Plot meshes
    print("\nPlotting meshes...")
    plot_mesh(uniform_mesh, 'uniform_mesh.png')
    plot_mesh(adaptive_mesh, 'adaptive_mesh.png')

    # Plot potential
    print("\nPlotting potential...")
    plot_potential(uniform_mesh, 'potential.png')

    # Plot wavefunctions
    print("\nPlotting wavefunctions...")
    plot_wavefunctions(uniform_mesh, uniform_eigenvectors, uniform_eigenvalues, 'uniform_wavefunctions.png')
    plot_wavefunctions(adaptive_mesh, adaptive_eigenvectors, adaptive_eigenvalues, 'adaptive_wavefunctions.png')

    print("\nDone!")

if __name__ == "__main__":
    main()
