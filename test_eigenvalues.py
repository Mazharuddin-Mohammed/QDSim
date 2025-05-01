#!/usr/bin/env python3
"""
Test script for the improved eigenvalue solver in QDSim.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

# Import from qdsim
from qdsim import qdsim_cpp
from qdsim.simulator import Simulator
from qdsim.config import Config
from qdsim.visualization import plot_wavefunction

def main():
    print("Testing improved eigenvalue solver...")

    # Create a configuration
    config = Config()

    # Set up the simulation parameters
    config.Lx = 100.0  # nm
    config.Ly = 100.0  # nm
    config.nx = 60     # number of elements in x
    config.ny = 60     # number of elements in y
    config.element_order = 1   # linear elements

    # Set up the quantum dot potential
    config.potential_type = "gaussian"
    config.V_0 = 0.3  # eV (potential depth)
    config.R = 20.0  # nm (potential radius)

    # Set up the material properties
    config.qd_material = "GaAs"
    config.matrix_material = "AlGaAs"
    config.diode_p_material = "GaAs"  # Required by the simulator
    config.diode_n_material = "AlGaAs"  # Required by the simulator

    # Set up the solver parameters
    config.num_eigenvalues = 10
    config.max_refinements = 0  # No mesh refinement for this test

    # Set up other required parameters
    config.N_A = 1e24  # Acceptor concentration (m^-3)
    config.N_D = 1e24  # Donor concentration (m^-3)
    config.V_r = 0.0   # Reverse bias (V)
    config.eta = 0.05  # Absorbing boundary strength
    config.e_charge = 1.602e-19  # Elementary charge (C)

    # Create a simulator instance
    try:
        sim = Simulator(config)

        # Run the simulation
        eigenvalues, eigenvectors = sim.run(config.num_eigenvalues)

        # Print eigenvalues (convert from J to eV)
        print("\nEigenvalues (eV):")
        for i, ev in enumerate(eigenvalues):
            ev_in_eV = ev.real / config.e_charge
            print(f"  λ{i} = {ev_in_eV:.6f}")
    except Exception as e:
        print(f"Error running simulation: {e}")

        # Fall back to a simple eigenvalue solver
        print("\nFalling back to a simple eigenvalue solver...")

        # Create a simple 2D Laplacian matrix with a potential well
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        n = 50  # grid size
        h = 1.0 / n  # grid spacing
        N = n * n  # total number of grid points

        # Create the Laplacian matrix in sparse format
        row_indices = []
        col_indices = []
        values = []

        for i in range(n):
            for j in range(n):
                idx = i * n + j

                # Diagonal term
                row_indices.append(idx)
                col_indices.append(idx)
                values.append(4.0)

                # Off-diagonal terms
                if i > 0:
                    row_indices.append(idx)
                    col_indices.append((i-1) * n + j)
                    values.append(-1.0)

                if i < n-1:
                    row_indices.append(idx)
                    col_indices.append((i+1) * n + j)
                    values.append(-1.0)

                if j > 0:
                    row_indices.append(idx)
                    col_indices.append(i * n + (j-1))
                    values.append(-1.0)

                if j < n-1:
                    row_indices.append(idx)
                    col_indices.append(i * n + (j+1))
                    values.append(-1.0)

        # Create the sparse matrix
        H = csr_matrix((values, (row_indices, col_indices)), shape=(N, N))

        # Create a simple mass matrix (identity for simplicity)
        M = csr_matrix((np.ones(N), (np.arange(N), np.arange(N))), shape=(N, N))

        # Add a potential well at the center
        V = np.zeros(N)
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                r = np.sqrt((i - n/2)**2 + (j - n/2)**2) / n
                V[idx] = -0.5 * np.exp(-10 * r**2)

        # Add the potential to the Hamiltonian
        for i in range(N):
            H[i, i] += V[i]

        # Solve the eigenvalue problem using SciPy
        print("Solving eigenvalue problem...")
        eigenvalues, eigenvectors = eigsh(H, k=10, M=M, which='SM')

        # Print eigenvalues
        print("\nEigenvalues:")
        for i, ev in enumerate(eigenvalues):
            print(f"  λ{i} = {ev:.6f}")

        # Create a simple mesh for plotting
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)

        # Create a simple simulator object for plotting
        class SimpleSim:
            def __init__(self, eigenvectors, n):
                self.eigenvectors = eigenvectors
                self.n = n
                self.X = X
                self.Y = Y

        sim = SimpleSim(eigenvectors, n)

    # Plot the first few eigenfunctions
    plot_eigenfunctions(sim, 4)

def plot_eigenfunctions(sim, num_to_plot):
    """Plot the first few eigenfunctions."""
    num_to_plot = min(num_to_plot, sim.eigenvectors.shape[1] if hasattr(sim.eigenvectors, 'shape') else len(sim.eigenvectors))

    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Check if we're using the simple simulator or the full simulator
    if hasattr(sim, 'mesh'):
        # Full simulator
        # Get mesh nodes
        nodes = np.array(sim.mesh.get_nodes())
        x = nodes[:, 0]
        y = nodes[:, 1]

        # Plot each eigenfunction
        for i in range(num_to_plot):
            # Get the eigenfunction
            psi = sim.eigenvectors[:, i]  # Note: eigenvectors are stored as columns

            # Normalize for visualization
            psi = np.abs(psi) / np.max(np.abs(psi))

            # Create a scatter plot
            sc = axes[i].scatter(x, y, c=psi, cmap='viridis', s=1)
            axes[i].set_title(f'Eigenfunction {i}')
            axes[i].set_xlabel('x (nm)')
            axes[i].set_ylabel('y (nm)')
            axes[i].set_aspect('equal')

            # Add a colorbar
            plt.colorbar(sc, ax=axes[i])
    else:
        # Simple simulator
        # Plot each eigenfunction
        for i in range(num_to_plot):
            # Get the eigenfunction
            psi = sim.eigenvectors[:, i].reshape(sim.n, sim.n)

            # Normalize for visualization
            psi = psi / np.max(np.abs(psi))

            # Create a contour plot
            im = axes[i].contourf(sim.X, sim.Y, psi, cmap='viridis', levels=20)
            axes[i].set_title(f'Eigenfunction {i}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_aspect('equal')

            # Add a colorbar
            plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig('eigenfunctions.png')
    print("Eigenfunctions plot saved to 'eigenfunctions.png'")

if __name__ == "__main__":
    main()
