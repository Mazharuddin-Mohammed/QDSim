#!/usr/bin/env python3
"""
Test script for the improved eigenvalue solver in QDSim.
This script directly tests the C++ eigenvalue solver without using the Simulator class.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

# Import from qdsim
from qdsim import qdsim_cpp

def main():
    print("Testing improved eigenvalue solver directly...")

    # Create a mesh
    lx, ly = 100.0, 100.0  # nm
    nx, ny = 60, 60        # number of elements
    order = 1              # linear elements
    mesh = qdsim_cpp.Mesh(lx, ly, nx, ny, order)
    print(f"Mesh created with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")

    # Define potential function
    def quantum_dot_potential(x, y):
        # Quantum dot potential (deeper for better bound states)
        R = 20.0  # nm
        depth = 0.3  # eV
        r = np.sqrt((x - lx/2)**2 + (y - ly/2)**2)
        return -depth * np.exp(-r**2 / (2 * R**2))

    # Define effective mass function
    def effective_mass(x, y):
        # GaAs effective mass
        return 0.067

    # Define cap function (for boundary conditions)
    def cap(x, y):
        # Absorbing boundary conditions
        width = 10.0  # nm
        strength = 0.05  # eV

        # Distance from boundaries
        dx = min(x, lx - x)
        dy = min(y, ly - y)

        # Apply cap only near boundaries
        if dx < width or dy < width:
            d = min(dx, dy)
            return strength * (1.0 - d/width)**2
        else:
            return 0.0

    # Create a PoissonSolver first (required by FEMSolver)
    # We need to provide function pointers for the charge density and permittivity
    # For now, we'll use zero functions
    def zero_func(x, y):
        return 0.0

    poisson = qdsim_cpp.PoissonSolver(mesh, zero_func, zero_func)

    # Create FEM solver
    # We need to provide function pointers for m_star, V, and cap
    fem = qdsim_cpp.FEMSolver(mesh, effective_mass, quantum_dot_potential, cap, poisson, order)

    # Manually set the potential at each node
    nodes = np.array(mesh.get_nodes())
    potential = np.zeros(mesh.get_num_nodes())
    for i in range(mesh.get_num_nodes()):
        x, y = nodes[i]
        potential[i] = quantum_dot_potential(x, y)

    # Set the potential in the FEM solver
    fem.set_potential(potential)

    # Assemble the matrices
    fem.assemble_matrices()

    # Create eigenvalue solver
    eigen = qdsim_cpp.EigenSolver(fem)

    # Solve for eigenvalues
    num_eigenvalues = 10
    print(f"Solving for {num_eigenvalues} eigenvalues...")
    eigen.solve(num_eigenvalues)

    # Get eigenvalues and eigenvectors
    eigenvalues = eigen.get_eigenvalues()
    eigenvectors = eigen.get_eigenvectors()

    # Print eigenvalues
    print("\nEigenvalues (eV):")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ{i} = {ev.real:.6f}")

    # Plot the first few eigenfunctions
    plot_eigenfunctions(mesh, eigenvectors, 4)

def plot_eigenfunctions(mesh, eigenvectors, num_to_plot):
    """Plot the first few eigenfunctions."""
    num_to_plot = min(num_to_plot, len(eigenvectors))

    # Get mesh nodes
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Plot each eigenfunction
    for i in range(num_to_plot):
        # Get the eigenfunction
        psi = eigenvectors[i]

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

    plt.tight_layout()
    plt.savefig('eigenfunctions_direct.png')
    print("Eigenfunctions plot saved to 'eigenfunctions_direct.png'")

if __name__ == "__main__":
    main()
