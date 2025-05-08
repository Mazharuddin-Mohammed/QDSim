#!/usr/bin/env python3
"""
Example script demonstrating the use of the SchrodingerSolver class.

This script creates a simple quantum dot potential and solves the Schrödinger
equation to find the energy levels and wavefunctions.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import QDSim
from frontend.qdsim import Mesh, SchrodingerSolver

def main():
    # Define the mesh parameters
    Lx = 100.0  # Width of the domain in nm
    Ly = 100.0  # Height of the domain in nm
    nx = 101    # Number of points in the x-direction
    ny = 101    # Number of points in the y-direction
    element_order = 1  # Linear elements

    # Create the mesh
    mesh = Mesh(Lx, Ly, nx, ny, element_order)

    # Define the effective mass function (GaAs)
    def m_star(x, y):
        return 0.067  # Effective mass of electrons in GaAs (relative to free electron mass)

    # Define the potential function (harmonic oscillator)
    def V(x, y):
        # Center of the domain
        x0 = Lx / 2.0
        y0 = Ly / 2.0

        # Harmonic oscillator potential
        k = 0.001  # Spring constant in eV/nm²
        return 0.5 * k * ((x - x0)**2 + (y - y0)**2)

    # Create the SchrodingerSolver
    solver = SchrodingerSolver(mesh, m_star, V, use_gpu=False)

    # Solve the Schrödinger equation
    num_eigenvalues = 5
    eigenvalues, eigenvectors = solver.solve(num_eigenvalues)

    # Print the eigenvalues
    print("Eigenvalues (energy levels) in eV:")
    for i, eigenvalue in enumerate(eigenvalues):
        print(f"E_{i} = {eigenvalue:.6f} eV")

    # Plot the wavefunctions
    plot_wavefunctions(mesh, eigenvectors)

    # Plot the potential
    plot_potential(mesh, V)

    plt.show()

def plot_wavefunctions(mesh, eigenvectors):
    """
    Plot the wavefunctions.

    Args:
        mesh: The mesh
        eigenvectors: The eigenvectors (wavefunctions)
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
            wavefunction[y_idx, x_idx] = np.abs(eigenvector[j])

        # Normalize the wavefunction
        wavefunction = wavefunction / np.sqrt(np.sum(wavefunction**2))

        # Plot the wavefunction
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        surf = ax.plot_surface(X, Y, wavefunction, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel(r'$|\psi|^2$')
        ax.set_title(fr'Wavefunction $\psi_{i}$')

    plt.tight_layout()

def plot_potential(mesh, V):
    """
    Plot the potential.

    Args:
        mesh: The mesh
        V: The potential function
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

    # Evaluate the potential on the grid
    potential = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            potential[j, i] = V(x[i], y[j])

    # Plot the potential
    fig = plt.figure(figsize=(10, 5))

    # 3D plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, potential, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_zlabel('V (eV)')
    ax1.set_title('Potential')

    # 2D plot
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(potential, extent=[0, Lx, 0, Ly], origin='lower', cmap=cm.viridis)
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_title('Potential')
    plt.colorbar(im, ax=ax2, label='V (eV)')

    plt.tight_layout()

if __name__ == "__main__":
    main()
