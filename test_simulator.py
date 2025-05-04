#!/usr/bin/env python3
"""
Test script for the QDSim simulator.

This script creates a simple quantum dot simulation and runs it to verify
that the C++ implementation is working correctly.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend'))

# Import the QDSim modules
from qdsim.config import Config
from qdsim.simulator import Simulator

def main():
    """Main function to test the simulator."""
    # Create a configuration
    config = Config()
    
    # Domain size
    config.Lx = 200e-9  # 200 nm
    config.Ly = 100e-9  # 100 nm
    
    # Mesh parameters
    config.nx = 101  # Number of elements in x direction
    config.ny = 51   # Number of elements in y direction
    config.element_order = 1  # Linear elements
    
    # Quantum dot parameters
    config.R = 10e-9  # 10 nm radius
    config.V_0 = 0.5 * 1.602e-19  # 0.5 eV depth
    config.potential_type = "gaussian"  # Gaussian potential
    
    # Diode parameters
    config.diode_p_material = "GaAs"
    config.diode_n_material = "GaAs"
    config.qd_material = "InAs"
    config.matrix_material = "GaAs"
    config.N_A = 1e24  # Acceptor concentration (m^-3)
    config.N_D = 1e24  # Donor concentration (m^-3)
    config.V_r = 0.0   # Reverse bias (V)
    
    # Physical constants
    config.e_charge = 1.602e-19  # Elementary charge (C)
    config.m_e = 9.109e-31  # Electron mass (kg)
    
    # Solver parameters
    config.tolerance = 1e-6
    config.max_iter = 100
    config.use_mpi = False
    
    # Create the simulator
    print("Creating simulator...")
    sim = Simulator(config)
    
    # Solve for the first 5 eigenvalues
    print("\nSolving for eigenvalues...")
    eigenvalues, eigenvectors = sim.solve(5)
    
    # Print the eigenvalues
    print("\nEigenvalues (eV):")
    for i, ev in enumerate(eigenvalues):
        print(f"  {i}: {ev/config.e_charge:.6f}")
    
    # Plot the potential and ground state wavefunction
    plot_results(sim, eigenvalues, eigenvectors)
    
def plot_results(sim, eigenvalues, eigenvectors):
    """Plot the potential and ground state wavefunction."""
    # Get the mesh nodes
    nodes = np.array(sim.mesh.get_nodes())
    
    # Create a grid for plotting
    x = np.unique(nodes[:, 0])
    y = np.unique(nodes[:, 1])
    X, Y = np.meshgrid(x, y)
    
    # Interpolate the potential and wavefunction onto the grid
    Z_pot = np.zeros_like(X)
    Z_wf = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Potential
            Z_pot[i, j] = sim.potential(X[i, j], Y[i, j])
            
            # Wavefunction (ground state)
            if eigenvectors is not None and eigenvectors.shape[1] > 0:
                try:
                    # Try to use the interpolator
                    Z_wf[i, j] = abs(sim.interpolator.interpolate(X[i, j], Y[i, j], eigenvectors[:, 0]))**2
                except Exception as e:
                    print(f"Warning: Failed to interpolate wavefunction: {e}")
                    Z_wf[i, j] = 0.0
    
    # Convert potential to eV
    Z_pot = Z_pot / sim.config.e_charge
    
    # Create the figure
    fig = plt.figure(figsize=(12, 6))
    
    # Plot the potential
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X*1e9, Y*1e9, Z_pot, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_zlabel('Potential (eV)')
    ax1.set_title('Potential')
    
    # Plot the wavefunction
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X*1e9, Y*1e9, Z_wf, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_zlabel('Probability density')
    ax2.set_title('Ground state wavefunction')
    
    # Adjust the view
    for ax in [ax1, ax2]:
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('qdsim_test_results.png', dpi=300)
    print("\nResults saved to qdsim_test_results.png")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
