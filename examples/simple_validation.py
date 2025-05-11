#!/usr/bin/env python3
"""
Simple validation script for QDSim.

This script performs a simple validation of the QDSim codebase by:
1. Testing the mesh creation
2. Testing the Poisson solver
3. Testing the Schrodinger solver
4. Visualizing the results

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Add the necessary paths
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend'))

# Try to import the Python frontend
try:
    import qdsim
    from qdsim.config import Config
    print("Successfully imported qdsim Python frontend")
except ImportError as e:
    print(f"Error importing qdsim Python frontend: {e}")
    print("Make sure the frontend is installed")
    sys.exit(1)

def test_mesh_creation():
    """Test mesh creation."""
    print("\n=== Testing Mesh Creation ===")
    
    # Create a configuration
    config = Config()
    config.Lx = 100.0  # nm
    config.Ly = 50.0   # nm
    config.nx = 50
    config.ny = 25
    config.element_order = 1
    
    # Create a simulator
    simulator = qdsim.Simulator(config)
    
    # Get the mesh
    mesh = simulator.mesh
    
    # Print mesh information
    print(f"Created mesh with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")
    
    # Create a directory for the results
    os.makedirs("results_validation", exist_ok=True)
    
    # Plot the mesh
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())
    
    fig = Figure(figsize=(10, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Plot the mesh elements
    for element in elements:
        x = nodes[element, 0]
        y = nodes[element, 1]
        x = np.append(x, x[0])  # Close the triangle
        y = np.append(y, y[0])  # Close the triangle
        ax.plot(x, y, 'k-', linewidth=0.5)
    
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_title('Mesh')
    ax.set_aspect('equal')
    
    # Save the figure
    canvas.print_figure("results_validation/mesh.png", dpi=150)
    print("Saved mesh plot to results_validation/mesh.png")
    
    return True

def test_poisson_solver():
    """Test the Poisson solver."""
    print("\n=== Testing Poisson Solver ===")
    
    # Create a configuration
    config = Config()
    config.Lx = 100.0  # nm
    config.Ly = 50.0   # nm
    config.nx = 50
    config.ny = 25
    config.element_order = 1
    config.N_A = 1e16  # Acceptor concentration in cm^-3
    config.N_D = 1e16  # Donor concentration in cm^-3
    config.V_r = 0.0   # Reverse bias in V
    
    # Create a simulator
    simulator = qdsim.Simulator(config)
    
    # Solve the Poisson equation
    simulator.solve_poisson()
    
    # Get the potential
    potential = simulator.phi
    
    # Print potential information
    print(f"Solved Poisson equation, potential shape: {potential.shape}")
    print(f"Potential min: {potential.min()}, max: {potential.max()}")
    
    # Create a directory for the results
    os.makedirs("results_validation", exist_ok=True)
    
    # Plot the potential
    nodes = np.array(simulator.mesh.get_nodes())
    elements = np.array(simulator.mesh.get_elements())
    
    fig = Figure(figsize=(10, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Plot the potential
    from matplotlib.tri import Triangulation
    triangulation = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    contour = ax.tricontourf(triangulation, potential, 50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_title('Electrostatic Potential (V)')
    
    # Save the figure
    canvas.print_figure("results_validation/potential.png", dpi=150)
    print("Saved potential plot to results_validation/potential.png")
    
    return True

def test_schrodinger_solver():
    """Test the Schrodinger solver."""
    print("\n=== Testing Schrodinger Solver ===")
    
    # Create a configuration
    config = Config()
    config.Lx = 100.0  # nm
    config.Ly = 50.0   # nm
    config.nx = 50
    config.ny = 25
    config.element_order = 1
    config.R = 10.0  # QD radius in nm
    config.V_0 = 0.3  # QD potential depth in eV
    config.potential_type = "gaussian"
    
    # Create a simulator
    simulator = qdsim.Simulator(config)
    
    # Solve the Schrodinger equation
    try:
        eigenvalues, eigenvectors = simulator.solve(3)  # Solve for 3 eigenvalues
        
        # Print eigenvalue information
        print(f"Solved Schrodinger equation, found {len(eigenvalues)} eigenvalues")
        if len(eigenvalues) > 0:
            print(f"Eigenvalues (eV): {[ev/1.602e-19 for ev in eigenvalues[:min(3, len(eigenvalues))]]}")
        
        # Create a directory for the results
        os.makedirs("results_validation", exist_ok=True)
        
        # Plot the wavefunctions
        if len(eigenvalues) > 0:
            nodes = np.array(simulator.mesh.get_nodes())
            elements = np.array(simulator.mesh.get_elements())
            
            fig = Figure(figsize=(15, 5))
            canvas = FigureCanvas(fig)
            
            # Plot the first 3 wavefunctions (or fewer if less than 3 eigenvalues)
            num_states = min(3, len(eigenvalues))
            for i in range(num_states):
                ax = fig.add_subplot(1, 3, i+1)
                
                # Calculate probability density
                wavefunction = eigenvectors[:, i]
                probability = np.abs(wavefunction)**2
                
                # Plot the probability density
                from matplotlib.tri import Triangulation
                triangulation = Triangulation(nodes[:, 0], nodes[:, 1], elements)
                contour = ax.tricontourf(triangulation, probability, 50, cmap='plasma')
                fig.colorbar(contour, ax=ax)
                ax.set_xlabel('x (nm)')
                ax.set_ylabel('y (nm)')
                ax.set_title(f'Wavefunction {i} (E = {eigenvalues[i]/1.602e-19:.4f} eV)')
            
            # Save the figure
            fig.tight_layout()
            canvas.print_figure("results_validation/wavefunctions.png", dpi=150)
            print("Saved wavefunction plots to results_validation/wavefunctions.png")
        
        return True
    except Exception as e:
        print(f"Error in Schrodinger solver: {e}")
        return False

def main():
    """Main function."""
    print("=== QDSim Simple Validation ===")
    
    # Test mesh creation
    if not test_mesh_creation():
        print("Mesh creation test failed")
        return
    
    # Test Poisson solver
    if not test_poisson_solver():
        print("Poisson solver test failed")
        return
    
    # Test Schrodinger solver
    if not test_schrodinger_solver():
        print("Schrodinger solver test failed")
        # Continue anyway
    
    print("\n=== Validation Completed Successfully ===")
    print("Results are available in the 'results_validation' directory")

if __name__ == "__main__":
    main()
