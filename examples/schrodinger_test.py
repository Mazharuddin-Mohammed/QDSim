#!/usr/bin/env python3
"""
Schrodinger solver test script for QDSim.

This script tests the SchrodingerSolver class in QDSim.

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

def test_schrodinger_solver():
    """Test the SchrodingerSolver class."""
    print("\n=== Testing SchrodingerSolver ===")
    
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
    try:
        simulator = qdsim.Simulator(config)
        print("Created simulator")
        
        # Solve the Schrodinger equation
        try:
            eigenvalues, eigenvectors = simulator.solve(3)  # Solve for 3 eigenvalues
            print("Solved Schrodinger equation")
            
            # Print eigenvalue information
            print(f"Found {len(eigenvalues)} eigenvalues")
            if len(eigenvalues) > 0:
                print(f"Eigenvalues (eV): {[ev/1.602e-19 for ev in eigenvalues[:min(3, len(eigenvalues))]]}")
            
            # Create a directory for the results
            os.makedirs("results_validation", exist_ok=True)
            
            # Plot the wavefunctions
            if len(eigenvalues) > 0:
                # Get the mesh
                mesh = simulator.mesh
                nodes = np.array(mesh.get_nodes())
                elements = np.array(mesh.get_elements())
                
                # Plot the first 3 wavefunctions (or fewer if less than 3 eigenvalues)
                num_states = min(3, len(eigenvalues))
                for i in range(num_states):
                    plt.figure(figsize=(10, 5))
                    
                    # Calculate probability density
                    wavefunction = eigenvectors[:, i]
                    probability = np.abs(wavefunction)**2
                    
                    # Plot the probability density
                    from matplotlib.tri import Triangulation
                    triangulation = Triangulation(nodes[:, 0], nodes[:, 1], elements)
                    contour = plt.tricontourf(triangulation, probability, 50, cmap='plasma')
                    plt.colorbar(contour)
                    plt.xlabel('x (nm)')
                    plt.ylabel('y (nm)')
                    plt.title(f'Wavefunction {i} (E = {eigenvalues[i]/1.602e-19:.4f} eV)')
                    
                    # Save the figure
                    plt.savefig(f"results_validation/wavefunction_{i}.png", dpi=150)
                    print(f"Saved wavefunction {i} plot to results_validation/wavefunction_{i}.png")
            
            return True
        except Exception as e:
            print(f"Error solving Schrodinger equation: {e}")
            return False
    except Exception as e:
        print(f"Error creating simulator: {e}")
        return False

def main():
    """Main function."""
    print("=== QDSim Schrodinger Test ===")
    
    # Test SchrodingerSolver
    if not test_schrodinger_solver():
        print("SchrodingerSolver test failed")
        return
    
    print("\n=== Test Completed Successfully ===")
    print("Results are available in the 'results_validation' directory")

if __name__ == "__main__":
    main()
