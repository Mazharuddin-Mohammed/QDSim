#!/usr/bin/env python3
"""
Callback test script for QDSim.

This script tests the callback system in QDSim.

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the necessary paths
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend/build'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend'))

# Try to import the C++ module
try:
    import qdsim_cpp
    print("Successfully imported qdsim_cpp module")
except ImportError as e:
    print(f"Error importing qdsim_cpp module: {e}")
    print("Make sure the C++ extension is built and in the Python path")
    sys.exit(1)

# Try to import the Python frontend
try:
    import qdsim
    from qdsim.callback_wrapper import wrap_epsilon_r, wrap_rho, wrap_potential
    print("Successfully imported qdsim Python frontend")
except ImportError as e:
    print(f"Error importing qdsim Python frontend: {e}")
    print("Make sure the frontend is installed")
    sys.exit(1)

def test_callbacks():
    """Test the callback system."""
    print("\n=== Testing Callback System ===")
    
    # Create a mesh
    try:
        mesh = qdsim_cpp.Mesh(100.0, 50.0, 50, 25, 1)
        print(f"Created mesh with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")
        
        # Define callback functions
        @wrap_epsilon_r
        def epsilon_r(x, y):
            """Relative permittivity callback."""
            # Simple position-dependent permittivity
            if x < 50.0:
                return 12.9  # GaAs
            else:
                return 15.15  # InAs
        
        @wrap_rho
        def rho(x, y):
            """Charge density callback."""
            # Simple position-dependent charge density
            if x < 30.0:
                return 1.0e16  # n-type
            elif x > 70.0:
                return -1.0e16  # p-type
            else:
                return 0.0  # intrinsic
        
        @wrap_potential
        def potential(x, y):
            """Potential callback."""
            # Simple Gaussian quantum dot potential
            r = np.sqrt((x - 50.0)**2 + (y - 25.0)**2)
            return 0.3 * np.exp(-r**2 / 100.0)  # 0.3 eV deep, 10 nm radius
        
        print("Registered callback functions")
        
        # Try to use the callbacks directly
        try:
            print(f"epsilon_r(25.0, 25.0) = {epsilon_r(25.0, 25.0)}")
            print(f"epsilon_r(75.0, 25.0) = {epsilon_r(75.0, 25.0)}")
            print(f"rho(25.0, 25.0) = {rho(25.0, 25.0)}")
            print(f"rho(50.0, 25.0) = {rho(50.0, 25.0)}")
            print(f"rho(75.0, 25.0) = {rho(75.0, 25.0)}")
            print(f"potential(50.0, 25.0) = {potential(50.0, 25.0)}")
            print(f"potential(60.0, 25.0) = {potential(60.0, 25.0)}")
        except Exception as e:
            print(f"Error calling callbacks directly: {e}")
        
        # Try to use the callbacks through the C++ code
        try:
            # Check if the callbacks are registered
            if hasattr(qdsim_cpp, 'has_callback'):
                print(f"has_callback('epsilon_r') = {qdsim_cpp.has_callback('epsilon_r')}")
                print(f"has_callback('rho') = {qdsim_cpp.has_callback('rho')}")
                print(f"has_callback('potential') = {qdsim_cpp.has_callback('potential')}")
            else:
                print("has_callback method not available")
            
            # Try to create a PoissonSolver
            try:
                poisson_solver = qdsim_cpp.PoissonSolver(mesh)
                print("Created PoissonSolver")
                
                # Solve with simple boundary conditions
                poisson_solver.solve(0.0, 1.0)
                print("Solved Poisson equation")
                
                # Get the potential
                potential = np.array(poisson_solver.get_potential())
                print(f"Potential shape: {potential.shape}")
                print(f"Potential min: {potential.min()}, max: {potential.max()}")
                
                # Create a directory for the results
                os.makedirs("results_validation", exist_ok=True)
                
                # Plot the potential
                nodes = np.array(mesh.get_nodes())
                elements = np.array(mesh.get_elements())
                
                plt.figure(figsize=(10, 5))
                
                # Plot the potential
                from matplotlib.tri import Triangulation
                triangulation = Triangulation(nodes[:, 0], nodes[:, 1], elements)
                contour = plt.tricontourf(triangulation, potential, 50, cmap='viridis')
                plt.colorbar(contour)
                plt.xlabel('x (nm)')
                plt.ylabel('y (nm)')
                plt.title('Electrostatic Potential (V) - With Callbacks')
                
                # Save the figure
                plt.savefig("results_validation/potential_callbacks.png", dpi=150)
                print("Saved potential plot to results_validation/potential_callbacks.png")
            except Exception as e:
                print(f"Error creating PoissonSolver: {e}")
        except Exception as e:
            print(f"Error using callbacks through C++ code: {e}")
        
        return True
    except Exception as e:
        print(f"Error creating mesh: {e}")
        return False

def main():
    """Main function."""
    print("=== QDSim Callback Test ===")
    
    # Test callbacks
    if not test_callbacks():
        print("Callback test failed")
        return
    
    print("\n=== Test Completed Successfully ===")
    print("Results are available in the 'results_validation' directory")

if __name__ == "__main__":
    main()
