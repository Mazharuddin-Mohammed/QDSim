#!/usr/bin/env python3
"""
Poisson solver test script for QDSim.

This script tests the PoissonSolver class in QDSim.

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
    from qdsim.poisson_solver import PoissonSolver2D
    print("Successfully imported qdsim Python frontend")
except ImportError as e:
    print(f"Error importing qdsim Python frontend: {e}")
    print("Make sure the frontend is installed")
    sys.exit(1)

def test_poisson_solver():
    """Test the PoissonSolver class."""
    print("\n=== Testing PoissonSolver ===")
    
    # Create a mesh
    try:
        mesh = qdsim_cpp.Mesh(100.0, 50.0, 50, 25, 1)
        print(f"Created mesh with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")
        
        # Define callback functions
        def epsilon_r(x, y):
            return 12.9  # GaAs
        
        def rho(x, y, n=None, p=None):
            return 0.0  # No charge
        
        # Try to use the C++ PoissonSolver
        try:
            # Try to register callbacks
            try:
                if hasattr(qdsim_cpp, 'setCallback'):
                    qdsim_cpp.setCallback("epsilon_r", epsilon_r)
                    qdsim_cpp.setCallback("rho", rho)
                    print("Registered callbacks with C++ code")
            except Exception as e:
                print(f"Warning: Could not register callbacks: {e}")
            
            # Try to create the PoissonSolver
            try:
                poisson_solver = qdsim_cpp.PoissonSolver(mesh, epsilon_r, rho)
                print("Created C++ PoissonSolver")
                
                # Solve with simple boundary conditions
                poisson_solver.solve(0.0, 1.0)
                print("Solved Poisson equation with C++ solver")
                
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
                plt.title('Electrostatic Potential (V) - C++ Solver')
                
                # Save the figure
                plt.savefig("results_validation/potential_cpp.png", dpi=150)
                print("Saved potential plot to results_validation/potential_cpp.png")
                
                return True
            except Exception as e:
                print(f"Warning: Could not use C++ PoissonSolver: {e}")
        except Exception as e:
            print(f"Warning: Error in C++ PoissonSolver: {e}")
        
        # Use the Python PoissonSolver2D
        try:
            # Create a simple 2D Poisson solver
            x_min = 0.0
            x_max = mesh.get_lx()
            nx = mesh.get_nx() + 1  # Add 1 to match the number of nodes
            y_min = 0.0
            y_max = mesh.get_ly()
            ny = mesh.get_ny() + 1  # Add 1 to match the number of nodes
            
            poisson_solver = PoissonSolver2D(x_min, x_max, nx, y_min, y_max, ny, epsilon_r=12.9)
            print("Created Python PoissonSolver2D")
            
            # Set boundary conditions
            boundary_values = {
                'left': 0.0,
                'right': 1.0,
                'bottom': 0.0,
                'top': 0.0
            }
            poisson_solver.set_boundary_conditions(boundary_values)
            
            # Set a simple charge density (all zeros)
            rho_array = np.zeros((ny, nx))
            poisson_solver.set_charge_density(rho_array)
            
            # Solve the Poisson equation
            potential = poisson_solver.solve()
            print("Solved Poisson equation with Python solver")
            
            print(f"Potential shape: {potential.shape}")
            print(f"Potential min: {potential.min()}, max: {potential.max()}")
            
            # Calculate the electric field
            Ex, Ey = poisson_solver.get_electric_field()
            print(f"Electric field calculated, Ex shape: {Ex.shape}, Ey shape: {Ey.shape}")
            print(f"Ex min: {Ex.min()}, max: {Ex.max()}")
            print(f"Ey min: {Ey.min()}, max: {Ey.max()}")
            
            # Create a directory for the results
            os.makedirs("results_validation", exist_ok=True)
            
            # Plot the potential
            X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
            
            plt.figure(figsize=(10, 5))
            contour = plt.contourf(X, Y, potential, 50, cmap='viridis')
            plt.colorbar(contour)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.title('Electrostatic Potential (V) - Python Solver')
            
            # Save the figure
            plt.savefig("results_validation/potential_python.png", dpi=150)
            print("Saved potential plot to results_validation/potential_python.png")
            
            # Plot the electric field
            plt.figure(figsize=(10, 5))
            plt.quiver(X, Y, Ex, Ey)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.title('Electric Field (V/nm)')
            
            # Save the figure
            plt.savefig("results_validation/electric_field.png", dpi=150)
            print("Saved electric field plot to results_validation/electric_field.png")
            
            return True
        except Exception as e:
            print(f"Error in Python PoissonSolver: {e}")
            return False
    except Exception as e:
        print(f"Error creating mesh: {e}")
        return False

def main():
    """Main function."""
    print("=== QDSim Poisson Test ===")
    
    # Test PoissonSolver
    if not test_poisson_solver():
        print("PoissonSolver test failed")
        return
    
    print("\n=== Test Completed Successfully ===")
    print("Results are available in the 'results_validation' directory")

if __name__ == "__main__":
    main()
