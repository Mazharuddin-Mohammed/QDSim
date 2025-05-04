#!/usr/bin/env python3
"""
Test script to identify the cause of the segmentation fault.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import time

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

# Import the C++ module directly
import qdsim_cpp as cpp

def test_mesh_only():
    """Test creating a mesh and plotting it."""
    print("Test 1: Creating a mesh and plotting it...")

    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)

    # Get the mesh nodes
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Plot the mesh
    plt.figure()
    plt.scatter(x, y, s=1)
    plt.title('Mesh')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig('test_mesh.png')
    plt.close()

    print("Test 1 completed successfully!")

def test_basic_solver():
    """Test creating a basic solver and plotting the results."""
    print("Test 2: Creating a basic solver and plotting the results...")

    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)

    # Create a basic solver
    solver = cpp.BasicSolver(mesh)

    # Solve the equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 0.7  # Voltage at the n-contact
    N_A = 1e18  # Acceptor doping concentration
    N_D = 1e18  # Donor doping concentration

    solver.solve(V_p, V_n, N_A, N_D)

    # Get the results
    potential = np.array(solver.get_potential())
    n = np.array(solver.get_n())
    p = np.array(solver.get_p())

    # Get the mesh nodes
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Plot the potential
    plt.figure()
    plt.tricontourf(x, y, potential, 20, cmap='viridis')
    plt.colorbar()
    plt.title('Potential (V)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig('test_potential.png')
    plt.close()

    print("Test 2 completed successfully!")

def test_improved_solver_no_plot():
    """Test creating an improved solver without plotting."""
    print("Test 3: Creating an improved solver without plotting...")

    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)

    # Define callback functions
    def epsilon_r(x, y):
        """Relative permittivity function."""
        return 12.9  # GaAs

    def rho(x, y, n, p):
        """Charge density function."""
        return 0.0  # Simplified: no charge density

    # Create an improved solver
    solver = cpp.ImprovedSelfConsistentSolver(mesh, epsilon_r, rho)

    # Solve the equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 0.7  # Voltage at the n-contact
    N_A = 1e18  # Acceptor doping concentration
    N_D = 1e18  # Donor doping concentration

    solver.solve(V_p, V_n, N_A, N_D)

    # Get the results
    potential = np.array(solver.get_potential())
    n = np.array(solver.get_n())
    p = np.array(solver.get_p())

    print("Test 3 completed successfully!")

def test_improved_solver_with_plot():
    """Test creating an improved solver and plotting the results."""
    print("Test 4: Creating an improved solver and plotting the results...")

    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)

    # Define callback functions
    def epsilon_r(x, y):
        """Relative permittivity function."""
        return 12.9  # GaAs

    def rho(x, y, n, p):
        """Charge density function."""
        return 0.0  # Simplified: no charge density

    # Create an improved solver
    solver = cpp.ImprovedSelfConsistentSolver(mesh, epsilon_r, rho)

    # Solve the equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 0.7  # Voltage at the n-contact
    N_A = 1e18  # Acceptor doping concentration
    N_D = 1e18  # Donor doping concentration

    solver.solve(V_p, V_n, N_A, N_D)

    # Get the results
    potential = np.array(solver.get_potential())
    n = np.array(solver.get_n())
    p = np.array(solver.get_p())

    # Get the mesh nodes
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Plot the potential
    plt.figure()
    plt.tricontourf(x, y, potential, 20, cmap='viridis')
    plt.colorbar()
    plt.title('Potential (V)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig('test_improved_potential.png')
    plt.close()

    print("Test 4 completed successfully!")

def test_improved_solver_with_plot_and_gc():
    """Test creating an improved solver, plotting the results, and forcing garbage collection."""
    print("Test 5: Creating an improved solver, plotting the results, and forcing garbage collection...")

    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)

    # Define callback functions
    def epsilon_r(x, y):
        """Relative permittivity function."""
        return 12.9  # GaAs

    def rho(x, y, n, p):
        """Charge density function."""
        return 0.0  # Simplified: no charge density

    # Create an improved solver
    solver = cpp.ImprovedSelfConsistentSolver(mesh, epsilon_r, rho)

    # Solve the equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 0.7  # Voltage at the n-contact
    N_A = 1e18  # Acceptor doping concentration
    N_D = 1e18  # Donor doping concentration

    solver.solve(V_p, V_n, N_A, N_D)

    # Get the results
    potential = np.array(solver.get_potential())
    n = np.array(solver.get_n())
    p = np.array(solver.get_p())

    # Get the mesh nodes
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Force garbage collection
    del solver
    del mesh
    gc.collect()

    # Plot the potential
    plt.figure()
    plt.tricontourf(x, y, potential, 20, cmap='viridis')
    plt.colorbar()
    plt.title('Potential (V)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig('test_improved_potential_gc.png')
    plt.close()

    print("Test 5 completed successfully!")

def main():
    """Main function to run the tests."""
    try:
        test_mesh_only()
        print()
    except Exception as e:
        print(f"Test 1 failed: {e}")

    try:
        test_basic_solver()
        print()
    except Exception as e:
        print(f"Test 2 failed: {e}")

    try:
        test_improved_solver_no_plot()
        print()
    except Exception as e:
        print(f"Test 3 failed: {e}")

    try:
        test_improved_solver_with_plot()
        print()
    except Exception as e:
        print(f"Test 4 failed: {e}")

    try:
        test_improved_solver_with_plot_and_gc()
        print()
    except Exception as e:
        print(f"Test 5 failed: {e}")

    print("All tests completed!")

    # Sleep for a few seconds to see if the segmentation fault occurs after a delay
    print("Sleeping for 5 seconds...")
    time.sleep(5)
    print("Done sleeping.")

if __name__ == "__main__":
    main()
