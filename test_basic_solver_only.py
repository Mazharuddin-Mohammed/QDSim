#!/usr/bin/env python3
"""
Test script that creates a mesh and a basic solver.
"""

import sys
import os

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

# Import the C++ module directly
import qdsim_cpp as cpp

def main():
    """
    Main function to test creating a mesh and a basic solver.
    """
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 50
    ny = 25
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)
    
    # Create a basic solver
    solver = cpp.BasicSolver(mesh)
    
    print("Successfully created a mesh and a basic solver.")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
