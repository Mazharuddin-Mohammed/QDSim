#!/usr/bin/env python3
"""
Minimal test script for QDSim.

This script performs a minimal test of the QDSim codebase by:
1. Testing the mesh creation
2. Visualizing the mesh

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the necessary paths
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend/build'))

# Try to import the C++ module
try:
    import qdsim_cpp
    print("Successfully imported qdsim_cpp module")
except ImportError as e:
    print(f"Error importing qdsim_cpp module: {e}")
    print("Make sure the C++ extension is built and in the Python path")
    sys.exit(1)

def test_mesh_creation():
    """Test mesh creation."""
    print("\n=== Testing Mesh Creation ===")
    
    # Create a mesh
    try:
        mesh = qdsim_cpp.Mesh(100.0, 50.0, 50, 25, 1)
        print(f"Created mesh with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")
        
        # Get the nodes and elements
        nodes = np.array(mesh.get_nodes())
        elements = np.array(mesh.get_elements())
        
        print(f"Nodes shape: {nodes.shape}")
        print(f"Elements shape: {elements.shape}")
        
        # Create a directory for the results
        os.makedirs("results_validation", exist_ok=True)
        
        # Save the nodes and elements to files
        np.savetxt("results_validation/nodes.txt", nodes)
        np.savetxt("results_validation/elements.txt", elements)
        print("Saved nodes and elements to files")
        
        # Plot the mesh
        plt.figure(figsize=(10, 5))
        
        # Plot the mesh elements
        for element in elements:
            x = nodes[element, 0]
            y = nodes[element, 1]
            x = np.append(x, x[0])  # Close the triangle
            y = np.append(y, y[0])  # Close the triangle
            plt.plot(x, y, 'k-', linewidth=0.5)
        
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.title('Mesh')
        plt.axis('equal')
        
        # Save the figure
        plt.savefig("results_validation/mesh.png", dpi=150)
        print("Saved mesh plot to results_validation/mesh.png")
        
        return True
    except Exception as e:
        print(f"Error creating mesh: {e}")
        return False

def main():
    """Main function."""
    print("=== QDSim Minimal Test ===")
    
    # Test mesh creation
    if not test_mesh_creation():
        print("Mesh creation test failed")
        return
    
    print("\n=== Test Completed Successfully ===")
    print("Results are available in the 'results_validation' directory")

if __name__ == "__main__":
    main()
