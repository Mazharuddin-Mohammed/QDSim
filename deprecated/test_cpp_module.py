#!/usr/bin/env python3
"""
Test script to check what's available in the qdsim_cpp module.
"""

import sys
import os

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

try:
    from qdsim import qdsim_cpp
    print("Successfully imported qdsim_cpp module")
    
    # Print all available attributes
    print("\nAvailable attributes:")
    for attr in dir(qdsim_cpp):
        if not attr.startswith('__'):
            print(f"  - {attr}")
    
    # Try to create a Mesh object
    print("\nTrying to create a Mesh object:")
    mesh = qdsim_cpp.Mesh(100e-9, 100e-9, 10, 10, 1)
    print(f"  - Created mesh with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")
    
    # Try to create a FEInterpolator object
    try:
        print("\nTrying to create a FEInterpolator object:")
        interpolator = qdsim_cpp.FEInterpolator(mesh)
        print("  - Successfully created FEInterpolator")
    except Exception as e:
        print(f"  - Failed to create FEInterpolator: {e}")
    
except ImportError as e:
    print(f"Error importing qdsim_cpp: {e}")
except Exception as e:
    print(f"Error: {e}")
