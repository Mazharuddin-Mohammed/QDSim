#!/usr/bin/env python3
"""
Test script to check what's available in the qdsim_cpp module.
"""

import sys
import os

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

# Print the Python path
print("Python path:")
for p in sys.path:
    print(f"  - {p}")

# Check if the module file exists
module_path = os.path.join(os.path.dirname(__file__), 'frontend', 'qdsim', 'qdsim_cpp.cpython-312-x86_64-linux-gnu.so')
print(f"\nChecking if module file exists at: {module_path}")
print(f"  - Exists: {os.path.exists(module_path)}")

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

except ImportError as e:
    print(f"Error importing qdsim_cpp: {e}")
except Exception as e:
    print(f"Error: {e}")
