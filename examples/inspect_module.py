"""
Inspect the QDSim Python module.

This script inspects the QDSim Python module to check what classes and functions
are available.
"""

import sys
import os

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Try to import the C++ extension directly
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build'))
    import qdsim_cpp
    print("Successfully imported qdsim_cpp")
    print("Available attributes:")
    for attr in dir(qdsim_cpp):
        if not attr.startswith('__'):
            print(f"  {attr}")
except ImportError as e:
    print(f"Failed to import qdsim_cpp: {e}")

# Import the Python module
from frontend import qdsim
print("\nSuccessfully imported qdsim")
print("Available attributes:")
for attr in dir(qdsim):
    if not attr.startswith('__'):
        print(f"  {attr}")
