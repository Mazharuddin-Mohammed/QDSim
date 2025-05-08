#!/usr/bin/env python3
"""
Simple test script to check if we can import the C++ module correctly.
"""

import sys
import os

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

try:
    # Try to import the C++ module directly
    import qdsim_cpp
    print("Successfully imported qdsim_cpp")
    
    # Check if SimpleSelfConsistentSolver is available
    if hasattr(qdsim_cpp, 'SimpleSelfConsistentSolver'):
        print("SimpleSelfConsistentSolver is available")
    else:
        print("SimpleSelfConsistentSolver is NOT available")
    
    # Check if create_simple_self_consistent_solver is available
    if hasattr(qdsim_cpp, 'create_simple_self_consistent_solver'):
        print("create_simple_self_consistent_solver is available")
    else:
        print("create_simple_self_consistent_solver is NOT available")
    
    # List all available attributes
    print("\nAvailable attributes:")
    for attr in dir(qdsim_cpp):
        if not attr.startswith('__'):
            print(f"- {attr}")
    
except ImportError as e:
    print(f"Error importing qdsim_cpp: {e}")
