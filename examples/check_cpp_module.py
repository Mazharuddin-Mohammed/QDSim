"""
Check if the FullPoissonDriftDiffusionSolver class is available in the C++ module.
"""

import sys
import os

# Add the build directory to the Python path
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build')
sys.path.append(build_dir)

# Try to import the C++ module
try:
    import qdsim_cpp
    print("Successfully imported qdsim_cpp")
    
    # Check if the FullPoissonDriftDiffusionSolver class is available
    if hasattr(qdsim_cpp, 'FullPoissonDriftDiffusionSolver'):
        print("FullPoissonDriftDiffusionSolver class is available")
    else:
        print("FullPoissonDriftDiffusionSolver class is NOT available")
        
    # Print all available attributes
    print("\nAvailable attributes:")
    for attr in dir(qdsim_cpp):
        if not attr.startswith('__'):
            print(f"  {attr}")
except ImportError as e:
    print(f"Failed to import qdsim_cpp: {e}")
