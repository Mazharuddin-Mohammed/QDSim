#!/usr/bin/env python3
"""
Error handling demonstration for QDSim.

This script demonstrates the error handling capabilities of QDSim,
including robust error handling and graceful degradation.

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path to import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qdsim

def main():
    """Main function demonstrating error handling capabilities."""
    print("QDSim Error Handling Demonstration")
    print("==================================")
    
    # 1. Input Validation
    print("\n1. Input Validation")
    
    # Create a simulator
    simulator = qdsim.Simulator()
    
    # Try to create a mesh with invalid parameters
    print("\nTrying to create a mesh with invalid parameters...")
    try:
        simulator.create_mesh(-10.0, 100.0, 101, 101)
    except ValueError as e:
        print(f"Caught error (as expected): {e}")
    
    # Try to create a mesh with valid parameters
    print("\nCreating a mesh with valid parameters...")
    simulator.create_mesh(100.0, 100.0, 101, 101)
    print("Mesh created successfully")
    
    # Try to set an invalid potential function
    print("\nTrying to set an invalid potential function...")
    try:
        def invalid_potential(x, y):
            # This will cause a division by zero error
            return 1.0 / (x * y)
        
        simulator.set_potential(invalid_potential)
    except Exception as e:
        print(f"Caught error (as expected): {e}")
    
    # Set a valid potential function
    print("\nSetting a valid potential function...")
    def valid_potential(x, y):
        # Simple harmonic oscillator potential
        return 0.5 * 1.0e-19 * (x**2 + y**2)
    
    simulator.set_potential(valid_potential)
    print("Potential function set successfully")
    
    # Try to set an invalid material
    print("\nTrying to set an invalid material...")
    try:
        simulator.set_material("InvalidMaterial")
    except ValueError as e:
        print(f"Caught error (as expected): {e}")
    
    # Set a valid material
    print("\nSetting a valid material...")
    simulator.set_material("GaAs")
    print("Material set successfully")
    
    # 2. Robust Error Handling
    print("\n2. Robust Error Handling")
    
    # Try to solve with invalid parameters
    print("\nTrying to solve with invalid parameters...")
    try:
        simulator.solve(num_states=-5)
    except ValueError as e:
        print(f"Caught error (as expected): {e}")
    
    # Solve with valid parameters
    print("\nSolving with valid parameters...")
    simulator.solve(num_states=5)
    print("Solver completed successfully")
    
    # 3. Graceful Degradation
    print("\n3. Graceful Degradation")
    
    # Create a simulator with a non-positive-definite mass matrix
    print("\nCreating a simulator with a problematic mass matrix...")
    simulator_bad = qdsim.Simulator()
    simulator_bad.create_mesh(100.0, 100.0, 101, 101)
    
    # Define a potential that will cause numerical issues
    def problematic_potential(x, y):
        # This potential has very large values that can cause numerical issues
        if abs(x) < 1e-10 and abs(y) < 1e-10:
            return 1.0e10  # Very large value
        return 0.0
    
    simulator_bad.set_potential(problematic_potential)
    simulator_bad.set_material("GaAs")
    
    # Try to solve with the problematic potential
    print("\nTrying to solve with a problematic potential...")
    try:
        # Enable automatic recovery
        simulator_bad.enable_recovery(True)
        simulator_bad.solve(num_states=5)
        print("Solver completed with automatic recovery")
        
        # Get eigenvalues
        eigenvalues = simulator_bad.get_eigenvalues()
        print(f"Eigenvalues: {eigenvalues}")
        
        # Check if eigenvalues are valid
        if all(np.isfinite(eigenvalues)):
            print("Recovery was successful - eigenvalues are valid")
        else:
            print("Recovery was partially successful - some eigenvalues may be invalid")
    except Exception as e:
        print(f"Recovery failed: {e}")
    
    # 4. Error Logging
    print("\n4. Error Logging")
    
    # Enable error logging
    print("\nEnabling error logging...")
    simulator.enable_logging("qdsim_error.log", log_level="INFO")
    print("Error logging enabled")
    
    # Generate some log messages
    print("\nGenerating log messages...")
    simulator.log_debug("This is a debug message")
    simulator.log_info("This is an info message")
    simulator.log_warning("This is a warning message")
    simulator.log_error("This is an error message")
    print("Log messages generated")
    
    # Show log file contents
    print("\nLog file contents:")
    with open("qdsim_error.log", "r") as f:
        print(f.read())
    
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main()
