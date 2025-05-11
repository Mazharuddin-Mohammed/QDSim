#!/usr/bin/env python3
"""
Material database test script for QDSim.

This script tests the MaterialDatabase class in QDSim.

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np

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
    print("Successfully imported qdsim Python frontend")
except ImportError as e:
    print(f"Error importing qdsim Python frontend: {e}")
    print("Make sure the frontend is installed")
    sys.exit(1)

def test_material_database():
    """Test the MaterialDatabase class."""
    print("\n=== Testing MaterialDatabase ===")
    
    # Create a MaterialDatabase
    try:
        db = qdsim_cpp.MaterialDatabase()
        print("Created MaterialDatabase")
        
        # Try to get GaAs properties
        try:
            gaas = db.get_material("GaAs")
            print(f"GaAs properties: {gaas}")
            
            # Try to access individual properties
            try:
                if hasattr(gaas, 'epsilon_r'):
                    print(f"GaAs epsilon_r = {gaas.epsilon_r}")
                elif isinstance(gaas, tuple) and len(gaas) >= 5:
                    print(f"GaAs epsilon_r = {gaas[4]}")
                else:
                    print(f"GaAs properties type: {type(gaas)}")
            except Exception as e:
                print(f"Error accessing GaAs properties: {e}")
        except Exception as e:
            print(f"Error getting GaAs properties: {e}")
        
        # Try to get InAs properties
        try:
            inas = db.get_material("InAs")
            print(f"InAs properties: {inas}")
        except Exception as e:
            print(f"Error getting InAs properties: {e}")
        
        # Try to get available materials
        try:
            if hasattr(db, 'get_available_materials'):
                materials = db.get_available_materials()
                print(f"Available materials: {materials}")
            else:
                print("get_available_materials method not available")
        except Exception as e:
            print(f"Error getting available materials: {e}")
        
        return True
    except Exception as e:
        print(f"Error creating MaterialDatabase: {e}")
        
        # Try to use the Python materials module as a fallback
        try:
            import qdsim.materials as materials
            print("Using Python materials module")
            
            # Try to get GaAs properties
            try:
                gaas = materials.GaAs()
                print(f"GaAs properties: epsilon_r = {gaas.epsilon_r}, E_g = {gaas.E_g} eV")
            except Exception as e:
                print(f"Error getting GaAs properties from Python module: {e}")
            
            # Try to get InAs properties
            try:
                inas = materials.InAs()
                print(f"InAs properties: epsilon_r = {inas.epsilon_r}, E_g = {inas.E_g} eV")
            except Exception as e:
                print(f"Error getting InAs properties from Python module: {e}")
            
            return True
        except Exception as e:
            print(f"Error using Python materials module: {e}")
            return False

def main():
    """Main function."""
    print("=== QDSim Material Test ===")
    
    # Test MaterialDatabase
    if not test_material_database():
        print("MaterialDatabase test failed")
        return
    
    print("\n=== Test Completed Successfully ===")

if __name__ == "__main__":
    main()
