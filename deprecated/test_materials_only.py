#!/usr/bin/env python3
"""
Test script for materials module only
Tests the Cython materials implementation
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '.')

def test_materials_import():
    """Test if we can import the materials module"""
    try:
        from qdsim_cython.core.materials import Material, create_material
        print("✓ Successfully imported materials module")
        return True
    except ImportError as e:
        print(f"❌ Failed to import materials: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing materials: {e}")
        return False

def test_material_creation():
    """Test creating and using Material objects"""
    try:
        from qdsim_cython.core.materials import Material, create_material
        
        # Test creating a default material
        material = create_material("GaAs")
        print(f"✓ Created material: {material}")
        
        # Test accessing properties
        print(f"  Electron mass: {material.m_e}")
        print(f"  Hole mass: {material.m_h}")
        print(f"  Bandgap: {material.E_g} eV")
        print(f"  Dielectric constant: {material.epsilon_r}")
        
        # Test setting properties
        material.m_e = 0.1
        material.E_g = 1.5
        print(f"  Modified electron mass: {material.m_e}")
        print(f"  Modified bandgap: {material.E_g} eV")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing material creation: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("TESTING CYTHON MATERIALS MODULE")
    print("="*60)
    
    success = True
    
    # Test 1: Import
    print("\n1. Testing module import...")
    if not test_materials_import():
        success = False
    
    # Test 2: Material creation and usage
    print("\n2. Testing material creation...")
    if not test_material_creation():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The Cython materials module is working correctly.")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("The Cython implementation needs more debugging.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
