#!/usr/bin/env python3
"""
Test script to debug execution issues
"""

import sys
import os

def test_basic():
    """Test basic functionality"""
    print("=== BASIC TEST ===")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print("✓ Basic test passed")

def test_imports():
    """Test importing modules"""
    print("\n=== IMPORT TEST ===")
    
    try:
        print("Testing sys...")
        import sys
        print("✓ sys imported")
        
        print("Testing os...")
        import os
        print("✓ os imported")
        
        print("Testing time...")
        import time
        print("✓ time imported")
        
        print("✓ All basic imports passed")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")

def test_compiled_module():
    """Test importing our compiled module"""
    print("\n=== COMPILED MODULE TEST ===")
    
    try:
        # Try to import the minimal materials module
        print("Attempting to import materials_minimal...")
        import materials_minimal
        print("✓ materials_minimal imported successfully!")
        
        # Test basic functionality
        print("Testing create_material...")
        mat = materials_minimal.create_material()
        print(f"✓ Material created: {mat}")
        
        print("✓ Compiled module test passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return False

def main():
    """Main test function"""
    print("QDSim Execution Debug Test")
    print("=" * 50)
    
    test_basic()
    test_imports()
    success = test_compiled_module()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
