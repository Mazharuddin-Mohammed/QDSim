#!/usr/bin/env python3
"""
Simple Cython Test - Identify Basic Issues

This script tests individual Cython modules one by one to identify
specific compilation and import issues.
"""

import sys
import os
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "frontend"))
sys.path.insert(0, str(Path(__file__).parent / "qdsim_cython"))

def test_basic_imports():
    """Test basic Python imports first"""
    print("🔧 Testing Basic Python Imports...")
    
    try:
        import sys
        print("✅ sys imported")
    except Exception as e:
        print(f"❌ sys failed: {e}")
    
    try:
        import os
        print("✅ os imported")
    except Exception as e:
        print(f"❌ os failed: {e}")
    
    try:
        import pathlib
        print("✅ pathlib imported")
    except Exception as e:
        print(f"❌ pathlib failed: {e}")

def test_numpy_import():
    """Test NumPy import"""
    print("\n🔧 Testing NumPy Import...")
    
    try:
        import numpy as np
        print(f"✅ NumPy imported: version {np.__version__}")
        return True
    except ImportError as e:
        print(f"❌ NumPy not available: {e}")
        return False

def test_frontend_import():
    """Test frontend import"""
    print("\n🔧 Testing Frontend Import...")
    
    try:
        import qdsim
        print("✅ QDSim frontend imported")
        return True
    except ImportError as e:
        print(f"❌ QDSim frontend failed: {e}")
        return False
    except Exception as e:
        print(f"❌ QDSim frontend error: {e}")
        return False

def test_cython_directory():
    """Test if Cython directory exists"""
    print("\n🔧 Testing Cython Directory Structure...")
    
    cython_dir = Path("qdsim_cython")
    if cython_dir.exists():
        print(f"✅ Cython directory exists: {cython_dir}")
        
        # Check subdirectories
        subdirs = ["core", "solvers", "gpu", "analysis", "visualization"]
        for subdir in subdirs:
            subdir_path = cython_dir / subdir
            if subdir_path.exists():
                print(f"✅ {subdir} directory exists")
            else:
                print(f"❌ {subdir} directory missing")
        
        return True
    else:
        print(f"❌ Cython directory not found: {cython_dir}")
        return False

def test_cython_files():
    """Test if Cython files exist"""
    print("\n🔧 Testing Cython Files...")
    
    cython_files = [
        "qdsim_cython/__init__.py",
        "qdsim_cython/core/__init__.py",
        "qdsim_cython/core/materials.pyx",
        "qdsim_cython/core/mesh.pyx",
        "qdsim_cython/core/physics.pyx",
        "qdsim_cython/core/interpolator.pyx",
        "qdsim_cython/solvers/__init__.py",
        "qdsim_cython/solvers/poisson.pyx",
        "qdsim_cython/solvers/schrodinger.pyx",
        "qdsim_cython/gpu/cuda_solver.pyx",
        "qdsim_cython/analysis/__init__.py",
        "qdsim_cython/analysis/quantum_analysis.pyx",
        "qdsim_cython/visualization/__init__.py"
    ]
    
    for file_path in cython_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")

def test_individual_cython_imports():
    """Test individual Cython module imports"""
    print("\n🔧 Testing Individual Cython Imports...")
    
    # Test core modules one by one
    modules_to_test = [
        "qdsim_cython",
        "qdsim_cython.core",
        "qdsim_cython.solvers",
        "qdsim_cython.analysis",
        "qdsim_cython.visualization"
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            print(f"❌ {module_name} import failed: {e}")
        except Exception as e:
            print(f"❌ {module_name} error: {e}")

def test_compiled_extensions():
    """Test if Cython extensions are compiled"""
    print("\n🔧 Testing Compiled Extensions...")
    
    import glob
    
    # Look for compiled extensions
    so_files = glob.glob("qdsim_cython/**/*.so", recursive=True)
    pyd_files = glob.glob("qdsim_cython/**/*.pyd", recursive=True)
    
    if so_files:
        print("✅ Found compiled .so files:")
        for so_file in so_files:
            print(f"   {so_file}")
    else:
        print("❌ No .so files found")
    
    if pyd_files:
        print("✅ Found compiled .pyd files:")
        for pyd_file in pyd_files:
            print(f"   {pyd_file}")
    else:
        print("ℹ️  No .pyd files found (normal on Linux)")
    
    return len(so_files) > 0 or len(pyd_files) > 0

def test_build_system():
    """Test build system files"""
    print("\n🔧 Testing Build System...")
    
    build_files = [
        "qdsim_cython/setup.py",
        "build_cython.sh"
    ]
    
    for file_path in build_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")

def main():
    """Main test function"""
    print("🚀 SIMPLE CYTHON TEST - IDENTIFYING ISSUES")
    print("=" * 60)
    
    # Run basic tests
    test_basic_imports()
    numpy_available = test_numpy_import()
    frontend_available = test_frontend_import()
    
    # Test Cython structure
    cython_dir_exists = test_cython_directory()
    test_cython_files()
    
    # Test if extensions are compiled
    extensions_compiled = test_compiled_extensions()
    
    # Test build system
    test_build_system()
    
    # Test imports if structure exists
    if cython_dir_exists:
        test_individual_cython_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"   NumPy Available: {'✅' if numpy_available else '❌'}")
    print(f"   Frontend Available: {'✅' if frontend_available else '❌'}")
    print(f"   Cython Directory: {'✅' if cython_dir_exists else '❌'}")
    print(f"   Extensions Compiled: {'✅' if extensions_compiled else '❌'}")
    
    if not extensions_compiled:
        print("\n🔧 NEXT STEPS:")
        print("   1. Cython extensions need to be compiled")
        print("   2. Run: ./build_cython.sh")
        print("   3. Or run: cd qdsim_cython && python setup.py build_ext --inplace")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
