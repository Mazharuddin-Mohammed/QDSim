#!/usr/bin/env python3
"""
Build Working Schrödinger Solver

This script builds the working Schrödinger solver with proper error handling.
"""

import sys
import os
from pathlib import Path

def build_working_solver():
    """Build the working Schrödinger solver"""
    print("🔧 Building Working Schrödinger Solver")
    print("=" * 50)
    
    try:
        from setuptools import setup, Extension
        from Cython.Build import cythonize
        import numpy as np
        
        print("✅ Build tools imported")
        
        # Define extension
        ext = Extension(
            'qdsim_cython.solvers.working_schrodinger_solver',
            ['solvers/working_schrodinger_solver.pyx'],
            include_dirs=[np.get_include()],
            language='c++',
            extra_compile_args=['-std=c++17', '-O2', '-fPIC'],
            extra_link_args=['-fPIC']
        )
        
        print("✅ Extension defined")
        
        # Build
        setup(
            ext_modules=cythonize([ext], compiler_directives={
                'language_level': 3,
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True
            }),
            script_name='build_working_solver.py',
            script_args=['build_ext', '--inplace']
        )
        
        print("✅ Working Schrödinger solver built successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import():
    """Test importing the built solver"""
    print("\n🔍 Testing Import")
    print("=" * 30)
    
    try:
        # Add to path
        sys.path.insert(0, '.')
        
        # Test import
        import qdsim_cython.solvers.working_schrodinger_solver as wss
        print("✅ Working solver imported successfully")
        
        # Test class access
        solver_class = wss.WorkingSchrodingerSolver
        print("✅ Solver class accessible")
        
        # Check methods
        methods = [
            'apply_open_system_boundary_conditions',
            'apply_dirac_delta_normalization',
            'configure_device_specific_solver',
            'apply_conservative_boundary_conditions',
            'apply_minimal_cap_boundaries'
        ]
        
        available_methods = 0
        for method in methods:
            if hasattr(solver_class, method):
                available_methods += 1
                print(f"✅ {method}: Available")
            else:
                print(f"❌ {method}: Missing")
        
        print(f"📊 Methods available: {available_methods}/{len(methods)}")
        
        return available_methods == len(methods)
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main build and test function"""
    print("🚀 WORKING SOLVER BUILD AND TEST")
    print("=" * 60)
    
    # Build solver
    build_success = build_working_solver()
    
    if build_success:
        # Test import
        import_success = test_import()
        
        if import_success:
            print("\n🎉 SUCCESS: Working solver built and tested!")
            print("   ✅ Compilation successful")
            print("   ✅ Import working")
            print("   ✅ All methods available")
            return True
        else:
            print("\n⚠️  BUILD SUCCESS but import issues")
            return False
    else:
        print("\n❌ BUILD FAILED")
        return False

if __name__ == "__main__":
    success = main()
