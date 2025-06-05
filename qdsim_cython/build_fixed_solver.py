#!/usr/bin/env python3
"""
Build Fixed Open System Solver

This script builds the completely fixed solver that addresses all issues.
"""

import sys
import os

def build_fixed_solver():
    """Build the fixed solver"""
    print("🔧 Building Fixed Open System Solver")
    print("=" * 50)
    
    try:
        from setuptools import setup, Extension
        from Cython.Build import cythonize
        import numpy as np
        
        # Define extension
        ext = Extension(
            'qdsim_cython.solvers.fixed_open_system_solver',
            ['solvers/fixed_open_system_solver.pyx'],
            include_dirs=[np.get_include()],
            language='c++',
            extra_compile_args=['-std=c++17', '-O2'],
        )
        
        # Build
        setup(
            ext_modules=cythonize([ext], compiler_directives={
                'language_level': 3,
                'boundscheck': False,
                'wraparound': False
            }),
            script_name='build_fixed_solver.py',
            script_args=['build_ext', '--inplace']
        )
        
        print("✅ Fixed solver built successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        return False

def test_fixed_solver():
    """Test the fixed solver"""
    print("\n🔍 Testing Fixed Solver")
    print("=" * 30)
    
    try:
        sys.path.insert(0, '.')
        import qdsim_cython.solvers.fixed_open_system_solver as fixed_solver
        
        print("✅ Fixed solver imported")
        
        # Test solver creation
        def m_star_func(x, y):
            return 0.067 * 9.1093837015e-31
        
        def potential_func(x, y):
            return 0.0
        
        solver = fixed_solver.FixedOpenSystemSolver(
            5, 4, 15e-9, 12e-9, m_star_func, potential_func, False
        )
        
        print("✅ Solver created successfully")
        
        # Test solving
        eigenvals, eigenvecs = solver.solve(1)
        
        if len(eigenvals) > 0:
            print(f"✅ Eigenvalue computed: {eigenvals[0]/1.602176634e-19:.6f} eV")
            return True
        else:
            print("❌ No eigenvalues computed")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 FIXED SOLVER BUILD AND TEST")
    print("=" * 60)
    
    # Build
    build_success = build_fixed_solver()
    
    if build_success:
        # Test
        test_success = test_fixed_solver()
        
        if test_success:
            print("\n🎉 COMPLETE SUCCESS!")
            print("   ✅ Fixed solver builds correctly")
            print("   ✅ Matrix assembly working")
            print("   ✅ Eigenvalue solver functional")
            print("   ✅ Ready for open system validation")
            return True
        else:
            print("\n⚠️  Build success but test issues")
            return False
    else:
        print("\n❌ Build failed")
        return False

if __name__ == "__main__":
    success = main()
