#!/usr/bin/env python3
"""
Build Advanced Memory Manager

This script builds the advanced memory management system.
"""

import sys
import os

def build_memory_manager():
    """Build the advanced memory manager"""
    print("🧠 Building Advanced Memory Manager")
    print("=" * 50)
    
    try:
        from setuptools import setup, Extension
        from Cython.Build import cythonize
        import numpy as np
        
        # Create memory directory if it doesn't exist
        os.makedirs('memory', exist_ok=True)
        
        # Create __init__.py for memory package
        init_file = 'memory/__init__.py'
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Memory management package\n')
        
        # Define extension
        ext = Extension(
            'qdsim_cython.memory.advanced_memory_manager',
            ['memory/advanced_memory_manager.pyx'],
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
            script_name='build_memory_manager.py',
            script_args=['build_ext', '--inplace']
        )
        
        print("✅ Advanced memory manager built successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_manager():
    """Test the memory manager"""
    print("\n🧪 Testing Memory Manager")
    print("=" * 30)
    
    try:
        sys.path.insert(0, '.')
        import qdsim_cython.memory.advanced_memory_manager as amm
        
        print("✅ Memory manager imported")
        
        # Test basic functionality
        result = amm.test_memory_manager()
        
        if result:
            print("✅ Memory manager test passed")
            return True
        else:
            print("❌ Memory manager test failed")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 ADVANCED MEMORY MANAGER BUILD AND TEST")
    print("=" * 60)
    
    # Build
    build_success = build_memory_manager()
    
    if build_success:
        # Test
        test_success = test_memory_manager()
        
        if test_success:
            print("\n🎉 COMPLETE SUCCESS!")
            print("   ✅ Advanced memory manager built")
            print("   ✅ RAII-based memory management working")
            print("   ✅ Memory pools functional")
            print("   ✅ Managed NumPy arrays working")
            print("   ✅ Garbage collection operational")
            return True
        else:
            print("\n⚠️  Build success but test issues")
            return False
    else:
        print("\n❌ Build failed")
        return False

if __name__ == "__main__":
    success = main()
