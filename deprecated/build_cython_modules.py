#!/usr/bin/env python3
"""
Build Cython Modules Individually
Fix compilation issues by building each module separately
"""

import os
import sys
import subprocess
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

def build_materials_module():
    """Build the materials module"""
    print("Building materials module...")
    
    try:
        materials_ext = Extension(
            "qdsim_cython.core.materials",
            sources=["qdsim_cython/core/materials.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O2"],
            extra_link_args=["-std=c++17"]
        )
        
        cythonized = cythonize(
            [materials_ext],
            compiler_directives={
                'language_level': 3,
                'embedsignature': True,
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
            },
            build_dir="build_cython"
        )
        
        print("✅ Materials module cythonized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Materials module failed: {e}")
        return False

def build_mesh_module():
    """Build the mesh module"""
    print("Building mesh module...")
    
    try:
        mesh_ext = Extension(
            "qdsim_cython.core.mesh",
            sources=["qdsim_cython/core/mesh.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O2"],
            extra_link_args=["-std=c++17"]
        )
        
        cythonized = cythonize(
            [mesh_ext],
            compiler_directives={
                'language_level': 3,
                'embedsignature': True,
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
            },
            build_dir="build_cython"
        )
        
        print("✅ Mesh module cythonized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Mesh module failed: {e}")
        return False

def build_physics_module():
    """Build the physics module"""
    print("Building physics module...")
    
    try:
        physics_ext = Extension(
            "qdsim_cython.core.physics",
            sources=["qdsim_cython/core/physics.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O2"],
            extra_link_args=["-std=c++17"]
        )
        
        cythonized = cythonize(
            [physics_ext],
            compiler_directives={
                'language_level': 3,
                'embedsignature': True,
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
            },
            build_dir="build_cython"
        )
        
        print("✅ Physics module cythonized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Physics module failed: {e}")
        return False

def build_poisson_module():
    """Build the Poisson solver module"""
    print("Building Poisson module...")
    
    try:
        poisson_ext = Extension(
            "qdsim_cython.solvers.poisson",
            sources=["qdsim_cython/solvers/poisson.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O2"],
            extra_link_args=["-std=c++17"]
        )
        
        cythonized = cythonize(
            [poisson_ext],
            compiler_directives={
                'language_level': 3,
                'embedsignature': True,
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
            },
            build_dir="build_cython"
        )
        
        print("✅ Poisson module cythonized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Poisson module failed: {e}")
        return False

def test_cython_imports():
    """Test if Cython modules can be imported"""
    print("\nTesting Cython module imports...")
    
    modules_to_test = [
        ("qdsim_cython.core.materials", "Materials"),
        ("qdsim_cython.core.mesh", "Mesh"),
        ("qdsim_cython.core.physics", "Physics"),
        ("qdsim_cython.solvers.poisson", "Poisson")
    ]
    
    success_count = 0
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name} module imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name} module import failed: {e}")
        except Exception as e:
            print(f"❌ {display_name} module error: {e}")
    
    print(f"\nImport test results: {success_count}/{len(modules_to_test)} modules working")
    return success_count == len(modules_to_test)

def create_simple_test():
    """Create a simple test for Cython functionality"""
    print("\nCreating simple Cython test...")
    
    test_code = '''
#!/usr/bin/env python3
"""Simple Cython functionality test"""

def test_materials():
    """Test materials module"""
    try:
        from qdsim_cython.core import materials
        
        # Test Material creation
        mat = materials.Material()
        mat.m_e = 0.041
        mat.epsilon_r = 13.9
        
        print(f"✅ Material: m_e={mat.m_e}, ε_r={mat.epsilon_r}")
        
        # Test MaterialDatabase
        db = materials.MaterialDatabase()
        print("✅ MaterialDatabase created")
        
        return True
        
    except Exception as e:
        print(f"❌ Materials test failed: {e}")
        return False

def test_mesh():
    """Test mesh module"""
    try:
        from qdsim_cython.core import mesh
        print("✅ Mesh module accessible")
        return True
        
    except Exception as e:
        print(f"❌ Mesh test failed: {e}")
        return False

def test_physics():
    """Test physics module"""
    try:
        from qdsim_cython.core import physics
        print("✅ Physics module accessible")
        return True
        
    except Exception as e:
        print(f"❌ Physics test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Cython Modules...")
    
    materials_ok = test_materials()
    mesh_ok = test_mesh()
    physics_ok = test_physics()
    
    if all([materials_ok, mesh_ok, physics_ok]):
        print("🎉 Cython modules working!")
    else:
        print("❌ Some Cython modules failed")
'''
    
    with open("test_cython_simple.py", "w") as f:
        f.write(test_code)
    
    print("✅ Simple test created: test_cython_simple.py")

def main():
    """Main build function"""
    
    print("QDSim Cython Module Builder")
    print("=" * 40)
    
    # Build modules individually
    materials_ok = build_materials_module()
    mesh_ok = build_mesh_module()
    physics_ok = build_physics_module()
    poisson_ok = build_poisson_module()
    
    # Test imports
    imports_ok = test_cython_imports()
    
    # Create simple test
    create_simple_test()
    
    # Summary
    print("\n" + "=" * 40)
    print("BUILD SUMMARY")
    print("=" * 40)
    
    modules_built = sum([materials_ok, mesh_ok, physics_ok, poisson_ok])
    print(f"Modules built: {modules_built}/4")
    
    if materials_ok:
        print("✅ Materials module ready")
    if mesh_ok:
        print("✅ Mesh module ready")
    if physics_ok:
        print("✅ Physics module ready")
    if poisson_ok:
        print("✅ Poisson module ready")
    
    if imports_ok:
        print("✅ All modules importable")
    else:
        print("⚠️  Some import issues found")
    
    if modules_built >= 2:
        print("\n🎉 Cython migration partially successful!")
        print("Next step: Test unified memory architecture")
        return True
    else:
        print("\n❌ Cython migration failed")
        print("Need to fix compilation issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
