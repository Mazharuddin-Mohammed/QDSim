#!/usr/bin/env python3
"""
Test Simple Eigensolvers - Direct Approach
Try to use the eigensolvers directly without complex setup
"""

import sys
import math

def test_direct_eigensolvers():
    """Test eigensolvers directly"""
    
    print("="*60)
    print("TESTING DIRECT EIGENSOLVERS")
    print("="*60)
    
    try:
        import qdsim_cpp
        print("✅ qdsim_cpp imported successfully!")
        
        # Test 1: Direct EigenSolver
        print("\n🎯 Testing EigenSolver directly...")
        eigen_solver = qdsim_cpp.EigenSolver()
        print("✅ EigenSolver created")
        
        # Check methods
        methods = [m for m in dir(eigen_solver) if not m.startswith('_')]
        print(f"EigenSolver methods: {methods}")
        
        # Test 2: Check BasicSolver
        print("\n🎯 Testing BasicSolver...")
        if hasattr(qdsim_cpp, 'BasicSolver'):
            basic_solver = qdsim_cpp.BasicSolver()
            print("✅ BasicSolver created")
            basic_methods = [m for m in dir(basic_solver) if not m.startswith('_')]
            print(f"BasicSolver methods: {basic_methods}")
        
        # Test 3: Check if there are any simple solve functions
        print("\n🎯 Checking module-level functions...")
        module_functions = [f for f in dir(qdsim_cpp) if not f.startswith('_') and callable(getattr(qdsim_cpp, f))]
        print(f"Module functions: {module_functions}")
        
        # Test 4: Try SimpleMesh + SimpleInterpolator
        print("\n🎯 Testing SimpleMesh approach...")
        if hasattr(qdsim_cpp, 'SimpleMesh'):
            simple_mesh = qdsim_cpp.SimpleMesh()
            print("✅ SimpleMesh created")
            simple_mesh_methods = [m for m in dir(simple_mesh) if not m.startswith('_')]
            print(f"SimpleMesh methods: {simple_mesh_methods}")
        
        # Test 5: Check create_improved_self_consistent_solver
        print("\n🎯 Testing improved solver creation...")
        if hasattr(qdsim_cpp, 'create_improved_self_consistent_solver'):
            print("✅ create_improved_self_consistent_solver found")
            # This might be a factory function that's easier to use
        
        return True
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mesh_and_interpolator():
    """Test mesh and interpolator combination"""
    
    print("\n" + "="*60)
    print("TESTING MESH + INTERPOLATOR")
    print("="*60)
    
    try:
        import qdsim_cpp
        
        # Create mesh
        print("Creating mesh...")
        mesh = qdsim_cpp.Mesh(50e-9, 50e-9, 16, 16, 1)
        print(f"✅ Mesh: {mesh.get_num_nodes()} nodes")
        
        # Create interpolator
        print("Creating FEInterpolator...")
        interpolator = qdsim_cpp.FEInterpolator(mesh)
        print("✅ FEInterpolator created")
        
        # Check interpolator methods
        interp_methods = [m for m in dir(interpolator) if not m.startswith('_')]
        print(f"FEInterpolator methods: {interp_methods}")
        
        # Test interpolation
        print("Testing interpolation...")
        if hasattr(interpolator, 'interpolate'):
            # Try to interpolate a simple function
            print("✅ Interpolate method found")
        
        return True, mesh, interpolator
        
    except Exception as e:
        print(f"❌ Mesh+Interpolator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_poisson_solver():
    """Test PoissonSolver as a simpler starting point"""
    
    print("\n" + "="*60)
    print("TESTING POISSON SOLVER")
    print("="*60)
    
    try:
        import qdsim_cpp
        
        # Create mesh
        mesh = qdsim_cpp.Mesh(50e-9, 50e-9, 16, 16, 1)
        print(f"✅ Mesh: {mesh.get_num_nodes()} nodes")
        
        # Try PoissonSolver
        print("Creating PoissonSolver...")
        
        # Check constructor signature
        try:
            poisson_solver = qdsim_cpp.PoissonSolver()
            print("✅ PoissonSolver created (no args)")
        except Exception as e1:
            print(f"❌ No-arg constructor failed: {e1}")
            
            try:
                # Try with mesh
                poisson_solver = qdsim_cpp.PoissonSolver(mesh)
                print("✅ PoissonSolver created (with mesh)")
            except Exception as e2:
                print(f"❌ Mesh constructor failed: {e2}")
                return False
        
        # Check methods
        poisson_methods = [m for m in dir(poisson_solver) if not m.startswith('_')]
        print(f"PoissonSolver methods: {poisson_methods}")
        
        return True, poisson_solver
        
    except Exception as e:
        print(f"❌ PoissonSolver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_material_database():
    """Test MaterialDatabase for proper material setup"""
    
    print("\n" + "="*60)
    print("TESTING MATERIAL DATABASE")
    print("="*60)
    
    try:
        import qdsim_cpp
        
        # Test MaterialDatabase
        print("Creating MaterialDatabase...")
        mat_db = qdsim_cpp.MaterialDatabase()
        print("✅ MaterialDatabase created")
        
        # Check methods
        db_methods = [m for m in dir(mat_db) if not m.startswith('_')]
        print(f"MaterialDatabase methods: {db_methods}")
        
        # Test Material creation
        print("Creating Material...")
        material = qdsim_cpp.Material()
        print("✅ Material created")
        
        # Set properties
        material.m_e = 0.041
        material.epsilon_r = 13.9
        print(f"✅ Material properties: m_e={material.m_e}, ε_r={material.epsilon_r}")
        
        return True, mat_db, material
        
    except Exception as e:
        print(f"❌ Material test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def main():
    """Main test function"""
    
    print("QDSim Simple Eigensolvers Test")
    print("Finding the simplest path to eigenvalue solving")
    print()
    
    # Test 1: Direct eigensolvers
    direct_success = test_direct_eigensolvers()
    
    # Test 2: Mesh and interpolator
    mesh_success, mesh, interpolator = test_mesh_and_interpolator()
    
    # Test 3: Poisson solver
    poisson_success, poisson_solver = test_poisson_solver()
    
    # Test 4: Material database
    material_success, mat_db, material = test_material_database()
    
    # Summary
    print("\n" + "="*60)
    print("SIMPLE TEST SUMMARY")
    print("="*60)
    
    if direct_success:
        print("✅ Direct eigensolvers accessible")
    if mesh_success:
        print("✅ Mesh and interpolator working")
    if poisson_success:
        print("✅ PoissonSolver working")
    if material_success:
        print("✅ Materials working")
    
    if all([direct_success, mesh_success, poisson_success, material_success]):
        print("\n🎉 All basic components working!")
        print("💡 Next step: Find correct API for eigenvalue solving")
        return 0
    else:
        print("\n⚠️  Some components not working")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
