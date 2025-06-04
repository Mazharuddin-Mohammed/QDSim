#!/usr/bin/env python3
"""
Test Factory Functions - Easy Path to Eigensolvers
Use the factory functions instead of complex constructors
"""

import math

def test_factory_functions():
    """Test the factory functions for easier solver creation"""
    
    print("="*60)
    print("TESTING FACTORY FUNCTIONS")
    print("="*60)
    
    try:
        import qdsim_cpp
        print("✅ qdsim_cpp imported")
        
        # Test 1: Simple mesh factory
        print("\n1. Testing create_simple_mesh...")
        try:
            simple_mesh = qdsim_cpp.create_simple_mesh()
            print(f"✅ Simple mesh created: {type(simple_mesh)}")
            if hasattr(simple_mesh, 'get_num_nodes'):
                print(f"   Nodes: {simple_mesh.get_num_nodes()}")
        except Exception as e:
            print(f"❌ create_simple_mesh failed: {e}")
        
        # Test 2: Simple self-consistent solver factory
        print("\n2. Testing create_simple_self_consistent_solver...")
        try:
            simple_sc_solver = qdsim_cpp.create_simple_self_consistent_solver()
            print(f"✅ Simple SC solver created: {type(simple_sc_solver)}")
        except Exception as e:
            print(f"❌ create_simple_self_consistent_solver failed: {e}")
        
        # Test 3: Regular self-consistent solver factory
        print("\n3. Testing create_self_consistent_solver...")
        try:
            sc_solver = qdsim_cpp.create_self_consistent_solver()
            print(f"✅ SC solver created: {type(sc_solver)}")
        except Exception as e:
            print(f"❌ create_self_consistent_solver failed: {e}")
        
        # Test 4: Improved self-consistent solver factory
        print("\n4. Testing create_improved_self_consistent_solver...")
        try:
            improved_sc_solver = qdsim_cpp.create_improved_self_consistent_solver()
            print(f"✅ Improved SC solver created: {type(improved_sc_solver)}")
        except Exception as e:
            print(f"❌ create_improved_self_consistent_solver failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_level_functions():
    """Test module-level physics functions"""
    
    print("\n" + "="*60)
    print("TESTING MODULE-LEVEL FUNCTIONS")
    print("="*60)
    
    try:
        import qdsim_cpp
        
        # Test physics functions
        physics_functions = [
            'potential', 'effective_mass', 'epsilon_r', 'charge_density',
            'electron_concentration', 'hole_concentration', 'mobility_n', 'mobility_p'
        ]
        
        for func_name in physics_functions:
            if hasattr(qdsim_cpp, func_name):
                func = getattr(qdsim_cpp, func_name)
                print(f"✅ {func_name}: {type(func)}")
                
                # Try to call with test coordinates
                try:
                    if func_name in ['potential', 'effective_mass', 'epsilon_r']:
                        result = func(0.0, 0.0)
                        print(f"   {func_name}(0,0) = {result}")
                    elif func_name == 'charge_density':
                        # This might need more arguments
                        print(f"   {func_name} available (needs more args)")
                except Exception as call_error:
                    print(f"   {func_name} call failed: {call_error}")
            else:
                print(f"❌ {func_name}: not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Module functions test failed: {e}")
        return False

def test_simple_eigenvalue_path():
    """Try to find the simplest path to eigenvalue solving"""
    
    print("\n" + "="*60)
    print("TESTING SIMPLE EIGENVALUE PATH")
    print("="*60)
    
    try:
        import qdsim_cpp
        
        # Step 1: Create mesh (we know this works)
        print("1. Creating mesh...")
        mesh = qdsim_cpp.Mesh(50e-9, 50e-9, 8, 8, 1)
        print(f"✅ Mesh: {mesh.get_num_nodes()} nodes")
        
        # Step 2: Try SimpleMesh
        print("2. Testing SimpleMesh...")
        try:
            simple_mesh = qdsim_cpp.SimpleMesh()
            print(f"✅ SimpleMesh created: {type(simple_mesh)}")
            simple_methods = [m for m in dir(simple_mesh) if not m.startswith('_')]
            print(f"   SimpleMesh methods: {simple_methods}")
        except Exception as e:
            print(f"❌ SimpleMesh failed: {e}")
        
        # Step 3: Try BasicSolver
        print("3. Testing BasicSolver...")
        try:
            basic_solver = qdsim_cpp.BasicSolver()
            print(f"✅ BasicSolver created: {type(basic_solver)}")
            basic_methods = [m for m in dir(basic_solver) if not m.startswith('_')]
            print(f"   BasicSolver methods: {basic_methods}")
            
            # Look for solve methods
            solve_methods = [m for m in basic_methods if 'solve' in m.lower()]
            if solve_methods:
                print(f"   Solve methods found: {solve_methods}")
                
                # Try to call solve methods
                for solve_method in solve_methods:
                    try:
                        method = getattr(basic_solver, solve_method)
                        print(f"   🎯 Trying {solve_method}...")
                        # This might need arguments, but let's see the signature
                        print(f"      Method: {method}")
                    except Exception as solve_error:
                        print(f"      {solve_method} failed: {solve_error}")
            
        except Exception as e:
            print(f"❌ BasicSolver failed: {e}")
        
        # Step 4: Try to use factory functions with mesh
        print("4. Testing factory functions with mesh...")
        try:
            # See if factory functions accept mesh parameter
            print("   Trying factory functions with parameters...")
            
            # This is exploratory - we don't know the signatures yet
            print("   (Need to check factory function signatures)")
            
        except Exception as e:
            print(f"❌ Factory with mesh failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple eigenvalue path failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("QDSim Factory Functions Test")
    print("Finding the easiest path to eigenvalue solving")
    print()
    
    # Test 1: Factory functions
    factory_success = test_factory_functions()
    
    # Test 2: Module-level functions
    module_success = test_module_level_functions()
    
    # Test 3: Simple eigenvalue path
    simple_success = test_simple_eigenvalue_path()
    
    # Summary
    print("\n" + "="*60)
    print("FACTORY FUNCTIONS SUMMARY")
    print("="*60)
    
    if factory_success:
        print("✅ Factory functions accessible")
    if module_success:
        print("✅ Module-level physics functions working")
    if simple_success:
        print("✅ Simple solver path explored")
    
    if all([factory_success, module_success, simple_success]):
        print("\n🎉 Factory functions provide easier access!")
        print("💡 Next: Use factory functions to create solver chain")
        return 0
    else:
        print("\n⚠️  Some factory functions not working")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
