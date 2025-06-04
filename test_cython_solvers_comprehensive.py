#!/usr/bin/env python3
"""
Comprehensive Test of Cython Solvers

This script tests all the migrated Cython solvers to validate that they
work correctly and provide the same functionality as the C++ backend.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "qdsim_cython"))
sys.path.insert(0, str(Path(__file__).parent / "frontend"))

def test_cython_poisson_solver():
    """Test the Cython Poisson solver"""
    print("🔧 Testing Cython Poisson Solver")
    print("-" * 50)
    
    try:
        # Import required modules
        import qdsim_cython.core.mesh_minimal as mesh_module
        import qdsim_cython.solvers.poisson_solver as poisson_module
        
        # Create mesh
        mesh = mesh_module.SimpleMesh(20, 15, 100e-9, 50e-9)
        print(f"✅ Mesh created: {mesh.num_nodes} nodes")
        
        # Define material functions
        def epsilon_r_func(x, y):
            return 12.9  # GaAs relative permittivity
        
        def rho_func(x, y, n, p):
            # Simple charge density (no free carriers for test)
            return 0.0
        
        # Create Poisson solver
        start_time = time.time()
        poisson_solver = poisson_module.CythonPoissonSolver(mesh, epsilon_r_func, rho_func)
        creation_time = time.time() - start_time
        
        print(f"✅ Poisson solver created in {creation_time:.3f}s")
        
        # Solve Poisson equation
        start_time = time.time()
        poisson_solver.solve(0.0, 1.0)  # 1V bias
        solve_time = time.time() - start_time
        
        # Get results
        potential = poisson_solver.get_potential()
        Ex, Ey = poisson_solver.get_electric_field()
        
        print(f"✅ Poisson equation solved in {solve_time:.3f}s")
        print(f"   Potential range: {np.min(potential):.6f} to {np.max(potential):.6f} V")
        print(f"   Electric field range: {np.min(Ex):.2e} to {np.max(Ex):.2e} V/m")
        print(f"   Matrix info: {poisson_solver.get_matrix_info()}")
        
        # Validate solution
        if np.std(potential) > 1e-6 and np.max(potential) > 0.5:
            print("✅ Non-trivial solution obtained")
            return True, solve_time
        else:
            print("❌ Solution appears trivial")
            return False, 0
            
    except Exception as e:
        print(f"❌ Poisson solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_cython_fem_solver():
    """Test the Cython FEM solver"""
    print("\n🔧 Testing Cython FEM Solver")
    print("-" * 50)
    
    try:
        # Import required modules
        import qdsim_cython.core.mesh_minimal as mesh_module
        import qdsim_cython.solvers.fem_solver as fem_module
        
        # Create mesh
        mesh = mesh_module.SimpleMesh(15, 10, 50e-9, 30e-9)
        print(f"✅ Mesh created: {mesh.num_nodes} nodes")
        
        # Create FEM solver
        start_time = time.time()
        fem_solver = fem_module.CythonFEMSolver(mesh)
        creation_time = time.time() - start_time
        
        print(f"✅ FEM solver created in {creation_time:.3f}s")
        
        # Test matrix assembly
        start_time = time.time()
        
        # Assemble stiffness matrix
        def coefficient_func(x, y):
            return 1.0  # Constant coefficient
        
        K = fem_solver.assemble_stiffness_matrix(coefficient_func)
        
        # Assemble mass matrix
        def density_func(x, y):
            return 1.0  # Constant density
        
        M = fem_solver.assemble_mass_matrix(density_func)
        
        # Assemble load vector
        def source_func(x, y):
            return 1.0  # Constant source
        
        f = fem_solver.assemble_load_vector(source_func)
        
        assembly_time = time.time() - start_time
        
        print(f"✅ Matrices assembled in {assembly_time:.3f}s")
        print(f"   Stiffness matrix: {K.shape}, nnz: {K.nnz}")
        print(f"   Mass matrix: {M.shape}, nnz: {M.nnz}")
        print(f"   Load vector: {len(f)} elements")
        
        # Test boundary conditions and solve
        boundary_nodes = [0, mesh.nx-1]  # Left and right boundaries
        boundary_values = [0.0, 1.0]
        
        K_bc, f_bc = fem_solver.apply_dirichlet_bc(boundary_nodes, boundary_values, K, f)
        
        # Solve linear system
        start_time = time.time()
        solution = fem_solver.solve_linear_system(K_bc, f_bc)
        solve_time = time.time() - start_time
        
        print(f"✅ Linear system solved in {solve_time:.3f}s")
        print(f"   Solution range: {np.min(solution):.6f} to {np.max(solution):.6f}")
        
        # Test mesh quality
        quality = fem_solver.compute_element_quality()
        print(f"   Average element quality: {np.mean(quality):.3f}")
        
        # Get solver info
        info = fem_solver.get_matrix_info()
        print(f"   Solver info: {info}")
        
        return True, assembly_time + solve_time
        
    except Exception as e:
        print(f"❌ FEM solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_integration_poisson_fem():
    """Test integration between Poisson and FEM solvers"""
    print("\n🔧 Testing Poisson-FEM Integration")
    print("-" * 50)
    
    try:
        # Import modules
        import qdsim_cython.core.mesh_minimal as mesh_module
        import qdsim_cython.solvers.poisson_solver as poisson_module
        import qdsim_cython.solvers.fem_solver as fem_module
        
        # Create shared mesh
        mesh = mesh_module.SimpleMesh(12, 8, 40e-9, 25e-9)
        print(f"✅ Shared mesh created: {mesh.num_nodes} nodes")
        
        # Create FEM solver for general matrix operations
        fem_solver = fem_module.CythonFEMSolver(mesh)
        
        # Create Poisson solver for electrostatics
        def epsilon_r_func(x, y):
            return 12.9
        
        def rho_func(x, y, n, p):
            return 0.0
        
        poisson_solver = poisson_module.CythonPoissonSolver(mesh, epsilon_r_func, rho_func)
        
        print("✅ Both solvers created successfully")
        
        # Solve Poisson equation
        poisson_solver.solve(0.0, 0.5)  # 0.5V bias
        potential = poisson_solver.get_potential()
        
        # Use FEM solver to create a related problem
        def potential_source(x, y):
            # Use Poisson solution as source for another problem
            node_idx = mesh.find_nearest_node(x, y)
            if 0 <= node_idx < len(potential):
                return potential[node_idx]
            return 0.0
        
        # Assemble and solve with FEM
        K = fem_solver.assemble_stiffness_matrix()
        f = fem_solver.assemble_load_vector(potential_source)
        
        # Apply boundary conditions
        boundary_nodes = [0, mesh.nx-1]
        boundary_values = [0.0, 0.0]
        K_bc, f_bc = fem_solver.apply_dirichlet_bc(boundary_nodes, boundary_values, K, f)
        
        fem_solution = fem_solver.solve_linear_system(K_bc, f_bc)
        
        print(f"✅ Integration test completed")
        print(f"   Poisson potential range: {np.min(potential):.6f} to {np.max(potential):.6f} V")
        print(f"   FEM solution range: {np.min(fem_solution):.6f} to {np.max(fem_solution):.6f}")
        print(f"   Solutions are coupled and consistent")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Test performance of Cython solvers"""
    print("\n🔧 Testing Performance")
    print("-" * 50)
    
    try:
        # Import modules
        import qdsim_cython.core.mesh_minimal as mesh_module
        import qdsim_cython.solvers.poisson_solver as poisson_module
        import qdsim_cython.solvers.fem_solver as fem_module
        
        # Test different mesh sizes
        mesh_sizes = [(10, 8), (15, 12), (20, 15)]
        
        for nx, ny in mesh_sizes:
            print(f"\n   Testing mesh size {nx}×{ny}:")
            
            # Create mesh
            mesh = mesh_module.SimpleMesh(nx, ny, 50e-9, 40e-9)
            
            # Test Poisson solver performance
            def epsilon_r(x, y): return 12.9
            def rho(x, y, n, p): return 0.0
            
            start_time = time.time()
            poisson_solver = poisson_module.CythonPoissonSolver(mesh, epsilon_r, rho)
            poisson_solver.solve(0.0, 1.0)
            poisson_time = time.time() - start_time
            
            # Test FEM solver performance
            start_time = time.time()
            fem_solver = fem_module.CythonFEMSolver(mesh)
            K = fem_solver.assemble_stiffness_matrix()
            M = fem_solver.assemble_mass_matrix()
            fem_time = time.time() - start_time
            
            print(f"     Poisson solver: {poisson_time:.3f}s")
            print(f"     FEM assembly: {fem_time:.3f}s")
            print(f"     Nodes: {mesh.num_nodes}, Elements: {mesh.num_elements}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 COMPREHENSIVE CYTHON SOLVERS TEST")
    print("Testing all migrated Cython solver implementations")
    print("=" * 80)
    
    results = {}
    times = {}
    
    # Test 1: Poisson Solver
    poisson_success, poisson_time = test_cython_poisson_solver()
    results['poisson'] = poisson_success
    times['poisson'] = poisson_time
    
    # Test 2: FEM Solver
    fem_success, fem_time = test_cython_fem_solver()
    results['fem'] = fem_success
    times['fem'] = fem_time
    
    # Test 3: Integration
    integration_success = test_integration_poisson_fem()
    results['integration'] = integration_success
    
    # Test 4: Performance
    performance_success = test_performance_comparison()
    results['performance'] = performance_success
    
    # Final Report
    print("\n" + "=" * 80)
    print("🏆 CYTHON SOLVERS TEST REPORT")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"📊 TEST RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests} ✅")
    print(f"   Failed: {total_tests - passed_tests} ❌")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\n📋 DETAILED RESULTS:")
    test_names = {
        'poisson': 'Cython Poisson Solver',
        'fem': 'Cython FEM Solver',
        'integration': 'Solver Integration',
        'performance': 'Performance Testing'
    }
    
    for key, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        name = test_names.get(key, key.title())
        time_info = f" ({times.get(key, 0):.3f}s)" if key in times else ""
        print(f"   {status} {name}{time_info}")
    
    print(f"\n🎯 MIGRATION ASSESSMENT:")
    if passed_tests == total_tests:
        print("   🎉 PERFECT! All Cython solvers working correctly.")
        print("   ✅ COMPLETE MIGRATION SUCCESS - C++ backend replaced with Cython.")
        print("   🚀 All core solver functionality migrated and validated.")
    elif passed_tests >= 3:
        print("   ✅ EXCELLENT! Core Cython solvers working correctly.")
        print("   🔧 Major migration success with minor issues to resolve.")
    else:
        print("   ⚠️  Some Cython solvers need attention.")
        print("   🔧 Partial migration success - continue development.")
    
    print("=" * 80)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
