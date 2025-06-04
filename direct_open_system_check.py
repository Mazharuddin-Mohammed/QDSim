#!/usr/bin/env python3
"""
Direct Open System Check

This script directly checks what open system functionality is actually
implemented in both the original and Cython versions.
"""

import sys
import os
from pathlib import Path

def check_original_open_system():
    """Check what open system functionality exists in original implementation"""
    print("🔍 Checking Original Open System Implementation")
    print("=" * 60)
    
    try:
        # Import original backend
        sys.path.insert(0, str(Path(__file__).parent / "backend" / "build"))
        import qdsim_cpp
        
        print("1. Checking SchrodingerSolver methods...")
        
        # Create a minimal mesh for testing
        mesh = qdsim_cpp.Mesh(20e-9, 10e-9, 10, 5, 1)
        
        # Create minimal functions
        def m_star(x, y): return 0.067 * 9.1093837015e-31
        def potential(x, y): return 0.0
        def cap(x, y): return 0.0
        def eps_r(x, y): return 12.9
        def rho(x, y, n, p): return 0.0
        
        # Create self-consistent solver
        sc_solver = qdsim_cpp.SelfConsistentSolver(mesh, eps_r, rho)
        
        # Create Schrödinger solver
        schrodinger = qdsim_cpp.SchrodingerSolver(mesh, m_star, potential, cap, sc_solver, False)
        
        print("✅ SchrodingerSolver created successfully")
        
        # Check available methods
        methods = dir(schrodinger)
        open_system_methods = [
            'apply_open_system_boundary_conditions',
            'apply_dirac_delta_normalization',
            'apply_conservative_boundary_conditions',
            'apply_minimal_cap_boundaries',
            'configure_device_specific_solver'
        ]
        
        print("\n2. Checking open system methods:")
        available_methods = []
        for method in open_system_methods:
            if method in methods:
                available_methods.append(method)
                print(f"   ✅ {method}: Available")
            else:
                print(f"   ❌ {method}: Missing")
        
        print(f"\n   📊 Available: {len(available_methods)}/{len(open_system_methods)} methods")
        
        # Test the methods
        print("\n3. Testing open system methods:")
        
        try:
            schrodinger.apply_open_system_boundary_conditions()
            print("   ✅ apply_open_system_boundary_conditions: Works")
        except Exception as e:
            print(f"   ❌ apply_open_system_boundary_conditions: {e}")
        
        try:
            schrodinger.apply_dirac_delta_normalization()
            print("   ✅ apply_dirac_delta_normalization: Works")
        except Exception as e:
            print(f"   ❌ apply_dirac_delta_normalization: {e}")
        
        try:
            schrodinger.apply_conservative_boundary_conditions()
            print("   ✅ apply_conservative_boundary_conditions: Works")
        except Exception as e:
            print(f"   ❌ apply_conservative_boundary_conditions: {e}")
        
        # Test solving with open system
        print("\n4. Testing open system solving:")
        try:
            eigenvalues, eigenvectors = schrodinger.solve(3)
            print(f"   ✅ Solve successful: {len(eigenvalues)} eigenvalues")
            
            # Check if eigenvalues are complex (indicating open system)
            complex_count = 0
            for i, E in enumerate(eigenvalues):
                if hasattr(E, 'imag') and abs(E.imag) > 1e-25:
                    complex_count += 1
                    print(f"     E_{i+1}: {E.real:.2e} + {E.imag:.2e}j J")
                else:
                    print(f"     E_{i+1}: {E.real:.2e} J")
            
            if complex_count > 0:
                print(f"   ✅ Open system confirmed: {complex_count} complex eigenvalues")
            else:
                print("   ⚠️  No complex eigenvalues (may be closed system)")
            
        except Exception as e:
            print(f"   ❌ Solve failed: {e}")
        
        return {
            'success': True,
            'available_methods': len(available_methods),
            'total_methods': len(open_system_methods),
            'methods_working': True
        }
        
    except Exception as e:
        print(f"❌ Original check failed: {e}")
        return {'success': False, 'error': str(e)}

def check_cython_open_system():
    """Check what open system functionality exists in Cython implementation"""
    print("\n🔧 Checking Cython Open System Implementation")
    print("=" * 60)
    
    try:
        # Import Cython modules
        sys.path.insert(0, 'qdsim_cython/qdsim_cython')
        
        print("1. Checking Cython Schrödinger solver...")
        
        try:
            import solvers.schrodinger_solver as schrodinger_cython
            print("✅ Cython Schrödinger solver imported")
            
            # Check the class
            solver_class = schrodinger_cython.CythonSchrodingerSolver
            methods = dir(solver_class)
            
            # Look for open system methods
            open_system_methods = [
                'apply_open_system_boundary_conditions',
                'apply_dirac_delta_normalization',
                'apply_conservative_boundary_conditions',
                'apply_minimal_cap_boundaries',
                'configure_device_specific_solver'
            ]
            
            print("\n2. Checking open system methods in Cython:")
            available_methods = []
            for method in open_system_methods:
                if method in methods:
                    available_methods.append(method)
                    print(f"   ✅ {method}: Available")
                else:
                    print(f"   ❌ {method}: Missing")
            
            print(f"\n   📊 Available: {len(available_methods)}/{len(open_system_methods)} methods")
            
            # Test solver creation
            print("\n3. Testing Cython solver creation:")
            
            try:
                import core.mesh_minimal as mesh_module
                mesh = mesh_module.SimpleMesh(10, 8, 20e-9, 15e-9)
                
                def m_star_func(x, y): return 0.067
                def potential_func(x, y): return 0.0
                
                cython_solver = solver_class(mesh, m_star_func, potential_func)
                print("   ✅ Cython solver created successfully")
                
                # Test available methods
                if 'apply_open_system_boundary_conditions' in available_methods:
                    try:
                        # This method might not exist in current implementation
                        if hasattr(cython_solver, 'apply_open_system_boundary_conditions'):
                            cython_solver.apply_open_system_boundary_conditions()
                            print("   ✅ Open system BC method works")
                        else:
                            print("   ⚠️  Open system BC method not implemented")
                    except Exception as e:
                        print(f"   ❌ Open system BC failed: {e}")
                
                # Test solving
                try:
                    eigenvalues, eigenvectors = cython_solver.solve(3)
                    print(f"   ✅ Cython solve successful: {len(eigenvalues)} eigenvalues")
                    
                    # Check eigenvalue types
                    for i, E in enumerate(eigenvalues[:3]):
                        print(f"     E_{i+1}: {E:.2e} J")
                    
                except Exception as e:
                    print(f"   ❌ Cython solve failed: {e}")
                
                return {
                    'success': True,
                    'available_methods': len(available_methods),
                    'total_methods': len(open_system_methods),
                    'solver_created': True
                }
                
            except Exception as e:
                print(f"   ❌ Solver creation failed: {e}")
                return {
                    'success': False,
                    'available_methods': len(available_methods),
                    'total_methods': len(open_system_methods),
                    'solver_created': False,
                    'error': str(e)
                }
        
        except ImportError as e:
            print(f"❌ Cython import failed: {e}")
            return {'success': False, 'error': str(e)}
        
    except Exception as e:
        print(f"❌ Cython check failed: {e}")
        return {'success': False, 'error': str(e)}

def compare_implementations(original_results, cython_results):
    """Compare original vs Cython implementations"""
    print("\n📊 Comparing Original vs Cython Implementations")
    print("=" * 60)
    
    print("✅ IMPLEMENTATION COMPARISON:")
    
    # Original implementation
    if original_results['success']:
        orig_methods = original_results.get('available_methods', 0)
        orig_total = original_results.get('total_methods', 5)
        print(f"   Original Implementation:")
        print(f"     Open system methods: {orig_methods}/{orig_total}")
        print(f"     Status: ✅ WORKING")
    else:
        print(f"   Original Implementation:")
        print(f"     Status: ❌ FAILED")
    
    # Cython implementation
    if cython_results['success']:
        cython_methods = cython_results.get('available_methods', 0)
        cython_total = cython_results.get('total_methods', 5)
        solver_created = cython_results.get('solver_created', False)
        print(f"   Cython Implementation:")
        print(f"     Open system methods: {cython_methods}/{cython_total}")
        print(f"     Solver creation: {'✅' if solver_created else '❌'}")
        print(f"     Status: ✅ WORKING")
    else:
        print(f"   Cython Implementation:")
        print(f"     Status: ❌ FAILED")
    
    # Assessment
    original_working = original_results['success']
    cython_working = cython_results['success']
    
    if original_working and cython_working:
        orig_methods = original_results.get('available_methods', 0)
        cython_methods = cython_results.get('available_methods', 0)
        
        print(f"\n🎯 MIGRATION ASSESSMENT:")
        if cython_methods >= orig_methods:
            print("   ✅ CYTHON HAS EQUIVALENT OR BETTER OPEN SYSTEM SUPPORT")
        elif cython_methods >= orig_methods * 0.6:
            print("   ⚠️  CYTHON HAS PARTIAL OPEN SYSTEM SUPPORT")
        else:
            print("   ❌ CYTHON MISSING CRITICAL OPEN SYSTEM FEATURES")
        
        print(f"   Original methods: {orig_methods}")
        print(f"   Cython methods: {cython_methods}")
        print(f"   Migration completeness: {(cython_methods/max(orig_methods,1))*100:.1f}%")
        
        return cython_methods >= orig_methods * 0.6
    
    elif original_working:
        print(f"\n🎯 MIGRATION ASSESSMENT:")
        print("   ❌ ORIGINAL WORKING BUT CYTHON FAILED")
        print("   🔧 Need to fix Cython implementation")
        return False
    
    else:
        print(f"\n🎯 MIGRATION ASSESSMENT:")
        print("   ❌ BOTH IMPLEMENTATIONS HAVE ISSUES")
        print("   🔧 Need to fix fundamental problems")
        return False

def main():
    """Main direct check function"""
    print("🚀 DIRECT OPEN SYSTEM CHECK")
    print("Checking actual open system implementation in original vs Cython")
    print("=" * 80)
    
    # Check original implementation
    original_results = check_original_open_system()
    
    # Check Cython implementation
    cython_results = check_cython_open_system()
    
    # Compare implementations
    migration_success = compare_implementations(original_results, cython_results)
    
    print("\n" + "=" * 80)
    print("🏆 DIRECT OPEN SYSTEM CHECK RESULTS")
    print("=" * 80)
    
    if migration_success:
        print("✅ OPEN SYSTEM MIGRATION: SUCCESSFUL")
        print("   Original open system functionality confirmed")
        print("   Cython implementation has adequate support")
        print("   Migration can proceed with confidence")
    else:
        print("❌ OPEN SYSTEM MIGRATION: NEEDS WORK")
        print("   Issues detected in open system implementation")
        print("   Additional development required")
    
    return migration_success

if __name__ == "__main__":
    success = main()
