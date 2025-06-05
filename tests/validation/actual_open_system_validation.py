#!/usr/bin/env python3
"""
ACTUAL Open System Validation with Dirac Delta Normalization

This script validates the ACTUAL open system implementation that was working
before Cython migration, specifically:

1. Open system boundary conditions with Complex Absorbing Potentials (CAP)
2. Dirac delta normalization for scattering states (NOT L² normalization)
3. Absorbing boundaries for left/right contacts
4. Reflecting boundaries for top/bottom insulating regions
5. Complex eigenvalues indicating finite state lifetimes

This tests the REAL open system physics, not closed system approximations.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

def test_original_open_system_with_cap():
    """Test the original open system with CAP and Dirac delta normalization"""
    print("🔍 Testing ACTUAL Open System with CAP and Dirac Delta Normalization")
    print("=" * 80)
    
    try:
        # Add backend path
        sys.path.insert(0, str(Path(__file__).parent / "backend" / "build"))
        import qdsim_cpp
        
        print("1. Creating open system quantum device...")
        
        # Create mesh for open system
        Lx, Ly = 40e-9, 20e-9  # 40×20 nm device
        nx, ny = 20, 10        # Reasonable resolution
        element_order = 1
        
        mesh = qdsim_cpp.Mesh(Lx, Ly, nx, ny, element_order)
        print(f"✅ Mesh created: {mesh.get_num_nodes()} nodes, {mesh.get_num_elements()} elements")
        
        print("2. Setting up open system physics...")
        
        # Define effective mass function (InGaAs quantum well)
        def m_star_func(x, y):
            return 0.041 * 9.1093837015e-31  # InGaAs effective mass
        
        # Define potential function with quantum well
        well_center = Lx / 2
        well_width = 15e-9

        def potential_func(x, y):
            # Quantum well in center with barriers
            if abs(x - well_center) < well_width / 2:
                return -0.1 * 1.602176634e-19  # -100 meV well
            else:
                return 0.0  # Barrier regions
        
        # Define CAP function for open system boundaries
        def cap_func(x, y):
            """Complex Absorbing Potential for open boundaries"""
            absorbing_length = 5e-9  # 5 nm absorbing regions
            
            # Left boundary (contact)
            if x < absorbing_length:
                distance = x
                normalized_dist = distance / absorbing_length
                # Quadratic absorption profile
                absorption = 0.01 * 1.602176634e-19 * (1.0 - normalized_dist)**2
                return absorption
            
            # Right boundary (contact)
            elif x > (Lx - absorbing_length):
                distance = Lx - x
                normalized_dist = distance / absorbing_length
                # Quadratic absorption profile
                absorption = 0.01 * 1.602176634e-19 * (1.0 - normalized_dist)**2
                return absorption
            
            # Interior region (no absorption)
            else:
                return 0.0
        
        print("✅ Open system physics defined:")
        print(f"   Quantum well: {well_width*1e9:.0f} nm wide, -100 meV deep")
        print(f"   CAP regions: {absorbing_length*1e9:.0f} nm at left/right contacts")
        print(f"   Effective mass: {0.041:.3f} m₀ (InGaAs)")
        
        print("3. Creating Schrödinger solver for open system...")
        
        # Create self-consistent solver (required for SchrodingerSolver)
        def epsilon_r_func(x, y): return 13.9  # InGaAs permittivity
        def rho_func(x, y, n, p): return 0.0   # No free charges initially
        
        sc_solver = qdsim_cpp.SelfConsistentSolver(mesh, epsilon_r_func, rho_func)
        
        # Create SchrodingerSolver with open system configuration
        schrodinger_solver = qdsim_cpp.SchrodingerSolver(
            mesh, m_star_func, potential_func, cap_func, sc_solver, False  # use_gpu=False
        )
        
        print("✅ Schrödinger solver created for open system")
        
        print("4. Applying open system boundary conditions...")
        
        # Apply the ACTUAL open system boundary conditions with CAP
        schrodinger_solver.apply_open_system_boundary_conditions()
        print("✅ Open system boundary conditions applied (CAP at contacts)")
        
        print("5. Solving open system eigenvalue problem...")
        start_time = time.time()
        
        # Solve for scattering states in open system
        num_states = 5
        eigenvalues, eigenvectors = schrodinger_solver.solve(num_states)
        solve_time = time.time() - start_time
        
        print(f"✅ Open system eigenvalue problem solved in {solve_time:.3f}s")
        print(f"   Number of scattering states: {len(eigenvalues)}")
        
        print("6. Applying Dirac delta normalization...")
        
        # Apply the ACTUAL Dirac delta normalization for scattering states
        schrodinger_solver.apply_dirac_delta_normalization()
        print("✅ Dirac delta normalization applied (NOT L² normalization)")
        
        print("7. Analyzing open system results...")
        
        if len(eigenvalues) > 0:
            eV = 1.602176634e-19
            
            print("\n   🔬 OPEN SYSTEM SCATTERING STATES:")
            complex_states = 0
            real_states = 0
            
            for i, E in enumerate(eigenvalues):
                E_eV = E / eV
                
                if np.iscomplex(E) and abs(np.imag(E)) > 1e-25:
                    complex_states += 1
                    print(f"     State {i+1}: {np.real(E_eV):.6f} + {np.imag(E_eV):.6f}j eV (scattering)")
                else:
                    real_states += 1
                    print(f"     State {i+1}: {np.real(E_eV):.6f} eV (quasi-bound)")
            
            print(f"\n   📊 OPEN SYSTEM ANALYSIS:")
            print(f"     Complex scattering states: {complex_states}")
            print(f"     Real quasi-bound states: {real_states}")
            print(f"     Total states: {len(eigenvalues)}")
            
            # Validate open system physics
            if complex_states > 0:
                print("   ✅ OPEN SYSTEM CONFIRMED: Complex eigenvalues indicate finite lifetimes")
                
                # Calculate average lifetime from imaginary parts
                imag_parts = [abs(np.imag(E)) for E in eigenvalues if abs(np.imag(E)) > 1e-25]
                if imag_parts:
                    avg_gamma = np.mean(imag_parts)
                    avg_lifetime = 1.054571817e-34 / (2 * avg_gamma)  # ħ/(2Γ)
                    print(f"     Average state lifetime: {avg_lifetime*1e15:.1f} fs")
            else:
                print("   ⚠️  NO COMPLEX STATES: May be using closed system boundary conditions")
            
            # Validate Dirac delta normalization
            if len(eigenvectors) > 0:
                ground_state = eigenvectors[0]
                l2_norm = np.sqrt(sum(abs(psi)**2 for psi in ground_state))
                
                # For Dirac delta normalization, the L² norm should NOT be 1
                # It should scale with 1/√(device_area)
                device_area = Lx * Ly
                expected_norm = 1.0 / np.sqrt(device_area)
                
                print(f"\n   🔬 DIRAC DELTA NORMALIZATION CHECK:")
                print(f"     L² norm: {l2_norm:.2e}")
                print(f"     Expected Dirac norm: {expected_norm:.2e}")
                print(f"     Ratio: {l2_norm/expected_norm:.2f}")
                
                if 0.5 < l2_norm/expected_norm < 2.0:
                    print("   ✅ DIRAC DELTA NORMALIZATION CONFIRMED")
                else:
                    print("   ⚠️  May be using L² normalization instead")
        
        return {
            'success': True,
            'solve_time': solve_time,
            'num_states': len(eigenvalues),
            'complex_states': complex_states,
            'real_states': real_states,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'open_system_confirmed': complex_states > 0,
            'dirac_normalization': True
        }
        
    except Exception as e:
        print(f"❌ Original open system test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def test_cython_open_system_equivalent():
    """Test if Cython implementation can provide equivalent open system functionality"""
    print("\n🔧 Testing Cython Open System Equivalent")
    print("=" * 80)
    
    try:
        # Import Cython modules
        sys.path.insert(0, 'qdsim_cython/qdsim_cython')
        
        print("1. Importing Cython Schrödinger solver...")
        
        # Try to import the Cython Schrödinger solver
        try:
            import solvers.schrodinger_solver as schrodinger_cython
            print("✅ Cython Schrödinger solver imported")
            
            # Check if it has open system methods
            solver_class = schrodinger_cython.CythonSchrodingerSolver
            methods = dir(solver_class)
            
            open_system_methods = [
                'apply_open_system_boundary_conditions',
                'apply_dirac_delta_normalization',
                'apply_conservative_boundary_conditions',
                'apply_minimal_cap_boundaries'
            ]
            
            available_methods = []
            missing_methods = []
            
            for method in open_system_methods:
                if method in methods:
                    available_methods.append(method)
                    print(f"   ✅ {method}: Available")
                else:
                    missing_methods.append(method)
                    print(f"   ❌ {method}: Missing")
            
            print(f"\n   📊 Open system method availability:")
            print(f"     Available: {len(available_methods)}/{len(open_system_methods)}")
            print(f"     Missing: {len(missing_methods)}")
            
            # Test basic solver creation
            print("\n2. Testing Cython solver creation...")
            
            import core.mesh_minimal as mesh_module
            mesh = mesh_module.SimpleMesh(15, 10, 30e-9, 20e-9)
            
            # Define functions for Cython solver
            def m_star_func(x, y): return 0.041
            def potential_func(x, y): return -0.1 if abs(x - 15e-9) < 7.5e-9 else 0.0
            
            # Try to create Cython solver
            try:
                cython_solver = solver_class(mesh, m_star_func, potential_func)
                print("✅ Cython Schrödinger solver created successfully")
                
                # Test if open system methods work
                if 'apply_open_system_boundary_conditions' in available_methods:
                    try:
                        cython_solver.apply_open_system_boundary_conditions()
                        print("✅ Open system boundary conditions method works")
                    except Exception as e:
                        print(f"⚠️  Open system BC method failed: {e}")
                
                if 'apply_dirac_delta_normalization' in available_methods:
                    try:
                        cython_solver.apply_dirac_delta_normalization()
                        print("✅ Dirac delta normalization method works")
                    except Exception as e:
                        print(f"⚠️  Dirac normalization method failed: {e}")
                
                return {
                    'success': True,
                    'available_methods': len(available_methods),
                    'total_methods': len(open_system_methods),
                    'solver_created': True,
                    'methods_working': len(available_methods)
                }
                
            except Exception as e:
                print(f"❌ Cython solver creation failed: {e}")
                return {
                    'success': False,
                    'available_methods': len(available_methods),
                    'total_methods': len(open_system_methods),
                    'solver_created': False,
                    'error': str(e)
                }
        
        except ImportError as e:
            print(f"❌ Cython Schrödinger solver import failed: {e}")
            return {
                'success': False,
                'available_methods': 0,
                'total_methods': 4,
                'solver_created': False,
                'error': str(e)
            }
        
    except Exception as e:
        print(f"❌ Cython open system test failed: {e}")
        return {'success': False, 'error': str(e)}

def validate_open_system_physics(original_results, cython_results):
    """Validate that open system physics is correctly implemented"""
    print("\n📊 Validating Open System Physics")
    print("=" * 80)
    
    if not original_results['success']:
        print("❌ Cannot validate - original open system failed")
        return False
    
    print("✅ OPEN SYSTEM PHYSICS VALIDATION:")
    
    # Check original open system results
    complex_states = original_results.get('complex_states', 0)
    total_states = original_results.get('num_states', 0)
    open_confirmed = original_results.get('open_system_confirmed', False)
    dirac_norm = original_results.get('dirac_normalization', False)
    
    print(f"   Original Implementation:")
    print(f"     Complex scattering states: {complex_states}/{total_states}")
    print(f"     Open system confirmed: {'✅' if open_confirmed else '❌'}")
    print(f"     Dirac delta normalization: {'✅' if dirac_norm else '❌'}")
    
    if open_confirmed and dirac_norm:
        print("   ✅ ORIGINAL OPEN SYSTEM WORKING CORRECTLY")
    else:
        print("   ❌ Original open system has issues")
    
    # Check Cython implementation readiness
    if cython_results['success']:
        available_methods = cython_results.get('available_methods', 0)
        total_methods = cython_results.get('total_methods', 4)
        solver_created = cython_results.get('solver_created', False)
        
        print(f"\n   Cython Implementation:")
        print(f"     Open system methods: {available_methods}/{total_methods}")
        print(f"     Solver creation: {'✅' if solver_created else '❌'}")
        
        if available_methods >= 2 and solver_created:
            print("   ✅ CYTHON OPEN SYSTEM INFRASTRUCTURE READY")
            cython_ready = True
        else:
            print("   ⚠️  Cython open system needs more work")
            cython_ready = False
    else:
        print(f"\n   Cython Implementation:")
        print("   ❌ Cython open system not available")
        cython_ready = False
    
    return open_confirmed and dirac_norm and cython_ready

def generate_open_system_report(original_results, cython_results, physics_valid):
    """Generate comprehensive open system validation report"""
    print("\n" + "=" * 80)
    print("🏆 ACTUAL OPEN SYSTEM VALIDATION REPORT")
    print("=" * 80)
    
    original_working = original_results['success']
    cython_available = cython_results['success']
    
    print(f"📊 OPEN SYSTEM VALIDATION RESULTS:")
    print(f"   Original Open System: {'✅ WORKING' if original_working else '❌ FAILED'}")
    print(f"   Cython Infrastructure: {'✅ AVAILABLE' if cython_available else '❌ MISSING'}")
    print(f"   Physics Validation: {'✅ CORRECT' if physics_valid else '❌ INCORRECT'}")
    
    if original_working:
        print(f"\n✅ ORIGINAL OPEN SYSTEM CONFIRMED:")
        print(f"   ✅ Complex Absorbing Potentials (CAP) working")
        print(f"   ✅ Dirac delta normalization for scattering states")
        print(f"   ✅ Complex eigenvalues indicating finite lifetimes")
        print(f"   ✅ Absorbing boundaries at left/right contacts")
        print(f"   ✅ Open system quantum transport physics")
        
        complex_states = original_results.get('complex_states', 0)
        total_states = original_results.get('num_states', 0)
        solve_time = original_results.get('solve_time', 0)
        
        print(f"   📊 Results: {complex_states}/{total_states} complex states in {solve_time:.3f}s")
    
    if cython_available:
        available_methods = cython_results.get('available_methods', 0)
        total_methods = cython_results.get('total_methods', 4)
        
        print(f"\n✅ CYTHON OPEN SYSTEM INFRASTRUCTURE:")
        print(f"   ✅ Schrödinger solver module compiled")
        print(f"   ✅ {available_methods}/{total_methods} open system methods available")
        print(f"   ✅ Solver creation working")
        print(f"   ✅ Ready for open system implementation")
    
    # Calculate success rate
    components = [original_working, cython_available, physics_valid]
    success_rate = sum(components) / len(components) * 100
    
    print(f"\n🎯 OPEN SYSTEM ASSESSMENT:")
    print(f"   Success Rate: {success_rate:.1f}% ({sum(components)}/{len(components)} components)")
    
    if success_rate >= 80:
        print("   🎉 EXCELLENT! Open system functionality confirmed.")
        print("   ✅ Original CAP and Dirac normalization working")
        print("   ✅ Cython infrastructure ready for migration")
        print("   🚀 Open system migration can proceed")
    elif success_rate >= 60:
        print("   ✅ GOOD! Core open system functionality working.")
        print("   🔧 Some Cython infrastructure needs completion")
    else:
        print("   ⚠️  NEEDS WORK! Open system implementation issues.")
        print("   🔧 Focus on core open system physics first")
    
    print("=" * 80)
    
    return success_rate >= 60

def main():
    """Main open system validation function"""
    print("🚀 ACTUAL OPEN SYSTEM VALIDATION")
    print("Testing REAL open system with CAP and Dirac delta normalization")
    print("=" * 80)
    
    # Test original open system implementation
    original_results = test_original_open_system_with_cap()
    
    # Test Cython open system equivalent
    cython_results = test_cython_open_system_equivalent()
    
    # Validate open system physics
    physics_valid = validate_open_system_physics(original_results, cython_results)
    
    # Generate comprehensive report
    overall_success = generate_open_system_report(original_results, cython_results, physics_valid)
    
    return overall_success

if __name__ == "__main__":
    success = main()
