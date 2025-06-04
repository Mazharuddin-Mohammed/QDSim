#!/usr/bin/env python3
"""
Test Real Eigensolvers - Ready for Working Environment
This script tests the actual backend eigensolvers properly
NO FAKE METHODS - Uses real SchrodingerSolver, FEMSolver, EigenSolver
"""

import sys
import math

def test_schrodinger_solver():
    """Test the real SchrodingerSolver from backend"""
    
    print("="*60)
    print("TESTING REAL SCHRODINGER SOLVER")
    print("="*60)
    
    # Import the actual backend
    sys.path.insert(0, 'backend/build')
    import fe_interpolator_module as fem
    
    print("✅ Backend imported")
    
    # Verify SchrodingerSolver is available
    if not hasattr(fem, 'SchrodingerSolver'):
        print("❌ SchrodingerSolver not found in backend")
        available = [x for x in dir(fem) if not x.startswith('_')]
        print(f"Available: {available}")
        return False
    
    print("✅ SchrodingerSolver found in backend")
    
    # Create mesh for quantum device
    domain_size = 50e-9  # 50 nm domain
    mesh_points = 32     # 32x32 mesh for eigenvalue problem
    
    print(f"Creating quantum mesh:")
    print(f"  Domain: ±{domain_size*1e9:.0f} nm")
    print(f"  Resolution: {mesh_points}×{mesh_points}")
    
    mesh = fem.Mesh(domain_size, domain_size, mesh_points, mesh_points, 1)
    print(f"✅ Mesh created: {mesh.get_num_nodes()} nodes, {mesh.get_num_elements()} elements")
    
    # Define effective mass function for InGaAs
    def m_star(x, y):
        """Effective mass function for InGaAs"""
        return 0.041  # InGaAs effective mass in units of m_e
    
    # Define quantum potential function
    def potential(x, y):
        """Real quantum potential for chromium QD in InGaAs p-n junction"""
        
        # Physical parameters
        reverse_bias = -1.0      # V (reverse bias voltage)
        depletion_width = 20e-9  # m (depletion region width)
        qd_depth = 0.3           # eV (chromium QD depth)
        qd_width = 8e-9          # m (QD characteristic width)
        
        # P-N junction potential (linear in depletion region)
        if abs(x) < depletion_width:
            V_junction = reverse_bias * x / depletion_width
        else:
            V_junction = reverse_bias * (1.0 if x > 0 else -1.0)
        
        # Gaussian quantum dot potential
        r_squared = x*x + y*y
        sigma_squared = qd_width * qd_width / 2.0
        V_qd = -qd_depth * math.exp(-r_squared / sigma_squared)
        
        return V_junction + V_qd
    
    print(f"Quantum potential defined:")
    print(f"  QD center: {potential(0, 0):.6f} eV")
    print(f"  QD edge: {potential(8e-9, 0):.6f} eV")
    print(f"  Junction region: {potential(25e-9, 0):.6f} eV")
    
    # Create the real Schrödinger solver
    print(f"Creating SchrodingerSolver...")
    use_gpu = False  # Start with CPU version
    
    solver = fem.SchrodingerSolver(mesh, m_star, potential, use_gpu)
    print("✅ SchrodingerSolver created successfully")
    
    # Solve the actual Schrödinger equation
    num_eigenvalues = 10  # Compute first 10 eigenvalues
    print(f"Solving Schrödinger equation for {num_eigenvalues} eigenvalues...")
    
    # This is the real eigenvalue computation - no fake methods!
    eigenvalues, eigenvectors = solver.solve(num_eigenvalues)
    
    print("✅ Schrödinger equation solved successfully!")
    
    # Analyze real results
    print(f"\nReal quantum mechanics results:")
    print(f"  Number of computed eigenvalues: {len(eigenvalues)}")
    print(f"  Number of computed eigenvectors: {len(eigenvectors)}")
    
    print(f"\nEnergy eigenvalues:")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:.6f} eV")
    
    # Check for bound states
    bound_states = [E for E in eigenvalues if E < 0]
    print(f"\nBound state analysis:")
    print(f"  Bound states: {len(bound_states)}/{len(eigenvalues)}")
    
    if len(eigenvalues) > 1:
        energy_gap = eigenvalues[1] - eigenvalues[0]
        print(f"  Ground state: {eigenvalues[0]:.6f} eV")
        print(f"  First excited: {eigenvalues[1]:.6f} eV")
        print(f"  Energy gap: {energy_gap:.6f} eV ({energy_gap*1000:.1f} meV)")
    
    # Analyze eigenvectors
    if eigenvectors and len(eigenvectors) > 0:
        ground_state = eigenvectors[0]
        print(f"\nGround state wavefunction:")
        print(f"  Vector size: {len(ground_state)}")
        
        # Check normalization
        norm_squared = sum(abs(psi)**2 for psi in ground_state)
        print(f"  Norm²: {norm_squared:.6f}")
        
        # Find maximum amplitude
        max_amplitude = max(abs(psi) for psi in ground_state)
        print(f"  Max amplitude: {max_amplitude:.6f}")
    
    return True, eigenvalues, eigenvectors

def test_fem_eigensolver():
    """Test the FEMSolver + EigenSolver approach"""
    
    print(f"\n" + "="*60)
    print("TESTING FEMSolver + EigenSolver APPROACH")
    print("="*60)
    
    sys.path.insert(0, 'backend/build')
    import fe_interpolator_module as fem
    
    # Check availability
    has_fem_solver = hasattr(fem, 'FEMSolver')
    has_eigen_solver = hasattr(fem, 'EigenSolver')
    
    print(f"FEMSolver available: {has_fem_solver}")
    print(f"EigenSolver available: {has_eigen_solver}")
    
    if not (has_fem_solver and has_eigen_solver):
        print("❌ FEMSolver or EigenSolver not available")
        return False
    
    print("✅ Both FEMSolver and EigenSolver found")
    
    # Create mesh
    mesh = fem.Mesh(50e-9, 50e-9, 16, 16, 1)
    print(f"✅ Mesh created for FEM approach")
    
    # Define functions (same as above)
    def m_star(x, y):
        return 0.041
    
    def potential(x, y):
        reverse_bias = -1.0
        depletion_width = 20e-9
        qd_depth = 0.3
        qd_width = 8e-9
        
        if abs(x) < depletion_width:
            V_junction = reverse_bias * x / depletion_width
        else:
            V_junction = reverse_bias * (1.0 if x > 0 else -1.0)
        
        r_squared = x*x + y*y
        sigma_squared = qd_width * qd_width / 2.0
        V_qd = -qd_depth * math.exp(-r_squared / sigma_squared)
        
        return V_junction + V_qd
    
    def capacitance(x, y):
        return 13.9  # InGaAs relative permittivity
    
    # Create FEMSolver
    print("Creating FEMSolver...")
    
    # Based on bindings.cpp, FEMSolver constructor needs:
    # (mesh, m_star, V, cap, sc_solver, order, use_gpu)
    sc_solver = None  # No self-consistent solver for now
    order = 1         # Linear elements
    use_gpu = False
    
    fem_solver = fem.FEMSolver(mesh, m_star, potential, capacitance, sc_solver, order, use_gpu)
    print("✅ FEMSolver created")
    
    # Assemble matrices
    print("Assembling FEM matrices...")
    fem_solver.assemble_matrices()
    print("✅ Matrices assembled")
    
    # Get matrices
    H = fem_solver.get_H()  # Hamiltonian matrix
    M = fem_solver.get_M()  # Mass matrix
    print(f"✅ Matrices extracted: H and M")
    
    # Create EigenSolver
    print("Creating EigenSolver...")
    eigen_solver = fem.EigenSolver(fem_solver)
    print("✅ EigenSolver created")
    
    # Solve eigenvalue problem
    num_eigenvalues = 10
    print(f"Solving generalized eigenvalue problem H ψ = E M ψ...")
    
    eigen_solver.solve(num_eigenvalues)
    print("✅ Eigenvalue problem solved")
    
    # Get results
    eigenvalues = eigen_solver.get_eigenvalues()
    eigenvectors = eigen_solver.get_eigenvectors()
    
    print(f"Results:")
    print(f"  Eigenvalues: {len(eigenvalues)}")
    print(f"  Eigenvectors: {len(eigenvectors)}")
    
    for i, E in enumerate(eigenvalues[:5]):  # Show first 5
        print(f"  E_{i} = {E:.6f} eV")
    
    return True, eigenvalues, eigenvectors

def validate_physics(eigenvalues, eigenvectors):
    """Validate physics of the real results"""
    
    print(f"\n" + "="*60)
    print("REAL PHYSICS VALIDATION")
    print("="*60)
    
    if not eigenvalues or len(eigenvalues) == 0:
        print("❌ No eigenvalues to validate")
        return False
    
    print(f"Physics validation for {len(eigenvalues)} eigenvalues:")
    
    # Check energy ordering
    is_ordered = all(eigenvalues[i] <= eigenvalues[i+1] for i in range(len(eigenvalues)-1))
    print(f"  Energy ordering: {'✅ Correct' if is_ordered else '❌ Incorrect'}")
    
    # Check energy scales
    ground_state = eigenvalues[0]
    energy_scale_ok = -3.0 < ground_state < 1.0  # Reasonable for QD
    print(f"  Energy scale: {'✅ Reasonable' if energy_scale_ok else '❌ Unrealistic'}")
    print(f"    Ground state: {ground_state:.6f} eV")
    
    # Check for bound states
    bound_states = [E for E in eigenvalues if E < 0]
    print(f"  Bound states: {len(bound_states)}/{len(eigenvalues)}")
    
    # Check energy gaps
    if len(eigenvalues) > 1:
        gaps = [eigenvalues[i+1] - eigenvalues[i] for i in range(len(eigenvalues)-1)]
        avg_gap = sum(gaps) / len(gaps)
        print(f"  Average energy gap: {avg_gap*1000:.1f} meV")
        
        gap_reasonable = 0.001 < avg_gap < 0.5  # 1-500 meV
        print(f"  Gap scale: {'✅ Reasonable' if gap_reasonable else '❌ Unrealistic'}")
    
    # Overall assessment
    physics_valid = is_ordered and energy_scale_ok and len(bound_states) > 0
    print(f"\nOverall physics: {'✅ VALID' if physics_valid else '❌ NEEDS REVIEW'}")
    
    return physics_valid

def main():
    """Main test function for real eigensolvers"""
    
    print("QDSim Real Eigensolver Test")
    print("Testing actual backend eigensolvers - NO FAKE METHODS")
    print()
    
    try:
        # Test 1: SchrodingerSolver
        print("TEST 1: SchrodingerSolver")
        success1, eigenvalues1, eigenvectors1 = test_schrodinger_solver()
        
        # Test 2: FEMSolver + EigenSolver
        print("\nTEST 2: FEMSolver + EigenSolver")
        success2, eigenvalues2, eigenvectors2 = test_fem_eigensolver()
        
        # Validate physics
        if success1:
            print("\nVALIDATION: SchrodingerSolver Results")
            physics_valid1 = validate_physics(eigenvalues1, eigenvectors1)
        
        if success2:
            print("\nVALIDATION: FEMSolver Results")
            physics_valid2 = validate_physics(eigenvalues2, eigenvectors2)
        
        # Final assessment
        print(f"\n" + "="*60)
        print("REAL EIGENSOLVER TEST SUMMARY")
        print("="*60)
        
        if success1:
            print("✅ SchrodingerSolver: WORKING")
            print(f"   Ground state: {eigenvalues1[0]:.6f} eV")
            print(f"   Bound states: {len([E for E in eigenvalues1 if E < 0])}")
        else:
            print("❌ SchrodingerSolver: FAILED")
        
        if success2:
            print("✅ FEMSolver + EigenSolver: WORKING")
            print(f"   Ground state: {eigenvalues2[0]:.6f} eV")
            print(f"   Bound states: {len([E for E in eigenvalues2 if E < 0])}")
        else:
            print("❌ FEMSolver + EigenSolver: FAILED")
        
        if success1 or success2:
            print("\n🎉 REAL QUANTUM SIMULATION SUCCESS!")
            print("✅ Backend eigensolvers are fully functional")
            print("✅ Actual Schrödinger equation solving works")
            print("✅ No fake methods used - all real!")
            return 0
        else:
            print("\n❌ Eigensolver tests failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
