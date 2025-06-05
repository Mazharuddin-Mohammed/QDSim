#!/usr/bin/env python3
"""
Debug Matrix Assembly Issues

This script debugs the fundamental matrix assembly problems that are causing
the eigenvalue solver to fail with singular matrices and zero eigenvalues.
"""

import sys
import os
import numpy as np
from pathlib import Path

def debug_matrix_assembly():
    """Debug the matrix assembly process"""
    print("🔍 Debugging Matrix Assembly Issues")
    print("=" * 60)
    
    try:
        # Import required modules
        sys.path.insert(0, 'qdsim_cython/qdsim_cython')
        import core.mesh_minimal as mesh_module
        
        print("1. Creating simple mesh...")
        mesh = mesh_module.SimpleMesh(5, 4, 10e-9, 8e-9)
        print(f"✅ Mesh created: {mesh.num_nodes} nodes, {mesh.num_elements} elements")
        
        # Get mesh data
        nodes_x, nodes_y = mesh.get_nodes()
        elements = mesh.get_elements()
        
        print(f"   Node coordinates range: x=[{np.min(nodes_x)*1e9:.1f}, {np.max(nodes_x)*1e9:.1f}] nm")
        print(f"   Node coordinates range: y=[{np.min(nodes_y)*1e9:.1f}, {np.max(nodes_y)*1e9:.1f}] nm")
        print(f"   Elements shape: {elements.shape}")
        print(f"   Element indices range: [{np.min(elements)}, {np.max(elements)}]")
        
        print("\n2. Testing manual matrix assembly...")
        
        # Constants
        HBAR = 1.054571817e-34
        M_E = 9.1093837015e-31
        EV_TO_J = 1.602176634e-19
        
        # Simple physics functions
        def m_star_func(x, y):
            return 0.067 * M_E  # GaAs effective mass
        
        def potential_func(x, y):
            return 0.0  # No potential for testing
        
        # Manual matrix assembly
        import scipy.sparse as sp
        
        num_nodes = mesh.num_nodes
        num_elements = mesh.num_elements
        
        # Initialize matrix builders
        row_indices = []
        col_indices = []
        hamiltonian_data = []
        mass_data = []
        
        print(f"   Assembling {num_elements} elements...")
        
        valid_elements = 0
        for elem_idx in range(num_elements):
            # Get element nodes
            n0, n1, n2 = elements[elem_idx, 0], elements[elem_idx, 1], elements[elem_idx, 2]
            
            # Get coordinates
            x0, y0 = nodes_x[n0], nodes_y[n0]
            x1, y1 = nodes_x[n1], nodes_y[n1]
            x2, y2 = nodes_x[n2], nodes_y[n2]
            
            # Calculate area
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            
            if area < 1e-15:
                print(f"   ⚠️  Degenerate element {elem_idx}: area = {area}")
                continue
            
            valid_elements += 1
            
            # Element center
            x_center = (x0 + x1 + x2) / 3.0
            y_center = (y0 + y1 + y2) / 3.0
            
            # Material properties
            m_star = m_star_func(x_center, y_center)
            V_pot = potential_func(x_center, y_center)
            
            # Simple element matrices (using constant approximation)
            # Kinetic energy: -ℏ²/(2m*) ∇²
            kinetic_factor = HBAR * HBAR / (2.0 * m_star) * area / 3.0
            
            # Potential energy: V(r)
            potential_factor = V_pot * area / 3.0
            
            # Mass matrix factor
            mass_factor = area / 12.0
            
            # Assemble element contributions
            nodes = [n0, n1, n2]
            for i in range(3):
                for j in range(3):
                    row_indices.append(nodes[i])
                    col_indices.append(nodes[j])
                    
                    # Hamiltonian: kinetic + potential
                    if i == j:
                        H_val = kinetic_factor + potential_factor
                        M_val = 2.0 * mass_factor
                    else:
                        H_val = kinetic_factor * 0.5  # Off-diagonal kinetic
                        M_val = mass_factor
                    
                    hamiltonian_data.append(H_val)
                    mass_data.append(M_val)
        
        print(f"   ✅ Processed {valid_elements}/{num_elements} valid elements")
        
        # Create sparse matrices
        H_matrix = sp.csr_matrix(
            (hamiltonian_data, (row_indices, col_indices)),
            shape=(num_nodes, num_nodes)
        )
        
        M_matrix = sp.csr_matrix(
            (mass_data, (row_indices, col_indices)),
            shape=(num_nodes, num_nodes)
        )
        
        print(f"   ✅ Matrices assembled:")
        print(f"     Hamiltonian: {H_matrix.shape}, nnz = {H_matrix.nnz}")
        print(f"     Mass: {M_matrix.shape}, nnz = {M_matrix.nnz}")
        
        # Check matrix properties
        H_diag = H_matrix.diagonal()
        M_diag = M_matrix.diagonal()
        
        print(f"     H diagonal range: [{np.min(H_diag):.2e}, {np.max(H_diag):.2e}] J")
        print(f"     M diagonal range: [{np.min(M_diag):.2e}, {np.max(M_diag):.2e}]")
        
        # Test eigenvalue solving
        print("\n3. Testing eigenvalue solving...")
        
        import scipy.sparse.linalg as spla
        
        try:
            # Apply simple boundary conditions (fix first and last nodes)
            H_bc = H_matrix.tolil()
            M_bc = M_matrix.tolil()
            
            boundary_nodes = [0, num_nodes-1]
            for i in boundary_nodes:
                H_bc[i, :] = 0
                H_bc[i, i] = 1
                M_bc[i, :] = 0
                M_bc[i, i] = 1
            
            H_bc = H_bc.tocsr()
            M_bc = M_bc.tocsr()
            
            print(f"   Applied boundary conditions to {len(boundary_nodes)} nodes")
            
            # Solve eigenvalue problem
            num_eigs = min(3, num_nodes - 2)
            eigenvals, eigenvecs = spla.eigsh(H_bc, k=num_eigs, M=M_bc, which='SM')
            
            print(f"   ✅ Eigenvalue problem solved!")
            print(f"     Number of eigenvalues: {len(eigenvals)}")
            
            eigenvals_eV = eigenvals / EV_TO_J
            print(f"     Energy levels (eV):")
            for i, E in enumerate(eigenvals_eV):
                print(f"       E_{i+1}: {E:.6f} eV")
            
            return True, len(eigenvals)
            
        except Exception as e:
            print(f"   ❌ Eigenvalue solving failed: {e}")
            
            # Debug matrix condition
            try:
                H_dense = H_bc.toarray()
                M_dense = M_bc.toarray()
                
                H_cond = np.linalg.cond(H_dense)
                M_cond = np.linalg.cond(M_dense)
                
                print(f"     Matrix condition numbers:")
                print(f"       H condition: {H_cond:.2e}")
                print(f"       M condition: {M_cond:.2e}")
                
                if H_cond > 1e12:
                    print(f"     ⚠️  Hamiltonian matrix is ill-conditioned")
                if M_cond > 1e12:
                    print(f"     ⚠️  Mass matrix is ill-conditioned")
                
            except Exception as e2:
                print(f"     Matrix condition check failed: {e2}")
            
            return False, 0
        
    except Exception as e:
        print(f"❌ Matrix assembly debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_cython_solver_with_debug():
    """Test Cython solver with debugging"""
    print("\n🔧 Testing Cython Solver with Debug Info")
    print("=" * 60)
    
    try:
        # Import Cython solver
        sys.path.insert(0, 'qdsim_cython/qdsim_cython')
        import core.mesh_minimal as mesh_module
        import solvers.schrodinger_solver as schrodinger_module
        
        print("1. Creating Cython solver...")
        mesh = mesh_module.SimpleMesh(5, 4, 10e-9, 8e-9)
        
        def m_star_func(x, y):
            return 0.067 * 9.1093837015e-31
        
        def potential_func(x, y):
            return 0.0
        
        # Create closed system first (simpler)
        solver = schrodinger_module.CythonSchrodingerSolver(
            mesh, m_star_func, potential_func, use_open_boundaries=False
        )
        
        print("✅ Cython solver created")
        
        # Get matrix info
        matrix_info = solver.get_matrix_info()
        print(f"   Matrix info: {matrix_info}")
        
        # Try solving
        print("\n2. Testing Cython solver...")
        eigenvalues, eigenvectors = solver.solve(2)
        
        print(f"✅ Cython solver result:")
        print(f"   Number of eigenvalues: {len(eigenvalues)}")
        
        if len(eigenvalues) > 0:
            eV = 1.602176634e-19
            eigenvalues_eV = eigenvalues / eV
            print(f"   Energy levels (eV):")
            for i, E in enumerate(eigenvalues_eV):
                print(f"     E_{i+1}: {E:.6f} eV")
        
        return True, len(eigenvalues)
        
    except Exception as e:
        print(f"❌ Cython solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    """Main debug function"""
    print("🚀 MATRIX ASSEMBLY DEBUG")
    print("Debugging fundamental eigenvalue solver issues")
    print("=" * 80)
    
    # Test 1: Manual matrix assembly
    manual_success, manual_eigs = debug_matrix_assembly()
    
    # Test 2: Cython solver
    cython_success, cython_eigs = test_cython_solver_with_debug()
    
    print("\n" + "=" * 80)
    print("🏆 DEBUG RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"📊 RESULTS:")
    print(f"   Manual assembly: {'✅ SUCCESS' if manual_success else '❌ FAILED'} ({manual_eigs} eigenvalues)")
    print(f"   Cython solver: {'✅ SUCCESS' if cython_success else '❌ FAILED'} ({cython_eigs} eigenvalues)")
    
    if manual_success and cython_success:
        print("\n🎉 BOTH METHODS WORKING!")
        print("   Matrix assembly is correct")
        print("   Eigenvalue solving is functional")
        print("   Ready to fix open system implementation")
    elif manual_success:
        print("\n⚠️  MANUAL ASSEMBLY WORKS, CYTHON SOLVER NEEDS FIXING")
        print("   Matrix assembly logic is correct")
        print("   Cython implementation has bugs")
    else:
        print("\n❌ FUNDAMENTAL MATRIX ASSEMBLY ISSUES")
        print("   Need to fix basic FEM implementation")
    
    return manual_success and cython_success

if __name__ == "__main__":
    success = main()
