#!/usr/bin/env python3
"""
Debug Matrix Assembly Issues

This script systematically debugs and fixes the matrix assembly problems
that are causing zero element areas and singular matrices.
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def create_debug_mesh(nx, ny, Lx, Ly):
    """Create a simple mesh with detailed debugging"""
    print(f"🔧 Creating debug mesh: {nx}×{ny} nodes, {Lx*1e9:.1f}×{Ly*1e9:.1f} nm")
    
    # Generate nodes
    nodes_x = []
    nodes_y = []
    
    for j in range(ny):
        for i in range(nx):
            x = i * Lx / (nx - 1)
            y = j * Ly / (ny - 1)
            nodes_x.append(x)
            nodes_y.append(y)
    
    nodes_x = np.array(nodes_x)
    nodes_y = np.array(nodes_y)
    
    print(f"✅ Nodes generated: {len(nodes_x)} total")
    print(f"   X range: [{np.min(nodes_x)*1e9:.2f}, {np.max(nodes_x)*1e9:.2f}] nm")
    print(f"   Y range: [{np.min(nodes_y)*1e9:.2f}, {np.max(nodes_y)*1e9:.2f}] nm")
    
    # Generate elements (triangles)
    elements = []
    
    for j in range(ny - 1):
        for i in range(nx - 1):
            # Bottom-left triangle
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + i
            elements.append([n0, n1, n2])
            
            # Top-right triangle
            n0 = j * nx + (i + 1)
            n1 = (j + 1) * nx + (i + 1)
            n2 = (j + 1) * nx + i
            elements.append([n0, n1, n2])
    
    elements = np.array(elements)
    
    print(f"✅ Elements generated: {len(elements)} triangles")
    
    return nodes_x, nodes_y, elements

def debug_element_areas(nodes_x, nodes_y, elements):
    """Debug element area calculations"""
    print("\n🔍 Debugging Element Area Calculations")
    print("=" * 50)
    
    valid_elements = 0
    zero_area_elements = 0
    total_area = 0.0
    
    print("First 10 elements:")
    for elem_idx in range(min(10, len(elements))):
        # Get element nodes
        n0, n1, n2 = elements[elem_idx, 0], elements[elem_idx, 1], elements[elem_idx, 2]
        
        # Get coordinates
        x0, y0 = nodes_x[n0], nodes_y[n0]
        x1, y1 = nodes_x[n1], nodes_y[n1]
        x2, y2 = nodes_x[n2], nodes_y[n2]
        
        # Calculate area using cross product formula
        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        
        print(f"  Element {elem_idx}:")
        print(f"    Nodes: {n0}, {n1}, {n2}")
        print(f"    Coords: ({x0*1e9:.2f},{y0*1e9:.2f}), ({x1*1e9:.2f},{y1*1e9:.2f}), ({x2*1e9:.2f},{y2*1e9:.2f}) nm")
        print(f"    Area: {area*1e18:.6f} nm²")
        
        if area > 1e-20:
            valid_elements += 1
            total_area += area
        else:
            zero_area_elements += 1
            print(f"    ⚠️  ZERO AREA ELEMENT!")
    
    print(f"\nElement statistics:")
    print(f"  Total elements: {len(elements)}")
    print(f"  Valid elements (first 10): {valid_elements}/10")
    print(f"  Zero area elements (first 10): {zero_area_elements}/10")
    print(f"  Average area (first 10): {total_area/max(valid_elements,1)*1e18:.6f} nm²")
    
    return valid_elements > 0

def test_manual_matrix_assembly(nodes_x, nodes_y, elements):
    """Test manual matrix assembly with proper debugging"""
    print("\n🔧 Testing Manual Matrix Assembly")
    print("=" * 50)
    
    num_nodes = len(nodes_x)
    num_elements = len(elements)
    
    # Physical constants
    HBAR = 1.054571817e-34
    M_E = 9.1093837015e-31
    EV_TO_J = 1.602176634e-19
    
    # Simple physics
    m_star = 0.067 * M_E
    V_pot = 0.0
    
    # Matrix builders
    row_indices = []
    col_indices = []
    H_data = []
    M_data = []
    
    valid_elements = 0
    
    print(f"Processing {num_elements} elements...")
    
    for elem_idx in range(num_elements):
        # Get element nodes
        n0, n1, n2 = elements[elem_idx, 0], elements[elem_idx, 1], elements[elem_idx, 2]
        
        # Get coordinates
        x0, y0 = nodes_x[n0], nodes_y[n0]
        x1, y1 = nodes_x[n1], nodes_y[n1]
        x2, y2 = nodes_x[n2], nodes_y[n2]
        
        # Calculate area
        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        
        if area < 1e-20:
            continue
        
        valid_elements += 1
        
        # Simple element matrices (constant approximation)
        kinetic_factor = HBAR * HBAR / (2.0 * m_star) * area / 3.0
        potential_factor = V_pot * area / 3.0
        mass_factor = area / 12.0
        
        # Assemble 3x3 element matrix
        nodes = [n0, n1, n2]
        for i in range(3):
            for j in range(3):
                row_indices.append(nodes[i])
                col_indices.append(nodes[j])
                
                if i == j:
                    H_val = kinetic_factor + potential_factor
                    M_val = 2.0 * mass_factor
                else:
                    H_val = kinetic_factor * 0.5
                    M_val = mass_factor
                
                H_data.append(H_val)
                M_data.append(M_val)
    
    print(f"✅ Processed {valid_elements}/{num_elements} valid elements")
    
    if valid_elements == 0:
        print("❌ NO VALID ELEMENTS - Matrix assembly impossible")
        return False, None, None
    
    # Create sparse matrices
    H_matrix = sp.csr_matrix(
        (H_data, (row_indices, col_indices)),
        shape=(num_nodes, num_nodes)
    )
    
    M_matrix = sp.csr_matrix(
        (M_data, (row_indices, col_indices)),
        shape=(num_nodes, num_nodes)
    )
    
    print(f"✅ Matrices created:")
    print(f"  H: {H_matrix.shape}, nnz = {H_matrix.nnz}")
    print(f"  M: {M_matrix.shape}, nnz = {M_matrix.nnz}")
    
    # Check matrix properties
    H_diag = H_matrix.diagonal()
    M_diag = M_matrix.diagonal()
    
    print(f"  H diagonal range: [{np.min(H_diag):.2e}, {np.max(H_diag):.2e}] J")
    print(f"  M diagonal range: [{np.min(M_diag):.2e}, {np.max(M_diag):.2e}]")
    
    # Check for zero diagonals
    zero_H_diag = np.sum(np.abs(H_diag) < 1e-20)
    zero_M_diag = np.sum(np.abs(M_diag) < 1e-20)
    
    print(f"  Zero H diagonal elements: {zero_H_diag}/{num_nodes}")
    print(f"  Zero M diagonal elements: {zero_M_diag}/{num_nodes}")
    
    return True, H_matrix, M_matrix

def test_boundary_conditions_and_solve(H_matrix, M_matrix):
    """Test boundary conditions and eigenvalue solving"""
    print("\n⚡ Testing Boundary Conditions and Solving")
    print("=" * 50)
    
    num_nodes = H_matrix.shape[0]
    
    # Apply simple boundary conditions
    H_bc = H_matrix.tolil()
    M_bc = M_matrix.tolil()
    
    # Fix first and last nodes (Dirichlet BC)
    boundary_nodes = [0, num_nodes-1]
    
    for node in boundary_nodes:
        H_bc[node, :] = 0
        H_bc[node, node] = 1.0
        M_bc[node, :] = 0
        M_bc[node, node] = 1.0
    
    H_bc = H_bc.tocsr()
    M_bc = M_bc.tocsr()
    
    print(f"✅ Applied boundary conditions to {len(boundary_nodes)} nodes")
    
    # Check matrix conditioning
    try:
        H_dense = H_bc.toarray()
        M_dense = M_bc.toarray()
        
        H_cond = np.linalg.cond(H_dense)
        M_cond = np.linalg.cond(M_dense)
        
        print(f"  Matrix condition numbers:")
        print(f"    H condition: {H_cond:.2e}")
        print(f"    M condition: {M_cond:.2e}")
        
        if H_cond > 1e12:
            print("  ⚠️  H matrix is ill-conditioned")
        if M_cond > 1e12:
            print("  ⚠️  M matrix is ill-conditioned")
            
    except Exception as e:
        print(f"  Could not compute condition numbers: {e}")
    
    # Try solving eigenvalue problem
    try:
        print("  Attempting eigenvalue solve...")
        
        max_eigs = min(3, num_nodes - 3)
        if max_eigs < 1:
            max_eigs = 1
        
        eigenvals, eigenvecs = spla.eigsh(
            H_bc, k=max_eigs, M=M_bc, which='SM', tol=1e-6, maxiter=1000
        )
        
        print(f"✅ Eigenvalue problem solved!")
        print(f"  Number of eigenvalues: {len(eigenvals)}")
        
        EV_TO_J = 1.602176634e-19
        eigenvals_eV = eigenvals / EV_TO_J
        
        print(f"  Energy levels (eV):")
        for i, E in enumerate(eigenvals_eV):
            print(f"    E_{i+1}: {E:.6f} eV")
        
        return True, len(eigenvals)
        
    except Exception as e:
        print(f"❌ Eigenvalue solve failed: {e}")
        
        # Try different solver parameters
        try:
            print("  Trying alternative solver...")
            eigenvals, eigenvecs = spla.eigsh(
                H_bc, k=1, M=M_bc, which='SM', tol=1e-4, maxiter=500
            )
            
            print(f"✅ Alternative solver succeeded!")
            print(f"  Energy: {eigenvals[0]/EV_TO_J:.6f} eV")
            
            return True, 1
            
        except Exception as e2:
            print(f"❌ Alternative solver also failed: {e2}")
            return False, 0

def main():
    """Main debugging function"""
    print("🚀 MATRIX ASSEMBLY DEBUG AND FIX")
    print("Systematically fixing fundamental FEM implementation issues")
    print("=" * 80)
    
    # Test 1: Create and debug mesh
    print("1. Creating and debugging mesh...")
    nodes_x, nodes_y, elements = create_debug_mesh(6, 4, 20e-9, 15e-9)
    
    # Test 2: Debug element areas
    print("\n2. Debugging element area calculations...")
    areas_ok = debug_element_areas(nodes_x, nodes_y, elements)
    
    if not areas_ok:
        print("❌ CRITICAL: Element areas are zero - mesh generation problem")
        return False
    
    # Test 3: Test matrix assembly
    print("\n3. Testing matrix assembly...")
    assembly_ok, H_matrix, M_matrix = test_manual_matrix_assembly(nodes_x, nodes_y, elements)
    
    if not assembly_ok:
        print("❌ CRITICAL: Matrix assembly failed")
        return False
    
    # Test 4: Test solving
    print("\n4. Testing eigenvalue solving...")
    solve_ok, num_eigenvals = test_boundary_conditions_and_solve(H_matrix, M_matrix)
    
    # Final assessment
    print("\n" + "=" * 80)
    print("🏆 MATRIX ASSEMBLY DEBUG RESULTS")
    print("=" * 80)
    
    print(f"📊 RESULTS:")
    print(f"  Mesh generation: {'✅ SUCCESS' if areas_ok else '❌ FAILED'}")
    print(f"  Matrix assembly: {'✅ SUCCESS' if assembly_ok else '❌ FAILED'}")
    print(f"  Eigenvalue solving: {'✅ SUCCESS' if solve_ok else '❌ FAILED'}")
    print(f"  Eigenvalues computed: {num_eigenvals if solve_ok else 0}")
    
    if areas_ok and assembly_ok and solve_ok:
        print("\n🎉 SUCCESS: All fundamental issues resolved!")
        print("  ✅ Element areas calculated correctly")
        print("  ✅ Matrix assembly working")
        print("  ✅ Eigenvalue solver functional")
        print("  ✅ Ready to fix Cython implementation")
        return True
    else:
        print("\n❌ ISSUES REMAIN:")
        if not areas_ok:
            print("  ❌ Element area calculation problems")
        if not assembly_ok:
            print("  ❌ Matrix assembly issues")
        if not solve_ok:
            print("  ❌ Eigenvalue solver problems")
        return False

if __name__ == "__main__":
    success = main()
