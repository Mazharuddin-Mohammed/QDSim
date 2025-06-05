#!/usr/bin/env python3
"""
Fix Matrix Assembly Issues

This script systematically debugs and fixes the matrix assembly problems.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def create_working_mesh(nx, ny, Lx, Ly):
    """Create a working mesh with proper debugging"""
    print(f"🔧 Creating mesh: {nx}×{ny} nodes, {Lx*1e9:.1f}×{Ly*1e9:.1f} nm")
    
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
    
    print(f"✅ Nodes: {len(nodes_x)} total")
    print(f"   X: [{np.min(nodes_x)*1e9:.2f}, {np.max(nodes_x)*1e9:.2f}] nm")
    print(f"   Y: [{np.min(nodes_y)*1e9:.2f}, {np.max(nodes_y)*1e9:.2f}] nm")
    
    # Generate triangular elements
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
    print(f"✅ Elements: {len(elements)} triangles")
    
    return nodes_x, nodes_y, elements

def debug_areas(nodes_x, nodes_y, elements):
    """Debug element area calculations"""
    print("\n🔍 Debugging Element Areas")
    print("=" * 40)
    
    valid_count = 0
    total_area = 0.0
    
    for i in range(min(5, len(elements))):
        n0, n1, n2 = elements[i]
        
        x0, y0 = nodes_x[n0], nodes_y[n0]
        x1, y1 = nodes_x[n1], nodes_y[n1]
        x2, y2 = nodes_x[n2], nodes_y[n2]
        
        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        
        print(f"Element {i}: area = {area*1e18:.6f} nm²")
        
        if area > 1e-20:
            valid_count += 1
            total_area += area
    
    print(f"Valid elements: {valid_count}/5")
    return valid_count > 0

def assemble_matrices(nodes_x, nodes_y, elements):
    """Assemble FEM matrices"""
    print("\n🔧 Assembling Matrices")
    print("=" * 30)
    
    num_nodes = len(nodes_x)
    
    # Constants
    HBAR = 1.054571817e-34
    M_E = 9.1093837015e-31
    m_star = 0.067 * M_E
    
    # Matrix builders
    rows, cols, H_data, M_data = [], [], [], []
    
    valid_elements = 0
    
    for elem_idx in range(len(elements)):
        n0, n1, n2 = elements[elem_idx]
        
        x0, y0 = nodes_x[n0], nodes_y[n0]
        x1, y1 = nodes_x[n1], nodes_y[n1]
        x2, y2 = nodes_x[n2], nodes_y[n2]
        
        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        
        if area < 1e-20:
            continue
        
        valid_elements += 1
        
        # Element matrices
        kinetic = HBAR * HBAR / (2.0 * m_star) * area / 3.0
        mass = area / 12.0
        
        nodes = [n0, n1, n2]
        for i in range(3):
            for j in range(3):
                rows.append(nodes[i])
                cols.append(nodes[j])
                
                if i == j:
                    H_data.append(kinetic)
                    M_data.append(2.0 * mass)
                else:
                    H_data.append(kinetic * 0.5)
                    M_data.append(mass)
    
    print(f"Valid elements: {valid_elements}/{len(elements)}")
    
    if valid_elements == 0:
        return None, None
    
    H = sp.csr_matrix((H_data, (rows, cols)), shape=(num_nodes, num_nodes))
    M = sp.csr_matrix((M_data, (rows, cols)), shape=(num_nodes, num_nodes))
    
    print(f"H: {H.shape}, nnz = {H.nnz}")
    print(f"M: {M.shape}, nnz = {M.nnz}")
    
    return H, M

def solve_eigenvalue_problem(H, M):
    """Solve eigenvalue problem"""
    print("\n⚡ Solving Eigenvalue Problem")
    print("=" * 35)
    
    num_nodes = H.shape[0]
    
    # Apply boundary conditions
    H_bc = H.tolil()
    M_bc = M.tolil()
    
    # Fix boundary nodes
    boundary_nodes = [0, num_nodes-1]
    for node in boundary_nodes:
        H_bc[node, :] = 0
        H_bc[node, node] = 1.0
        M_bc[node, :] = 0
        M_bc[node, node] = 1.0
    
    H_bc = H_bc.tocsr()
    M_bc = M_bc.tocsr()
    
    try:
        eigenvals, eigenvecs = spla.eigsh(
            H_bc, k=2, M=M_bc, which='SM', tol=1e-6
        )
        
        EV_TO_J = 1.602176634e-19
        print(f"✅ Solved: {len(eigenvals)} eigenvalues")
        for i, E in enumerate(eigenvals):
            print(f"  E_{i+1}: {E/EV_TO_J:.6f} eV")
        
        return True, eigenvals
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, None

def main():
    """Main function"""
    print("🚀 FIXING MATRIX ASSEMBLY")
    print("=" * 50)
    
    # Create mesh
    nodes_x, nodes_y, elements = create_working_mesh(5, 4, 15e-9, 12e-9)
    
    # Debug areas
    areas_ok = debug_areas(nodes_x, nodes_y, elements)
    if not areas_ok:
        print("❌ Area calculation failed")
        return False
    
    # Assemble matrices
    H, M = assemble_matrices(nodes_x, nodes_y, elements)
    if H is None:
        print("❌ Matrix assembly failed")
        return False
    
    # Solve
    solve_ok, eigenvals = solve_eigenvalue_problem(H, M)
    
    print(f"\n🎯 RESULT: {'✅ SUCCESS' if solve_ok else '❌ FAILED'}")
    return solve_ok

if __name__ == "__main__":
    success = main()
