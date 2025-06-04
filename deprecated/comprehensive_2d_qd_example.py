#!/usr/bin/env python3
"""
Comprehensive 2D Quantum Dot Example for QDSim

This script provides a comprehensive example for simulating a 2D quantum dot
embedded in a PN junction using QDSim. It demonstrates the workflow from
mesh creation to visualization of results, with detailed explanations.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
import time

# Add the necessary paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build'))

# Import QDSim modules
try:
    import qdsim_cpp
    print("Successfully imported C++ backend")
except ImportError as e:
    print(f"Warning: Could not import C++ backend: {e}")
    print("Make sure the C++ backend is built and in the Python path")
    sys.exit(1)

# Try to import the Python frontend
try:
    from frontend.qdsim.config import Config
    from frontend.qdsim.simulator import Simulator
    print("Successfully imported Python frontend")
except ImportError as e:
    print(f"Warning: Could not import Python frontend: {e}")
    print("Using minimal implementation")

def create_custom_colormap():
    """Create a custom colormap for visualizing potentials and wavefunctions."""
    # Define custom colors for the colormap
    colors = [(0.0, 'darkblue'), (0.25, 'blue'), (0.5, 'white'), (0.75, 'red'), (1.0, 'darkred')]
    
    # Create the colormap
    cmap_name = 'custom_diverging'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    return cm

def main():
    """Main function implementing the comprehensive 2D quantum dot example."""
    print("\n=== Comprehensive 2D Quantum Dot Example ===\n")
    
    # Step 1: Create a mesh
    print("Step 1: Creating a mesh")
    Lx = 100.0  # Width of the domain (nm)
    Ly = 100.0  # Height of the domain (nm)
    nx = 50     # Number of elements in x-direction
    ny = 50     # Number of elements in y-direction
    element_order = 1  # Use linear elements for simplicity
    
    mesh = qdsim_cpp.Mesh(Lx, Ly, nx, ny, element_order)
    print(f"Mesh created with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")
    
    # Step 2: Define quantum dot parameters
    print("Step 2: Defining quantum dot parameters")
    
    qd_radius = 5.0  # Quantum dot radius (nm)
    qd_depth = 0.3   # Quantum dot potential depth (eV)
    qd_x = 0.0       # x-position of quantum dot center (nm)
    qd_y = 0.0       # y-position of quantum dot center (nm)
    
    # Step 3: Define PN junction parameters
    print("Step 3: Defining PN junction parameters")
    
    N_A = 1e24  # Acceptor concentration (m^-3)
    N_D = 1e24  # Donor concentration (m^-3)
    junction_position = 0.0  # Position of the junction (nm)
    
    # Step 4: Get mesh nodes and elements
    print("Step 4: Getting mesh nodes and elements")
    
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())
    
    print(f"Nodes shape: {nodes.shape}")
    print(f"Elements shape: {elements.shape}")
    
    # Step 5: Create potentials
    print("Step 5: Creating potentials")
    
    # Create the PN junction potential (simplified analytical model)
    pn_potential = np.zeros(len(nodes))
    
    # Built-in potential
    kT = 8.617e-5 * 300  # eV at 300K
    n_i = 1e16  # Intrinsic carrier concentration (m^-3, approximate)
    V_bi = kT * np.log(N_A * N_D / n_i**2)
    
    # Depletion width
    epsilon_r = 12.9  # GaAs
    epsilon_0 = 8.854e-12  # F/m
    q = 1.602e-19  # C
    W = np.sqrt(2 * epsilon_r * epsilon_0 * V_bi / (q * (N_A * N_D / (N_A + N_D))))
    W_p = W * N_D / (N_A + N_D)
    W_n = W * N_A / (N_A + N_D)
    
    print(f"Built-in potential: {V_bi:.3f} V")
    print(f"Depletion width: {W*1e9:.3f} nm (p-side: {W_p*1e9:.3f} nm, n-side: {W_n*1e9:.3f} nm)")
    
    # Create the PN junction potential
    for i in range(len(nodes)):
        x, y = nodes[i]
        if x < junction_position - W_p:
            # P-side outside depletion region
            pn_potential[i] = 0
        elif x > junction_position + W_n:
            # N-side outside depletion region
            pn_potential[i] = V_bi
        else:
            # Inside depletion region - quadratic potential
            if x < junction_position:
                # P-side depletion region
                pn_potential[i] = V_bi * (1 - ((junction_position - x) / W_p)**2)
            else:
                # N-side depletion region
                pn_potential[i] = V_bi * (1 - ((x - junction_position) / W_n)**2)
    
    # Create the quantum dot potential
    qd_potential = np.zeros(len(nodes))
    for i in range(len(nodes)):
        x, y = nodes[i]
        r = np.sqrt((x - qd_x)**2 + (y - qd_y)**2)
        if r < 3*qd_radius:  # Truncate at 3*radius for efficiency
            qd_potential[i] = -qd_depth * np.exp(-r**2 / (2*qd_radius**2))
    
    # Combine the potentials
    combined_potential = pn_potential + qd_potential
    
    # Step 6: Solve the Schrödinger equation
    print("Step 6: Solving the Schrödinger equation")
    
    # Define the effective mass function
    def m_star(x, y):
        """Effective mass function."""
        # Check if the point is inside the quantum dot
        r = np.sqrt((x - qd_x)**2 + (y - qd_y)**2)
        if r < qd_radius:
            return 0.023  # InAs effective mass
        else:
            return 0.067  # GaAs effective mass
    
    # Define the potential function
    def potential(x, y):
        """Potential function."""
        # Find the nearest node
        distances = np.sqrt((nodes[:, 0] - x)**2 + (nodes[:, 1] - y)**2)
        idx = np.argmin(distances)
        return combined_potential[idx]
    
    # Try to use the C++ SchrodingerSolver
    try:
        # Create a SchrodingerSolver
        print("Creating SchrodingerSolver...")
        schrodinger_solver = qdsim_cpp.create_schrodinger_solver(mesh, m_star, potential, False)
        
        # Solve for the lowest 5 eigenstates
        print("Solving for eigenstates...")
        num_states = 5
        start_time = time.time()
        eigenvalues, eigenvectors = schrodinger_solver.solve(num_states)
        end_time = time.time()
        
        print(f"Solved for {len(eigenvalues)} eigenstates in {end_time - start_time:.2f} seconds")
        print("Eigenvalues (eV):")
        for i, e in enumerate(eigenvalues):
            print(f"  E_{i} = {e:.6f}")
        
        # Convert eigenvectors to numpy arrays
        eigenvectors_np = []
        for i in range(len(eigenvectors)):
            eigenvectors_np.append(np.array(eigenvectors[i]))
        
        print(f"Eigenvector shapes: {[ev.shape for ev in eigenvectors_np]}")
        
    except Exception as e:
        print(f"Warning: Failed to use C++ SchrodingerSolver: {e}")
        print("Using simplified approach...")
        
        # Create a simplified Hamiltonian matrix
        print("Creating simplified Hamiltonian matrix...")
        num_nodes = len(nodes)
        H = np.zeros((num_nodes, num_nodes))
        
        # Create a simplified mass matrix
        M = np.zeros((num_nodes, num_nodes))
        
        # Fill the diagonal elements
        for i in range(num_nodes):
            H[i, i] = combined_potential[i]
            M[i, i] = 1.0
        
        # Solve the eigenvalue problem
        print("Solving simplified eigenvalue problem...")
        from scipy.sparse.linalg import eigsh
        
        start_time = time.time()
        eigenvalues, eigenvectors_np = eigsh(H, k=5, M=M, which='SM')
        end_time = time.time()
        
        print(f"Solved for {len(eigenvalues)} eigenstates in {end_time - start_time:.2f} seconds")
        print("Eigenvalues (eV):")
        for i, e in enumerate(eigenvalues):
            print(f"  E_{i} = {e:.6f}")
        
        print(f"Eigenvector shape: {eigenvectors_np.shape}")
    
    # Step 7: Visualize the results
    print("Step 7: Visualizing the results")
    
    # Create a directory for the results
    results_dir = "results_comprehensive_2d_qd"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a custom colormap
    cmap = create_custom_colormap()
    
    # Create a triangulation for plotting
    x = nodes[:, 0]
    y = nodes[:, 1]
    triangles = elements
    
    # Create the triangulation
    triang = Triangulation(x, y, triangles)
    
    # Plot the PN junction potential
    plt.figure(figsize=(10, 8))
    plt.tripcolor(triang, pn_potential, shading='gouraud', cmap=cmap)
    plt.colorbar(label='Potential (V)')
    plt.title('PN Junction Potential')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig(os.path.join(results_dir, "pn_potential.png"), dpi=300)
    
    # Plot the quantum dot potential
    plt.figure(figsize=(10, 8))
    plt.tripcolor(triang, qd_potential, shading='gouraud', cmap=cmap)
    plt.colorbar(label='Potential (eV)')
    plt.title('Quantum Dot Potential')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig(os.path.join(results_dir, "qd_potential.png"), dpi=300)
    
    # Plot the combined potential
    plt.figure(figsize=(10, 8))
    plt.tripcolor(triang, combined_potential, shading='gouraud', cmap=cmap)
    plt.colorbar(label='Potential (V)')
    plt.title('Combined Potential (PN Junction + Quantum Dot)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig(os.path.join(results_dir, "combined_potential.png"), dpi=300)
    
    # Plot the wavefunctions
    for i in range(min(5, len(eigenvectors_np))):
        plt.figure(figsize=(10, 8))
        plt.tripcolor(triang, np.abs(eigenvectors_np[i])**2, shading='gouraud', cmap='viridis')
        plt.colorbar(label='Probability density')
        plt.title(f'Wavefunction {i} (E = {eigenvalues[i]:.6f} eV)')
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.savefig(os.path.join(results_dir, f"wavefunction_{i}.png"), dpi=300)
    
    # Create a combined visualization
    plt.figure(figsize=(15, 10))
    
    # Plot combined potential
    plt.subplot(2, 3, 1)
    plt.tripcolor(triang, combined_potential, shading='gouraud', cmap=cmap)
    plt.colorbar(label='Potential (V)')
    plt.title('Combined Potential')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    
    # Plot wavefunctions
    for i in range(min(5, len(eigenvectors_np))):
        plt.subplot(2, 3, i+2)
        plt.tripcolor(triang, np.abs(eigenvectors_np[i])**2, shading='gouraud', cmap='viridis')
        plt.colorbar(label='Probability density')
        plt.title(f'Wavefunction {i} (E = {eigenvalues[i]:.6f} eV)')
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combined_visualization.png"), dpi=300)
    
    print(f"\nResults saved to {results_dir}")
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
