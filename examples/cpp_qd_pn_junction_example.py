#!/usr/bin/env python3
"""
Example script demonstrating the use of the C++ PNJunction class with a quantum dot.

This script creates a P-N junction using the C++ implementation, adds a quantum dot,
and solves the Schrödinger equation to find the eigenstates.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qdsim_cpp as qdc
import sys
import os

# Add the parent directory to the path so we can import from frontend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from frontend.qdsim.visualization import plot_potential_3d, plot_wavefunction_3d
from frontend.qdsim.materials import Material

def main():
    # Create a mesh
    Lx = 200.0  # nm
    Ly = 100.0  # nm
    nx = 201
    ny = 101
    mesh = qdc.Mesh(Lx, Ly, nx, ny, 1)  # P1 elements
    
    # P-N junction parameters
    epsilon_r = 12.9  # Relative permittivity of GaAs
    N_A = 1e16  # Acceptor concentration (cm^-3)
    N_D = 1e16  # Donor concentration (cm^-3)
    T = 300.0  # Temperature (K)
    junction_position = 0.0  # Junction at x = 0
    V_r = 1.0  # Reverse bias voltage (V)
    
    # Convert doping concentrations from cm^-3 to nm^-3
    N_A_nm3 = N_A * 1e-21
    N_D_nm3 = N_D * 1e-21
    
    # Create the P-N junction
    print("Creating P-N junction...")
    pn_junction = qdc.PNJunction(mesh, epsilon_r, N_A_nm3, N_D_nm3, T, junction_position, V_r)
    
    # Print junction parameters
    print(f"Built-in potential: {pn_junction.V_bi:.3f} V")
    print(f"Depletion width: {pn_junction.W:.3f} nm")
    print(f"P-side depletion width: {pn_junction.W_p:.3f} nm")
    print(f"N-side depletion width: {pn_junction.W_n:.3f} nm")
    print(f"Intrinsic carrier concentration: {pn_junction.n_i:.3e} nm^-3")
    
    # Solve the Poisson equation
    print("Solving Poisson equation...")
    pn_junction.solve()
    
    # Create a grid for visualization
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)
    
    # Calculate P-N junction potential on the grid
    print("Calculating P-N junction potential on grid...")
    pn_potential = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            pn_potential[i, j] = pn_junction.get_potential(X[i, j], Y[i, j])
    
    # Define quantum dot parameters
    qd_position = (0.0, 0.0)  # Position of the quantum dot (at the junction)
    qd_radius = 10.0  # Radius of the quantum dot (nm)
    qd_depth = 0.3  # Depth of the quantum dot potential (eV)
    
    # Define quantum dot potential function
    def qd_potential_func(x, y):
        r = np.sqrt((x - qd_position[0])**2 + (y - qd_position[1])**2)
        if r <= qd_radius:
            return -qd_depth
        else:
            return 0.0
    
    # Calculate quantum dot potential on the grid
    print("Calculating quantum dot potential on grid...")
    qd_potential = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            qd_potential[i, j] = qd_potential_func(X[i, j], Y[i, j])
    
    # Combine P-N junction and quantum dot potentials
    combined_potential = pn_potential + qd_potential
    
    # Define materials
    matrix_material = Material(
        name="GaAs",
        m_e=0.067,
        m_h=0.45,
        E_g=1.42,
        epsilon_r=12.9,
        Delta_E_c=0.0
    )
    
    qd_material = Material(
        name="InAs",
        m_e=0.023,
        m_h=0.41,
        E_g=0.35,
        epsilon_r=15.15,
        Delta_E_c=0.7
    )
    
    # Define effective mass function
    def m_star_func(x, y):
        r = np.sqrt((x - qd_position[0])**2 + (y - qd_position[1])**2)
        if r <= qd_radius:
            return qd_material.m_e
        else:
            return matrix_material.m_e
    
    # Define combined potential function
    def combined_potential_func(x, y):
        # P-N junction potential
        pn_pot = pn_junction.get_potential(x, y)
        
        # Quantum dot potential
        qd_pot = qd_potential_func(x, y)
        
        return pn_pot + qd_pot
    
    # Define capacitance function (not used in this example)
    def cap_func(x, y):
        return 0.0
    
    # Create a self-consistent solver
    print("Creating self-consistent solver...")
    
    # Define callback functions for the self-consistent solver
    def epsilon_r_func(x, y):
        r = np.sqrt((x - qd_position[0])**2 + (y - qd_position[1])**2)
        if r <= qd_radius:
            return qd_material.epsilon_r
        else:
            return matrix_material.epsilon_r
    
    def rho_func(x, y, n, p):
        # This is a simplified charge density function
        # In a real implementation, this would calculate the charge density
        # based on the doping concentrations and carrier concentrations
        d = x - junction_position
        if d < 0:
            # P-side
            return -qdc.e_charge * N_A_nm3
        else:
            # N-side
            return qdc.e_charge * N_D_nm3
    
    def n_conc_func(x, y, phi, mat):
        # Electron concentration
        d = x - junction_position
        if d < 0:
            # P-side
            return pn_junction.n_i**2 / N_A_nm3
        else:
            # N-side
            return N_D_nm3
    
    def p_conc_func(x, y, phi, mat):
        # Hole concentration
        d = x - junction_position
        if d < 0:
            # P-side
            return N_A_nm3
        else:
            # N-side
            return pn_junction.n_i**2 / N_D_nm3
    
    def mu_n_func(x, y, mat):
        # Electron mobility
        return 0.3  # Example value
    
    def mu_p_func(x, y, mat):
        # Hole mobility
        return 0.02  # Example value
    
    # Create the self-consistent solver
    sc_solver = qdc.create_self_consistent_solver(
        mesh, epsilon_r_func, rho_func, n_conc_func, p_conc_func, mu_n_func, mu_p_func
    )
    
    # Solve the self-consistent problem
    print("Solving self-consistent problem...")
    sc_solver.solve(0, -V_r, N_A_nm3, N_D_nm3)
    
    # Create the FEM solver
    print("Creating FEM solver...")
    fem_solver = qdc.FEMSolver(mesh, m_star_func, combined_potential_func, cap_func, sc_solver, 1, False)
    
    # Assemble the matrices
    print("Assembling matrices...")
    fem_solver.assemble_matrices()
    
    # Create the eigenvalue solver
    print("Creating eigenvalue solver...")
    eigen_solver = qdc.EigenSolver(fem_solver)
    
    # Solve the eigenvalue problem
    print("Solving eigenvalue problem...")
    num_eigenpairs = 10
    eigen_solver.solve(num_eigenpairs)
    
    # Get the eigenvalues and eigenvectors
    eigenvalues = eigen_solver.get_eigenvalues()
    eigenvectors = eigen_solver.get_eigenvectors()
    
    # Print the eigenvalues
    print("Eigenvalues (eV):")
    for i, ev in enumerate(eigenvalues):
        print(f"  {i+1}: {ev:.6f}")
    
    # Plot the combined potential
    print("Plotting combined potential...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_potential_3d(ax, X, Y, combined_potential, title="Combined P-N Junction and Quantum Dot Potential")
    plt.savefig("cpp_qd_pn_combined_potential.png", dpi=300, bbox_inches='tight')
    
    # Plot the wavefunctions
    print("Plotting wavefunctions...")
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the first 4 eigenstates
    for i in range(min(4, num_eigenpairs)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Reshape the eigenvector to match the grid
        wavefunction = np.zeros((ny, nx))
        for j in range(ny):
            for k in range(nx):
                idx = j * nx + k
                if idx < len(eigenvectors[i]):
                    wavefunction[j, k] = abs(eigenvectors[i][idx])**2
        
        plot_wavefunction_3d(ax, X, Y, wavefunction, title=f"Eigenstate {i+1}, E = {eigenvalues[i]:.6f} eV")
    
    plt.tight_layout()
    plt.savefig("cpp_qd_pn_wavefunctions.png", dpi=300, bbox_inches='tight')
    
    print("Done! Plots saved as png files.")

if __name__ == "__main__":
    main()
