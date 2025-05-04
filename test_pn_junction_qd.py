#!/usr/bin/env python3
"""
Test script for simulating a p-n junction with a quantum dot using the ImprovedSelfConsistentSolver.
"""

import sys
import os
import numpy as np

# Add the backend build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend/build'))

# Import the C++ module directly
import qdsim_cpp as cpp

def main():
    """
    Main function to simulate a p-n junction with a quantum dot using the ImprovedSelfConsistentSolver.
    """
    # Create a mesh with higher resolution
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 100
    ny = 50
    element_order = 1
    mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)
    
    # Define quantum dot parameters
    qd_x = 0.0   # QD position (at the junction)
    qd_y = Ly/2  # QD position (at the center of the device)
    qd_radius = 5.0  # QD radius (nm)
    qd_depth = 0.3   # QD potential depth (eV)
    
    # Define callback functions
    def epsilon_r(x, y):
        """Relative permittivity function."""
        # GaAs relative permittivity
        return 12.9
    
    def rho(x, y, n, p):
        """Charge density function."""
        q = 1.602e-19  # Elementary charge in C
        if len(n) == 0 or len(p) == 0:
            return 0.0
        
        # Find the nearest node
        nodes = np.array(mesh.get_nodes())
        distances = np.sqrt((nodes[:, 0] - x)**2 + (nodes[:, 1] - y)**2)
        idx = np.argmin(distances)
        
        # Return the charge density at the nearest node
        return q * (p[idx] - n[idx])
    
    # Create the ImprovedSelfConsistentSolver using the helper function
    sc_solver = cpp.create_improved_self_consistent_solver(mesh, epsilon_r, rho)
    
    # Solve the self-consistent Poisson-drift-diffusion equations
    V_p = 0.0  # Voltage at the p-contact
    V_n = 0.7  # Voltage at the n-contact (forward bias)
    N_A = 1e18  # Acceptor doping concentration
    N_D = 1e18  # Donor doping concentration
    tolerance = 1e-6
    max_iter = 100
    
    print("Solving the self-consistent Poisson-drift-diffusion equations...")
    sc_solver.solve(V_p, V_n, N_A, N_D, tolerance, max_iter)
    
    # Get the results
    potential = np.array(sc_solver.get_potential())
    n = np.array(sc_solver.get_n())
    p = np.array(sc_solver.get_p())
    
    print(f"Potential shape: {potential.shape}")
    print(f"Electron concentration shape: {n.shape}")
    print(f"Hole concentration shape: {p.shape}")
    
    # Print some statistics
    print(f"Potential min: {potential.min()}, max: {potential.max()}")
    print(f"Electron concentration min: {n.min()}, max: {n.max()}")
    print(f"Hole concentration min: {p.min()}, max: {p.max()}")
    
    # Get the mesh nodes
    nodes = np.array(mesh.get_nodes())
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # Add the quantum dot potential
    qd_potential = np.zeros_like(potential)
    for i in range(len(nodes)):
        # Distance from the quantum dot center
        r = np.sqrt((x[i] - qd_x)**2 + (y[i] - qd_y)**2)
        
        # Gaussian potential well
        if r < 3*qd_radius:  # Truncate at 3*radius for efficiency
            qd_potential[i] = -qd_depth * np.exp(-r**2 / (2*qd_radius**2))
    
    # Combine the p-n junction potential with the quantum dot potential
    combined_potential = potential + qd_potential
    
    # Save the results to files
    print("Saving results to files...")
    
    # Create a directory for the results if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save the mesh
    np.savetxt("results/mesh_x.txt", x)
    np.savetxt("results/mesh_y.txt", y)
    
    # Save the potentials
    np.savetxt("results/pn_potential.txt", potential)
    np.savetxt("results/qd_potential.txt", qd_potential)
    np.savetxt("results/combined_potential.txt", combined_potential)
    
    # Save the carrier concentrations
    np.savetxt("results/electron_concentration.txt", n)
    np.savetxt("results/hole_concentration.txt", p)
    
    print("Results saved to the 'results' directory.")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
