#!/usr/bin/env python3
"""
Minimal P-N Junction Example

This example demonstrates a simple P-N junction model using the C++ implementation.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import qdsim_cpp as qdc

def main():
    """Main function."""
    # Create a mesh
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 101
    ny = 51
    mesh = qdc.Mesh(Lx, Ly, nx, ny, 1)  # P1 elements

    # P-N junction parameters
    epsilon_r = 12.9  # Relative permittivity of GaAs
    N_A = 1e-9  # Acceptor concentration (nm^-3)
    N_D = 1e-9  # Donor concentration (nm^-3)
    junction_position = Lx / 2  # Junction at the center of the domain
    V_r = 0.5  # Reverse bias (V)

    # Define callback functions
    def epsilon_r_callback(x, y):
        return epsilon_r

    def rho_callback(x, y, n, p):
        # Charge density in C/nm^3
        q = 1.602e-19  # Elementary charge in C
        
        # Simple doping-based charge density
        if x < junction_position:
            return -q * N_A  # p-type region
        else:
            return q * N_D   # n-type region

    # Create the BasicSolver (simplest solver)
    print("Creating BasicSolver...")
    solver = qdc.BasicSolver(mesh)

    # Solve with the basic solver
    print("Solving with BasicSolver...")
    solver.solve(0.0, -V_r, N_A, N_D)

    # Get the potential
    print("Getting results...")
    potential = solver.get_potential()

    # Create a grid for visualization
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Create interpolators
    print("Creating interpolators...")
    simple_mesh = qdc.create_simple_mesh(mesh)
    interpolator = qdc.SimpleInterpolator(simple_mesh)

    # Calculate potential on grid
    print("Calculating potential on grid...")
    potential_grid = np.zeros((ny, nx))
    mid_y = ny // 2

    # Interpolate potential along the middle of the domain
    for j in range(nx):
        try:
            potential_grid[mid_y, j] = interpolator.interpolate(x[j], y[mid_y], potential)
        except:
            potential_grid[mid_y, j] = 0.0

    # Plot potential
    plt.figure(figsize=(10, 6))
    plt.plot(x, potential_grid[mid_y, :], 'b-', linewidth=2)
    plt.axvline(x=junction_position, color='k', linestyle='--', label='Junction')
    plt.xlabel('x (nm)')
    plt.ylabel('Potential (V)')
    plt.title('P-N Junction Potential')
    plt.grid(True)
    plt.legend()
    plt.savefig('minimal_pn_junction.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
