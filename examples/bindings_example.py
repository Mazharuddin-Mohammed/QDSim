#!/usr/bin/env python3
"""
Example demonstrating the enhanced C++ bindings.

This example shows how to use the enhanced C++ bindings, including:
1. STL and Eigen conversions
2. Memory management for callbacks
3. Error handling

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from qdsim_cpp import (
    Mesh, MaterialDatabase, Material, SelfConsistentSolver,
    create_self_consistent_solver, clear_callbacks, clear_callback,
    has_callback, CallbackException
)

def main():
    """Main function demonstrating the enhanced C++ bindings."""
    print("Demonstrating enhanced C++ bindings")
    
    # 1. STL and Eigen conversions
    print("\n1. STL and Eigen conversions")
    
    # Create a mesh
    Lx, Ly = 100.0, 100.0  # Domain size in nm
    nx, ny = 50, 50        # Number of elements in each direction
    element_order = 1      # Linear elements
    mesh = Mesh(Lx, Ly, nx, ny, element_order)
    print(f"Created mesh with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")
    
    # Create a material database and get a material
    mat_db = MaterialDatabase()
    gaas = mat_db.get_material("GaAs")
    print(f"GaAs properties: m_e={gaas.m_e}, m_h={gaas.m_h}, E_g={gaas.E_g}, epsilon_r={gaas.epsilon_r}")
    
    # Create a custom material
    custom_mat = Material()
    custom_mat.m_e = 0.067       # Electron effective mass (relative to free electron mass)
    custom_mat.m_h = 0.45        # Hole effective mass (relative to free electron mass)
    custom_mat.E_g = 1.42        # Band gap energy (eV)
    custom_mat.Delta_E_c = 0.7   # Conduction band offset (eV)
    custom_mat.epsilon_r = 12.9  # Relative permittivity
    custom_mat.mu_n = 0.85       # Electron mobility (m²/V·s)
    custom_mat.mu_p = 0.04       # Hole mobility (m²/V·s)
    custom_mat.N_c = 4.7e17      # Effective density of states in conduction band (1/cm³)
    custom_mat.N_v = 7.0e18      # Effective density of states in valence band (1/cm³)
    
    print(f"Custom material: {custom_mat}")
    
    # 2. Memory management for callbacks
    print("\n2. Memory management for callbacks")
    
    # Define callback functions
    def epsilon_r_callback(x, y):
        """Relative permittivity callback."""
        return 12.9  # GaAs
    
    def rho_callback(x, y, n, p):
        """Charge density callback."""
        q = 1.602e-19  # Elementary charge in C
        return q * (p - n)  # Charge density in C/nm³
    
    def n_conc_callback(x, y, phi, mat):
        """Electron concentration callback."""
        kT = 0.0259  # eV at 300K
        return mat.N_c * np.exp(-phi / kT)
    
    def p_conc_callback(x, y, phi, mat):
        """Hole concentration callback."""
        kT = 0.0259  # eV at 300K
        return mat.N_v * np.exp((phi - mat.E_g) / kT)
    
    def mu_n_callback(x, y, mat):
        """Electron mobility callback."""
        return mat.mu_n
    
    def mu_p_callback(x, y, mat):
        """Hole mobility callback."""
        return mat.mu_p
    
    # Create a SelfConsistentSolver with the callbacks
    solver = create_self_consistent_solver(
        mesh,
        epsilon_r_callback,
        rho_callback,
        n_conc_callback,
        p_conc_callback,
        mu_n_callback,
        mu_p_callback
    )
    
    # Check if callbacks are registered
    print(f"Has epsilon_r callback: {has_callback('epsilon_r')}")
    print(f"Has rho callback: {has_callback('rho')}")
    print(f"Has n_conc callback: {has_callback('n_conc')}")
    
    # Clear a specific callback
    clear_callback("mu_p")
    print(f"After clearing mu_p callback, has mu_p callback: {has_callback('mu_p')}")
    
    # Clear all callbacks
    clear_callbacks()
    print(f"After clearing all callbacks, has epsilon_r callback: {has_callback('epsilon_r')}")
    
    # 3. Error handling
    print("\n3. Error handling")
    
    # Define a callback that raises an exception
    def error_callback(x, y):
        """Callback that raises an exception."""
        raise ValueError("This is a test error")
    
    # Create a SelfConsistentSolver with the error callback
    try:
        solver = create_self_consistent_solver(
            mesh,
            error_callback,
            rho_callback,
            n_conc_callback,
            p_conc_callback,
            mu_n_callback,
            mu_p_callback
        )
        
        # Solve the Poisson-drift-diffusion equations
        # This will trigger the error callback
        solver.solve(0.0, 1.0, 1e16, 1e16)
    except CallbackException as e:
        print(f"Caught CallbackException: {e}")
    finally:
        # Clean up
        clear_callbacks()

if __name__ == "__main__":
    main()
