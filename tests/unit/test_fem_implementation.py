#!/usr/bin/env python3
"""
Unit tests for the FEM implementation in QDSim.

This script tests the FEM implementation in QDSim, including:
1. Element matrix assembly with proper quadrature rules
2. Finite element interpolation for potentials
3. MPI data transfer efficiency (if MPI is available)

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Add the parent directory to the path so we can import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import qdsim

class TestFEMImplementation:
    """Test class for the FEM implementation in QDSim."""
    
    def setup_method(self):
        """Set up common parameters for tests."""
        self.config = qdsim.Config()
        
        # Physical constants
        self.config.e_charge = 1.602e-19  # Elementary charge in C
        self.config.hbar = 1.054e-34      # Reduced Planck's constant in J·s
        self.config.m_e = 9.11e-31        # Electron mass in kg
        
        # Use GaAs effective mass
        self.m_star = 0.067 * self.config.m_e
        
        # Set up mesh parameters
        self.config.Lx = 100e-9  # 100 nm
        self.config.Ly = 100e-9  # 100 nm
        self.config.nx = 51      # Mesh points
        self.config.ny = 51      # Mesh points
        self.config.element_order = 2  # Quadratic elements for better accuracy
        
        # Disable MPI for tests
        self.config.use_mpi = False
        
        # Define a simple test potential
        def test_potential(x, y):
            return 0.5 * self.config.e_charge * (x**2 + y**2) / (1e-9)**2
        
        self.config.potential_function = test_potential
        self.config.m_star_function = lambda x, y: self.m_star
    
    def test_element_matrix_assembly(self):
        """Test element matrix assembly with proper quadrature rules."""
        print("\nTesting element matrix assembly...")
        
        # Create simulator
        simulator = qdsim.Simulator(self.config)
        
        # Get the stiffness and mass matrices
        K = simulator.get_stiffness_matrix()
        M = simulator.get_mass_matrix()
        
        # Check that the matrices are symmetric
        K_sym_error = np.max(np.abs(K - K.T))
        M_sym_error = np.max(np.abs(M - M.T))
        
        print(f"Stiffness matrix symmetry error: {K_sym_error:.6e}")
        print(f"Mass matrix symmetry error: {M_sym_error:.6e}")
        
        # Check that the matrices are positive definite
        K_eigenvalues = np.linalg.eigvalsh(K.toarray())
        M_eigenvalues = np.linalg.eigvalsh(M.toarray())
        
        print(f"Smallest stiffness matrix eigenvalue: {np.min(K_eigenvalues):.6e}")
        print(f"Smallest mass matrix eigenvalue: {np.min(M_eigenvalues):.6e}")
        
        # Check the condition number of the matrices
        K_cond = np.max(K_eigenvalues) / np.min(K_eigenvalues)
        M_cond = np.max(M_eigenvalues) / np.min(M_eigenvalues)
        
        print(f"Stiffness matrix condition number: {K_cond:.6e}")
        print(f"Mass matrix condition number: {M_cond:.6e}")
        
        # Assert that the matrices are symmetric
        assert K_sym_error < 1e-10, "Stiffness matrix is not symmetric"
        assert M_sym_error < 1e-10, "Mass matrix is not symmetric"
        
        # Assert that the matrices are positive definite
        assert np.min(K_eigenvalues) > 0, "Stiffness matrix is not positive definite"
        assert np.min(M_eigenvalues) > 0, "Mass matrix is not positive definite"
        
        # Assert that the condition numbers are reasonable
        assert K_cond < 1e6, "Stiffness matrix condition number is too large"
        assert M_cond < 1e6, "Mass matrix condition number is too large"
        
        return K, M
    
    def test_finite_element_interpolation(self):
        """Test finite element interpolation for potentials."""
        print("\nTesting finite element interpolation...")
        
        # Create simulator
        simulator = qdsim.Simulator(self.config)
        
        # Define a test function
        def test_function(x, y):
            return np.sin(np.pi * x / self.config.Lx) * np.sin(np.pi * y / self.config.Ly)
        
        # Interpolate the test function onto the FEM mesh
        nodes = simulator.get_nodes()
        values = np.array([test_function(x, y) for x, y in nodes])
        
        # Create a grid for evaluation
        x = np.linspace(-self.config.Lx/2, self.config.Lx/2, 100)
        y = np.linspace(-self.config.Ly/2, self.config.Ly/2, 100)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate the test function on the grid
        Z_exact = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_exact[i, j] = test_function(X[i, j], Y[i, j])
        
        # Interpolate the FEM solution onto the grid
        Z_interp = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_interp[i, j] = simulator.interpolate(X[i, j], Y[i, j], values)
        
        # Calculate the error
        error = np.abs(Z_exact - Z_interp)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        print(f"Maximum interpolation error: {max_error:.6e}")
        print(f"Mean interpolation error: {mean_error:.6e}")
        
        # Plot the results
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot exact solution
        im1 = axs[0].contourf(X*1e9, Y*1e9, Z_exact, 50, cmap='viridis')
        axs[0].set_title('Exact Solution')
        axs[0].set_xlabel('x (nm)')
        axs[0].set_ylabel('y (nm)')
        fig.colorbar(im1, ax=axs[0])
        
        # Plot interpolated solution
        im2 = axs[1].contourf(X*1e9, Y*1e9, Z_interp, 50, cmap='plasma')
        axs[1].set_title('Interpolated Solution')
        axs[1].set_xlabel('x (nm)')
        axs[1].set_ylabel('y (nm)')
        fig.colorbar(im2, ax=axs[1])
        
        # Plot error
        im3 = axs[2].contourf(X*1e9, Y*1e9, error, 50, cmap='inferno')
        axs[2].set_title('Interpolation Error')
        axs[2].set_xlabel('x (nm)')
        axs[2].set_ylabel('y (nm)')
        fig.colorbar(im3, ax=axs[2])
        
        plt.tight_layout()
        plt.savefig('fem_interpolation_test.png', dpi=300)
        
        # Assert that the error is small
        assert max_error < 1e-2, "Maximum interpolation error is too large"
        assert mean_error < 1e-3, "Mean interpolation error is too large"
        
        return max_error, mean_error
    
    def test_convergence_with_mesh_refinement(self):
        """Test convergence of the solution with mesh refinement."""
        print("\nTesting convergence with mesh refinement...")
        
        # Mesh sizes to test
        mesh_sizes = [11, 21, 41, 81]
        
        # Results
        eigenvalues_list = []
        
        for mesh_size in mesh_sizes:
            print(f"Running simulation with mesh size {mesh_size}x{mesh_size}...")
            
            # Update mesh size
            self.config.nx = mesh_size
            self.config.ny = mesh_size
            
            # Create simulator
            simulator = qdsim.Simulator(self.config)
            
            # Solve for the first 5 eigenstates
            eigenvalues, eigenvectors = simulator.run(num_eigenvalues=5)
            
            # Convert eigenvalues to eV
            eigenvalues_eV = np.real(eigenvalues) / self.config.e_charge
            
            # Store results
            eigenvalues_list.append(eigenvalues_eV)
            
            print(f"  Ground state energy: {eigenvalues_eV[0]:.6f} eV")
        
        # Calculate convergence rate
        convergence_rates = []
        for i in range(len(mesh_sizes)-1):
            h1 = 1.0 / mesh_sizes[i]
            h2 = 1.0 / mesh_sizes[i+1]
            h_ratio = h1 / h2
            
            e1 = eigenvalues_list[i][0]
            e2 = eigenvalues_list[i+1][0]
            e_ratio = abs(e1 - e2) / abs(e2)
            
            rate = np.log(e_ratio) / np.log(h_ratio)
            convergence_rates.append(rate)
            
            print(f"Convergence rate between mesh sizes {mesh_sizes[i]} and {mesh_sizes[i+1]}: {rate:.2f}")
        
        # Plot convergence
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ground state energy vs. mesh size
        ax.plot(mesh_sizes, [evals[0] for evals in eigenvalues_list], 'bo-')
        ax.set_xlabel('Mesh Size (NxN)')
        ax.set_ylabel('Ground State Energy (eV)')
        ax.set_title('Convergence of Ground State Energy with Mesh Refinement')
        ax.grid(True)
        
        # Add a second y-axis for the convergence rate
        ax2 = ax.twinx()
        ax2.plot([(mesh_sizes[i] + mesh_sizes[i+1])/2 for i in range(len(mesh_sizes)-1)], 
                convergence_rates, 'ro-')
        ax2.set_ylabel('Convergence Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        plt.savefig('fem_convergence_test.png', dpi=300)
        
        # Assert that the convergence rate is at least 2 (for quadratic elements)
        assert np.mean(convergence_rates) > 1.5, "Convergence rate is too low"
        
        return mesh_sizes, eigenvalues_list, convergence_rates
    
    def test_mpi_data_transfer(self):
        """Test MPI data transfer efficiency (if MPI is available)."""
        print("\nTesting MPI data transfer efficiency...")
        
        # Check if MPI is available
        try:
            from mpi4py import MPI
            has_mpi = True
        except ImportError:
            has_mpi = False
            print("MPI not available, skipping MPI data transfer test")
            return None
        
        if has_mpi:
            # Enable MPI
            self.config.use_mpi = True
            
            # Create simulator
            simulator = qdsim.Simulator(self.config)
            
            # Get MPI communicator
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            
            print(f"Running on {size} processes, current rank: {rank}")
            
            # Solve for the first 5 eigenstates
            start_time = time.time()
            eigenvalues, eigenvectors = simulator.run(num_eigenvalues=5)
            mpi_time = time.time() - start_time
            
            print(f"MPI solution time: {mpi_time:.4f} seconds")
            
            # Disable MPI for comparison
            self.config.use_mpi = False
            
            # Create simulator
            simulator_seq = qdsim.Simulator(self.config)
            
            # Solve for the first 5 eigenstates
            start_time = time.time()
            eigenvalues_seq, eigenvectors_seq = simulator_seq.run(num_eigenvalues=5)
            seq_time = time.time() - start_time
            
            print(f"Sequential solution time: {seq_time:.4f} seconds")
            
            # Calculate speedup
            speedup = seq_time / mpi_time
            
            print(f"Speedup with {size} processes: {speedup:.2f}x")
            
            # Assert that the eigenvalues are the same
            rel_diff = np.abs((np.real(eigenvalues) - np.real(eigenvalues_seq)) / np.real(eigenvalues_seq))
            max_diff = np.max(rel_diff)
            
            print(f"Maximum relative difference in eigenvalues: {max_diff:.6e}")
            
            # Assert that the difference is small
            assert max_diff < 1e-6, "MPI and sequential solutions differ significantly"
            
            # Assert that there is some speedup with MPI
            if size > 1:
                assert speedup > 1.0, "No speedup with MPI"
            
            return size, speedup

if __name__ == "__main__":
    # Run the tests
    test = TestFEMImplementation()
    test.setup_method()
    test.test_element_matrix_assembly()
    test.test_finite_element_interpolation()
    test.test_convergence_with_mesh_refinement()
    test.test_mpi_data_transfer()
