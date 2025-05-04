#!/usr/bin/env python3
"""
Validation test for a chromium quantum dot in an AlGaAs P-N diode.

This script simulates a realistic case of a chromium quantum dot embedded
at the interface of an AlGaAs P-N junction diode under reverse bias.
The simulation domain is 200nm long and 100nm wide, with the quantum dot
placed at the middle of the P-N junction interface.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pytest

# Add the parent directory to the path so we can import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import qdsim

class TestChromiumQDAlGaAsPNDiode:
    """Test class for validating a chromium QD in an AlGaAs P-N diode."""
    
    def setup_method(self):
        """Set up parameters for the simulation."""
        self.config = qdsim.Config()
        
        # Physical constants
        self.config.e_charge = 1.602e-19  # Elementary charge in C
        self.config.hbar = 1.054e-34      # Reduced Planck's constant in J·s
        self.config.m_e = 9.11e-31        # Electron mass in kg
        self.config.k_B = 1.381e-23       # Boltzmann constant in J/K
        self.config.epsilon_0 = 8.85e-12  # Vacuum permittivity in F/m
        
        # Material parameters for AlGaAs
        self.epsilon_r = 12.9             # Relative permittivity of AlGaAs
        self.m_star = 0.067 * self.config.m_e  # Effective mass in AlGaAs
        self.E_g = 1.424 * self.config.e_charge  # Band gap of GaAs in J
        
        # P-N junction parameters
        self.N_A = 1e24  # Acceptor doping concentration in m^-3 (1e18 cm^-3)
        self.N_D = 1e24  # Donor doping concentration in m^-3 (1e18 cm^-3)
        self.V_bi = 1.2 * self.config.e_charge  # Built-in potential in J
        self.V_reverse = 2.0 * self.config.e_charge  # Reverse bias in J
        
        # Quantum dot parameters
        self.QD_depth = 0.3 * self.config.e_charge  # QD potential depth in J
        self.QD_radius = 5e-9  # QD radius in m (5 nm)
        
        # Simulation domain parameters
        self.config.Lx = 200e-9  # 200 nm length
        self.config.Ly = 100e-9  # 100 nm width
        
        # Mesh parameters
        self.config.nx = 201  # Mesh points in x direction
        self.config.ny = 101  # Mesh points in y direction
        self.config.element_order = 2  # Quadratic elements for better accuracy
        
        # Disable MPI for tests
        self.config.use_mpi = False
        
        # Define the P-N junction and QD potentials
        self.define_potentials()
        
    def define_potentials(self):
        """Define the P-N junction and QD potentials."""
        # P-N junction parameters
        V_total = self.V_bi + self.V_reverse  # Total potential across junction
        
        # Calculate depletion width
        W_D = np.sqrt(2 * self.epsilon_r * self.config.epsilon_0 * V_total / 
                     (self.config.e_charge * (1/self.N_A + 1/self.N_D)))
        
        # Depletion region extends into p and n regions proportionally to doping
        W_p = W_D * self.N_D / (self.N_A + self.N_D)
        W_n = W_D * self.N_A / (self.N_A + self.N_D)
        
        print(f"Total depletion width: {W_D*1e9:.2f} nm")
        print(f"P-side depletion width: {W_p*1e9:.2f} nm")
        print(f"N-side depletion width: {W_n*1e9:.2f} nm")
        
        # P-N junction interface is at x = 0
        # P region: x < 0, N region: x > 0
        
        def pn_junction_potential(x, y):
            """
            Calculate the P-N junction potential.
            
            Args:
                x: x-coordinate in m
                y: y-coordinate in m
                
            Returns:
                Potential in J
            """
            # Convert to nm for easier calculation
            x_nm = x * 1e9
            
            # P-N junction interface is at x = 0
            if x < -W_p:
                # P region outside depletion region
                return 0
            elif x > W_n:
                # N region outside depletion region
                return -V_total
            else:
                # Inside depletion region - quadratic potential
                if x <= 0:
                    # P-side of depletion region
                    return -V_total * (x + W_p)**2 / (2 * W_p * W_D)
                else:
                    # N-side of depletion region
                    return -V_total * (1 - (x - W_n)**2 / (2 * W_n * W_D))
        
        def quantum_dot_potential(x, y):
            """
            Calculate the quantum dot potential.
            
            Args:
                x: x-coordinate in m
                y: y-coordinate in m
                
            Returns:
                Potential in J
            """
            # QD is centered at the P-N junction interface (0, 0)
            r = np.sqrt(x**2 + y**2)
            return -self.QD_depth * np.exp(-r**2 / (2 * self.QD_radius**2))
        
        def combined_potential(x, y):
            """
            Combine the P-N junction and QD potentials.
            
            Args:
                x: x-coordinate in m
                y: y-coordinate in m
                
            Returns:
                Combined potential in J
            """
            return pn_junction_potential(x, y) + quantum_dot_potential(x, y)
        
        # Store the potential functions
        self.pn_junction_potential = pn_junction_potential
        self.quantum_dot_potential = quantum_dot_potential
        self.combined_potential = combined_potential
        
        # Set the potential function for the simulator
        self.config.potential_function = combined_potential
        
        # Set the effective mass function
        self.config.m_star_function = lambda x, y: self.m_star
    
    def test_chromium_qd_algaas_pn_diode(self):
        """Test the chromium QD in AlGaAs P-N diode simulation."""
        print("\nTesting chromium QD in AlGaAs P-N diode...")
        
        # Create the simulator
        simulator = qdsim.Simulator(self.config)
        
        # Solve for the first 10 eigenstates
        eigenvalues, eigenvectors = simulator.run(num_eigenvalues=10)
        
        # Convert eigenvalues to eV
        eigenvalues_eV = np.real(eigenvalues) / self.config.e_charge
        
        # Print eigenvalues
        print("Eigenvalues (eV):")
        for i, E in enumerate(eigenvalues_eV):
            print(f"  E_{i}: {E:.6f} eV")
        
        # Check if there are bound states (negative eigenvalues)
        bound_states = [i for i, E in enumerate(eigenvalues_eV) if E < 0]
        print(f"Number of bound states: {len(bound_states)}")
        
        # Plot the potentials
        self.plot_potentials(simulator)
        
        # Plot the wavefunctions
        self.plot_wavefunctions(simulator, bound_states)
        
        # Plot the probability densities
        self.plot_probability_densities(simulator, bound_states)
        
        # Assert that there are bound states
        assert len(bound_states) > 0, "No bound states found in the quantum dot"
        
        # Assert that the eigenvalues are in a reasonable range
        assert np.all(eigenvalues_eV > -1.0), "Eigenvalues too negative"
        assert np.all(eigenvalues_eV < 1.0), "Eigenvalues too positive"
        
        return eigenvalues_eV, bound_states
    
    def plot_potentials(self, simulator):
        """
        Plot the P-N junction, QD, and combined potentials.
        
        Args:
            simulator: QDSim simulator instance
        """
        # Create a grid for plotting
        x = np.linspace(-self.config.Lx/2, self.config.Lx/2, 200)
        y = np.linspace(-self.config.Ly/2, self.config.Ly/2, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate potentials on the grid
        pn_potential = np.zeros_like(X)
        qd_potential = np.zeros_like(X)
        combined_potential = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pn_potential[i, j] = self.pn_junction_potential(X[i, j], Y[i, j])
                qd_potential[i, j] = self.quantum_dot_potential(X[i, j], Y[i, j])
                combined_potential[i, j] = self.combined_potential(X[i, j], Y[i, j])
        
        # Convert to eV for plotting
        pn_potential /= self.config.e_charge
        qd_potential /= self.config.e_charge
        combined_potential /= self.config.e_charge
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot P-N junction potential
        im1 = axs[0].contourf(X*1e9, Y*1e9, pn_potential, 50, cmap='viridis')
        axs[0].set_title('P-N Junction Potential')
        axs[0].set_xlabel('x (nm)')
        axs[0].set_ylabel('y (nm)')
        fig.colorbar(im1, ax=axs[0], label='Potential (eV)')
        
        # Plot QD potential
        im2 = axs[1].contourf(X*1e9, Y*1e9, qd_potential, 50, cmap='plasma')
        axs[1].set_title('Quantum Dot Potential')
        axs[1].set_xlabel('x (nm)')
        axs[1].set_ylabel('y (nm)')
        fig.colorbar(im2, ax=axs[1], label='Potential (eV)')
        
        # Plot combined potential
        im3 = axs[2].contourf(X*1e9, Y*1e9, combined_potential, 50, cmap='inferno')
        axs[2].set_title('Combined Potential')
        axs[2].set_xlabel('x (nm)')
        axs[2].set_ylabel('y (nm)')
        fig.colorbar(im3, ax=axs[2], label='Potential (eV)')
        
        # Add a line at x = 0 to indicate the P-N junction interface
        for ax in axs:
            ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('chromium_qd_algaas_pn_diode_potentials.png', dpi=300)
        
        # Also plot 1D slices of the potentials
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot along x-axis (y = 0)
        axs[0].plot(x*1e9, pn_potential[50, :], 'b-', label='P-N Junction')
        axs[0].plot(x*1e9, qd_potential[50, :], 'r-', label='Quantum Dot')
        axs[0].plot(x*1e9, combined_potential[50, :], 'g-', label='Combined')
        axs[0].set_title('Potential along y = 0')
        axs[0].set_xlabel('x (nm)')
        axs[0].set_ylabel('Potential (eV)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot along y-axis (x = 0)
        axs[1].plot(y*1e9, pn_potential[:, 100], 'b-', label='P-N Junction')
        axs[1].plot(y*1e9, qd_potential[:, 100], 'r-', label='Quantum Dot')
        axs[1].plot(y*1e9, combined_potential[:, 100], 'g-', label='Combined')
        axs[1].set_title('Potential along x = 0')
        axs[1].set_xlabel('y (nm)')
        axs[1].set_ylabel('Potential (eV)')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('chromium_qd_algaas_pn_diode_potential_slices.png', dpi=300)
    
    def plot_wavefunctions(self, simulator, bound_states):
        """
        Plot the wavefunctions of the bound states.
        
        Args:
            simulator: QDSim simulator instance
            bound_states: List of indices of bound states
        """
        if not bound_states:
            print("No bound states to plot")
            return
        
        # Create a figure with subplots for each bound state
        n_states = len(bound_states)
        n_cols = min(3, n_states)
        n_rows = (n_states + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axs = np.array([axs])
        axs = axs.flatten()
        
        # Plot each bound state
        for i, state_idx in enumerate(bound_states):
            if i < len(axs):
                simulator.plot_wavefunction(axs[i], state_idx=state_idx)
                axs[i].set_title(f'Wavefunction of State {state_idx}')
        
        # Hide unused subplots
        for i in range(n_states, len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('chromium_qd_algaas_pn_diode_wavefunctions.png', dpi=300)
    
    def plot_probability_densities(self, simulator, bound_states):
        """
        Plot the probability densities of the bound states.
        
        Args:
            simulator: QDSim simulator instance
            bound_states: List of indices of bound states
        """
        if not bound_states:
            print("No bound states to plot")
            return
        
        # Create a figure with subplots for each bound state
        n_states = len(bound_states)
        n_cols = min(3, n_states)
        n_rows = (n_states + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axs = np.array([axs])
        axs = axs.flatten()
        
        # Plot each bound state
        for i, state_idx in enumerate(bound_states):
            if i < len(axs):
                simulator.plot_probability_density(axs[i], state_idx=state_idx)
                axs[i].set_title(f'Probability Density of State {state_idx}')
        
        # Hide unused subplots
        for i in range(n_states, len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('chromium_qd_algaas_pn_diode_probability_densities.png', dpi=300)
    
    def test_reverse_bias_sweep(self):
        """Test the effect of reverse bias on the QD energy levels."""
        print("\nTesting reverse bias sweep...")
        
        # Reverse bias values to test
        reverse_biases = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # in V
        
        # Results
        eigenvalues_list = []
        bound_states_list = []
        
        for V_reverse in reverse_biases:
            print(f"Running simulation with reverse bias {V_reverse} V...")
            
            # Update reverse bias
            self.V_reverse = V_reverse * self.config.e_charge
            
            # Redefine potentials with new reverse bias
            self.define_potentials()
            
            # Create simulator
            simulator = qdsim.Simulator(self.config)
            
            # Solve for the first 10 eigenstates
            eigenvalues, eigenvectors = simulator.run(num_eigenvalues=10)
            
            # Convert eigenvalues to eV
            eigenvalues_eV = np.real(eigenvalues) / self.config.e_charge
            
            # Find bound states
            bound_states = [i for i, E in enumerate(eigenvalues_eV) if E < 0]
            
            # Store results
            eigenvalues_list.append(eigenvalues_eV)
            bound_states_list.append(bound_states)
            
            print(f"  Number of bound states: {len(bound_states)}")
            print(f"  Ground state energy: {eigenvalues_eV[0]:.6f} eV")
        
        # Plot energy levels vs. reverse bias
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each energy level
        for level in range(min(5, len(eigenvalues_list[0]))):
            energies = [evals[level] for evals in eigenvalues_list]
            ax.plot(reverse_biases, energies, 'o-', label=f'Level {level}')
        
        ax.set_xlabel('Reverse Bias (V)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Energy Levels vs. Reverse Bias')
        ax.legend()
        ax.grid(True)
        
        # Add a horizontal line at E = 0 to indicate the bound/unbound threshold
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('chromium_qd_algaas_pn_diode_bias_sweep.png', dpi=300)
        
        # Plot number of bound states vs. reverse bias
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_bound_states = [len(bs) for bs in bound_states_list]
        ax.plot(reverse_biases, num_bound_states, 'bo-')
        ax.set_xlabel('Reverse Bias (V)')
        ax.set_ylabel('Number of Bound States')
        ax.set_title('Number of Bound States vs. Reverse Bias')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('chromium_qd_algaas_pn_diode_bound_states.png', dpi=300)
        
        # Assert that the number of bound states decreases with increasing reverse bias
        assert num_bound_states[0] >= num_bound_states[-1], "Number of bound states should decrease with increasing reverse bias"
        
        return reverse_biases, eigenvalues_list, bound_states_list
    
    def test_qd_depth_sweep(self):
        """Test the effect of QD potential depth on the energy levels."""
        print("\nTesting QD potential depth sweep...")
        
        # QD potential depth values to test
        qd_depths = [0.1, 0.2, 0.3, 0.4, 0.5]  # in eV
        
        # Results
        eigenvalues_list = []
        bound_states_list = []
        
        # Fix reverse bias
        self.V_reverse = 2.0 * self.config.e_charge
        
        for depth in qd_depths:
            print(f"Running simulation with QD depth {depth} eV...")
            
            # Update QD depth
            self.QD_depth = depth * self.config.e_charge
            
            # Redefine potentials with new QD depth
            self.define_potentials()
            
            # Create simulator
            simulator = qdsim.Simulator(self.config)
            
            # Solve for the first 10 eigenstates
            eigenvalues, eigenvectors = simulator.run(num_eigenvalues=10)
            
            # Convert eigenvalues to eV
            eigenvalues_eV = np.real(eigenvalues) / self.config.e_charge
            
            # Find bound states
            bound_states = [i for i, E in enumerate(eigenvalues_eV) if E < 0]
            
            # Store results
            eigenvalues_list.append(eigenvalues_eV)
            bound_states_list.append(bound_states)
            
            print(f"  Number of bound states: {len(bound_states)}")
            print(f"  Ground state energy: {eigenvalues_eV[0]:.6f} eV")
        
        # Plot energy levels vs. QD depth
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each energy level
        for level in range(min(5, len(eigenvalues_list[0]))):
            energies = [evals[level] for evals in eigenvalues_list]
            ax.plot(qd_depths, energies, 'o-', label=f'Level {level}')
        
        ax.set_xlabel('QD Potential Depth (eV)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Energy Levels vs. QD Potential Depth')
        ax.legend()
        ax.grid(True)
        
        # Add a horizontal line at E = 0 to indicate the bound/unbound threshold
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('chromium_qd_algaas_pn_diode_depth_sweep.png', dpi=300)
        
        # Plot number of bound states vs. QD depth
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_bound_states = [len(bs) for bs in bound_states_list]
        ax.plot(qd_depths, num_bound_states, 'ro-')
        ax.set_xlabel('QD Potential Depth (eV)')
        ax.set_ylabel('Number of Bound States')
        ax.set_title('Number of Bound States vs. QD Potential Depth')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('chromium_qd_algaas_pn_diode_depth_bound_states.png', dpi=300)
        
        # Assert that the number of bound states increases with increasing QD depth
        assert num_bound_states[0] <= num_bound_states[-1], "Number of bound states should increase with increasing QD depth"
        
        return qd_depths, eigenvalues_list, bound_states_list

if __name__ == "__main__":
    # Run the tests
    test = TestChromiumQDAlGaAsPNDiode()
    test.setup_method()
    test.test_chromium_qd_algaas_pn_diode()
    test.test_reverse_bias_sweep()
    test.test_qd_depth_sweep()
