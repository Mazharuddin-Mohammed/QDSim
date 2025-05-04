#!/usr/bin/env python3
"""
Documentation Demonstration for QDSim.

This script demonstrates the comprehensive documentation features of QDSim,
including code documentation, user guide examples, and theory implementation.

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time

# Add the parent directory to the path to import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qdsim

def main():
    """Main function demonstrating documentation features."""
    print("QDSim Documentation Demonstration")
    print("=================================")
    
    # Enable logging to demonstrate error messages
    qdsim.enable_logging("qdsim_demo.log", log_level="DEBUG")
    
    # 1. Basic Usage (from User Guide)
    print("\n1. Basic Usage (from User Guide)")
    
    # Create a simulator
    simulator = qdsim.Simulator()
    
    # Create a mesh
    Lx = 100.0  # Domain size in nm
    Ly = 100.0
    nx = 101    # Number of mesh points
    ny = 101
    simulator.create_mesh(Lx, Ly, nx, ny)
    print(f"Created mesh with {nx}x{ny} points")
    
    # Define a harmonic oscillator potential
    def harmonic_potential(x, y):
        # Parameters
        k = 0.1  # Spring constant in eV/nm²
        
        # Convert to J
        k_J = k * 1.602e-19  # Convert eV/nm² to J/m²
        
        # Calculate potential
        return k_J * (x**2 + y**2)
    
    # Set the potential
    simulator.set_potential(harmonic_potential)
    print("Set harmonic oscillator potential")
    
    # Set the material to GaAs
    simulator.set_material("GaAs")
    print("Set material to GaAs")
    
    # Solve for the first 10 eigenstates
    print("Solving for the first 10 eigenstates...")
    start_time = time.time()
    simulator.solve(num_states=10)
    end_time = time.time()
    print(f"Solved in {end_time - start_time:.2f} seconds")
    
    # Get eigenvalues
    eigenvalues = simulator.get_eigenvalues()
    
    # Convert eigenvalues to eV
    eigenvalues_eV = np.array(eigenvalues) / 1.602e-19
    
    # Print eigenvalues
    print("\nEigenvalues (eV):")
    for i, e in enumerate(eigenvalues_eV):
        print(f"  E{i} = {e:.6f} eV")
    
    # Plot potential
    fig, ax = plt.subplots(figsize=(10, 8))
    simulator.plot_potential(ax, convert_to_eV=True)
    plt.savefig("potential.png", dpi=300, bbox_inches='tight')
    print("Saved potential plot to potential.png")
    
    # Plot ground state wavefunction
    fig, ax = plt.subplots(figsize=(10, 8))
    simulator.plot_wavefunction(ax, state_idx=0)
    plt.savefig("wavefunction_0.png", dpi=300, bbox_inches='tight')
    print("Saved ground state wavefunction plot to wavefunction_0.png")
    
    # Plot first excited state wavefunction
    fig, ax = plt.subplots(figsize=(10, 8))
    simulator.plot_wavefunction(ax, state_idx=1)
    plt.savefig("wavefunction_1.png", dpi=300, bbox_inches='tight')
    print("Saved first excited state wavefunction plot to wavefunction_1.png")
    
    # 2. Advanced Features (from User Guide)
    print("\n2. Advanced Features (from User Guide)")
    
    # Create a custom material
    custom_material = {
        "m_e": 0.1,           # Electron effective mass
        "m_h": 0.5,           # Hole effective mass
        "E_g": 1.5,           # Bandgap in eV
        "epsilon_r": 12.0,    # Dielectric constant
        "Delta_E_c": 0.7,     # Conduction band offset in eV
        "Delta_E_v": 0.3      # Valence band offset in eV
    }
    
    # Add the custom material to the database
    simulator.add_material("CustomMaterial", custom_material)
    print("Added custom material 'CustomMaterial'")
    
    # Use the custom material
    simulator.set_material("CustomMaterial")
    print("Set material to 'CustomMaterial'")
    
    # Solve with the custom material
    print("Solving with the custom material...")
    start_time = time.time()
    simulator.solve(num_states=10)
    end_time = time.time()
    print(f"Solved in {end_time - start_time:.2f} seconds")
    
    # Get eigenvalues
    eigenvalues = simulator.get_eigenvalues()
    
    # Convert eigenvalues to eV
    eigenvalues_eV = np.array(eigenvalues) / 1.602e-19
    
    # Print eigenvalues
    print("\nEigenvalues with custom material (eV):")
    for i, e in enumerate(eigenvalues_eV):
        print(f"  E{i} = {e:.6f} eV")
    
    # 3. Theory Implementation (from Theory Documentation)
    print("\n3. Theory Implementation (from Theory Documentation)")
    
    # Create a new simulator for theory demonstration
    simulator_theory = qdsim.Simulator()
    
    # Create a mesh
    simulator_theory.create_mesh(Lx, Ly, nx, ny)
    
    # Define a quantum dot potential based on the theory
    def quantum_dot_potential(x, y):
        # Parameters
        V0 = 0.3  # Potential depth (eV)
        radius = 10.0  # Quantum dot radius (nm)
        
        # Convert to J
        V0_J = V0 * 1.602e-19  # Convert eV to J
        
        # Calculate distance from origin
        r = np.sqrt(x**2 + y**2)
        
        # Gaussian potential well
        return -V0_J * np.exp(-r**2 / (2 * radius**2))
    
    # Set the potential
    simulator_theory.set_potential(quantum_dot_potential)
    print("Set quantum dot potential")
    
    # Set the material to GaAs
    simulator_theory.set_material("GaAs")
    print("Set material to GaAs")
    
    # Enable spin-orbit coupling
    simulator_theory.enable_spin_orbit_coupling(
        spin_orbit_type="rashba",
        rashba_parameter=0.05,  # eV·nm
        dresselhaus_parameter=0.0
    )
    print("Enabled Rashba spin-orbit coupling")
    
    # Solve with spin-orbit coupling
    print("Solving with spin-orbit coupling...")
    start_time = time.time()
    simulator_theory.solve(num_states=10)
    end_time = time.time()
    print(f"Solved in {end_time - start_time:.2f} seconds")
    
    # Get eigenvalues
    eigenvalues = simulator_theory.get_eigenvalues()
    
    # Convert eigenvalues to eV
    eigenvalues_eV = np.array(eigenvalues) / 1.602e-19
    
    # Print eigenvalues
    print("\nEigenvalues with spin-orbit coupling (eV):")
    for i, e in enumerate(eigenvalues_eV):
        print(f"  E{i} = {e:.6f} eV")
    
    # Calculate spin-orbit splitting
    print("\nSpin-orbit splitting:")
    for i in range(0, len(eigenvalues_eV), 2):
        if i + 1 < len(eigenvalues_eV):
            splitting = eigenvalues_eV[i+1] - eigenvalues_eV[i]
            print(f"  E{i+1} - E{i} = {splitting*1000:.3f} meV")
    
    # Plot spin-up and spin-down wavefunctions
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    simulator_theory.plot_wavefunction(axes[0], state_idx=0, title="Spin-up Wavefunction")
    simulator_theory.plot_wavefunction(axes[1], state_idx=1, title="Spin-down Wavefunction")
    plt.savefig("spin_orbit_wavefunctions.png", dpi=300, bbox_inches='tight')
    print("Saved spin-orbit wavefunctions plot to spin_orbit_wavefunctions.png")
    
    # 4. Interactive Visualization
    print("\n4. Interactive Visualization")
    
    # Create an interactive visualization
    create_interactive_visualization(simulator_theory)
    
    print("\nDemonstration completed successfully!")

def create_interactive_visualization(simulator):
    """
    Create an interactive visualization with sliders and radio buttons.
    
    Args:
        simulator: Simulator object with solved eigenstates
    """
    # Get eigenvalues and eigenvectors
    eigenvalues = simulator.get_eigenvalues()
    eigenvectors = simulator.get_eigenvectors()
    
    # Convert eigenvalues to eV
    eigenvalues_eV = np.array(eigenvalues) / 1.602e-19
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    # Initial state index
    state_idx = 0
    
    # Plot initial wavefunction
    plot = simulator.plot_wavefunction(ax, state_idx=state_idx)
    
    # Add slider for state index
    ax_state = plt.axes([0.25, 0.1, 0.65, 0.03])
    state_slider = Slider(
        ax=ax_state,
        label='State Index',
        valmin=0,
        valmax=len(eigenvalues_eV) - 1,
        valinit=state_idx,
        valstep=1
    )
    
    # Add radio buttons for plot type
    ax_radio = plt.axes([0.025, 0.5, 0.15, 0.15])
    radio = RadioButtons(
        ax=ax_radio,
        labels=['Wavefunction', 'Probability', 'Potential'],
        active=0
    )
    
    # Update function for slider
    def update_state(val):
        ax.clear()
        state_idx = int(val)
        if radio.value_selected == 'Wavefunction':
            simulator.plot_wavefunction(ax, state_idx=state_idx)
        elif radio.value_selected == 'Probability':
            simulator.plot_probability(ax, state_idx=state_idx)
        else:  # Potential
            simulator.plot_potential(ax, convert_to_eV=True)
        
        # Update title
        if radio.value_selected != 'Potential':
            ax.set_title(f"State {state_idx}: E = {eigenvalues_eV[state_idx]:.6f} eV")
        
        fig.canvas.draw_idle()
    
    # Update function for radio buttons
    def update_plot_type(label):
        ax.clear()
        state_idx = int(state_slider.val)
        if label == 'Wavefunction':
            simulator.plot_wavefunction(ax, state_idx=state_idx)
        elif label == 'Probability':
            simulator.plot_probability(ax, state_idx=state_idx)
        else:  # Potential
            simulator.plot_potential(ax, convert_to_eV=True)
        
        # Update title
        if label != 'Potential':
            ax.set_title(f"State {state_idx}: E = {eigenvalues_eV[state_idx]:.6f} eV")
        
        fig.canvas.draw_idle()
    
    # Register update functions
    state_slider.on_changed(update_state)
    radio.on_clicked(update_plot_type)
    
    # Add reset button
    ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(ax_reset, 'Reset', hovercolor='0.975')
    
    def reset(event):
        state_slider.reset()
    
    reset_button.on_clicked(reset)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
