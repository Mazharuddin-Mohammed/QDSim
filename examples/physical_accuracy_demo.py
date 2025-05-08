#!/usr/bin/env python3
"""
Physical accuracy demonstration for QDSim.

This script demonstrates the physical accuracy enhancements in QDSim,
including realistic physical parameters, expanded material database,
and additional quantum effects like spin-orbit coupling.

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path to import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qdsim

def main():
    """Main function demonstrating physical accuracy enhancements."""
    print("QDSim Physical Accuracy Demonstration")
    print("=====================================")
    
    # Create a simulator
    simulator = qdsim.Simulator()
    
    # 1. Realistic Physical Parameters
    print("\n1. Realistic Physical Parameters")
    
    # Create mesh
    Lx = 100.0  # Domain size in nm
    Ly = 100.0
    nx = 101    # Number of mesh points
    ny = 101
    simulator.create_mesh(Lx, Ly, nx, ny)
    
    # Define a realistic quantum dot potential
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
    
    # Set potential
    simulator.set_potential(quantum_dot_potential)
    
    # Set material (GaAs)
    simulator.set_material("GaAs")
    
    # Solve for eigenstates
    print("Solving for eigenstates with realistic parameters...")
    simulator.solve(num_states=5)
    
    # Get results
    eigenvalues = simulator.get_eigenvalues()
    
    # Convert eigenvalues to eV
    eigenvalues_eV = np.array(eigenvalues) / 1.602e-19
    
    # Print eigenvalues
    print("Eigenvalues (eV):")
    for i, e in enumerate(eigenvalues_eV):
        print(f"  E{i} = {e:.6f} eV")
    
    # Plot potential and ground state wavefunction
    fig, ax = plt.subplots(figsize=(10, 8))
    simulator.plot_potential(ax, convert_to_eV=True)
    plt.savefig("realistic_potential.png", dpi=300, bbox_inches='tight')
    print("Saved realistic potential to realistic_potential.png")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    simulator.plot_wavefunction(ax, state_idx=0)
    plt.savefig("realistic_wavefunction.png", dpi=300, bbox_inches='tight')
    print("Saved realistic wavefunction to realistic_wavefunction.png")
    
    # 2. Expanded Material Database
    print("\n2. Expanded Material Database")
    
    # Get available materials
    materials = simulator.get_available_materials()
    print(f"Available materials: {', '.join(materials)}")
    
    # Print properties of GaAs
    print("\nProperties of GaAs:")
    gaas = simulator.get_material_properties("GaAs")
    print(f"  Electron effective mass: {gaas['m_e']:.3f} m_0")
    print(f"  Hole effective mass: {gaas['m_h']:.3f} m_0")
    print(f"  Bandgap: {gaas['E_g']:.3f} eV")
    print(f"  Dielectric constant: {gaas['epsilon_r']:.1f}")
    print(f"  Lattice constant: {gaas['lattice_constant']:.6f} nm")
    print(f"  Spin-orbit splitting: {gaas['spin_orbit_splitting']:.3f} eV")
    
    # Create AlGaAs alloy with different compositions
    print("\nProperties of AlGaAs alloys with different compositions:")
    compositions = [0.1, 0.2, 0.3, 0.4, 0.5]
    bandgaps = []
    
    for x in compositions:
        algaas = simulator.create_alloy("AlAs", "GaAs", x)
        bandgaps.append(algaas['E_g'])
        print(f"  Al{x:.1f}Ga{1-x:.1f}As bandgap: {algaas['E_g']:.3f} eV")
    
    # Plot bandgap vs. composition
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(compositions, bandgaps, 'bo-')
    ax.set_xlabel('Al Composition (x)')
    ax.set_ylabel('Bandgap (eV)')
    ax.set_title('AlGaAs Bandgap vs. Composition')
    ax.grid(True)
    plt.savefig("algaas_bandgap.png", dpi=300, bbox_inches='tight')
    print("Saved AlGaAs bandgap plot to algaas_bandgap.png")
    
    # Get temperature-dependent properties
    print("\nTemperature-dependent properties of GaAs:")
    temperatures = [77, 150, 300, 400]
    bandgaps_T = []
    
    for T in temperatures:
        gaas_T = simulator.get_material_at_temperature("GaAs", T)
        bandgaps_T.append(gaas_T['E_g'])
        print(f"  GaAs bandgap at {T} K: {gaas_T['E_g']:.3f} eV")
    
    # Plot bandgap vs. temperature
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temperatures, bandgaps_T, 'ro-')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Bandgap (eV)')
    ax.set_title('GaAs Bandgap vs. Temperature')
    ax.grid(True)
    plt.savefig("gaas_bandgap_temperature.png", dpi=300, bbox_inches='tight')
    print("Saved GaAs temperature-dependent bandgap plot to gaas_bandgap_temperature.png")
    
    # 3. Spin-Orbit Coupling
    print("\n3. Spin-Orbit Coupling")
    
    # Create a new simulator for spin-orbit coupling
    simulator_so = qdsim.Simulator()
    
    # Create mesh
    simulator_so.create_mesh(Lx, Ly, nx, ny)
    
    # Set potential
    simulator_so.set_potential(quantum_dot_potential)
    
    # Set material (InGaAs, which has stronger spin-orbit coupling)
    simulator_so.set_material("InGaAs")
    
    # Enable spin-orbit coupling
    simulator_so.enable_spin_orbit_coupling(
        spin_orbit_type="rashba",
        rashba_parameter=0.05,  # eV·nm
        dresselhaus_parameter=0.0
    )
    
    # Solve for eigenstates with spin-orbit coupling
    print("Solving for eigenstates with spin-orbit coupling...")
    simulator_so.solve(num_states=10)
    
    # Get results
    eigenvalues_so = simulator_so.get_eigenvalues()
    
    # Convert eigenvalues to eV
    eigenvalues_so_eV = np.array(eigenvalues_so) / 1.602e-19
    
    # Print eigenvalues
    print("Eigenvalues with spin-orbit coupling (eV):")
    for i, e in enumerate(eigenvalues_so_eV):
        print(f"  E{i} = {e:.6f} eV")
    
    # Calculate spin-orbit splitting
    spin_orbit_splittings = []
    for i in range(0, len(eigenvalues_so_eV), 2):
        if i + 1 < len(eigenvalues_so_eV):
            splitting = eigenvalues_so_eV[i+1] - eigenvalues_so_eV[i]
            spin_orbit_splittings.append(splitting)
            print(f"  Spin-orbit splitting E{i+1}-E{i}: {splitting*1000:.3f} meV")
    
    # Plot spin-up and spin-down wavefunctions
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    simulator_so.plot_wavefunction(axes[0], state_idx=0, title="Spin-up Wavefunction")
    simulator_so.plot_wavefunction(axes[1], state_idx=1, title="Spin-down Wavefunction")
    plt.savefig("spin_orbit_wavefunctions.png", dpi=300, bbox_inches='tight')
    print("Saved spin-orbit wavefunctions to spin_orbit_wavefunctions.png")
    
    # Plot energy level diagram
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Plot energy levels
    for i, energy in enumerate(eigenvalues_so_eV[:10]):
        # Plot horizontal line for energy level
        ax.plot([-0.2, 0.2], [energy, energy], 'b-', linewidth=2)
        
        # Add label
        ax.text(0.25, energy, f'E{i} = {energy:.6f} eV', 
               va='center', ha='left')
        
        # Add spin label
        spin_label = "↑" if i % 2 == 0 else "↓"
        ax.text(-0.3, energy, spin_label, va='center', ha='center', fontsize=16)
    
    # Plot spin-orbit splitting arrows
    for i in range(len(spin_orbit_splittings)):
        base_idx = i * 2
        # Plot arrow for spin-orbit splitting
        ax.arrow(-0.1, eigenvalues_so_eV[base_idx], 0, 
                spin_orbit_splittings[i] - 0.005,
                head_width=0.03, head_length=0.005, fc='r', ec='r', 
                length_includes_head=True)
        
        # Add splitting label
        ax.text(-0.4, eigenvalues_so_eV[base_idx] + spin_orbit_splittings[i]/2, 
               f'ΔE = {spin_orbit_splittings[i]*1000:.3f} meV', 
               va='center', ha='right', color='r')
    
    # Set labels and title
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Energy Level Diagram with Spin-Orbit Coupling')
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("spin_orbit_energy_levels.png", dpi=300, bbox_inches='tight')
    print("Saved spin-orbit energy level diagram to spin_orbit_energy_levels.png")
    
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main()
