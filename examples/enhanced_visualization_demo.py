#!/usr/bin/env python3
"""
Enhanced visualization and analysis demo for QDSim.

This script demonstrates the enhanced visualization and analysis capabilities
of QDSim, including 3D visualization, interactive controls, and result analysis.

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
    """Main function demonstrating enhanced visualization and analysis."""
    print("QDSim Enhanced Visualization and Analysis Demo")
    
    # Create a simulator
    simulator = qdsim.Simulator()
    
    # Set up a quantum dot simulation
    # Parameters for a GaAs/AlGaAs quantum dot
    Lx = 100.0  # Domain size in nm
    Ly = 100.0
    nx = 101    # Number of mesh points
    ny = 101
    
    # Create mesh
    simulator.create_mesh(Lx, Ly, nx, ny)
    
    # Define potentials
    # P-N junction potential
    def pn_junction_potential(x, y):
        # Parameters
        V_bi = 1.0  # Built-in potential (eV)
        depletion_width = 50.0  # Depletion width (nm)
        
        # Convert to V
        V_bi_J = V_bi * 1.602e-19  # Convert eV to J
        
        # Calculate potential
        if abs(x) < depletion_width / 2:
            # Linear potential in depletion region
            return V_bi_J * (x / (depletion_width / 2) + 1) / 2
        elif x <= -depletion_width / 2:
            # P-side
            return 0
        else:
            # N-side
            return V_bi_J
    
    # Quantum dot potential (Gaussian well)
    def quantum_dot_potential(x, y):
        # Parameters
        depth = 0.3  # Potential depth (eV)
        radius = 10.0  # Quantum dot radius (nm)
        
        # Convert to J
        depth_J = depth * 1.602e-19  # Convert eV to J
        
        # Calculate distance from origin
        r = np.sqrt(x**2 + y**2)
        
        # Gaussian potential well
        return -depth_J * np.exp(-r**2 / (2 * radius**2))
    
    # Combined potential
    def combined_potential(x, y):
        return pn_junction_potential(x, y) + quantum_dot_potential(x, y)
    
    # Set potential
    simulator.set_potential(combined_potential)
    
    # Set material (GaAs)
    simulator.set_material("GaAs")
    
    # Solve for eigenstates
    print("Solving for eigenstates...")
    num_states = 10
    simulator.solve(num_states=num_states)
    
    # Get results
    mesh = simulator.get_mesh()
    eigenvectors = simulator.get_eigenvectors()
    eigenvalues = simulator.get_eigenvalues()
    potential = simulator.get_potential_values()
    
    # 1. Enhanced 3D Visualization
    print("\n1. Enhanced 3D Visualization")
    print("Creating 3D visualization of wavefunction and potential...")
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Create combined visualization
    axes = qdsim.plot_combined_visualization(
        fig=fig,
        mesh=mesh,
        eigenvector=eigenvectors[:, 0],  # Ground state
        potential_values=potential,
        eigenvalue=eigenvalues[0],
        use_nm=True,
        center_coords=True,
        convert_to_eV=True,
        azimuth=30,
        elevation=30,
        plot_type='surface'
    )
    
    # Save figure
    fig.savefig("enhanced_visualization.png", dpi=300, bbox_inches='tight')
    print("Saved 3D visualization to enhanced_visualization.png")
    
    # 2. Energy Level Analysis
    print("\n2. Energy Level Analysis")
    print("Creating energy level report...")
    
    # Create energy level report
    energy_report, energy_fig = qdsim.create_energy_level_report(
        eigenvalues=eigenvalues,
        convert_to_eV=True
    )
    
    # Print energy level information
    print(f"Ground state energy: {energy_report['ground_state_energy']:.6f} eV")
    print(f"Average level spacing: {energy_report['avg_spacing']:.6f} eV")
    print(f"Energy levels: {', '.join([f'{e:.6f}' for e in energy_report['energies']])}")
    
    # Save figure
    energy_fig.savefig("energy_levels.png", dpi=300, bbox_inches='tight')
    print("Saved energy level diagram to energy_levels.png")
    
    # 3. Transition Analysis
    print("\n3. Transition Analysis")
    print("Creating transition probability report...")
    
    # Create transition probability report
    transition_report, transition_figs = qdsim.create_transition_probability_report(
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        mesh=mesh,
        convert_to_eV=True
    )
    
    # Save figures
    for i, fig in enumerate(transition_figs):
        fig.savefig(f"transition_analysis_{i+1}.png", dpi=300, bbox_inches='tight')
    print(f"Saved {len(transition_figs)} transition analysis figures")
    
    # 4. Wavefunction Localization Analysis
    print("\n4. Wavefunction Localization Analysis")
    print("Creating wavefunction localization report...")
    
    # Create wavefunction localization report
    localization_report, localization_fig = qdsim.create_wavefunction_localization_report(
        eigenvectors=eigenvectors,
        mesh=mesh,
        eigenvalues=eigenvalues,
        convert_to_eV=True
    )
    
    # Save figure
    localization_fig.savefig("wavefunction_localization.png", dpi=300, bbox_inches='tight')
    print("Saved wavefunction localization analysis to wavefunction_localization.png")
    
    # Export results to CSV
    qdsim.export_results_to_csv(localization_report, "localization_results.csv")
    print("Exported localization results to localization_results.csv")
    
    # 5. Interactive Visualization
    print("\n5. Interactive Visualization")
    print("Launching interactive visualization...")
    print("Close the visualization window to continue.")
    
    # Show interactive visualization
    qdsim.show_enhanced_visualization(
        mesh=mesh,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        potential=potential
    )
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
