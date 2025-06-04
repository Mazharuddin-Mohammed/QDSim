#!/usr/bin/env python3
"""
Test script to verify the dimensional consistency and parameter scaling fixes.
This script runs simulations with different configurations and checks that
potentials and energies are within physically meaningful ranges.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction, plot_potential, plot_energy_shift

# Create output directory
os.makedirs('test_results', exist_ok=True)

def run_test_case(config, case_name):
    """Run a test case with the given configuration."""
    print(f"\nRunning test case: {case_name}")
    
    # Create simulator
    sim = Simulator(config)
    
    # Run simulation
    eigenvalues, eigenvectors = sim.run(num_eigenvalues=3)
    
    # Convert eigenvalues to eV for display
    eigenvalues_eV = np.real(eigenvalues) / config.e_charge
    
    # Print results
    print(f"  Domain size: {config.Lx*1e9:.1f} x {config.Ly*1e9:.1f} nm")
    print(f"  QD radius: {config.R*1e9:.1f} nm")
    print(f"  Potential type: {config.potential_type}")
    print(f"  Potential depth: {config.V_0:.2f} eV")
    print(f"  Eigenvalues (eV): {', '.join([f'{e:.4f}' for e in eigenvalues_eV])}")
    
    # Create figure for visualization
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot potential
    ax_pot = fig.add_subplot(gs[0, 0:2])
    plot_potential(ax_pot, sim.mesh, sim.phi, 
                  title=f"Potential - {case_name}", 
                  convert_to_eV=True)
    
    # Plot ground state wavefunction
    ax_wf0 = fig.add_subplot(gs[0, 2])
    plot_wavefunction(ax_wf0, sim.mesh, eigenvectors[:, 0], 
                     title=f"Ground State (E = {eigenvalues_eV[0]:.4f} eV)")
    
    # Plot first excited state wavefunction
    ax_wf1 = fig.add_subplot(gs[1, 0])
    plot_wavefunction(ax_wf1, sim.mesh, eigenvectors[:, 1], 
                     title=f"1st Excited State (E = {eigenvalues_eV[1]:.4f} eV)")
    
    # Plot second excited state wavefunction
    ax_wf2 = fig.add_subplot(gs[1, 1])
    plot_wavefunction(ax_wf2, sim.mesh, eigenvectors[:, 2], 
                     title=f"2nd Excited State (E = {eigenvalues_eV[2]:.4f} eV)")
    
    # Plot 1D slice of potential
    ax_slice = fig.add_subplot(gs[1, 2])
    nodes = np.array(sim.mesh.get_nodes())
    phi = sim.phi
    
    # Find nodes close to y=0
    y_tolerance = config.Ly / 50
    central_nodes = [i for i, node in enumerate(nodes) if abs(node[1]) < y_tolerance]
    
    # Sort by x coordinate
    central_nodes.sort(key=lambda i: nodes[i][0])
    
    # Extract x coordinates and potential values
    x_coords = [nodes[i][0] for i in central_nodes]
    pot_values = [phi[i] for i in central_nodes]
    
    # Convert to eV and nm
    x_coords_nm = np.array(x_coords) * 1e9
    pot_values_eV = np.array(pot_values) / config.e_charge
    
    # Plot
    ax_slice.plot(x_coords_nm, pot_values_eV)
    ax_slice.set_xlabel('x (nm)')
    ax_slice.set_ylabel('Potential (eV)')
    ax_slice.set_title('Potential (1D slice at y≈0)')
    ax_slice.grid(True)
    
    # Add horizontal lines for eigenvalues
    for i, e in enumerate(eigenvalues_eV):
        ax_slice.axhline(y=e, color=f'C{i+1}', linestyle='--', 
                        label=f'E{i} = {e:.4f} eV')
    
    ax_slice.legend()
    
    # Save figure
    fig.tight_layout()
    fig.savefig(f'test_results/{case_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.close(fig)
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'simulator': sim
    }

# Test cases
test_cases = [
    # 1. Square potential with shallow depth
    {
        'name': 'Square Shallow',
        'config': {
            'potential_type': 'square',
            'V_0': 0.3,
            'R': 15e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40
        }
    },
    # 2. Square potential with deep depth
    {
        'name': 'Square Deep',
        'config': {
            'potential_type': 'square',
            'V_0': 1.5,
            'R': 15e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40
        }
    },
    # 3. Gaussian potential with shallow depth
    {
        'name': 'Gaussian Shallow',
        'config': {
            'potential_type': 'gaussian',
            'V_0': 0.3,
            'R': 15e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40
        }
    },
    # 4. Gaussian potential with deep depth
    {
        'name': 'Gaussian Deep',
        'config': {
            'potential_type': 'gaussian',
            'V_0': 1.5,
            'R': 15e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40
        }
    },
    # 5. Small QD
    {
        'name': 'Small QD',
        'config': {
            'potential_type': 'square',
            'V_0': 0.5,
            'R': 5e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40
        }
    },
    # 6. Large QD
    {
        'name': 'Large QD',
        'config': {
            'potential_type': 'square',
            'V_0': 0.5,
            'R': 30e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40
        }
    }
]

# Run all test cases
results = {}
for case in test_cases:
    # Create base config
    config = Config()
    config.use_mpi = False
    config.max_refinements = 2
    config.adaptive_threshold = 0.05
    
    # Apply case-specific config
    for key, value in case['config'].items():
        setattr(config, key, value)
    
    # Validate config
    config.validate()
    
    # Run test
    results[case['name']] = run_test_case(config, case['name'])

# Create comparison figure for eigenvalues
fig_compare = plt.figure(figsize=(12, 8))
ax_compare = fig_compare.add_subplot(111)

# Plot eigenvalues for each case
x_positions = np.arange(len(test_cases))
width = 0.2
for i in range(3):  # For the first 3 eigenvalues
    eigenvalues = [np.real(results[case['name']]['eigenvalues'][i]) / 1.602e-19 
                  for case in test_cases]
    ax_compare.bar(x_positions + i*width, eigenvalues, width, 
                  label=f'E{i}')

# Set labels
ax_compare.set_xlabel('Test Case')
ax_compare.set_ylabel('Energy (eV)')
ax_compare.set_title('Comparison of Eigenvalues Across Test Cases')
ax_compare.set_xticks(x_positions + width)
ax_compare.set_xticklabels([case['name'] for case in test_cases])
ax_compare.legend()
ax_compare.grid(True, linestyle='--', alpha=0.7)

# Save comparison figure
fig_compare.tight_layout()
fig_compare.savefig('test_results/eigenvalue_comparison.png', dpi=300)

print("\nAll tests completed. Results saved to 'test_results' directory.")
