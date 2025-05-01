#!/usr/bin/env python3
"""
Test script to verify that potentials are properly scaled and stay within
physically meaningful ranges (below ~100 eV).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qdsim import Simulator, Config
from qdsim.visualization import plot_potential

# Create output directory
os.makedirs('test_results_potential', exist_ok=True)

def test_potential_scaling(potential_type, V_0_values, title):
    """Test potential scaling with different potential depths."""
    print(f"\nTesting {potential_type} potential scaling:")

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    fig.suptitle(f"{title} - Potential Scaling Test", fontsize=16)

    # 1D plot for comparison
    ax_1d = fig.add_subplot(gs[0, :])
    ax_1d.set_xlabel('Position (nm)')
    ax_1d.set_ylabel('Potential (eV)')
    ax_1d.set_title('1D Potential Profile (y=0)')
    ax_1d.grid(True)

    # Create x-axis for 1D plot
    x = np.linspace(-50, 50, 1000)  # nm
    y = np.zeros_like(x)

    # Plot potentials for each V_0 (up to 3 for the 2D plots)
    results = []
    for i, V_0 in enumerate(V_0_values):
        print(f"  Testing V_0 = {V_0} eV")

        # Create config
        config = Config()
        config.use_mpi = False
        config.potential_type = potential_type
        config.V_0 = V_0
        config.R = 15e-9
        config.nx = 30
        config.ny = 30
        config.element_order = 1
        config.max_refinements = 0

        # Validate config
        try:
            config.validate()
        except AssertionError as e:
            print(f"  Validation failed: {e}")
            continue

        # Create simulator
        sim = Simulator(config)

        # Get potential values
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

        # Plot 1D slice
        ax_1d.plot(x_coords_nm, pot_values_eV, 'o-', label=f'V_0 = {V_0} eV')

        # Plot 2D potential (only for the first 3 values to fit in the grid)
        if i < 3:
            ax_2d = fig.add_subplot(gs[1, i])
            im = plot_potential(ax_2d, sim.mesh, phi,
                               title=f'V_0 = {V_0} eV',
                               convert_to_eV=True)

        # Store results
        results.append({
            'V_0': V_0,
            'x_coords': x_coords_nm,
            'pot_values': pot_values_eV,
            'min_pot': np.min(pot_values_eV),
            'max_pot': np.max(pot_values_eV)
        })

    # Add legend to 1D plot
    ax_1d.legend()

    # Save figure
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'test_results_potential/{potential_type}_scaling.png', dpi=300)
    plt.close(fig)

    # Create summary figure
    fig_summary = plt.figure(figsize=(10, 6))
    ax_summary = fig_summary.add_subplot(111)
    ax_summary.set_xlabel('Configured V_0 (eV)')
    ax_summary.set_ylabel('Actual Potential Range (eV)')
    ax_summary.set_title(f'{title} - Potential Range vs. Configured V_0')

    # Plot min and max potential values
    v0_values = [r['V_0'] for r in results]
    min_values = [r['min_pot'] for r in results]
    max_values = [r['max_pot'] for r in results]

    ax_summary.plot(v0_values, min_values, 'o-', label='Min Potential')
    ax_summary.plot(v0_values, max_values, 's-', label='Max Potential')

    # Add reference line (y=x)
    ax_summary.plot([0, max(v0_values)], [0, max(v0_values)], 'k--', alpha=0.5, label='y=x')

    # Add horizontal line at 100 eV
    ax_summary.axhline(y=100, color='r', linestyle='--', label='100 eV Limit')

    ax_summary.legend()
    ax_summary.grid(True)

    # Save summary figure
    fig_summary.tight_layout()
    fig_summary.savefig(f'test_results_potential/{potential_type}_summary.png', dpi=300)
    plt.close(fig_summary)

    return results

# Test square potential with various depths
square_results = test_potential_scaling(
    'square',
    [0.5, 1.0, 5.0, 10.0, 50.0, 200.0],  # Including values beyond the 100 eV limit
    'Square Potential'
)

# Test gaussian potential with various depths
gaussian_results = test_potential_scaling(
    'gaussian',
    [0.5, 1.0, 5.0, 10.0, 50.0, 200.0],  # Including values beyond the 100 eV limit
    'Gaussian Potential'
)

# Create comparison figure
fig_compare = plt.figure(figsize=(12, 8))
ax_compare = fig_compare.add_subplot(111)
ax_compare.set_xlabel('Configured V_0 (eV)')
ax_compare.set_ylabel('Maximum Potential (eV)')
ax_compare.set_title('Maximum Potential vs. Configured V_0')

# Plot max potential values for both potential types
square_v0 = [r['V_0'] for r in square_results]
square_max = [r['max_pot'] for r in square_results]
gaussian_v0 = [r['V_0'] for r in gaussian_results]
gaussian_max = [r['max_pot'] for r in gaussian_results]

ax_compare.plot(square_v0, square_max, 'o-', label='Square Potential')
ax_compare.plot(gaussian_v0, gaussian_max, 's-', label='Gaussian Potential')

# Add reference line (y=x)
max_v0 = max(max(square_v0), max(gaussian_v0))
ax_compare.plot([0, max_v0], [0, max_v0], 'k--', alpha=0.5, label='y=x')

# Add horizontal line at 100 eV
ax_compare.axhline(y=100, color='r', linestyle='--', label='100 eV Limit')

ax_compare.legend()
ax_compare.grid(True)

# Save comparison figure
fig_compare.tight_layout()
fig_compare.savefig('test_results_potential/potential_comparison.png', dpi=300)

print("\nAll potential scaling tests completed. Results saved to 'test_results_potential' directory.")
