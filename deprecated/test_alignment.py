#!/usr/bin/env python3
"""
Test script to verify the consistency of mesh, material alignment, and
coordinate systems in visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction, plot_potential

# Create output directory
os.makedirs('test_results_alignment', exist_ok=True)

def test_alignment_test(config, title):
    """Test coordinate alignment between mesh, potential, and wavefunctions."""
    print(f"\nTesting coordinate alignment: {title}")

    # Create simulator
    sim = Simulator(config)

    # Run simulation
    eigenvalues, eigenvectors = sim.run(num_eigenvalues=3)

    # Convert eigenvalues to eV for display
    eigenvalues_eV = np.real(eigenvalues) / config.e_charge
    print(f"  Eigenvalues (eV): {', '.join([f'{e:.4f}' for e in eigenvalues_eV])}")

    # Get mesh data
    nodes = np.array(sim.mesh.get_nodes())
    elements = np.array(sim.mesh.get_elements())

    # Create figure for raw coordinates
    fig_raw = plt.figure(figsize=(15, 10))
    gs_raw = GridSpec(2, 3, figure=fig_raw)
    fig_raw.suptitle(f"{title} - Raw Coordinates", fontsize=16)

    # Plot mesh
    ax_mesh = fig_raw.add_subplot(gs_raw[0, 0])
    ax_mesh.triplot(nodes[:, 0]*1e9, nodes[:, 1]*1e9, elements, 'k-', lw=0.5, alpha=0.5)
    ax_mesh.set_xlabel('x (nm)')
    ax_mesh.set_ylabel('y (nm)')
    ax_mesh.set_title('Mesh')
    ax_mesh.set_aspect('equal')

    # Plot potential
    ax_pot = fig_raw.add_subplot(gs_raw[0, 1:])
    plot_potential(ax_pot, sim.mesh, sim.phi,
                  title='Potential',
                  convert_to_eV=True,
                  use_nm=True,
                  center_coords=False)

    # Plot wavefunctions
    for i in range(3):
        ax_wf = fig_raw.add_subplot(gs_raw[1, i])
        plot_wavefunction(ax_wf, sim.mesh, eigenvectors[:, i],
                         title=f'Wavefunction {i} (E = {eigenvalues_eV[i]:.4f} eV)',
                         use_nm=True,
                         center_coords=False)

    # Save raw coordinates figure
    fig_raw.tight_layout(rect=[0, 0, 1, 0.95])
    fig_raw.savefig(f'test_results_alignment/{title.replace(" ", "_").lower()}_raw.png', dpi=300)
    plt.close(fig_raw)

    # Create figure for centered coordinates
    fig_centered = plt.figure(figsize=(15, 10))
    gs_centered = GridSpec(2, 3, figure=fig_centered)
    fig_centered.suptitle(f"{title} - Centered Coordinates", fontsize=16)

    # Calculate center
    x_center = np.mean(nodes[:, 0])
    y_center = np.mean(nodes[:, 1])
    print(f"  Mesh center: ({x_center*1e9:.2f}, {y_center*1e9:.2f}) nm")

    # Plot mesh
    ax_mesh = fig_centered.add_subplot(gs_centered[0, 0])
    ax_mesh.triplot((nodes[:, 0] - x_center)*1e9,
                   (nodes[:, 1] - y_center)*1e9,
                   elements, 'k-', lw=0.5, alpha=0.5)
    ax_mesh.set_xlabel('x (nm)')
    ax_mesh.set_ylabel('y (nm)')
    ax_mesh.set_title('Mesh (Centered)')
    ax_mesh.set_aspect('equal')
    ax_mesh.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_mesh.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Plot potential
    ax_pot = fig_centered.add_subplot(gs_centered[0, 1:])
    plot_potential(ax_pot, sim.mesh, sim.phi,
                  title='Potential (Centered)',
                  convert_to_eV=True,
                  use_nm=True,
                  center_coords=True)

    # Plot wavefunctions
    for i in range(3):
        ax_wf = fig_centered.add_subplot(gs_centered[1, i])
        plot_wavefunction(ax_wf, sim.mesh, eigenvectors[:, i],
                         title=f'Wavefunction {i} (E = {eigenvalues_eV[i]:.4f} eV)',
                         use_nm=True,
                         center_coords=True)

    # Save centered coordinates figure
    fig_centered.tight_layout(rect=[0, 0, 1, 0.95])
    fig_centered.savefig(f'test_results_alignment/{title.replace(" ", "_").lower()}_centered.png', dpi=300)
    plt.close(fig_centered)

    # Create overlay figure
    fig_overlay = plt.figure(figsize=(15, 5))
    gs_overlay = GridSpec(1, 3, figure=fig_overlay)
    fig_overlay.suptitle(f"{title} - Potential and Wavefunction Overlay", fontsize=16)

    # Create grid for 2D visualization
    x_range = np.max(nodes[:, 0]) - np.min(nodes[:, 0])
    y_range = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
    x_grid = np.linspace(np.min(nodes[:, 0]), np.max(nodes[:, 0]), 100)
    y_grid = np.linspace(np.min(nodes[:, 1]), np.max(nodes[:, 1]), 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Plot overlays for first 3 wavefunctions
    for i in range(3):
        ax_overlay = fig_overlay.add_subplot(gs_overlay[0, i])

        # Plot potential contours
        im_pot = plot_potential(ax_overlay, sim.mesh, sim.phi,
                              title=f'Overlay - Wavefunction {i}',
                              convert_to_eV=True,
                              use_nm=True,
                              center_coords=True,
                              vmin=-config.V_0,
                              vmax=0.1)

        # Plot wavefunction contours
        wf_values = np.abs(eigenvectors[:, i])**2
        wf_max = np.max(wf_values)
        if wf_max > 0:
            wf_values = wf_values / wf_max

        # Create a new colormap for the wavefunction
        from matplotlib.colors import LinearSegmentedColormap
        wf_cmap = LinearSegmentedColormap.from_list(
            'wf_cmap', [(0, 'white'), (1, 'red')], N=100)

        # Plot wavefunction contours
        ax_overlay.tricontour(
            (nodes[:, 0] - x_center)*1e9,
            (nodes[:, 1] - y_center)*1e9,
            elements, wf_values,
            levels=5, colors='r', linewidths=1.5, alpha=0.7
        )

        # Add horizontal and vertical lines at origin
        ax_overlay.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_overlay.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    # Save overlay figure
    fig_overlay.tight_layout(rect=[0, 0, 1, 0.95])
    fig_overlay.savefig(f'test_results_alignment/{title.replace(" ", "_").lower()}_overlay.png', dpi=300)
    plt.close(fig_overlay)

    return {
        'eigenvalues': eigenvalues_eV,
        'x_center': x_center,
        'y_center': y_center
    }

# Test cases
test_cases = [
    {
        'name': 'Square Centered',
        'config': {
            'potential_type': 'square',
            'V_0': 0.5,
            'R': 15e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40,
            'Lx': 100e-9,
            'Ly': 100e-9,
            'junction_position': 0.0
        }
    },
    {
        'name': 'Square Offset',
        'config': {
            'potential_type': 'square',
            'V_0': 0.5,
            'R': 15e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40,
            'Lx': 100e-9,
            'Ly': 100e-9,
            'junction_position': 20e-9
        }
    },
    {
        'name': 'Gaussian Centered',
        'config': {
            'potential_type': 'gaussian',
            'V_0': 0.5,
            'R': 15e-9,
            'element_order': 2,
            'nx': 40,
            'ny': 40,
            'Lx': 100e-9,
            'Ly': 100e-9,
            'junction_position': 0.0
        }
    }
]

# Run all test cases
results = {}
for case in test_cases:
    # Create base config
    config = Config()
    config.use_mpi = False
    config.max_refinements = 1
    config.adaptive_threshold = 0.05

    # Apply case-specific config
    for key, value in case['config'].items():
        setattr(config, key, value)

    # Validate config
    config.validate()

    # Run test
    results[case['name']] = test_alignment_test(config, case['name'])

print("\nAll alignment tests completed. Results saved to 'test_results_alignment' directory.")
