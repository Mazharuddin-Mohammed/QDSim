#!/usr/bin/env python3
"""
Enhance the deep QD simulation with mesh visualization, separate 2D potentials,
and mesh refinement visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction

# Create output directory
os.makedirs('results_enhanced', exist_ok=True)

# Configuration for a high-resolution 2D simulation with deeper QD
config = Config()
config.Lx = 200e-9  # 200 nm domain width
config.Ly = 200e-9  # 200 nm domain height
config.nx = 50      # Increased mesh points for better resolution
config.ny = 50
config.element_order = 2  # Use quadratic elements for better accuracy
config.max_refinements = 4  # More refinements for convergence
config.adaptive_threshold = 0.01  # Lower threshold for more aggressive refinement
config.use_mpi = False
config.potential_type = "square"  # Can be "square" or "gaussian"
config.R = 15e-9    # QD radius: 15 nm
config.V_0 = 1.0 * 1.602e-19  # Deeper potential: 1.0 eV (increased from 0.3 eV)
config.V_bi = 1.0 * 1.602e-19  # Built-in potential: 1.0 eV
config.N_A = 5e23   # Acceptor concentration: 5e23 m^-3
config.N_D = 5e23   # Donor concentration: 5e23 m^-3
config.eta = 0.1 * 1.602e-19  # CAP strength: 0.1 eV
config.hbar = 1.054e-34  # Reduced Planck constant
config.matrix_material = "GaAs"
config.qd_material = "InAs"

# Define a more realistic pn junction depletion region
config.depletion_width = 50e-9  # 50 nm depletion width
config.junction_position = 0.0  # Junction at the center of the domain

# Voltage shifts to simulate
V_r_values = np.linspace(-0.5, 0.5, 5)  # 5 voltage values from -0.5V to 0.5V

# Number of energy levels to compute
num_eigenvalues = 5  # Increased to see more bound states

# More realistic pn junction potential model
def pn_potential_1d(x, V_bi, V_r, depletion_width, junction_position):
    """Calculate the potential profile of a pn junction."""
    V = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < junction_position - depletion_width/2:
            V[i] = 0  # p-side
        elif xi > junction_position + depletion_width/2:
            V[i] = V_bi - V_r  # n-side
        else:
            # Quadratic potential in depletion region
            pos = 2 * (xi - junction_position) / depletion_width
            V[i] = (V_bi - V_r) * (pos**2 + pos + 1) / 4
    return V

# QD potential models
def square_qd_potential(x, y, R, V_0):
    r = np.sqrt(x**2 + y**2)
    return np.where(r <= R, -V_0, 0)

def gaussian_qd_potential(x, y, R, V_0):
    r2 = x**2 + y**2
    return -V_0 * np.exp(-r2/(2*R**2))

# 2D potential model for pn junction
def pn_potential_2d(X, Y, V_bi, V_r, depletion_width, junction_position):
    """Calculate the 2D potential of a pn junction."""
    V = np.zeros_like(X)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            xi = X[i, j]
            if xi < junction_position - depletion_width/2:
                V[i, j] = 0  # p-side
            elif xi > junction_position + depletion_width/2:
                V[i, j] = V_bi - V_r  # n-side
            else:
                # Quadratic potential in depletion region
                pos = 2 * (xi - junction_position) / depletion_width
                V[i, j] = (V_bi - V_r) * (pos**2 + pos + 1) / 4
    return V

# 2D QD potential
def qd_potential_2d(X, Y, R, V_0, potential_type):
    """Calculate the 2D potential of a quantum dot."""
    if potential_type == "square":
        r = np.sqrt(X**2 + Y**2)
        return np.where(r <= R, -V_0, 0)
    else:  # gaussian
        r2 = X**2 + Y**2
        return -V_0 * np.exp(-r2/(2*R**2))

# Create 2D grid for potential maps
X, Y = np.meshgrid(np.linspace(-config.Lx/2, config.Lx/2, 100),
                   np.linspace(-config.Ly/2, config.Ly/2, 100))

# Create a figure for separate potentials and mesh
fig_potentials = plt.figure(figsize=(15, 12))
gs = GridSpec(2, 3, figure=fig_potentials)

# Plot pn junction potential
ax_pn = fig_potentials.add_subplot(gs[0, 0])
V_pn = pn_potential_2d(X, Y, config.V_bi/1.602e-19, 0.0,
                     config.depletion_width, config.junction_position)
im_pn = ax_pn.contourf(X*1e9, Y*1e9, V_pn, 50, cmap='viridis')
plt.colorbar(im_pn, ax=ax_pn, label='Potential (eV)')
ax_pn.set_title('pn Junction Potential')
ax_pn.set_xlabel('x (nm)')
ax_pn.set_ylabel('y (nm)')

# Plot QD potential
ax_qd = fig_potentials.add_subplot(gs[0, 1])
V_qd = qd_potential_2d(X, Y, config.R, config.V_0/1.602e-19, config.potential_type)
im_qd = ax_qd.contourf(X*1e9, Y*1e9, V_qd, 50, cmap='viridis')
plt.colorbar(im_qd, ax=ax_qd, label='Potential (eV)')
ax_qd.set_title('Quantum Dot Potential')
ax_qd.set_xlabel('x (nm)')
ax_qd.set_ylabel('y (nm)')

# Plot combined potential
ax_combined = fig_potentials.add_subplot(gs[0, 2])
V_combined = V_pn + V_qd
im_combined = ax_combined.contourf(X*1e9, Y*1e9, V_combined, 50, cmap='viridis')
plt.colorbar(im_combined, ax=ax_combined, label='Potential (eV)')
ax_combined.set_title('Combined Potential')
ax_combined.set_xlabel('x (nm)')
ax_combined.set_ylabel('y (nm)')

# Run simulation for zero bias
print("Simulating with mesh visualization...")
config.V_r = 0.0  # Zero bias
sim = Simulator(config)
eigenvalues, eigenvectors = sim.run(num_eigenvalues)

# Get mesh data
nodes = np.array(sim.mesh.get_nodes())
elements = np.array(sim.mesh.get_elements())

# Plot initial mesh
ax_mesh_initial = fig_potentials.add_subplot(gs[1, 0])
triang = Triangulation(nodes[:, 0]*1e9, nodes[:, 1]*1e9, elements)
ax_mesh_initial.triplot(triang, 'k-', lw=0.5)
ax_mesh_initial.set_title('Mesh Visualization')
ax_mesh_initial.set_xlabel('x (nm)')
ax_mesh_initial.set_ylabel('y (nm)')
ax_mesh_initial.set_aspect('equal')

# Plot mesh with potential
ax_mesh_potential = fig_potentials.add_subplot(gs[1, 1])
im_mesh_pot = ax_mesh_potential.tricontourf(triang, sim.phi, 50, cmap='viridis')
plt.colorbar(im_mesh_pot, ax=ax_mesh_potential, label='Potential (V)')
ax_mesh_potential.triplot(triang, 'k-', lw=0.2, alpha=0.3)
ax_mesh_potential.set_title('Mesh with Potential')
ax_mesh_potential.set_xlabel('x (nm)')
ax_mesh_potential.set_ylabel('y (nm)')
ax_mesh_potential.set_aspect('equal')

# Plot ground state wavefunction
ax_wf = fig_potentials.add_subplot(gs[1, 2])
plot_wavefunction(ax_wf, sim.mesh, eigenvectors[:, 0])
ax_wf.set_title(f'Ground State (E = {np.real(eigenvalues[0])/1.602e-19:.4f} eV)')
ax_wf.set_aspect('equal')

fig_potentials.tight_layout()
fig_potentials.savefig('results_enhanced/potentials_and_mesh.png', dpi=300, bbox_inches='tight')

# Create a figure for mesh refinement analysis
fig_refinement = plt.figure(figsize=(15, 10))
gs_ref = GridSpec(2, 3, figure=fig_refinement)

# Plot mesh density around QD
ax_density = fig_refinement.add_subplot(gs_ref[0, 0])
# Calculate mesh density (inverse of triangle area)
density = np.zeros(len(elements))
for i, element in enumerate(elements):
    # Get vertices
    v1 = nodes[element[0]]
    v2 = nodes[element[1]]
    v3 = nodes[element[2]]
    # Calculate area
    area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))
    density[i] = 1.0 / area if area > 0 else 0

# Plot mesh density
im_density = ax_density.tripcolor(triang, density, cmap='viridis')
plt.colorbar(im_density, ax=ax_density, label='Mesh Density')
ax_density.set_title('Mesh Density')
ax_density.set_xlabel('x (nm)')
ax_density.set_ylabel('y (nm)')
ax_density.set_aspect('equal')

# Plot mesh with QD region highlighted
ax_qd_region = fig_refinement.add_subplot(gs_ref[0, 1])
ax_qd_region.triplot(triang, 'k-', lw=0.5, alpha=0.3)
# Highlight QD region
circle = plt.Circle((0, 0), config.R*1e9, color='r', fill=False, lw=2)
ax_qd_region.add_patch(circle)
ax_qd_region.set_title('QD Region Highlighted')
ax_qd_region.set_xlabel('x (nm)')
ax_qd_region.set_ylabel('y (nm)')
ax_qd_region.set_aspect('equal')

# Plot mesh with depletion region highlighted
ax_depletion = fig_refinement.add_subplot(gs_ref[0, 2])
ax_depletion.triplot(triang, 'k-', lw=0.5, alpha=0.3)
# Highlight depletion region
rect = plt.Rectangle((-config.depletion_width/2*1e9, -config.Ly/2*1e9),
                     config.depletion_width*1e9, config.Ly*1e9,
                     color='b', fill=False, lw=2)
ax_depletion.add_patch(rect)
ax_depletion.set_title('Depletion Region Highlighted')
ax_depletion.set_xlabel('x (nm)')
ax_depletion.set_ylabel('y (nm)')
ax_depletion.set_aspect('equal')

# Plot wavefunction probability densities for first three states
for i in range(min(3, num_eigenvalues)):
    ax_wf = fig_refinement.add_subplot(gs_ref[1, i])
    plot_wavefunction(ax_wf, sim.mesh, eigenvectors[:, i])
    ax_wf.set_title(f'Energy Level {i+1} (E = {np.real(eigenvalues[i])/1.602e-19:.4f} eV)')
    ax_wf.set_aspect('equal')

fig_refinement.tight_layout()
fig_refinement.savefig('results_enhanced/mesh_refinement_analysis.png', dpi=300, bbox_inches='tight')

# Create a figure for mesh quality analysis
fig_quality = plt.figure(figsize=(15, 10))
gs_qual = GridSpec(2, 2, figure=fig_quality)

# Calculate mesh quality metrics
quality = np.zeros(len(elements))
for i, element in enumerate(elements):
    # Get vertices
    v1 = nodes[element[0]]
    v2 = nodes[element[1]]
    v3 = nodes[element[2]]
    # Calculate edge lengths
    e1 = np.sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)
    e2 = np.sqrt((v3[0] - v2[0])**2 + (v3[1] - v2[1])**2)
    e3 = np.sqrt((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)
    # Calculate area
    area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))
    # Calculate quality (ratio of area to sum of squared edge lengths)
    quality[i] = 4.0 * np.sqrt(3) * area / (e1**2 + e2**2 + e3**2) if (e1**2 + e2**2 + e3**2) > 0 else 0

# Plot mesh quality
ax_quality = fig_quality.add_subplot(gs_qual[0, 0])
im_quality = ax_quality.tripcolor(triang, quality, cmap='viridis')
plt.colorbar(im_quality, ax=ax_quality, label='Mesh Quality')
ax_quality.set_title('Mesh Quality')
ax_quality.set_xlabel('x (nm)')
ax_quality.set_ylabel('y (nm)')
ax_quality.set_aspect('equal')

# Plot mesh quality histogram
ax_hist = fig_quality.add_subplot(gs_qual[0, 1])
# Check if we have valid quality values
if np.any(quality > 0) and np.any(np.isfinite(quality)):
    # Filter out zeros and infinities
    valid_quality = quality[np.logical_and(quality > 0, np.isfinite(quality))]
    if len(valid_quality) > 0:
        ax_hist.hist(valid_quality, bins=min(10, len(valid_quality)))
        ax_hist.set_title('Mesh Quality Histogram')
        ax_hist.set_xlabel('Quality')
        ax_hist.set_ylabel('Count')
    else:
        ax_hist.text(0.5, 0.5, 'No valid quality values',
                    ha='center', va='center', transform=ax_hist.transAxes)
else:
    ax_hist.text(0.5, 0.5, 'No valid quality values',
                ha='center', va='center', transform=ax_hist.transAxes)

# Plot potential along x-axis
ax_pot_x = fig_quality.add_subplot(gs_qual[1, 0])
x = np.linspace(-config.Lx/2, config.Lx/2, 200)
y = np.zeros_like(x)
V_pn_1d = pn_potential_1d(x, config.V_bi/1.602e-19, 0.0,
                        config.depletion_width, config.junction_position)
if config.potential_type == "square":
    V_qd_1d = square_qd_potential(x, y, config.R, config.V_0/1.602e-19)
else:
    V_qd_1d = gaussian_qd_potential(x, y, config.R, config.V_0/1.602e-19)
V_combined_1d = V_pn_1d + V_qd_1d
ax_pot_x.plot(x*1e9, V_pn_1d, 'b-', label='pn Junction')
ax_pot_x.plot(x*1e9, V_qd_1d, 'r-', label='Quantum Dot')
ax_pot_x.plot(x*1e9, V_combined_1d, 'g-', label='Combined')
ax_pot_x.set_title('Potential along x-axis (y=0)')
ax_pot_x.set_xlabel('x (nm)')
ax_pot_x.set_ylabel('Potential (eV)')
ax_pot_x.legend()
ax_pot_x.grid(True)

# Plot potential along y-axis
ax_pot_y = fig_quality.add_subplot(gs_qual[1, 1])
y = np.linspace(-config.Ly/2, config.Ly/2, 200)
x = np.zeros_like(y)
if config.potential_type == "square":
    V_qd_1d_y = square_qd_potential(x, y, config.R, config.V_0/1.602e-19)
else:
    V_qd_1d_y = gaussian_qd_potential(x, y, config.R, config.V_0/1.602e-19)
ax_pot_y.plot(y*1e9, V_qd_1d_y, 'r-', label='Quantum Dot')
ax_pot_y.set_title('Potential along y-axis (x=0)')
ax_pot_y.set_xlabel('y (nm)')
ax_pot_y.set_ylabel('Potential (eV)')
ax_pot_y.legend()
ax_pot_y.grid(True)

fig_quality.tight_layout()
fig_quality.savefig('results_enhanced/mesh_quality_analysis.png', dpi=300, bbox_inches='tight')

print("Enhanced visualizations saved to 'results_enhanced' directory.")
