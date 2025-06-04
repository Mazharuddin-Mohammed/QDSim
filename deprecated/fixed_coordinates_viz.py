#!/usr/bin/env python3
"""
Fixed coordinate system visualization for the deep QD simulation.
Ensures all plots use the same coordinate system with the origin at (0,0).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction

# Create output directory
os.makedirs('results_fixed', exist_ok=True)

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

# Run simulation for zero bias to get mesh coordinates first
print("Running initial simulation to get mesh coordinates...")
config.V_r = 0.0  # Zero bias
sim = Simulator(config)
eigenvalues, eigenvectors = sim.run(3)  # Get first three states

# Get mesh data
nodes = np.array(sim.mesh.get_nodes())
elements = np.array(sim.mesh.get_elements())

# Print mesh statistics
print(f"Mesh statistics:")
print(f"  Number of nodes: {len(nodes)}")
print(f"  Number of elements: {len(elements)}")
print(f"  Node coordinate ranges: x=[{np.min(nodes[:,0])*1e9:.1f}, {np.max(nodes[:,0])*1e9:.1f}] nm, "
      f"y=[{np.min(nodes[:,1])*1e9:.1f}, {np.max(nodes[:,1])*1e9:.1f}] nm")

# Shift coordinates to center the domain at (0,0)
# Calculate the center of the mesh
x_center = (np.min(nodes[:,0]) + np.max(nodes[:,0])) / 2
y_center = (np.min(nodes[:,1]) + np.max(nodes[:,1])) / 2
print(f"Mesh center: ({x_center*1e9:.1f}, {y_center*1e9:.1f}) nm")

# Shift nodes to center at (0,0)
nodes_centered = nodes.copy()
nodes_centered[:,0] -= x_center
nodes_centered[:,1] -= y_center

# Define domain limits based on centered mesh coordinates
x_min = np.min(nodes_centered[:,0])
x_max = np.max(nodes_centered[:,0])
y_min = np.min(nodes_centered[:,1])
y_max = np.max(nodes_centered[:,1])

# Convert to nm for plotting
x_min_nm = x_min * 1e9
x_max_nm = x_max * 1e9
y_min_nm = y_min * 1e9
y_max_nm = y_max * 1e9

print(f"Centered mesh domain limits: x=[{x_min_nm:.1f}, {x_max_nm:.1f}] nm, y=[{y_min_nm:.1f}, {y_max_nm:.1f}] nm")

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

# Create 2D grid for potential maps - use the same grid for all plots
num_points = 200
X, Y = np.meshgrid(np.linspace(x_min, x_max, num_points), 
                   np.linspace(y_min, y_max, num_points))

# Common function to set consistent axis limits
def set_consistent_limits(ax):
    ax.set_xlim(x_min_nm, x_max_nm)
    ax.set_ylim(y_min_nm, y_max_nm)
    ax.set_aspect('equal')
    # Add origin lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Create a figure for separate potentials and mesh
fig_potentials = plt.figure(figsize=(15, 10))
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
set_consistent_limits(ax_pn)

# Plot QD potential
ax_qd = fig_potentials.add_subplot(gs[0, 1])
V_qd = qd_potential_2d(X, Y, config.R, config.V_0/1.602e-19, config.potential_type)
im_qd = ax_qd.contourf(X*1e9, Y*1e9, V_qd, 50, cmap='viridis')
plt.colorbar(im_qd, ax=ax_qd, label='Potential (eV)')
ax_qd.set_title('Quantum Dot Potential')
ax_qd.set_xlabel('x (nm)')
ax_qd.set_ylabel('y (nm)')
set_consistent_limits(ax_qd)

# Plot combined potential
ax_combined = fig_potentials.add_subplot(gs[0, 2])
V_combined = V_pn + V_qd
im_combined = ax_combined.contourf(X*1e9, Y*1e9, V_combined, 50, cmap='viridis')
plt.colorbar(im_combined, ax=ax_combined, label='Potential (eV)')
ax_combined.set_title('Combined Potential')
ax_combined.set_xlabel('x (nm)')
ax_combined.set_ylabel('y (nm)')
set_consistent_limits(ax_combined)

# Plot mesh
ax_mesh = fig_potentials.add_subplot(gs[1, 0])
triang = Triangulation(nodes_centered[:, 0]*1e9, nodes_centered[:, 1]*1e9, elements)
ax_mesh.triplot(triang, 'k-', lw=0.5)
ax_mesh.set_title('Mesh Visualization')
ax_mesh.set_xlabel('x (nm)')
ax_mesh.set_ylabel('y (nm)')
set_consistent_limits(ax_mesh)

# Plot mesh with potential
ax_mesh_pot = fig_potentials.add_subplot(gs[1, 1])
# We need to interpolate the potential onto the centered mesh
# For simplicity, we'll just use the analytical potential
X_mesh, Y_mesh = np.meshgrid(np.linspace(x_min, x_max, 100), 
                            np.linspace(y_min, y_max, 100))
V_mesh_pn = pn_potential_2d(X_mesh, Y_mesh, config.V_bi/1.602e-19, 0.0, 
                          config.depletion_width, config.junction_position)
V_mesh_qd = qd_potential_2d(X_mesh, Y_mesh, config.R, config.V_0/1.602e-19, config.potential_type)
V_mesh_combined = V_mesh_pn + V_mesh_qd
im_mesh_pot = ax_mesh_pot.contourf(X_mesh*1e9, Y_mesh*1e9, V_mesh_combined, 50, cmap='viridis')
plt.colorbar(im_mesh_pot, ax=ax_mesh_pot, label='Potential (eV)')
ax_mesh_pot.triplot(triang, 'k-', lw=0.2, alpha=0.3)
ax_mesh_pot.set_title('Mesh with Potential')
ax_mesh_pot.set_xlabel('x (nm)')
ax_mesh_pot.set_ylabel('y (nm)')
set_consistent_limits(ax_mesh_pot)

# Plot ground state wavefunction
ax_wf = fig_potentials.add_subplot(gs[1, 2])
# We need to create a custom plot since the built-in function doesn't support centered coordinates
wf_values = np.abs(eigenvectors[:, 0])**2
im_wf = ax_wf.tripcolor(triang, wf_values, cmap='viridis')
plt.colorbar(im_wf, ax=ax_wf, label='Probability Density')
ax_wf.set_title(f'Ground State (E = {np.real(eigenvalues[0])/1.602e-19:.4f} eV)')
ax_wf.set_xlabel('x (nm)')
ax_wf.set_ylabel('y (nm)')
set_consistent_limits(ax_wf)

fig_potentials.tight_layout()
fig_potentials.savefig('results_fixed/potentials_and_mesh.png', dpi=300, bbox_inches='tight')

# Create a figure for mesh with highlighted regions
fig_regions = plt.figure(figsize=(15, 5))
gs_reg = GridSpec(1, 3, figure=fig_regions)

# Plot mesh with QD region highlighted
ax_qd_region = fig_regions.add_subplot(gs_reg[0, 0])
ax_qd_region.triplot(triang, 'k-', lw=0.5, alpha=0.3)
# Highlight QD region
circle = plt.Circle((0, 0), config.R*1e9, color='r', fill=False, lw=2)
ax_qd_region.add_patch(circle)
ax_qd_region.set_title('QD Region Highlighted')
ax_qd_region.set_xlabel('x (nm)')
ax_qd_region.set_ylabel('y (nm)')
set_consistent_limits(ax_qd_region)

# Plot mesh with depletion region highlighted
ax_depletion = fig_regions.add_subplot(gs_reg[0, 1])
ax_depletion.triplot(triang, 'k-', lw=0.5, alpha=0.3)
# Highlight depletion region
rect = plt.Rectangle(((-config.depletion_width/2 + config.junction_position)*1e9, y_min_nm), 
                     config.depletion_width*1e9, (y_max - y_min)*1e9, 
                     color='b', fill=False, lw=2)
ax_depletion.add_patch(rect)
ax_depletion.set_title('Depletion Region Highlighted')
ax_depletion.set_xlabel('x (nm)')
ax_depletion.set_ylabel('y (nm)')
set_consistent_limits(ax_depletion)

# Plot mesh with both regions highlighted
ax_both = fig_regions.add_subplot(gs_reg[0, 2])
ax_both.triplot(triang, 'k-', lw=0.5, alpha=0.3)
# Highlight QD region
circle = plt.Circle((0, 0), config.R*1e9, color='r', fill=False, lw=2, label='QD')
ax_both.add_patch(circle)
# Highlight depletion region
rect = plt.Rectangle(((-config.depletion_width/2 + config.junction_position)*1e9, y_min_nm), 
                     config.depletion_width*1e9, (y_max - y_min)*1e9, 
                     color='b', fill=False, lw=2, label='Depletion')
ax_both.add_patch(rect)
ax_both.set_title('Both Regions Highlighted')
ax_both.set_xlabel('x (nm)')
ax_both.set_ylabel('y (nm)')
set_consistent_limits(ax_both)
# Add legend
ax_both.legend(handles=[
    plt.Line2D([0], [0], color='r', lw=2, label='QD Region'),
    plt.Line2D([0], [0], color='b', lw=2, label='Depletion Region')
], loc='upper right')

fig_regions.tight_layout()
fig_regions.savefig('results_fixed/mesh_regions.png', dpi=300, bbox_inches='tight')

# Create a figure for wavefunctions
fig_wavefunctions = plt.figure(figsize=(15, 5))
gs_wf = GridSpec(1, 3, figure=fig_wavefunctions)

# Plot first three wavefunctions
for i in range(min(3, len(eigenvalues))):
    ax_wf = fig_wavefunctions.add_subplot(gs_wf[0, i])
    wf_values = np.abs(eigenvectors[:, i])**2
    im_wf = ax_wf.tripcolor(triang, wf_values, cmap='viridis')
    plt.colorbar(im_wf, ax=ax_wf, label='Probability Density')
    ax_wf.set_title(f'Energy Level {i+1} (E = {np.real(eigenvalues[i])/1.602e-19:.4f} eV)')
    ax_wf.set_xlabel('x (nm)')
    ax_wf.set_ylabel('y (nm)')
    set_consistent_limits(ax_wf)

fig_wavefunctions.tight_layout()
fig_wavefunctions.savefig('results_fixed/wavefunctions.png', dpi=300, bbox_inches='tight')

# Create a figure for 1D slices
fig_slices = plt.figure(figsize=(15, 10))
gs_slices = GridSpec(2, 2, figure=fig_slices)

# Plot potential along x-axis (y=0)
ax_x = fig_slices.add_subplot(gs_slices[0, 0])
x_1d = np.linspace(x_min, x_max, num_points)
y_1d = np.zeros_like(x_1d)
V_pn_1d = pn_potential_2d(x_1d.reshape(-1, 1), y_1d.reshape(-1, 1), 
                        config.V_bi/1.602e-19, 0.0, 
                        config.depletion_width, config.junction_position).flatten()
V_qd_1d = qd_potential_2d(x_1d.reshape(-1, 1), y_1d.reshape(-1, 1), 
                        config.R, config.V_0/1.602e-19, config.potential_type).flatten()
V_combined_1d = V_pn_1d + V_qd_1d
ax_x.plot(x_1d*1e9, V_pn_1d, 'b-', label='pn Junction')
ax_x.plot(x_1d*1e9, V_qd_1d, 'r-', label='Quantum Dot')
ax_x.plot(x_1d*1e9, V_combined_1d, 'g-', label='Combined')
ax_x.set_title('Potential along x-axis (y=0)')
ax_x.set_xlabel('x (nm)')
ax_x.set_ylabel('Potential (eV)')
ax_x.legend()
ax_x.grid(True)
ax_x.set_xlim(x_min_nm, x_max_nm)

# Plot potential along y-axis (x=0)
ax_y = fig_slices.add_subplot(gs_slices[0, 1])
y_1d = np.linspace(y_min, y_max, num_points)
x_1d = np.zeros_like(y_1d)
V_pn_1d_y = pn_potential_2d(x_1d.reshape(-1, 1), y_1d.reshape(-1, 1), 
                          config.V_bi/1.602e-19, 0.0, 
                          config.depletion_width, config.junction_position).flatten()
V_qd_1d_y = qd_potential_2d(x_1d.reshape(-1, 1), y_1d.reshape(-1, 1), 
                          config.R, config.V_0/1.602e-19, config.potential_type).flatten()
V_combined_1d_y = V_pn_1d_y + V_qd_1d_y
ax_y.plot(y_1d*1e9, V_pn_1d_y, 'b-', label='pn Junction')
ax_y.plot(y_1d*1e9, V_qd_1d_y, 'r-', label='Quantum Dot')
ax_y.plot(y_1d*1e9, V_combined_1d_y, 'g-', label='Combined')
ax_y.set_title('Potential along y-axis (x=0)')
ax_y.set_xlabel('y (nm)')
ax_y.set_ylabel('Potential (eV)')
ax_y.legend()
ax_y.grid(True)
ax_y.set_xlim(y_min_nm, y_max_nm)

# Plot wavefunction along x-axis (y=0)
ax_wf_x = fig_slices.add_subplot(gs_slices[1, 0])
# Extract wavefunction values along x-axis
# This is an approximation since we're using a mesh
x_coords = nodes_centered[:, 0]
y_coords = nodes_centered[:, 1]
# Find nodes close to y=0
y_zero_indices = np.where(np.abs(y_coords) < 1e-9)[0]
if len(y_zero_indices) > 0:
    # Sort by x coordinate
    sorted_indices = y_zero_indices[np.argsort(x_coords[y_zero_indices])]
    x_sorted = x_coords[sorted_indices]
    # Plot wavefunction values for first three states
    for i in range(min(3, len(eigenvalues))):
        wf_values = np.abs(eigenvectors[sorted_indices, i])**2
        if np.max(wf_values) > 0:
            ax_wf_x.plot(x_sorted*1e9, wf_values / np.max(wf_values), 
                        label=f'Level {i+1}')
    ax_wf_x.set_title('Normalized Wavefunction Probability along x-axis (y=0)')
    ax_wf_x.set_xlabel('x (nm)')
    ax_wf_x.set_ylabel('Normalized Probability')
    ax_wf_x.legend()
    ax_wf_x.grid(True)
    ax_wf_x.set_xlim(x_min_nm, x_max_nm)
else:
    ax_wf_x.text(0.5, 0.5, 'No nodes found at y=0', 
                ha='center', va='center', transform=ax_wf_x.transAxes)

# Plot wavefunction along y-axis (x=0)
ax_wf_y = fig_slices.add_subplot(gs_slices[1, 1])
# Extract wavefunction values along y-axis
# Find nodes close to x=0
x_zero_indices = np.where(np.abs(x_coords) < 1e-9)[0]
if len(x_zero_indices) > 0:
    # Sort by y coordinate
    sorted_indices = x_zero_indices[np.argsort(y_coords[x_zero_indices])]
    y_sorted = y_coords[sorted_indices]
    # Plot wavefunction values for first three states
    for i in range(min(3, len(eigenvalues))):
        wf_values = np.abs(eigenvectors[sorted_indices, i])**2
        if np.max(wf_values) > 0:
            ax_wf_y.plot(y_sorted*1e9, wf_values / np.max(wf_values), 
                        label=f'Level {i+1}')
    ax_wf_y.set_title('Normalized Wavefunction Probability along y-axis (x=0)')
    ax_wf_y.set_xlabel('y (nm)')
    ax_wf_y.set_ylabel('Normalized Probability')
    ax_wf_y.legend()
    ax_wf_y.grid(True)
    ax_wf_y.set_xlim(y_min_nm, y_max_nm)
else:
    ax_wf_y.text(0.5, 0.5, 'No nodes found at x=0', 
                ha='center', va='center', transform=ax_wf_y.transAxes)

fig_slices.tight_layout()
fig_slices.savefig('results_fixed/potential_wavefunction_slices.png', dpi=300, bbox_inches='tight')

# Create a figure showing the energy levels and bound state analysis
fig_energy = plt.figure(figsize=(15, 10))
gs_energy = GridSpec(2, 2, figure=fig_energy)

# Plot energy levels
ax_energy = fig_energy.add_subplot(gs_energy[0, 0])
energy_levels = [np.real(e)/1.602e-19 for e in eigenvalues]
ax_energy.bar(range(1, len(energy_levels) + 1), energy_levels)
ax_energy.set_title('Energy Levels')
ax_energy.set_xlabel('Level Number')
ax_energy.set_ylabel('Energy (eV)')
ax_energy.grid(True)

# Plot linewidths
ax_linewidth = fig_energy.add_subplot(gs_energy[0, 1])
linewidths = [-2 * np.imag(e)/1.602e-19 for e in eigenvalues]
ax_linewidth.bar(range(1, len(linewidths) + 1), linewidths)
ax_linewidth.set_title('Linewidths (Γ)')
ax_linewidth.set_xlabel('Level Number')
ax_linewidth.set_ylabel('Linewidth (eV)')
ax_linewidth.grid(True)

# Calculate localization (inverse participation ratio)
ax_localization = fig_energy.add_subplot(gs_energy[1, 0])
localization = []
for i in range(len(eigenvalues)):
    psi_squared = np.abs(eigenvectors[:, i])**2
    localization_value = np.sum(psi_squared**2) / (np.sum(psi_squared)**2) * 100
    localization.append(localization_value)
ax_localization.bar(range(1, len(localization) + 1), localization)
ax_localization.set_title('Wavefunction Localization')
ax_localization.set_xlabel('Level Number')
ax_localization.set_ylabel('Localization (%)')
ax_localization.grid(True)

# Calculate probability in QD
ax_probability = fig_energy.add_subplot(gs_energy[1, 1])
probability_in_qd = []
for i in range(len(eigenvalues)):
    psi_squared = np.abs(eigenvectors[:, i])**2
    # Find nodes in QD region
    in_qd = []
    for j, node in enumerate(nodes_centered):
        r = np.sqrt(node[0]**2 + node[1]**2)
        if r <= config.R:
            in_qd.append(j)
    
    if len(in_qd) > 0:
        probability = np.sum(psi_squared[in_qd]) / np.sum(psi_squared) * 100
    else:
        probability = 0
    probability_in_qd.append(probability)

ax_probability.bar(range(1, len(probability_in_qd) + 1), probability_in_qd)
ax_probability.set_title('Probability in QD')
ax_probability.set_xlabel('Level Number')
ax_probability.set_ylabel('Probability (%)')
ax_probability.axhline(y=50, color='r', linestyle='--', label='Bound State Threshold')
ax_probability.legend()
ax_probability.grid(True)

fig_energy.tight_layout()
fig_energy.savefig('results_fixed/energy_analysis.png', dpi=300, bbox_inches='tight')

print("Fixed coordinate visualizations saved to 'results_fixed' directory.")
