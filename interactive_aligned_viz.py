#!/usr/bin/env python3
"""
Interactive and perfectly aligned visualization for the deep QD simulation.
Ensures all plots use the same coordinate system with perfect alignment between
potentials and wavefunctions. Also provides interactive 3D visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots

# Create output directory
os.makedirs('results_interactive', exist_ok=True)

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

# Calculate potentials on the grid
V_pn = pn_potential_2d(X, Y, config.V_bi/1.602e-19, 0.0, 
                     config.depletion_width, config.junction_position)
V_qd = qd_potential_2d(X, Y, config.R, config.V_0/1.602e-19, config.potential_type)
V_combined = V_pn + V_qd

# Common function to set consistent axis limits
def set_consistent_limits(ax):
    ax.set_xlim(x_min_nm, x_max_nm)
    ax.set_ylim(y_min_nm, y_max_nm)
    ax.set_aspect('equal')
    # Add origin lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Create a triangulation for the centered mesh
triang = Triangulation(nodes_centered[:, 0]*1e9, nodes_centered[:, 1]*1e9, elements)

# Interpolate the wavefunctions onto the regular grid for perfect alignment
from scipy.interpolate import griddata

# Function to interpolate mesh data onto a regular grid
def interpolate_to_grid(nodes, values, X, Y):
    points = nodes[:, :2]  # Use only x and y coordinates
    return griddata(points, values, (X, Y), method='cubic', fill_value=0)

# Interpolate wavefunctions onto the regular grid
wf_grid = []
for i in range(min(3, len(eigenvalues))):
    wf_values = np.abs(eigenvectors[:, i])**2
    # Interpolate onto the regular grid
    wf_interp = interpolate_to_grid(nodes_centered, wf_values, X, Y)
    wf_grid.append(wf_interp)

# Create a figure for perfectly aligned potentials and wavefunctions
fig_aligned = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig_aligned)

# Plot pn junction potential
ax_pn = fig_aligned.add_subplot(gs[0, 0])
im_pn = ax_pn.contourf(X*1e9, Y*1e9, V_pn, 50, cmap='viridis')
plt.colorbar(im_pn, ax=ax_pn, label='Potential (eV)')
ax_pn.set_title('pn Junction Potential')
ax_pn.set_xlabel('x (nm)')
ax_pn.set_ylabel('y (nm)')
set_consistent_limits(ax_pn)

# Plot QD potential
ax_qd = fig_aligned.add_subplot(gs[0, 1])
im_qd = ax_qd.contourf(X*1e9, Y*1e9, V_qd, 50, cmap='viridis')
plt.colorbar(im_qd, ax=ax_qd, label='Potential (eV)')
ax_qd.set_title('Quantum Dot Potential')
ax_qd.set_xlabel('x (nm)')
ax_qd.set_ylabel('y (nm)')
set_consistent_limits(ax_qd)

# Plot combined potential
ax_combined = fig_aligned.add_subplot(gs[0, 2])
im_combined = ax_combined.contourf(X*1e9, Y*1e9, V_combined, 50, cmap='viridis')
plt.colorbar(im_combined, ax=ax_combined, label='Potential (eV)')
ax_combined.set_title('Combined Potential')
ax_combined.set_xlabel('x (nm)')
ax_combined.set_ylabel('y (nm)')
set_consistent_limits(ax_combined)

# Plot first three wavefunctions on the same grid
for i in range(min(3, len(eigenvalues))):
    ax_wf = fig_aligned.add_subplot(gs[1, i])
    im_wf = ax_wf.contourf(X*1e9, Y*1e9, wf_grid[i], 50, cmap='viridis')
    plt.colorbar(im_wf, ax=ax_wf, label='Probability Density')
    ax_wf.set_title(f'Energy Level {i+1} (E = {np.real(eigenvalues[i])/1.602e-19:.4f} eV)')
    ax_wf.set_xlabel('x (nm)')
    ax_wf.set_ylabel('y (nm)')
    set_consistent_limits(ax_wf)

fig_aligned.tight_layout()
fig_aligned.savefig('results_interactive/aligned_potentials_wavefunctions.png', dpi=300, bbox_inches='tight')

# Create a figure for overlaid potentials and wavefunctions
fig_overlay = plt.figure(figsize=(15, 5))
gs_overlay = GridSpec(1, 3, figure=fig_overlay)

# Plot overlaid potentials and wavefunctions
for i in range(min(3, len(eigenvalues))):
    ax_overlay = fig_overlay.add_subplot(gs_overlay[0, i])
    
    # Plot potential contours
    contour = ax_overlay.contour(X*1e9, Y*1e9, V_combined, 10, colors='k', alpha=0.5)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot wavefunction as filled contours
    im_wf = ax_overlay.contourf(X*1e9, Y*1e9, wf_grid[i], 50, cmap='viridis', alpha=0.7)
    plt.colorbar(im_wf, ax=ax_overlay, label='Probability Density')
    
    ax_overlay.set_title(f'Energy Level {i+1} (E = {np.real(eigenvalues[i])/1.602e-19:.4f} eV)')
    ax_overlay.set_xlabel('x (nm)')
    ax_overlay.set_ylabel('y (nm)')
    set_consistent_limits(ax_overlay)

fig_overlay.tight_layout()
fig_overlay.savefig('results_interactive/overlaid_potentials_wavefunctions.png', dpi=300, bbox_inches='tight')

# Create interactive 3D plots
print("Creating interactive 3D plots...")

# 3D plot of potential
fig_3d_pot = plt.figure(figsize=(10, 8))
ax_3d_pot = fig_3d_pot.add_subplot(111, projection='3d')
surf_pot = ax_3d_pot.plot_surface(X*1e9, Y*1e9, V_combined, cmap='viridis', 
                                 linewidth=0, antialiased=True)
plt.colorbar(surf_pot, ax=ax_3d_pot, label='Potential (eV)')
ax_3d_pot.set_title('3D Combined Potential')
ax_3d_pot.set_xlabel('x (nm)')
ax_3d_pot.set_ylabel('y (nm)')
ax_3d_pot.set_zlabel('Potential (eV)')
fig_3d_pot.savefig('results_interactive/3d_potential.png', dpi=300, bbox_inches='tight')

# 3D plots of wavefunctions
for i in range(min(3, len(eigenvalues))):
    fig_3d_wf = plt.figure(figsize=(10, 8))
    ax_3d_wf = fig_3d_wf.add_subplot(111, projection='3d')
    surf_wf = ax_3d_wf.plot_surface(X*1e9, Y*1e9, wf_grid[i], cmap='viridis', 
                                   linewidth=0, antialiased=True)
    plt.colorbar(surf_wf, ax=ax_3d_wf, label='Probability Density')
    ax_3d_wf.set_title(f'3D Wavefunction - Energy Level {i+1} (E = {np.real(eigenvalues[i])/1.602e-19:.4f} eV)')
    ax_3d_wf.set_xlabel('x (nm)')
    ax_3d_wf.set_ylabel('y (nm)')
    ax_3d_wf.set_zlabel('Probability Density')
    fig_3d_wf.savefig(f'results_interactive/3d_wavefunction_{i+1}.png', dpi=300, bbox_inches='tight')

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
# Get the middle index for y=0
mid_y_idx = num_points // 2
# Plot wavefunction values for first three states
for i in range(min(3, len(eigenvalues))):
    # Extract the slice at y=0
    wf_slice = wf_grid[i][mid_y_idx, :]
    if np.max(wf_slice) > 0:
        ax_wf_x.plot(x_1d*1e9, wf_slice / np.max(wf_slice), label=f'Level {i+1}')
ax_wf_x.set_title('Normalized Wavefunction Probability along x-axis (y=0)')
ax_wf_x.set_xlabel('x (nm)')
ax_wf_x.set_ylabel('Normalized Probability')
ax_wf_x.legend()
ax_wf_x.grid(True)
ax_wf_x.set_xlim(x_min_nm, x_max_nm)

# Plot wavefunction along y-axis (x=0)
ax_wf_y = fig_slices.add_subplot(gs_slices[1, 1])
# Get the middle index for x=0
mid_x_idx = num_points // 2
# Plot wavefunction values for first three states
for i in range(min(3, len(eigenvalues))):
    # Extract the slice at x=0
    wf_slice = wf_grid[i][:, mid_x_idx]
    if np.max(wf_slice) > 0:
        ax_wf_y.plot(y_1d*1e9, wf_slice / np.max(wf_slice), label=f'Level {i+1}')
ax_wf_y.set_title('Normalized Wavefunction Probability along y-axis (x=0)')
ax_wf_y.set_xlabel('y (nm)')
ax_wf_y.set_ylabel('Normalized Probability')
ax_wf_y.legend()
ax_wf_y.grid(True)
ax_wf_y.set_xlim(y_min_nm, y_max_nm)

fig_slices.tight_layout()
fig_slices.savefig('results_interactive/potential_wavefunction_slices.png', dpi=300, bbox_inches='tight')

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
    # Create a mask for the QD region
    r = np.sqrt(X**2 + Y**2)
    qd_mask = r <= config.R
    
    # Calculate probability in QD
    total_prob = np.sum(wf_grid[i])
    qd_prob = np.sum(wf_grid[i] * qd_mask)
    
    if total_prob > 0:
        probability = qd_prob / total_prob * 100
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
fig_energy.savefig('results_interactive/energy_analysis.png', dpi=300, bbox_inches='tight')

print("All visualizations saved to 'results_interactive' directory.")
print("Displaying interactive 3D plots...")

# Create interactive 3D plots for combined potential and ground state
fig_interactive = plt.figure(figsize=(15, 10))
gs_interactive = GridSpec(1, 2, figure=fig_interactive)

# 3D plot of potential
ax_3d_pot_int = fig_interactive.add_subplot(gs_interactive[0, 0], projection='3d')
surf_pot_int = ax_3d_pot_int.plot_surface(X*1e9, Y*1e9, V_combined, cmap='viridis', 
                                        linewidth=0, antialiased=True)
plt.colorbar(surf_pot_int, ax=ax_3d_pot_int, label='Potential (eV)')
ax_3d_pot_int.set_title('3D Combined Potential')
ax_3d_pot_int.set_xlabel('x (nm)')
ax_3d_pot_int.set_ylabel('y (nm)')
ax_3d_pot_int.set_zlabel('Potential (eV)')

# 3D plot of ground state wavefunction
ax_3d_wf_int = fig_interactive.add_subplot(gs_interactive[0, 1], projection='3d')
surf_wf_int = ax_3d_wf_int.plot_surface(X*1e9, Y*1e9, wf_grid[0], cmap='viridis', 
                                      linewidth=0, antialiased=True)
plt.colorbar(surf_wf_int, ax=ax_3d_wf_int, label='Probability Density')
ax_3d_wf_int.set_title(f'3D Ground State Wavefunction (E = {np.real(eigenvalues[0])/1.602e-19:.4f} eV)')
ax_3d_wf_int.set_xlabel('x (nm)')
ax_3d_wf_int.set_ylabel('y (nm)')
ax_3d_wf_int.set_zlabel('Probability Density')

fig_interactive.tight_layout()
plt.show()  # This will display the interactive plot

print("Interactive visualization complete.")
