#!/usr/bin/env python3
"""
Simple mesh and potential visualization for the deep QD simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction

# Create output directory
os.makedirs('results_mesh', exist_ok=True)

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

# Run simulation for zero bias
print("Simulating with mesh visualization...")
config.V_r = 0.0  # Zero bias
sim = Simulator(config)
eigenvalues, eigenvectors = sim.run(1)  # Just get the ground state

# Get mesh data
nodes = np.array(sim.mesh.get_nodes())
elements = np.array(sim.mesh.get_elements())

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

# Plot mesh
ax_mesh = fig_potentials.add_subplot(gs[1, 0])
triang = Triangulation(nodes[:, 0]*1e9, nodes[:, 1]*1e9, elements)
ax_mesh.triplot(triang, 'k-', lw=0.5)
ax_mesh.set_title('Mesh Visualization')
ax_mesh.set_xlabel('x (nm)')
ax_mesh.set_ylabel('y (nm)')
ax_mesh.set_aspect('equal')

# Plot mesh with potential
ax_mesh_pot = fig_potentials.add_subplot(gs[1, 1])
im_mesh_pot = ax_mesh_pot.tricontourf(triang, sim.phi, 50, cmap='viridis')
plt.colorbar(im_mesh_pot, ax=ax_mesh_pot, label='Potential (V)')
ax_mesh_pot.triplot(triang, 'k-', lw=0.2, alpha=0.3)
ax_mesh_pot.set_title('Mesh with Potential')
ax_mesh_pot.set_xlabel('x (nm)')
ax_mesh_pot.set_ylabel('y (nm)')
ax_mesh_pot.set_aspect('equal')

# Plot ground state wavefunction
ax_wf = fig_potentials.add_subplot(gs[1, 2])
plot_wavefunction(ax_wf, sim.mesh, eigenvectors[:, 0])
ax_wf.set_title(f'Ground State (E = {np.real(eigenvalues[0])/1.602e-19:.4f} eV)')
ax_wf.set_aspect('equal')

fig_potentials.tight_layout()
fig_potentials.savefig('results_mesh/potentials_and_mesh.png', dpi=300, bbox_inches='tight')

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
ax_qd_region.set_aspect('equal')

# Plot mesh with depletion region highlighted
ax_depletion = fig_regions.add_subplot(gs_reg[0, 1])
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

# Plot mesh with both regions highlighted
ax_both = fig_regions.add_subplot(gs_reg[0, 2])
ax_both.triplot(triang, 'k-', lw=0.5, alpha=0.3)
# Highlight QD region
circle = plt.Circle((0, 0), config.R*1e9, color='r', fill=False, lw=2, label='QD')
ax_both.add_patch(circle)
# Highlight depletion region
rect = plt.Rectangle((-config.depletion_width/2*1e9, -config.Ly/2*1e9), 
                     config.depletion_width*1e9, config.Ly*1e9, 
                     color='b', fill=False, lw=2, label='Depletion')
ax_both.add_patch(rect)
ax_both.set_title('Both Regions Highlighted')
ax_both.set_xlabel('x (nm)')
ax_both.set_ylabel('y (nm)')
ax_both.set_aspect('equal')
# Add legend
ax_both.legend(handles=[
    plt.Line2D([0], [0], color='r', lw=2, label='QD Region'),
    plt.Line2D([0], [0], color='b', lw=2, label='Depletion Region')
], loc='upper right')

fig_regions.tight_layout()
fig_regions.savefig('results_mesh/mesh_regions.png', dpi=300, bbox_inches='tight')

print("Mesh visualizations saved to 'results_mesh' directory.")
