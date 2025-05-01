"""
Visualization functions for QDSim.

This module provides functions for visualizing simulation results,
including wavefunctions, potentials, and electric fields.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation

def plot_wavefunction(ax, mesh, eigenvector, use_nm=True, center_coords=True, title=None):
    """
    Plot wavefunction probability density.

    Args:
        ax: Matplotlib axis
        mesh: Mesh object
        eigenvector: Eigenvector to plot
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        title: Custom title for the plot
    """
    # Get mesh data
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())

    # Center coordinates if requested
    if center_coords:
        x_center = np.mean(nodes[:, 0])
        y_center = np.mean(nodes[:, 1])
        nodes_plot = nodes.copy()
        nodes_plot[:, 0] -= x_center
        nodes_plot[:, 1] -= y_center
    else:
        nodes_plot = nodes

    # Convert to nm if requested
    scale = 1e9 if use_nm else 1.0
    nodes_plot = nodes_plot * scale

    # Calculate probability density
    psi = np.abs(eigenvector) ** 2

    # Normalize for better visualization
    psi_max = np.max(psi)
    if psi_max > 0:
        psi_norm = psi / psi_max
    else:
        psi_norm = psi

    # Create a triangulation for plotting
    triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)

    # Plot
    im = ax.tricontourf(triang, psi_norm, cmap='viridis', levels=50)

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Normalized Probability Density')

    # Set labels
    ax.set_xlabel('x (nm)' if use_nm else 'x (m)')
    ax.set_ylabel('y (nm)' if use_nm else 'y (m)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Wavefunction Density')

    # Set equal aspect ratio
    ax.set_aspect('equal')

    return im

def plot_electric_field(ax, mesh, poisson_solver, use_nm=True, center_coords=True, title=None):
    """
    Plot electric field vectors.

    Args:
        ax: Matplotlib axis
        mesh: Mesh object
        poisson_solver: Poisson solver object
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        title: Custom title for the plot
    """
    # Get mesh data
    nodes = np.array(mesh.get_nodes())

    # Center coordinates if requested
    if center_coords:
        x_center = np.mean(nodes[:, 0])
        y_center = np.mean(nodes[:, 1])
        nodes_plot = nodes.copy()
        nodes_plot[:, 0] -= x_center
        nodes_plot[:, 1] -= y_center
    else:
        nodes_plot = nodes

    # Convert to nm if requested
    scale = 1e9 if use_nm else 1.0
    nodes_plot = nodes_plot * scale

    # If poisson_solver is None, return empty plot
    if poisson_solver is None:
        ax.text(0.5, 0.5, "No electric field data available",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('x (nm)' if use_nm else 'x (m)')
        ax.set_ylabel('y (nm)' if use_nm else 'y (m)')
        ax.set_title(title if title else 'Electric Field')
        ax.set_aspect('equal')
        return

    # Get electric field
    E = np.array([poisson_solver.get_electric_field(x, y) for x, y in nodes])

    # Scale electric field for better visualization
    E_mag = np.sqrt(E[:, 0]**2 + E[:, 1]**2)
    E_max = np.max(E_mag)
    if E_max > 0:
        scale_factor = 1.0 / E_max
    else:
        scale_factor = 1.0

    # Plot
    ax.quiver(nodes_plot[:, 0], nodes_plot[:, 1],
              E[:, 0] * scale_factor, E[:, 1] * scale_factor,
              E_mag, cmap='viridis', scale=25)

    # Set labels
    ax.set_xlabel('x (nm)' if use_nm else 'x (m)')
    ax.set_ylabel('y (nm)' if use_nm else 'y (m)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Electric Field')

    # Set equal aspect ratio
    ax.set_aspect('equal')

def plot_potential(ax, mesh, potential_values, use_nm=True, center_coords=True,
                  title=None, convert_to_eV=True, vmin=None, vmax=None):
    """
    Plot potential.

    Args:
        ax: Matplotlib axis
        mesh: Mesh object
        potential_values: Potential values to plot
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        title: Custom title for the plot
        convert_to_eV: If True, convert potential from J to eV
        vmin, vmax: Min and max values for colorbar
    """
    # Get mesh data
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())

    # Center coordinates if requested
    if center_coords:
        x_center = np.mean(nodes[:, 0])
        y_center = np.mean(nodes[:, 1])
        nodes_plot = nodes.copy()
        nodes_plot[:, 0] -= x_center
        nodes_plot[:, 1] -= y_center
    else:
        nodes_plot = nodes

    # Convert to nm if requested
    scale = 1e9 if use_nm else 1.0
    nodes_plot = nodes_plot * scale

    # Convert potential to eV if requested
    if convert_to_eV:
        potential_plot = potential_values / 1.602e-19  # Convert J to eV
        potential_label = 'Potential (eV)'
    else:
        potential_plot = potential_values
        potential_label = 'Potential (V)'

    # Set colormap limits
    if vmin is None:
        vmin = np.min(potential_plot)
    if vmax is None:
        vmax = np.max(potential_plot)

    # Ensure reasonable limits
    if vmax - vmin < 1e-6:
        vmin -= 0.5
        vmax += 0.5

    # Create a triangulation for plotting
    triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)

    # Plot
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.tricontourf(triang, potential_plot, cmap='viridis', norm=norm, levels=50)

    # Add colorbar
    plt.colorbar(im, ax=ax, label=potential_label)

    # Set labels
    ax.set_xlabel('x (nm)' if use_nm else 'x (m)')
    ax.set_ylabel('y (nm)' if use_nm else 'y (m)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Potential')

    # Set equal aspect ratio
    ax.set_aspect('equal')

    return im

def plot_energy_shift(ax, V_r_values, energies, linewidths=None, title=None):
    """
    Plot energy shift vs. voltage.

    Args:
        ax: Matplotlib axis
        V_r_values: Reverse bias voltage values
        energies: Energy values (eV)
        linewidths: Linewidth values (eV), optional
        title: Custom title for the plot
    """
    # Plot energies
    ax.plot(V_r_values, energies, 'o-', label='Energy (eV)')

    # Plot linewidths if provided
    if linewidths is not None:
        ax2 = ax.twinx()
        ax2.plot(V_r_values, linewidths, 's-', color='red', label='Linewidth (eV)')
        ax2.set_ylabel('Linewidth (eV)')
        ax2.legend(loc='upper right')

    # Set labels
    ax.set_xlabel('Reverse Bias (V)')
    ax.set_ylabel('Energy (eV)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Energy vs. Voltage')

    # Add legend
    ax.legend(loc='upper left')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
