"""
Visualization module for QDSim.

This module provides functions for visualizing quantum dot simulation results,
including wavefunctions, potentials, electric fields, and energy shifts.
It supports both 2D and 3D visualizations.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

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

    # Plot
    im = ax.tricontourf(nodes_plot[:, 0], nodes_plot[:, 1], elements, psi_norm,
                        cmap='viridis', levels=50)

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

    # Plot
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.tricontourf(nodes_plot[:, 0], nodes_plot[:, 1], elements,
                        potential_plot, cmap='viridis', norm=norm, levels=50)

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

# 3D Visualization Functions

def plot_potential_3d(simulator, ax=None, cmap='viridis', alpha=0.8, view_angle=(30, 45),
                   resolution=50, title='Potential'):
    """
    Plot the potential in 3D.

    Args:
        simulator: The simulator object
        ax: Matplotlib axis to plot on (if None, a new one is created)
        cmap: Colormap to use
        alpha: Transparency of the surface
        view_angle: Tuple of (elevation, azimuth) for the 3D view
        resolution: Number of points in each dimension for the grid
        title: Title of the plot

    Returns:
        The matplotlib axis
    """
    # Create axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    # Get domain size
    Lx = simulator.config.Lx
    Ly = simulator.config.Ly

    # Create a grid for plotting
    x = np.linspace(0, Lx, resolution)
    y = np.linspace(0, Ly, resolution)
    X, Y = np.meshgrid(x, y)

    # Interpolate the potential onto the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = simulator.potential(X[i, j], Y[i, j])

    # Convert potential to eV
    Z = Z / simulator.config.e_charge

    # Plot the potential
    surf = ax.plot_surface(X*1e9, Y*1e9, Z, cmap=cmap, alpha=alpha)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_zlabel('Potential (eV)')
    ax.set_title(title)

    # Set the view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    return ax

def plot_wavefunction_3d(simulator, state_idx=0, ax=None, cmap='plasma', alpha=0.8,
                      view_angle=(30, 45), resolution=50, title=None):
    """
    Plot a wavefunction in 3D.

    Args:
        simulator: The simulator object
        state_idx: Index of the state to plot
        ax: Matplotlib axis to plot on (if None, a new one is created)
        cmap: Colormap to use
        alpha: Transparency of the surface
        view_angle: Tuple of (elevation, azimuth) for the 3D view
        resolution: Number of points in each dimension for the grid
        title: Title of the plot (if None, a default title is used)

    Returns:
        The matplotlib axis
    """
    # Create axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    # Get domain size
    Lx = simulator.config.Lx
    Ly = simulator.config.Ly

    # Create a grid for plotting
    x = np.linspace(0, Lx, resolution)
    y = np.linspace(0, Ly, resolution)
    X, Y = np.meshgrid(x, y)

    # Check if we have eigenvectors
    if simulator.eigenvectors is None or simulator.eigenvectors.shape[1] <= state_idx:
        print(f"Warning: No eigenvector available for state {state_idx}")
        return ax

    # Interpolate the wavefunction onto the grid
    Z = np.zeros_like(X)
    try:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Get the wavefunction value at this point
                wf_value = simulator.interpolator.interpolate(X[i, j], Y[i, j], simulator.eigenvectors[:, state_idx])
                # Calculate the probability density
                Z[i, j] = abs(wf_value)**2
    except Exception as e:
        print(f"Warning: Failed to interpolate wavefunction: {e}")
        return ax

    # Normalize the wavefunction for better visualization
    if np.max(Z) > 0:
        Z = Z / np.max(Z)

    # Plot the wavefunction
    surf = ax.plot_surface(X*1e9, Y*1e9, Z, cmap=cmap, alpha=alpha)
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_zlabel('Probability density')

    # Set the title
    if title is None:
        if simulator.eigenvalues is not None and len(simulator.eigenvalues) > state_idx:
            energy = simulator.eigenvalues[state_idx] / simulator.config.e_charge
            title = f'State {state_idx} (E = {energy:.6f} eV)'
        else:
            title = f'State {state_idx}'
    ax.set_title(title)

    # Set the view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    return ax

def create_simulation_dashboard(simulator, num_states=3, resolution=50, figsize=(15, 10)):
    """
    Create a comprehensive dashboard of simulation results.

    Args:
        simulator: The simulator object
        num_states: Number of states to plot
        resolution: Resolution for the grid interpolation
        figsize: Size of the figure

    Returns:
        The matplotlib figure
    """
    # Create the figure
    fig = plt.figure(figsize=figsize)

    # Create a grid layout
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Plot the potential in 3D
    ax_pot_3d = fig.add_subplot(gs[0, 0], projection='3d')
    plot_potential_3d(simulator, ax=ax_pot_3d, resolution=resolution)

    # Plot the potential in 2D
    ax_pot_2d = fig.add_subplot(gs[0, 1])
    nodes = np.array(simulator.mesh.get_nodes())
    elements = np.array(simulator.mesh.get_elements())
    potential_values = np.array(simulator.sc_solver.get_potential())
    plot_potential(ax_pot_2d, simulator.mesh, potential_values)

    # Plot the energy levels
    ax_energy = fig.add_subplot(gs[0, 2])
    if simulator.eigenvalues is not None and len(simulator.eigenvalues) > 0:
        # Convert eigenvalues to eV
        energies = [ev.real / simulator.config.e_charge for ev in simulator.eigenvalues[:num_states]]

        # Plot the energy levels
        for i, energy in enumerate(energies):
            ax_energy.axhline(y=energy, color='blue', linestyle='-', linewidth=2)
            ax_energy.text(1.02, energy, f'{energy:.6f} eV', va='center')

        # Set the labels and title
        ax_energy.set_xlabel('State')
        ax_energy.set_ylabel('Energy (eV)')
        ax_energy.set_title('Energy Levels')

        # Set the x-axis limits and ticks
        ax_energy.set_xlim(0, 1)
        ax_energy.set_xticks([])

        # Set the y-axis limits with some padding
        if len(energies) > 0:
            y_min = min(energies) - 0.1 * (max(energies) - min(energies))
            y_max = max(energies) + 0.1 * (max(energies) - min(energies))
            ax_energy.set_ylim(y_min, y_max)
    else:
        ax_energy.text(0.5, 0.5, 'No eigenvalues available',
                      ha='center', va='center', transform=ax_energy.transAxes)

    # Plot the wavefunctions
    for i in range(min(num_states, 3)):
        if simulator.eigenvectors is not None and simulator.eigenvectors.shape[1] > i:
            # 3D plot
            ax_wf_3d = fig.add_subplot(gs[1, i], projection='3d')
            plot_wavefunction_3d(simulator, state_idx=i, ax=ax_wf_3d, resolution=resolution)
        else:
            # Create a placeholder
            ax_wf = fig.add_subplot(gs[1, i])
            ax_wf.text(0.5, 0.5, f'No wavefunction available for state {i}',
                      ha='center', va='center', transform=ax_wf.transAxes)

    # Adjust the layout
    plt.tight_layout()

    return fig

def save_simulation_results(simulator, filename_prefix='qdsim_results', format='png', dpi=300):
    """
    Save comprehensive simulation results to files.

    Args:
        simulator: The simulator object
        filename_prefix: Prefix for the filenames
        format: File format (png, pdf, svg, etc.)
        dpi: Resolution for raster formats

    Returns:
        List of saved filenames
    """
    filenames = []

    # Create the dashboard
    fig_dashboard = create_simulation_dashboard(simulator)
    dashboard_filename = f"{filename_prefix}_dashboard.{format}"
    fig_dashboard.savefig(dashboard_filename, dpi=dpi, bbox_inches='tight')
    filenames.append(dashboard_filename)
    plt.close(fig_dashboard)

    # Save individual potential plots
    fig_pot = plt.figure(figsize=(8, 6))
    ax_pot = fig_pot.add_subplot(111, projection='3d')
    plot_potential_3d(simulator, ax=ax_pot)
    pot_filename = f"{filename_prefix}_potential.{format}"
    fig_pot.savefig(pot_filename, dpi=dpi, bbox_inches='tight')
    filenames.append(pot_filename)
    plt.close(fig_pot)

    # Save individual wavefunction plots for the first few states
    num_states = min(5, len(simulator.eigenvalues) if simulator.eigenvalues is not None else 0)
    for i in range(num_states):
        fig_wf = plt.figure(figsize=(8, 6))
        ax_wf = fig_wf.add_subplot(111, projection='3d')
        plot_wavefunction_3d(simulator, state_idx=i, ax=ax_wf)
        wf_filename = f"{filename_prefix}_wavefunction_{i}.{format}"
        fig_wf.savefig(wf_filename, dpi=dpi, bbox_inches='tight')
        filenames.append(wf_filename)
        plt.close(fig_wf)

    print(f"Saved {len(filenames)} files with prefix '{filename_prefix}'")
    return filenames