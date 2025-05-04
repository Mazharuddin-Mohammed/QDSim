"""
Enhanced visualization module for QDSim.

This module provides advanced visualization capabilities for quantum dot simulations,
including 3D visualization with proper alignment between wavefunctions, densities,
and potentials, as well as tools for analyzing simulation results.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import warnings

# Suppress matplotlib deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Custom colormaps for better visualization
def create_custom_colormaps():
    """Create custom colormaps for better visualization."""
    # Wavefunction colormap (viridis-based but with more contrast)
    wf_colors = plt.cm.viridis(np.linspace(0, 1, 256))
    wf_colors[:10, 3] = np.linspace(0, 1, 10)  # Add some transparency for low values
    wavefunction_cmap = LinearSegmentedColormap.from_list('wavefunction_cmap', wf_colors)
    
    # Potential colormap (blue to red, with white in the middle)
    potential_cmap = LinearSegmentedColormap.from_list(
        'potential_cmap', 
        [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    )
    
    return wavefunction_cmap, potential_cmap

# Create custom colormaps
WAVEFUNCTION_CMAP, POTENTIAL_CMAP = create_custom_colormaps()

def interpolate_to_grid(nodes, values, grid_size=100, method='cubic'):
    """
    Interpolate irregular mesh data to a regular grid for smoother visualization.
    
    Args:
        nodes: Mesh nodes (x, y coordinates)
        values: Values at each node
        grid_size: Size of the regular grid
        method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        X, Y: Meshgrid coordinates
        Z: Interpolated values on the grid
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # Create regular grid
    xi = np.linspace(np.min(x), np.max(x), grid_size)
    yi = np.linspace(np.min(y), np.max(y), grid_size)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolate values to grid
    Z = griddata((x, y), values, (X, Y), method=method, fill_value=0)
    
    return X, Y, Z

def plot_enhanced_wavefunction_3d(ax, mesh, eigenvector, use_nm=True, center_coords=True, 
                                 title=None, azimuth=30, elevation=30, 
                                 plot_type='surface', alpha=0.8, cmap=None):
    """
    Enhanced 3D plot of wavefunction probability density.
    
    Args:
        ax: Matplotlib 3D axis
        mesh: Mesh object
        eigenvector: Eigenvector to plot
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        title: Custom title for the plot
        azimuth: Azimuth angle for 3D view in degrees
        elevation: Elevation angle for 3D view in degrees
        plot_type: Type of plot ('surface', 'wireframe', 'contour')
        alpha: Transparency level (0-1)
        cmap: Custom colormap (if None, use default wavefunction colormap)
    
    Returns:
        surf: The surface plot object
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
    
    # Use custom colormap if not specified
    if cmap is None:
        cmap = WAVEFUNCTION_CMAP
    
    # Create a triangulation for 3D plotting
    triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)
    
    # For smoother visualization, interpolate to a regular grid
    X, Y, Z = interpolate_to_grid(nodes_plot, psi_norm, grid_size=100)
    
    # Plot based on the selected plot type
    if plot_type == 'wireframe':
        surf = ax.plot_wireframe(X, Y, Z, cmap=cmap, alpha=alpha, 
                                rstride=2, cstride=2, linewidth=0.5)
    elif plot_type == 'contour':
        # Plot contour on the bottom of the 3D plot
        offset = np.min(Z) - 0.1 * (np.max(Z) - np.min(Z))
        surf = ax.contourf(X, Y, Z, zdir='z', offset=offset, 
                          cmap=cmap, alpha=alpha, levels=20)
        # Add contour lines on the surface
        ax.contour(X, Y, Z, zdir='z', offset=offset, 
                  colors='k', alpha=0.3, levels=10)
    else:  # Default to surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha, 
                              edgecolor='none', rstride=1, cstride=1)
    
    # Add colorbar
    plt.colorbar(surf, ax=ax, label='Normalized Probability Density')
    
    # Set labels
    ax.set_xlabel('x (nm)' if use_nm else 'x (m)')
    ax.set_ylabel('y (nm)' if use_nm else 'y (m)')
    ax.set_zlabel('Probability Density')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Wavefunction Density')
    
    # Set view angle
    ax.view_init(elevation, azimuth)
    
    return surf

def plot_enhanced_potential_3d(ax, mesh, potential_values, use_nm=True, center_coords=True,
                              title=None, convert_to_eV=True, vmin=None, vmax=None,
                              azimuth=30, elevation=30, plot_type='surface', alpha=0.8,
                              cmap=None):
    """
    Enhanced 3D plot of potential.
    
    Args:
        ax: Matplotlib 3D axis
        mesh: Mesh object
        potential_values: Potential values to plot
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        title: Custom title for the plot
        convert_to_eV: If True, convert potential from J to eV
        vmin, vmax: Min and max values for colorbar
        azimuth: Azimuth angle for 3D view in degrees
        elevation: Elevation angle for 3D view in degrees
        plot_type: Type of plot ('surface', 'wireframe', 'contour')
        alpha: Transparency level (0-1)
        cmap: Custom colormap (if None, use default potential colormap)
    
    Returns:
        surf: The surface plot object
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
    
    # Use custom colormap if not specified
    if cmap is None:
        cmap = POTENTIAL_CMAP
    
    # Create a triangulation for 3D plotting
    triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)
    
    # For smoother visualization, interpolate to a regular grid
    X, Y, Z = interpolate_to_grid(nodes_plot, potential_plot, grid_size=100)
    
    # Set up normalization for the colormap
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot based on the selected plot type
    if plot_type == 'wireframe':
        surf = ax.plot_wireframe(X, Y, Z, cmap=cmap, norm=norm, alpha=alpha,
                                rstride=2, cstride=2, linewidth=0.5)
    elif plot_type == 'contour':
        # Plot contour on the bottom of the 3D plot
        offset = np.min(Z) - 0.1 * (np.max(Z) - np.min(Z))
        surf = ax.contourf(X, Y, Z, zdir='z', offset=offset, 
                          cmap=cmap, norm=norm, alpha=alpha, levels=20)
        # Add contour lines on the surface
        ax.contour(X, Y, Z, zdir='z', offset=offset, 
                  colors='k', alpha=0.3, levels=10)
    else:  # Default to surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm, alpha=alpha,
                              edgecolor='none', rstride=1, cstride=1)
    
    # Add colorbar
    plt.colorbar(surf, ax=ax, label=potential_label)
    
    # Set labels
    ax.set_xlabel('x (nm)' if use_nm else 'x (m)')
    ax.set_ylabel('y (nm)' if use_nm else 'y (m)')
    ax.set_zlabel(potential_label)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Potential')
    
    # Set view angle
    ax.view_init(elevation, azimuth)
    
    return surf

def plot_combined_visualization(fig, mesh, eigenvector, potential_values, eigenvalue=None,
                               use_nm=True, center_coords=True, convert_to_eV=True,
                               azimuth=30, elevation=30, plot_type='surface'):
    """
    Create a combined visualization with aligned wavefunction and potential.
    
    Args:
        fig: Matplotlib figure
        mesh: Mesh object
        eigenvector: Eigenvector to plot
        potential_values: Potential values to plot
        eigenvalue: Eigenvalue corresponding to the eigenvector (optional)
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        convert_to_eV: If True, convert potential from J to eV
        azimuth: Azimuth angle for 3D view in degrees
        elevation: Elevation angle for 3D view in degrees
        plot_type: Type of plot ('surface', 'wireframe', 'contour')
    
    Returns:
        axes: List of axes objects
    """
    # Create a 2x2 grid of subplots
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    
    # Create 3D axes for wavefunction and potential
    ax_wf = fig.add_subplot(gs[0, 0], projection='3d')
    ax_pot = fig.add_subplot(gs[0, 1], projection='3d')
    
    # Create 2D axes for cross-sections
    ax_wf_x = fig.add_subplot(gs[1, 0])
    ax_wf_y = fig.add_subplot(gs[1, 1])
    
    # Plot wavefunction in 3D
    energy_str = ""
    if eigenvalue is not None:
        energy_val = eigenvalue / 1.602e-19 if convert_to_eV else eigenvalue
        energy_str = f" (E = {energy_val:.6f} eV)" if convert_to_eV else f" (E = {energy_val:.6e} J)"
    
    plot_enhanced_wavefunction_3d(
        ax=ax_wf,
        mesh=mesh,
        eigenvector=eigenvector,
        use_nm=use_nm,
        center_coords=center_coords,
        title=f"Wavefunction{energy_str}",
        azimuth=azimuth,
        elevation=elevation,
        plot_type=plot_type
    )
    
    # Plot potential in 3D
    plot_enhanced_potential_3d(
        ax=ax_pot,
        mesh=mesh,
        potential_values=potential_values,
        use_nm=use_nm,
        center_coords=center_coords,
        title="Potential",
        convert_to_eV=convert_to_eV,
        azimuth=azimuth,
        elevation=elevation,
        plot_type=plot_type
    )
    
    # Get mesh data for cross-sections
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
    
    # Calculate probability density
    psi = np.abs(eigenvector) ** 2
    
    # Normalize for better visualization
    psi_max = np.max(psi)
    if psi_max > 0:
        psi_norm = psi / psi_max
    else:
        psi_norm = psi
    
    # Convert potential to eV if requested
    if convert_to_eV:
        potential_plot = potential_values / 1.602e-19  # Convert J to eV
    else:
        potential_plot = potential_values
    
    # Interpolate to a regular grid
    X, Y, Z_wf = interpolate_to_grid(nodes_plot, psi_norm)
    _, _, Z_pot = interpolate_to_grid(nodes_plot, potential_plot)
    
    # Get grid dimensions
    nx, ny = X.shape
    
    # Plot cross-section along x-axis (at y=0)
    y_mid_idx = ny // 2
    ax_wf_x.plot(X[:, y_mid_idx], Z_wf[:, y_mid_idx], 'b-', label='Wavefunction')
    
    # Add potential to the same plot (scaled to fit)
    pot_min = np.min(Z_pot)
    pot_max = np.max(Z_pot)
    pot_range = pot_max - pot_min
    
    if pot_range > 0:
        # Scale potential to [0, 1] for overlay
        pot_scaled = (Z_pot[:, y_mid_idx] - pot_min) / pot_range
        ax_wf_x.plot(X[:, y_mid_idx], pot_scaled, 'r--', label='Potential')
    
    ax_wf_x.set_xlabel('x (nm)' if use_nm else 'x (m)')
    ax_wf_x.set_ylabel('Normalized Value')
    ax_wf_x.set_title('X Cross-section (y=0)')
    ax_wf_x.legend()
    ax_wf_x.grid(True, linestyle='--', alpha=0.7)
    
    # Plot cross-section along y-axis (at x=0)
    x_mid_idx = nx // 2
    ax_wf_y.plot(Y[x_mid_idx, :], Z_wf[x_mid_idx, :], 'b-', label='Wavefunction')
    
    if pot_range > 0:
        # Scale potential to [0, 1] for overlay
        pot_scaled = (Z_pot[x_mid_idx, :] - pot_min) / pot_range
        ax_wf_y.plot(Y[x_mid_idx, :], pot_scaled, 'r--', label='Potential')
    
    ax_wf_y.set_xlabel('y (nm)' if use_nm else 'y (m)')
    ax_wf_y.set_ylabel('Normalized Value')
    ax_wf_y.set_title('Y Cross-section (x=0)')
    ax_wf_y.legend()
    ax_wf_y.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    return [ax_wf, ax_pot, ax_wf_x, ax_wf_y]

def create_energy_level_diagram(ax, eigenvalues, potential_min=None, convert_to_eV=True, 
                               show_labels=True, show_transitions=True):
    """
    Create an energy level diagram.
    
    Args:
        ax: Matplotlib axis
        eigenvalues: List of eigenvalues
        potential_min: Minimum potential value (for reference)
        convert_to_eV: If True, convert energies from J to eV
        show_labels: If True, show energy labels
        show_transitions: If True, show transition arrows
    """
    # Convert energies to eV if requested
    if convert_to_eV:
        energies = np.array(eigenvalues) / 1.602e-19
        energy_unit = 'eV'
    else:
        energies = np.array(eigenvalues)
        energy_unit = 'J'
    
    # Sort energies
    sorted_indices = np.argsort(energies)
    sorted_energies = energies[sorted_indices]
    
    # Set reference level (potential minimum)
    if potential_min is not None:
        if convert_to_eV:
            ref_level = potential_min / 1.602e-19
        else:
            ref_level = potential_min
    else:
        ref_level = sorted_energies[0] - 0.1 * (sorted_energies[-1] - sorted_energies[0])
    
    # Plot energy levels
    for i, energy in enumerate(sorted_energies):
        # Plot horizontal line for energy level
        ax.plot([-0.2, 0.2], [energy, energy], 'b-', linewidth=2)
        
        # Add label
        if show_labels:
            ax.text(0.25, energy, f'E{sorted_indices[i]} = {energy:.6f} {energy_unit}', 
                   va='center', ha='left')
    
    # Plot transitions if requested
    if show_transitions and len(sorted_energies) > 1:
        for i in range(len(sorted_energies) - 1):
            # Plot arrow for transition
            ax.arrow(0, sorted_energies[i], 0, sorted_energies[i+1] - sorted_energies[i] - 0.02,
                    head_width=0.05, head_length=0.02, fc='r', ec='r', length_includes_head=True)
            
            # Add transition energy label
            transition_energy = sorted_energies[i+1] - sorted_energies[i]
            ax.text(-0.3, sorted_energies[i] + transition_energy/2, 
                   f'ΔE = {transition_energy:.6f} {energy_unit}', 
                   va='center', ha='right', color='r')
    
    # Plot potential reference if provided
    if potential_min is not None:
        ax.axhline(y=ref_level, color='k', linestyle='--', alpha=0.5)
        ax.text(0.25, ref_level, f'V_min = {ref_level:.6f} {energy_unit}', 
               va='center', ha='left', alpha=0.5)
    
    # Set labels and title
    ax.set_ylabel(f'Energy ({energy_unit})')
    ax.set_title('Energy Level Diagram')
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax

def calculate_transition_probabilities(eigenvectors, operator=None):
    """
    Calculate transition probabilities between states.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        operator: Operator matrix for transitions (if None, use identity)
    
    Returns:
        prob_matrix: Matrix of transition probabilities
    """
    n_states = eigenvectors.shape[1]
    prob_matrix = np.zeros((n_states, n_states))
    
    # If no operator is provided, use identity (overlap integral)
    if operator is None:
        operator = np.eye(eigenvectors.shape[0])
    
    # Calculate transition probabilities
    for i in range(n_states):
        for j in range(n_states):
            # Calculate matrix element <ψi|O|ψj>
            if np.iscomplexobj(eigenvectors):
                matrix_element = np.abs(np.vdot(eigenvectors[:, i], 
                                              operator @ eigenvectors[:, j])) ** 2
            else:
                matrix_element = np.abs(eigenvectors[:, i].T @ operator @ eigenvectors[:, j]) ** 2
            
            prob_matrix[i, j] = matrix_element
    
    return prob_matrix

def plot_transition_matrix(ax, prob_matrix, eigenvalues=None, convert_to_eV=True,
                          cmap='viridis', show_values=True):
    """
    Plot transition probability matrix.
    
    Args:
        ax: Matplotlib axis
        prob_matrix: Matrix of transition probabilities
        eigenvalues: List of eigenvalues (optional)
        convert_to_eV: If True, convert energies from J to eV
        cmap: Colormap for the plot
        show_values: If True, show probability values in the cells
    """
    n_states = prob_matrix.shape[0]
    
    # Create labels for axes
    if eigenvalues is not None:
        if convert_to_eV:
            energies = np.array(eigenvalues) / 1.602e-19
            labels = [f'E{i}\n({energies[i]:.4f} eV)' for i in range(n_states)]
        else:
            labels = [f'E{i}\n({eigenvalues[i]:.2e} J)' for i in range(n_states)]
    else:
        labels = [f'State {i}' for i in range(n_states)]
    
    # Plot the matrix
    im = ax.imshow(prob_matrix, cmap=cmap, vmin=0, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Transition Probability')
    
    # Add labels
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values in the cells
    if show_values:
        for i in range(n_states):
            for j in range(n_states):
                text_color = 'white' if prob_matrix[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{prob_matrix[i, j]:.2f}', 
                       ha="center", va="center", color=text_color)
    
    # Set title and labels
    ax.set_title('Transition Probability Matrix')
    ax.set_xlabel('Final State')
    ax.set_ylabel('Initial State')
    
    return im
