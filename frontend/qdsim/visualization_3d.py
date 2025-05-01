"""
3D visualization functions for QDSim.

This module provides functions for 3D visualization of simulation results,
including wavefunctions, potentials, and electric fields.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_wavefunction(ax, mesh, eigenvector, use_nm=True, center_coords=True, 
                         title=None, azimuth=30, elevation=30, wireframe=False):
    """
    Plot wavefunction probability density in 3D.

    Args:
        ax: Matplotlib 3D axis
        mesh: Mesh object
        eigenvector: Eigenvector to plot
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        title: Custom title for the plot
        azimuth: Azimuth angle for 3D view in degrees
        elevation: Elevation angle for 3D view in degrees
        wireframe: If True, plot as wireframe instead of surface
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

    # Create a triangulation for 3D plotting
    triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)

    # Plot
    if wireframe:
        # Wireframe plot
        surf = ax.plot_trisurf(triang, psi_norm, cmap='viridis', 
                              edgecolor='black', linewidth=0.2, alpha=0.8)
    else:
        # Surface plot
        surf = ax.plot_trisurf(triang, psi_norm, cmap='viridis', 
                              edgecolor='none', alpha=0.8)
    
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

def plot_3d_potential(ax, mesh, potential_values, use_nm=True, center_coords=True,
                     title=None, convert_to_eV=True, vmin=None, vmax=None,
                     azimuth=30, elevation=30, wireframe=False):
    """
    Plot potential in 3D.

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
        wireframe: If True, plot as wireframe instead of surface
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

    # Create a triangulation for 3D plotting
    triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)

    # Plot
    norm = Normalize(vmin=vmin, vmax=vmax)
    if wireframe:
        # Wireframe plot
        surf = ax.plot_trisurf(triang, potential_plot, cmap='viridis', norm=norm,
                              edgecolor='black', linewidth=0.2, alpha=0.8)
    else:
        # Surface plot
        surf = ax.plot_trisurf(triang, potential_plot, cmap='viridis', norm=norm,
                              edgecolor='none', alpha=0.8)
    
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

def create_interactive_3d_plot(mesh, eigenvectors, eigenvalues, potential, 
                              use_nm=True, center_coords=True, convert_to_eV=True):
    """
    Create an interactive 3D plot with sliders for azimuth and elevation angles.

    Args:
        mesh: Mesh object
        eigenvectors: Eigenvectors to plot
        eigenvalues: Eigenvalues corresponding to eigenvectors
        potential: Potential values to plot
        use_nm: If True, convert coordinates to nm
        center_coords: If True, center the coordinates at (0,0)
        convert_to_eV: If True, convert potential from J to eV
    """
    from matplotlib.widgets import Slider, RadioButtons
    
    # Create figure and axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial values
    azimuth = 30
    elevation = 30
    state_idx = 0
    plot_type = 'wavefunction'
    wireframe = False
    
    # Create plot
    def update_plot():
        ax.clear()
        
        if plot_type == 'wavefunction':
            # Plot wavefunction
            if state_idx < len(eigenvectors[0]):
                energy_str = f"{np.real(eigenvalues[state_idx])/1.602e-19:.6f} eV"
                if np.iscomplex(eigenvalues[state_idx]):
                    energy_str += f" + {np.imag(eigenvalues[state_idx])/1.602e-19:.6f}i eV"
                
                plot_3d_wavefunction(
                    ax=ax,
                    mesh=mesh,
                    eigenvector=eigenvectors[:, state_idx],
                    use_nm=use_nm,
                    center_coords=center_coords,
                    title=f"State {state_idx} (E = {energy_str})",
                    azimuth=azimuth,
                    elevation=elevation,
                    wireframe=wireframe
                )
        else:
            # Plot potential
            plot_3d_potential(
                ax=ax,
                mesh=mesh,
                potential_values=potential,
                use_nm=use_nm,
                center_coords=center_coords,
                title="Potential",
                convert_to_eV=convert_to_eV,
                azimuth=azimuth,
                elevation=elevation,
                wireframe=wireframe
            )
        
        fig.canvas.draw_idle()
    
    # Initial plot
    update_plot()
    
    # Add sliders for azimuth and elevation
    ax_azimuth = plt.axes([0.25, 0.05, 0.65, 0.03])
    ax_elevation = plt.axes([0.25, 0.1, 0.65, 0.03])
    
    s_azimuth = Slider(ax_azimuth, 'Azimuth', 0, 360, valinit=azimuth)
    s_elevation = Slider(ax_elevation, 'Elevation', 0, 90, valinit=elevation)
    
    def update_azimuth(val):
        nonlocal azimuth
        azimuth = val
        update_plot()
    
    def update_elevation(val):
        nonlocal elevation
        elevation = val
        update_plot()
    
    s_azimuth.on_changed(update_azimuth)
    s_elevation.on_changed(update_elevation)
    
    # Add radio buttons for plot type
    ax_plot_type = plt.axes([0.025, 0.5, 0.15, 0.15])
    radio_plot_type = RadioButtons(ax_plot_type, ('wavefunction', 'potential'))
    
    def update_plot_type(val):
        nonlocal plot_type
        plot_type = val
        update_plot()
    
    radio_plot_type.on_clicked(update_plot_type)
    
    # Add radio buttons for wireframe
    ax_wireframe = plt.axes([0.025, 0.3, 0.15, 0.15])
    radio_wireframe = RadioButtons(ax_wireframe, ('surface', 'wireframe'))
    
    def update_wireframe(val):
        nonlocal wireframe
        wireframe = (val == 'wireframe')
        update_plot()
    
    radio_wireframe.on_clicked(update_wireframe)
    
    # Add slider for state index (if plotting wavefunction)
    if len(eigenvectors) > 0:
        ax_state = plt.axes([0.25, 0.15, 0.65, 0.03])
        s_state = Slider(ax_state, 'State', 0, len(eigenvectors[0])-1, valinit=state_idx, valstep=1)
        
        def update_state(val):
            nonlocal state_idx
            state_idx = int(val)
            update_plot()
        
        s_state.on_changed(update_state)
    
    plt.show()
