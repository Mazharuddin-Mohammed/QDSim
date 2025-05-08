#!/usr/bin/env python3
"""
3D Visualization Tool

This script provides interactive 3D visualization of simulation results.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import qdsim_cpp as qdc
import sys
import os

class Visualizer3D:
    """Interactive 3D visualization tool for QDSim results."""
    
    def __init__(self, mesh, potential, n_conc, p_conc):
        """Initialize the visualizer with simulation results."""
        self.mesh = mesh
        self.potential = potential
        self.n_conc = n_conc
        self.p_conc = p_conc
        
        # Create interpolators
        self.simple_mesh = qdc.create_simple_mesh(mesh)
        self.interpolator = qdc.SimpleInterpolator(self.simple_mesh)
        
        # Get mesh dimensions
        self.Lx = mesh.get_lx()
        self.Ly = mesh.get_ly()
        
        # Create grid for visualization
        self.nx = 101
        self.ny = 51
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize data arrays
        self.potential_grid = np.zeros((self.ny, self.nx))
        self.electron_conc = np.zeros((self.ny, self.nx))
        self.hole_conc = np.zeros((self.ny, self.nx))
        
        # Interpolate data onto grid
        self._interpolate_data()
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize plot
        self.surf = None
        self.current_data = 'potential'
        self.plot_data()
        
        # Add controls
        self._add_controls()
        
        # Add animation
        self.animation = None
        self.angle = 0
        
    def _interpolate_data(self):
        """Interpolate data onto grid."""
        for i in range(self.ny):
            for j in range(self.nx):
                xi, yi = self.X[i, j], self.Y[i, j]
                
                try:
                    self.potential_grid[i, j] = self.interpolator.interpolate(xi, yi, self.potential)
                    self.electron_conc[i, j] = self.interpolator.interpolate(xi, yi, self.n_conc)
                    self.hole_conc[i, j] = self.interpolator.interpolate(xi, yi, self.p_conc)
                except:
                    pass
    
    def _add_controls(self):
        """Add interactive controls to the figure."""
        # Add data selection radio buttons
        rax = plt.axes([0.05, 0.7, 0.15, 0.15])
        self.radio = RadioButtons(rax, ('Potential', 'Electron Concentration', 'Hole Concentration'))
        self.radio.on_clicked(self.update_data)
        
        # Add view angle slider
        ax_angle = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.slider_angle = Slider(ax_angle, 'View Angle', 0, 360, valinit=30)
        self.slider_angle.on_changed(self.update_view)
        
        # Add elevation angle slider
        ax_elev = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider_elev = Slider(ax_elev, 'Elevation', 0, 90, valinit=30)
        self.slider_elev.on_changed(self.update_elevation)
        
        # Add animation button
        ax_anim = plt.axes([0.8, 0.15, 0.1, 0.04])
        self.button_anim = Button(ax_anim, 'Animate')
        self.button_anim.on_clicked(self.toggle_animation)
    
    def update_data(self, label):
        """Update the data being displayed."""
        if label == 'Potential':
            self.current_data = 'potential'
        elif label == 'Electron Concentration':
            self.current_data = 'electron'
        elif label == 'Hole Concentration':
            self.current_data = 'hole'
        
        self.plot_data()
    
    def update_view(self, angle):
        """Update the view angle."""
        self.ax.view_init(elev=self.slider_elev.val, azim=angle)
        self.fig.canvas.draw_idle()
    
    def update_elevation(self, elev):
        """Update the elevation angle."""
        self.ax.view_init(elev=elev, azim=self.slider_angle.val)
        self.fig.canvas.draw_idle()
    
    def toggle_animation(self, event):
        """Toggle animation on/off."""
        if self.animation is None:
            self.animation = animation.FuncAnimation(
                self.fig, self.animate, frames=np.arange(0, 360, 2),
                interval=50, blit=False)
            self.button_anim.label.set_text('Stop')
        else:
            self.animation.event_source.stop()
            self.animation = None
            self.button_anim.label.set_text('Animate')
        
        self.fig.canvas.draw_idle()
    
    def animate(self, angle):
        """Animation function."""
        self.ax.view_init(elev=self.slider_elev.val, azim=angle)
        self.slider_angle.set_val(angle)
        return self.surf,
    
    def plot_data(self):
        """Plot the current data."""
        # Clear the axis
        self.ax.clear()
        
        # Get the data to plot
        if self.current_data == 'potential':
            Z = self.potential_grid
            title = 'Electrostatic Potential (V)'
            cmap = cm.viridis
        elif self.current_data == 'electron':
            Z = np.log10(self.electron_conc)
            title = 'Electron Concentration (log10(nm^-3))'
            cmap = cm.Blues
        elif self.current_data == 'hole':
            Z = np.log10(self.hole_conc)
            title = 'Hole Concentration (log10(nm^-3))'
            cmap = cm.Reds
        
        # Create the surface plot
        self.surf = self.ax.plot_surface(
            self.X, self.Y, Z, cmap=cmap,
            linewidth=0, antialiased=True)
        
        # Set labels and title
        self.ax.set_xlabel('x (nm)')
        self.ax.set_ylabel('y (nm)')
        self.ax.set_zlabel(title)
        self.ax.set_title(title)
        
        # Add a color bar
        if hasattr(self, 'cbar'):
            self.cbar.remove()
        self.cbar = self.fig.colorbar(self.surf, ax=self.ax, shrink=0.5, aspect=5)
        
        # Set the view angle
        self.ax.view_init(elev=self.slider_elev.val, azim=self.slider_angle.val)
        
        # Draw the figure
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the figure."""
        plt.show()
    
    def save(self, filename):
        """Save the figure to a file."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')


def main():
    """Main function."""
    # Check if a file was specified
    if len(sys.argv) < 2:
        print("Usage: python visualization_3d.py <simulation_file.npz>")
        return
    
    # Load simulation results
    sim_file = sys.argv[1]
    if not os.path.exists(sim_file):
        print(f"Error: File {sim_file} not found")
        return
    
    try:
        data = np.load(sim_file)
        mesh = data['mesh'].item()
        potential = data['potential']
        n_conc = data['n_conc']
        p_conc = data['p_conc']
    except:
        print(f"Error: Could not load simulation results from {sim_file}")
        return
    
    # Create visualizer
    visualizer = Visualizer3D(mesh, potential, n_conc, p_conc)
    
    # Show the visualization
    visualizer.show()


if __name__ == "__main__":
    main()
