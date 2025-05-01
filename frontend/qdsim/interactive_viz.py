"""
Interactive visualization module for QDSim.

This module provides interactive visualization capabilities for quantum dot simulations,
including the ability to zoom, pan, and rotate visualizations using mouse controls.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

from .visualization import plot_wavefunction, plot_potential, plot_electric_field


class InteractiveVisualizer:
    """
    Interactive visualizer for quantum dot simulations.
    
    This class provides interactive visualization capabilities for quantum dot simulations,
    including the ability to zoom, pan, and rotate visualizations using mouse controls.
    """
    
    def __init__(self, mesh, eigenvectors=None, eigenvalues=None, potential=None, poisson_solver=None):
        """
        Initialize the interactive visualizer.
        
        Args:
            mesh: Mesh object
            eigenvectors: List of eigenvectors (wavefunctions)
            eigenvalues: List of eigenvalues (energies)
            potential: Potential values
            poisson_solver: Poisson solver object for electric field visualization
        """
        self.mesh = mesh
        self.eigenvectors = eigenvectors if eigenvectors is not None else []
        self.eigenvalues = eigenvalues if eigenvalues is not None else []
        self.potential = potential
        self.poisson_solver = poisson_solver
        
        # Convert to numpy arrays if needed
        if self.eigenvectors and not isinstance(self.eigenvectors, np.ndarray):
            self.eigenvectors = np.array(self.eigenvectors)
        if self.eigenvalues and not isinstance(self.eigenvalues, np.ndarray):
            self.eigenvalues = np.array(self.eigenvalues)
        if self.potential is not None and not isinstance(self.potential, np.ndarray):
            self.potential = np.array(self.potential)
        
        # Visualization parameters
        self.use_nm = True
        self.center_coords = True
        self.convert_to_eV = True
        self.current_state = 0 if self.eigenvectors else None
        self.view_mode = '2D'  # '2D' or '3D'
        self.plot_type = 'wavefunction'  # 'wavefunction', 'potential', 'electric_field'
        
        # 3D visualization parameters
        self.azimuth = 30
        self.elevation = 30
        self.wireframe = False
        
        # Animation parameters
        self.animation_running = False
        self.animation_speed = 100  # ms between frames
        self.animation_obj = None
        
    def show_interactive_plot(self):
        """
        Show an interactive plot with controls for visualization parameters.
        
        This method creates a matplotlib figure with interactive controls for
        adjusting visualization parameters such as the state index, view mode,
        and plot type.
        """
        if not HAS_TKINTER:
            print("Tkinter is not available. Using matplotlib's interactive mode instead.")
            self._show_matplotlib_interactive()
            return
            
        # Create Tkinter window
        root = tk.Tk()
        root.title("QDSim Interactive Visualization")
        root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control frame
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Create plot frame
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add controls
        ttk.Label(control_frame, text="Visualization Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Plot type selection
        ttk.Label(control_frame, text="Plot Type:").pack(anchor=tk.W, pady=(10, 0))
        plot_type_var = tk.StringVar(value=self.plot_type)
        plot_types = ['wavefunction', 'potential', 'electric_field']
        plot_type_menu = ttk.Combobox(control_frame, textvariable=plot_type_var, values=plot_types, state="readonly")
        plot_type_menu.pack(fill=tk.X, pady=5)
        plot_type_menu.bind("<<ComboboxSelected>>", lambda e: self._update_plot_type(plot_type_var.get()))
        
        # View mode selection
        ttk.Label(control_frame, text="View Mode:").pack(anchor=tk.W, pady=(10, 0))
        view_mode_var = tk.StringVar(value=self.view_mode)
        view_modes = ['2D', '3D']
        view_mode_frame = ttk.Frame(control_frame)
        view_mode_frame.pack(fill=tk.X, pady=5)
        for mode in view_modes:
            ttk.Radiobutton(view_mode_frame, text=mode, variable=view_mode_var, value=mode, 
                           command=lambda: self._update_view_mode(view_mode_var.get())).pack(side=tk.LEFT, padx=5)
        
        # State selection (if eigenvectors are available)
        if self.eigenvectors is not None and len(self.eigenvectors) > 0:
            ttk.Label(control_frame, text="Quantum State:").pack(anchor=tk.W, pady=(10, 0))
            state_var = tk.IntVar(value=self.current_state)
            state_slider = ttk.Scale(control_frame, from_=0, to=len(self.eigenvectors)-1, 
                                    variable=state_var, orient=tk.HORIZONTAL)
            state_slider.pack(fill=tk.X, pady=5)
            state_slider.bind("<ButtonRelease-1>", lambda e: self._update_state(int(state_var.get())))
            
            # State label
            state_label = ttk.Label(control_frame, text=f"State: {self.current_state}")
            state_label.pack(anchor=tk.W)
            
            # Energy label (if eigenvalues are available)
            if self.eigenvalues is not None and len(self.eigenvalues) > 0:
                energy_label = ttk.Label(control_frame, 
                                        text=f"Energy: {self.eigenvalues[self.current_state]:.4f} eV")
                energy_label.pack(anchor=tk.W)
                
            # Animation controls
            ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            ttk.Label(control_frame, text="Animation:").pack(anchor=tk.W, pady=(10, 0))
            
            animation_frame = ttk.Frame(control_frame)
            animation_frame.pack(fill=tk.X, pady=5)
            
            start_button = ttk.Button(animation_frame, text="Start", 
                                     command=lambda: self._toggle_animation(start_button, speed_slider))
            start_button.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(control_frame, text="Speed:").pack(anchor=tk.W)
            speed_var = tk.DoubleVar(value=self.animation_speed)
            speed_slider = ttk.Scale(control_frame, from_=50, to=500, variable=speed_var, orient=tk.HORIZONTAL)
            speed_slider.pack(fill=tk.X, pady=5)
            speed_slider.bind("<ButtonRelease-1>", lambda e: self._update_animation_speed(speed_var.get()))
        
        # 3D view controls
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="3D View Controls:").pack(anchor=tk.W, pady=(10, 0))
        
        # Azimuth control
        ttk.Label(control_frame, text="Azimuth:").pack(anchor=tk.W)
        azimuth_var = tk.DoubleVar(value=self.azimuth)
        azimuth_slider = ttk.Scale(control_frame, from_=0, to=360, variable=azimuth_var, orient=tk.HORIZONTAL)
        azimuth_slider.pack(fill=tk.X, pady=5)
        azimuth_slider.bind("<ButtonRelease-1>", lambda e: self._update_3d_view(azimuth=azimuth_var.get()))
        
        # Elevation control
        ttk.Label(control_frame, text="Elevation:").pack(anchor=tk.W)
        elevation_var = tk.DoubleVar(value=self.elevation)
        elevation_slider = ttk.Scale(control_frame, from_=0, to=90, variable=elevation_var, orient=tk.HORIZONTAL)
        elevation_slider.pack(fill=tk.X, pady=5)
        elevation_slider.bind("<ButtonRelease-1>", lambda e: self._update_3d_view(elevation=elevation_var.get()))
        
        # Wireframe toggle
        wireframe_var = tk.BooleanVar(value=self.wireframe)
        ttk.Checkbutton(control_frame, text="Wireframe", variable=wireframe_var, 
                       command=lambda: self._update_3d_view(wireframe=wireframe_var.get())).pack(anchor=tk.W, pady=5)
        
        # Display options
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Display Options:").pack(anchor=tk.W, pady=(10, 0))
        
        # Use nm toggle
        use_nm_var = tk.BooleanVar(value=self.use_nm)
        ttk.Checkbutton(control_frame, text="Use nm units", variable=use_nm_var, 
                       command=lambda: self._update_display_options(use_nm=use_nm_var.get())).pack(anchor=tk.W, pady=5)
        
        # Center coordinates toggle
        center_coords_var = tk.BooleanVar(value=self.center_coords)
        ttk.Checkbutton(control_frame, text="Center coordinates", variable=center_coords_var, 
                       command=lambda: self._update_display_options(center_coords=center_coords_var.get())).pack(anchor=tk.W, pady=5)
        
        # Convert to eV toggle
        convert_to_eV_var = tk.BooleanVar(value=self.convert_to_eV)
        ttk.Checkbutton(control_frame, text="Convert to eV", variable=convert_to_eV_var, 
                       command=lambda: self._update_display_options(convert_to_eV=convert_to_eV_var.get())).pack(anchor=tk.W, pady=5)
        
        # Initial plot
        self._update_plot()
        
        # Start Tkinter event loop
        root.mainloop()
    
    def _show_matplotlib_interactive(self):
        """
        Show an interactive plot using matplotlib's interactive mode.
        
        This method is used as a fallback when Tkinter is not available.
        """
        plt.ion()  # Turn on interactive mode
        
        self.fig = plt.figure(figsize=(12, 8))
        
        # Create main axes for the plot
        self.ax = self.fig.add_subplot(111, projection='3d' if self.view_mode == '3D' else None)
        
        # Create control axes
        self.fig.subplots_adjust(bottom=0.3)
        
        # Add plot type radio buttons
        plot_type_ax = self.fig.add_axes([0.15, 0.05, 0.15, 0.15])
        plot_type_radio = RadioButtons(plot_type_ax, ('wavefunction', 'potential', 'electric_field'))
        plot_type_radio.on_clicked(self._update_plot_type)
        
        # Add view mode radio buttons
        view_mode_ax = self.fig.add_axes([0.35, 0.05, 0.15, 0.15])
        view_mode_radio = RadioButtons(view_mode_ax, ('2D', '3D'))
        view_mode_radio.on_clicked(self._update_view_mode)
        
        # Add state slider if eigenvectors are available
        if self.eigenvectors is not None and len(self.eigenvectors) > 0:
            state_ax = self.fig.add_axes([0.55, 0.1, 0.3, 0.03])
            state_slider = Slider(state_ax, 'State', 0, len(self.eigenvectors)-1, 
                                 valinit=self.current_state, valstep=1)
            state_slider.on_changed(self._update_state)
        
        # Initial plot
        self._update_plot()
        
        plt.show()
    
    def _update_plot_type(self, plot_type):
        """
        Update the plot type.
        
        Args:
            plot_type: The new plot type ('wavefunction', 'potential', or 'electric_field')
        """
        self.plot_type = plot_type
        self._update_plot()
    
    def _update_view_mode(self, view_mode):
        """
        Update the view mode.
        
        Args:
            view_mode: The new view mode ('2D' or '3D')
        """
        self.view_mode = view_mode
        self._update_plot()
    
    def _update_state(self, state):
        """
        Update the current quantum state.
        
        Args:
            state: The new state index
        """
        self.current_state = int(state)
        self._update_plot()
    
    def _update_3d_view(self, azimuth=None, elevation=None, wireframe=None):
        """
        Update the 3D view parameters.
        
        Args:
            azimuth: The new azimuth angle (degrees)
            elevation: The new elevation angle (degrees)
            wireframe: Whether to use wireframe mode
        """
        if azimuth is not None:
            self.azimuth = azimuth
        if elevation is not None:
            self.elevation = elevation
        if wireframe is not None:
            self.wireframe = wireframe
        
        if self.view_mode == '3D':
            self._update_plot()
    
    def _update_display_options(self, use_nm=None, center_coords=None, convert_to_eV=None):
        """
        Update the display options.
        
        Args:
            use_nm: Whether to use nanometers for coordinates
            center_coords: Whether to center the coordinates
            convert_to_eV: Whether to convert energies to eV
        """
        if use_nm is not None:
            self.use_nm = use_nm
        if center_coords is not None:
            self.center_coords = center_coords
        if convert_to_eV is not None:
            self.convert_to_eV = convert_to_eV
        
        self._update_plot()
    
    def _update_animation_speed(self, speed):
        """
        Update the animation speed.
        
        Args:
            speed: The new animation speed (ms between frames)
        """
        self.animation_speed = speed
        if self.animation_running and self.animation_obj is not None:
            self.animation_obj.event_source.interval = speed
    
    def _toggle_animation(self, button, speed_slider=None):
        """
        Toggle the animation on/off.
        
        Args:
            button: The animation button
            speed_slider: The animation speed slider
        """
        if self.animation_running:
            # Stop animation
            if self.animation_obj is not None:
                self.animation_obj.event_source.stop()
                self.animation_obj = None
            self.animation_running = False
            button.config(text="Start")
            if speed_slider is not None:
                speed_slider.config(state="normal")
        else:
            # Start animation
            self.animation_running = True
            button.config(text="Stop")
            if speed_slider is not None:
                speed_slider.config(state="disabled")
            
            # Create animation
            def update_frame(frame):
                self.current_state = frame % len(self.eigenvectors)
                self._update_plot()
                return []
            
            self.animation_obj = animation.FuncAnimation(
                self.fig, update_frame, frames=len(self.eigenvectors), 
                interval=self.animation_speed, blit=True)
    
    def _update_plot(self):
        """
        Update the plot based on the current visualization parameters.
        """
        # Clear the figure
        self.fig.clear()
        
        # Create new axes with the appropriate projection
        if self.view_mode == '3D':
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        
        # Plot based on the selected plot type
        if self.plot_type == 'wavefunction' and self.eigenvectors is not None and len(self.eigenvectors) > 0:
            self._plot_wavefunction()
        elif self.plot_type == 'potential' and self.potential is not None:
            self._plot_potential()
        elif self.plot_type == 'electric_field' and self.poisson_solver is not None:
            self._plot_electric_field()
        else:
            self.ax.text(0.5, 0.5, f"No data available for {self.plot_type} plot",
                        ha='center', va='center', transform=self.ax.transAxes)
        
        # Update the figure
        self.fig.tight_layout()
        if hasattr(self, 'canvas'):
            self.canvas.draw()
    
    def _plot_wavefunction(self):
        """
        Plot the wavefunction.
        """
        if self.current_state >= len(self.eigenvectors):
            self.current_state = 0
        
        eigenvector = self.eigenvectors[self.current_state]
        
        # Set title with energy if available
        if self.eigenvalues is not None and len(self.eigenvalues) > self.current_state:
            energy = self.eigenvalues[self.current_state]
            energy_str = f"{energy:.4f} eV" if self.convert_to_eV else f"{energy:.4e} J"
            title = f"Wavefunction (State {self.current_state}, Energy = {energy_str})"
        else:
            title = f"Wavefunction (State {self.current_state})"
        
        if self.view_mode == '2D':
            # Use the existing 2D plotting function
            plot_wavefunction(self.ax, self.mesh, eigenvector, 
                             use_nm=self.use_nm, center_coords=self.center_coords, 
                             title=title)
        else:
            # 3D plot
            self._plot_wavefunction_3d(eigenvector, title)
    
    def _plot_wavefunction_3d(self, eigenvector, title):
        """
        Plot the wavefunction in 3D.
        
        Args:
            eigenvector: The eigenvector to plot
            title: The plot title
        """
        # Get mesh data
        nodes = np.array(self.mesh.get_nodes())
        elements = np.array(self.mesh.get_elements())
        
        # Center coordinates if requested
        if self.center_coords:
            x_center = np.mean(nodes[:, 0])
            y_center = np.mean(nodes[:, 1])
            nodes_plot = nodes.copy()
            nodes_plot[:, 0] -= x_center
            nodes_plot[:, 1] -= y_center
        else:
            nodes_plot = nodes
        
        # Convert to nm if requested
        scale = 1e9 if self.use_nm else 1.0
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
        from matplotlib.tri import Triangulation
        triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)
        
        # Plot
        if self.wireframe:
            # Wireframe plot
            self.ax.plot_trisurf(triang, psi_norm, cmap='viridis', 
                               edgecolor='black', linewidth=0.2, alpha=0.8)
        else:
            # Surface plot
            surf = self.ax.plot_trisurf(triang, psi_norm, cmap='viridis', 
                                      edgecolor='none', alpha=0.8)
            self.fig.colorbar(surf, ax=self.ax, label='Normalized Probability Density')
        
        # Set labels
        self.ax.set_xlabel('x (nm)' if self.use_nm else 'x (m)')
        self.ax.set_ylabel('y (nm)' if self.use_nm else 'y (m)')
        self.ax.set_zlabel('Probability Density')
        
        # Set title
        self.ax.set_title(title)
        
        # Set view angle
        self.ax.view_init(self.elevation, self.azimuth)
    
    def _plot_potential(self):
        """
        Plot the potential.
        """
        # Set title
        title = "Potential"
        
        if self.view_mode == '2D':
            # Use the existing 2D plotting function
            plot_potential(self.ax, self.mesh, self.potential, 
                          use_nm=self.use_nm, center_coords=self.center_coords, 
                          title=title, convert_to_eV=self.convert_to_eV)
        else:
            # 3D plot
            self._plot_potential_3d(title)
    
    def _plot_potential_3d(self, title):
        """
        Plot the potential in 3D.
        
        Args:
            title: The plot title
        """
        # Get mesh data
        nodes = np.array(self.mesh.get_nodes())
        elements = np.array(self.mesh.get_elements())
        
        # Center coordinates if requested
        if self.center_coords:
            x_center = np.mean(nodes[:, 0])
            y_center = np.mean(nodes[:, 1])
            nodes_plot = nodes.copy()
            nodes_plot[:, 0] -= x_center
            nodes_plot[:, 1] -= y_center
        else:
            nodes_plot = nodes
        
        # Convert to nm if requested
        scale = 1e9 if self.use_nm else 1.0
        nodes_plot = nodes_plot * scale
        
        # Convert potential to eV if requested
        if self.convert_to_eV:
            potential_plot = self.potential / 1.602e-19  # Convert J to eV
            potential_label = 'Potential (eV)'
        else:
            potential_plot = self.potential
            potential_label = 'Potential (J)'
        
        # Create a triangulation for 3D plotting
        from matplotlib.tri import Triangulation
        triang = Triangulation(nodes_plot[:, 0], nodes_plot[:, 1], elements)
        
        # Plot
        if self.wireframe:
            # Wireframe plot
            self.ax.plot_trisurf(triang, potential_plot, cmap='viridis', 
                               edgecolor='black', linewidth=0.2, alpha=0.8)
        else:
            # Surface plot
            surf = self.ax.plot_trisurf(triang, potential_plot, cmap='viridis', 
                                      edgecolor='none', alpha=0.8)
            self.fig.colorbar(surf, ax=self.ax, label=potential_label)
        
        # Set labels
        self.ax.set_xlabel('x (nm)' if self.use_nm else 'x (m)')
        self.ax.set_ylabel('y (nm)' if self.use_nm else 'y (m)')
        self.ax.set_zlabel(potential_label)
        
        # Set title
        self.ax.set_title(title)
        
        # Set view angle
        self.ax.view_init(self.elevation, self.azimuth)
    
    def _plot_electric_field(self):
        """
        Plot the electric field.
        """
        # Set title
        title = "Electric Field"
        
        if self.view_mode == '2D':
            # Use the existing 2D plotting function
            plot_electric_field(self.ax, self.mesh, self.poisson_solver, 
                               use_nm=self.use_nm, center_coords=self.center_coords, 
                               title=title)
        else:
            # 3D plot - not implemented yet
            self.ax.text(0.5, 0.5, "3D electric field visualization not implemented",
                        ha='center', va='center', transform=self.ax.transAxes)


def show_interactive_visualization(mesh, eigenvectors=None, eigenvalues=None, potential=None, poisson_solver=None):
    """
    Show an interactive visualization of quantum dot simulation results.
    
    This function creates an interactive visualization of quantum dot simulation results,
    including wavefunctions, potentials, and electric fields.
    
    Args:
        mesh: Mesh object
        eigenvectors: List of eigenvectors (wavefunctions)
        eigenvalues: List of eigenvalues (energies)
        potential: Potential values
        poisson_solver: Poisson solver object for electric field visualization
    """
    visualizer = InteractiveVisualizer(mesh, eigenvectors, eigenvalues, potential, poisson_solver)
    visualizer.show_interactive_plot()
