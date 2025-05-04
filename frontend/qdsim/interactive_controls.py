"""
Interactive controls module for QDSim.

This module provides interactive controls for visualizing quantum dot simulations,
including camera controls, parameter sliders, and result analysis tools.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .enhanced_visualization import (
    plot_enhanced_wavefunction_3d,
    plot_enhanced_potential_3d,
    plot_combined_visualization,
    create_energy_level_diagram,
    calculate_transition_probabilities,
    plot_transition_matrix
)

class InteractiveControls:
    """
    Interactive controls for quantum dot simulations.
    
    This class provides interactive controls for visualizing quantum dot simulations,
    including camera controls, parameter sliders, and result analysis tools.
    """
    
    def __init__(self, mesh, eigenvectors=None, eigenvalues=None, potential=None, poisson_solver=None):
        """
        Initialize the interactive controls.
        
        Args:
            mesh: Mesh object
            eigenvectors: List of eigenvectors (wavefunctions)
            eigenvalues: List of eigenvalues (energies)
            potential: Potential values
            poisson_solver: Poisson solver object for electric field visualization
        """
        self.mesh = mesh
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.potential = potential
        self.poisson_solver = poisson_solver
        
        # Default parameters
        self.current_state = 0
        self.view_mode = '3D'
        self.plot_type = 'wavefunction'
        self.azimuth = 30
        self.elevation = 30
        self.surface_type = 'surface'
        self.use_nm = True
        self.center_coords = True
        self.convert_to_eV = True
        self.show_cross_sections = True
        self.show_energy_diagram = False
        self.show_transition_matrix = False
        
        # Figure and axes
        self.fig = None
        self.ax = None
        self.canvas = None
        
        # For tkinter
        self.root = None
        self.plot_frame = None
        self.control_frame = None
        
    def create_tkinter_window(self):
        """
        Create a tkinter window for interactive visualization.
        
        Returns:
            root: Tkinter root window
            plot_frame: Frame for the plot
            control_frame: Frame for the controls
        """
        # Create root window
        root = tk.Tk()
        root.title("QDSim Interactive Visualization")
        root.geometry("1200x800")
        
        # Create frames
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a PanedWindow to allow resizing
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create plot frame (left side)
        plot_frame = ttk.Frame(paned_window)
        paned_window.add(plot_frame, weight=3)
        
        # Create control frame (right side)
        control_frame = ttk.Frame(paned_window)
        paned_window.add(control_frame, weight=1)
        
        # Store references
        self.root = root
        self.plot_frame = plot_frame
        self.control_frame = control_frame
        
        return root, plot_frame, control_frame
    
    def create_matplotlib_figure(self):
        """
        Create a matplotlib figure for visualization.
        
        Returns:
            fig: Matplotlib figure
        """
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        # Store reference
        self.fig = fig
        
        return fig
    
    def add_tkinter_controls(self):
        """
        Add controls to the tkinter window.
        """
        # Add title
        ttk.Label(self.control_frame, text="Visualization Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Create a notebook for tabbed controls
        notebook = ttk.Notebook(self.control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        view_tab = ttk.Frame(notebook)
        analysis_tab = ttk.Frame(notebook)
        
        notebook.add(view_tab, text="View")
        notebook.add(analysis_tab, text="Analysis")
        
        # === View Tab ===
        
        # Plot type selection
        ttk.Label(view_tab, text="Plot Type:").pack(anchor=tk.W, pady=(10, 0))
        plot_type_var = tk.StringVar(value=self.plot_type)
        plot_types = ['wavefunction', 'potential', 'combined']
        plot_type_menu = ttk.Combobox(view_tab, textvariable=plot_type_var, values=plot_types, state="readonly")
        plot_type_menu.pack(fill=tk.X, pady=5)
        plot_type_menu.bind("<<ComboboxSelected>>", lambda e: self._update_plot_type(plot_type_var.get()))
        
        # View mode selection
        ttk.Label(view_tab, text="View Mode:").pack(anchor=tk.W, pady=(10, 0))
        view_mode_var = tk.StringVar(value=self.view_mode)
        view_modes = ['2D', '3D']
        view_mode_menu = ttk.Combobox(view_tab, textvariable=view_mode_var, values=view_modes, state="readonly")
        view_mode_menu.pack(fill=tk.X, pady=5)
        view_mode_menu.bind("<<ComboboxSelected>>", lambda e: self._update_view_mode(view_mode_var.get()))
        
        # Surface type selection (for 3D)
        ttk.Label(view_tab, text="Surface Type:").pack(anchor=tk.W, pady=(10, 0))
        surface_type_var = tk.StringVar(value=self.surface_type)
        surface_types = ['surface', 'wireframe', 'contour']
        surface_type_menu = ttk.Combobox(view_tab, textvariable=surface_type_var, values=surface_types, state="readonly")
        surface_type_menu.pack(fill=tk.X, pady=5)
        surface_type_menu.bind("<<ComboboxSelected>>", lambda e: self._update_surface_type(surface_type_var.get()))
        
        # Camera controls (for 3D)
        ttk.Label(view_tab, text="Camera Controls:").pack(anchor=tk.W, pady=(10, 0))
        
        # Azimuth slider
        ttk.Label(view_tab, text="Azimuth:").pack(anchor=tk.W)
        azimuth_var = tk.DoubleVar(value=self.azimuth)
        azimuth_slider = ttk.Scale(view_tab, from_=0, to=360, variable=azimuth_var, orient=tk.HORIZONTAL)
        azimuth_slider.pack(fill=tk.X, pady=5)
        azimuth_slider.bind("<ButtonRelease-1>", lambda e: self._update_camera(azimuth=azimuth_var.get()))
        
        # Elevation slider
        ttk.Label(view_tab, text="Elevation:").pack(anchor=tk.W)
        elevation_var = tk.DoubleVar(value=self.elevation)
        elevation_slider = ttk.Scale(view_tab, from_=0, to=90, variable=elevation_var, orient=tk.HORIZONTAL)
        elevation_slider.pack(fill=tk.X, pady=5)
        elevation_slider.bind("<ButtonRelease-1>", lambda e: self._update_camera(elevation=elevation_var.get()))
        
        # State selection (if eigenvectors are available)
        if self.eigenvectors is not None and len(self.eigenvectors) > 0:
            ttk.Label(view_tab, text="State:").pack(anchor=tk.W, pady=(10, 0))
            state_var = tk.IntVar(value=self.current_state)
            state_slider = ttk.Scale(view_tab, from_=0, to=len(self.eigenvectors[0])-1, variable=state_var, orient=tk.HORIZONTAL)
            state_slider.pack(fill=tk.X, pady=5)
            state_slider.bind("<ButtonRelease-1>", lambda e: self._update_state(state_var.get()))
            
            # Add state label to show current state and energy
            state_label_var = tk.StringVar(value=self._get_state_label(self.current_state))
            state_label = ttk.Label(view_tab, textvariable=state_label_var)
            state_label.pack(anchor=tk.W, pady=5)
            
            # Update state label when state changes
            state_slider.bind("<ButtonRelease-1>", lambda e: [self._update_state(state_var.get()), 
                                                            state_label_var.set(self._get_state_label(state_var.get()))])
        
        # Display options
        ttk.Label(view_tab, text="Display Options:").pack(anchor=tk.W, pady=(10, 0))
        
        # Use nm checkbox
        use_nm_var = tk.BooleanVar(value=self.use_nm)
        use_nm_check = ttk.Checkbutton(view_tab, text="Use nanometers", variable=use_nm_var)
        use_nm_check.pack(anchor=tk.W, pady=2)
        use_nm_check.config(command=lambda: self._update_display_option('use_nm', use_nm_var.get()))
        
        # Center coordinates checkbox
        center_coords_var = tk.BooleanVar(value=self.center_coords)
        center_coords_check = ttk.Checkbutton(view_tab, text="Center coordinates", variable=center_coords_var)
        center_coords_check.pack(anchor=tk.W, pady=2)
        center_coords_check.config(command=lambda: self._update_display_option('center_coords', center_coords_var.get()))
        
        # Convert to eV checkbox
        convert_to_eV_var = tk.BooleanVar(value=self.convert_to_eV)
        convert_to_eV_check = ttk.Checkbutton(view_tab, text="Convert to eV", variable=convert_to_eV_var)
        convert_to_eV_check.pack(anchor=tk.W, pady=2)
        convert_to_eV_check.config(command=lambda: self._update_display_option('convert_to_eV', convert_to_eV_var.get()))
        
        # Show cross sections checkbox
        cross_sections_var = tk.BooleanVar(value=self.show_cross_sections)
        cross_sections_check = ttk.Checkbutton(view_tab, text="Show cross sections", variable=cross_sections_var)
        cross_sections_check.pack(anchor=tk.W, pady=2)
        cross_sections_check.config(command=lambda: self._update_display_option('show_cross_sections', cross_sections_var.get()))
        
        # === Analysis Tab ===
        
        # Energy diagram
        ttk.Label(analysis_tab, text="Energy Analysis:").pack(anchor=tk.W, pady=(10, 0))
        
        # Show energy diagram checkbox
        energy_diagram_var = tk.BooleanVar(value=self.show_energy_diagram)
        energy_diagram_check = ttk.Checkbutton(analysis_tab, text="Show energy level diagram", variable=energy_diagram_var)
        energy_diagram_check.pack(anchor=tk.W, pady=2)
        energy_diagram_check.config(command=lambda: self._update_analysis_option('show_energy_diagram', energy_diagram_var.get()))
        
        # Transition matrix
        ttk.Label(analysis_tab, text="Transition Analysis:").pack(anchor=tk.W, pady=(10, 0))
        
        # Show transition matrix checkbox
        transition_matrix_var = tk.BooleanVar(value=self.show_transition_matrix)
        transition_matrix_check = ttk.Checkbutton(analysis_tab, text="Show transition probability matrix", variable=transition_matrix_var)
        transition_matrix_check.pack(anchor=tk.W, pady=2)
        transition_matrix_check.config(command=lambda: self._update_analysis_option('show_transition_matrix', transition_matrix_var.get()))
        
        # Add buttons for common views
        ttk.Label(view_tab, text="Preset Views:").pack(anchor=tk.W, pady=(10, 0))
        
        # Create a frame for the buttons
        button_frame = ttk.Frame(view_tab)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Add buttons
        ttk.Button(button_frame, text="Top", command=lambda: self._set_preset_view(90, 0)).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="Front", command=lambda: self._set_preset_view(0, 0)).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(button_frame, text="Side", command=lambda: self._set_preset_view(0, 90)).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(button_frame, text="30°", command=lambda: self._set_preset_view(30, 30)).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="45°", command=lambda: self._set_preset_view(45, 45)).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(button_frame, text="60°", command=lambda: self._set_preset_view(60, 30)).grid(row=1, column=2, padx=2, pady=2)
        
        # Add a button to reset the view
        ttk.Button(view_tab, text="Reset View", command=self._reset_view).pack(fill=tk.X, pady=10)
        
        # Add a button to save the current figure
        ttk.Button(view_tab, text="Save Figure", command=self._save_figure).pack(fill=tk.X, pady=10)
    
    def _get_state_label(self, state_idx):
        """
        Get a label for the current state with energy information.
        
        Args:
            state_idx: Index of the state
            
        Returns:
            label: String label for the state
        """
        if self.eigenvalues is not None and state_idx < len(self.eigenvalues):
            if self.convert_to_eV:
                energy = self.eigenvalues[state_idx] / 1.602e-19
                return f"State {state_idx}: E = {energy:.6f} eV"
            else:
                return f"State {state_idx}: E = {self.eigenvalues[state_idx]:.6e} J"
        else:
            return f"State {state_idx}"
    
    def _update_plot_type(self, plot_type):
        """
        Update the plot type.
        
        Args:
            plot_type: New plot type
        """
        self.plot_type = plot_type
        self._update_plot()
    
    def _update_view_mode(self, view_mode):
        """
        Update the view mode.
        
        Args:
            view_mode: New view mode
        """
        self.view_mode = view_mode
        
        # Recreate the figure with the new view mode
        self.fig.clear()
        
        if self.view_mode == '3D':
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        
        self._update_plot()
    
    def _update_surface_type(self, surface_type):
        """
        Update the surface type.
        
        Args:
            surface_type: New surface type
        """
        self.surface_type = surface_type
        self._update_plot()
    
    def _update_camera(self, azimuth=None, elevation=None):
        """
        Update the camera position.
        
        Args:
            azimuth: New azimuth angle
            elevation: New elevation angle
        """
        if azimuth is not None:
            self.azimuth = azimuth
        if elevation is not None:
            self.elevation = elevation
        
        if self.view_mode == '3D' and hasattr(self.ax, 'view_init'):
            self.ax.view_init(self.elevation, self.azimuth)
            self.canvas.draw_idle()
    
    def _update_state(self, state_idx):
        """
        Update the current state.
        
        Args:
            state_idx: New state index
        """
        self.current_state = int(state_idx)
        self._update_plot()
    
    def _update_display_option(self, option, value):
        """
        Update a display option.
        
        Args:
            option: Option name
            value: New value
        """
        setattr(self, option, value)
        self._update_plot()
    
    def _update_analysis_option(self, option, value):
        """
        Update an analysis option.
        
        Args:
            option: Option name
            value: New value
        """
        setattr(self, option, value)
        self._update_plot()
    
    def _set_preset_view(self, elevation, azimuth):
        """
        Set a preset camera view.
        
        Args:
            elevation: Elevation angle
            azimuth: Azimuth angle
        """
        self.elevation = elevation
        self.azimuth = azimuth
        
        if self.view_mode == '3D' and hasattr(self.ax, 'view_init'):
            self.ax.view_init(self.elevation, self.azimuth)
            self.canvas.draw_idle()
    
    def _reset_view(self):
        """
        Reset the view to default settings.
        """
        self.azimuth = 30
        self.elevation = 30
        self.surface_type = 'surface'
        self.use_nm = True
        self.center_coords = True
        self.convert_to_eV = True
        self.show_cross_sections = True
        
        self._update_plot()
    
    def _save_figure(self):
        """
        Save the current figure to a file.
        """
        from tkinter import filedialog
        
        # Ask for filename
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {filename}")
    
    def _update_plot(self):
        """
        Update the plot based on current settings.
        """
        # Clear the figure
        self.fig.clear()
        
        # Create the appropriate plot based on settings
        if self.plot_type == 'wavefunction' and self.eigenvectors is not None:
            if self.view_mode == '3D':
                self.ax = self.fig.add_subplot(111, projection='3d')
                plot_enhanced_wavefunction_3d(
                    ax=self.ax,
                    mesh=self.mesh,
                    eigenvector=self.eigenvectors[:, self.current_state],
                    use_nm=self.use_nm,
                    center_coords=self.center_coords,
                    title=self._get_state_label(self.current_state),
                    azimuth=self.azimuth,
                    elevation=self.elevation,
                    plot_type=self.surface_type
                )
            else:
                from .visualization import plot_wavefunction
                self.ax = self.fig.add_subplot(111)
                plot_wavefunction(
                    ax=self.ax,
                    mesh=self.mesh,
                    eigenvector=self.eigenvectors[:, self.current_state],
                    use_nm=self.use_nm,
                    center_coords=self.center_coords,
                    title=self._get_state_label(self.current_state)
                )
        
        elif self.plot_type == 'potential' and self.potential is not None:
            if self.view_mode == '3D':
                self.ax = self.fig.add_subplot(111, projection='3d')
                plot_enhanced_potential_3d(
                    ax=self.ax,
                    mesh=self.mesh,
                    potential_values=self.potential,
                    use_nm=self.use_nm,
                    center_coords=self.center_coords,
                    title="Potential",
                    convert_to_eV=self.convert_to_eV,
                    azimuth=self.azimuth,
                    elevation=self.elevation,
                    plot_type=self.surface_type
                )
            else:
                from .visualization import plot_potential
                self.ax = self.fig.add_subplot(111)
                plot_potential(
                    ax=self.ax,
                    mesh=self.mesh,
                    potential_values=self.potential,
                    use_nm=self.use_nm,
                    center_coords=self.center_coords,
                    title="Potential",
                    convert_to_eV=self.convert_to_eV
                )
        
        elif self.plot_type == 'combined' and self.eigenvectors is not None and self.potential is not None:
            if self.view_mode == '3D':
                # Use the combined visualization function
                plot_combined_visualization(
                    fig=self.fig,
                    mesh=self.mesh,
                    eigenvector=self.eigenvectors[:, self.current_state],
                    potential_values=self.potential,
                    eigenvalue=self.eigenvalues[self.current_state] if self.eigenvalues is not None else None,
                    use_nm=self.use_nm,
                    center_coords=self.center_coords,
                    convert_to_eV=self.convert_to_eV,
                    azimuth=self.azimuth,
                    elevation=self.elevation,
                    plot_type=self.surface_type
                )
                # No need to set self.ax as the function creates multiple axes
            else:
                # Create a 2x1 grid for wavefunction and potential
                gs = gridspec.GridSpec(2, 1)
                
                # Plot wavefunction
                ax1 = self.fig.add_subplot(gs[0])
                from .visualization import plot_wavefunction
                plot_wavefunction(
                    ax=ax1,
                    mesh=self.mesh,
                    eigenvector=self.eigenvectors[:, self.current_state],
                    use_nm=self.use_nm,
                    center_coords=self.center_coords,
                    title=self._get_state_label(self.current_state)
                )
                
                # Plot potential
                ax2 = self.fig.add_subplot(gs[1])
                from .visualization import plot_potential
                plot_potential(
                    ax=ax2,
                    mesh=self.mesh,
                    potential_values=self.potential,
                    use_nm=self.use_nm,
                    center_coords=self.center_coords,
                    title="Potential",
                    convert_to_eV=self.convert_to_eV
                )
                
                self.ax = ax1  # Set the primary axis
        
        # Add analysis plots if requested
        if self.show_energy_diagram and self.eigenvalues is not None:
            # Create a new figure for the energy diagram
            energy_fig = plt.figure(figsize=(6, 8))
            energy_ax = energy_fig.add_subplot(111)
            
            # Get potential minimum if available
            potential_min = np.min(self.potential) if self.potential is not None else None
            
            # Create energy level diagram
            create_energy_level_diagram(
                ax=energy_ax,
                eigenvalues=self.eigenvalues,
                potential_min=potential_min,
                convert_to_eV=self.convert_to_eV,
                show_labels=True,
                show_transitions=True
            )
            
            energy_fig.tight_layout()
            plt.show()
        
        if self.show_transition_matrix and self.eigenvectors is not None:
            # Calculate transition probabilities
            prob_matrix = calculate_transition_probabilities(self.eigenvectors)
            
            # Create a new figure for the transition matrix
            trans_fig = plt.figure(figsize=(8, 6))
            trans_ax = trans_fig.add_subplot(111)
            
            # Plot transition matrix
            plot_transition_matrix(
                ax=trans_ax,
                prob_matrix=prob_matrix,
                eigenvalues=self.eigenvalues,
                convert_to_eV=self.convert_to_eV,
                show_values=True
            )
            
            trans_fig.tight_layout()
            plt.show()
        
        # Update the canvas
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()
    
    def show_interactive_visualization(self):
        """
        Show an interactive visualization with controls.
        """
        try:
            # Create tkinter window
            self.create_tkinter_window()
            
            # Create matplotlib figure
            self.fig = self.create_matplotlib_figure()
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
            toolbar.update()
            
            # Add controls
            self.add_tkinter_controls()
            
            # Initial plot
            self._update_plot()
            
            # Start the main loop
            self.root.mainloop()
        
        except Exception as e:
            print(f"Error in interactive visualization: {e}")
            print("Falling back to matplotlib interactive mode")
            self.show_matplotlib_interactive()
    
    def show_matplotlib_interactive(self):
        """
        Show an interactive visualization using matplotlib's interactive mode.
        This is a fallback when tkinter is not available.
        """
        plt.ion()  # Turn on interactive mode
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(10, 8))
        
        if self.view_mode == '3D':
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        
        # Initial plot
        self._update_plot()
        
        # Show the plot
        plt.show()


def show_enhanced_visualization(mesh, eigenvectors=None, eigenvalues=None, potential=None, poisson_solver=None):
    """
    Show an enhanced interactive visualization of quantum dot simulation results.
    
    This function creates an interactive visualization of quantum dot simulation results,
    including wavefunctions, potentials, and analysis tools.
    
    Args:
        mesh: Mesh object
        eigenvectors: List of eigenvectors (wavefunctions)
        eigenvalues: List of eigenvalues (energies)
        potential: Potential values
        poisson_solver: Poisson solver object for electric field visualization
    """
    controls = InteractiveControls(mesh, eigenvectors, eigenvalues, potential, poisson_solver)
    controls.show_interactive_visualization()
