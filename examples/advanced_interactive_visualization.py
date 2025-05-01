#!/usr/bin/env python3
"""
Advanced Interactive Visualization Demo

This script demonstrates the advanced interactive visualization capabilities of QDSim,
including 3D visualization, animation, and custom controls.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
import tkinter as tk
from tkinter import ttk

# Add the parent directory to the path so we can import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qdsim

def create_custom_visualization(results):
    """
    Create a custom interactive visualization using the InteractiveVisualizer class.
    
    Args:
        results: Simulation results dictionary
    """
    # Extract results
    mesh = results["mesh"]
    eigenvectors = results["eigenvectors"]
    eigenvalues = results["eigenvalues"]
    potential = results["potential"]
    poisson_solver = results["poisson_solver"]
    
    # Create an interactive visualizer
    visualizer = qdsim.InteractiveVisualizer(
        mesh=mesh,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        potential=potential,
        poisson_solver=poisson_solver
    )
    
    # Customize visualization parameters
    visualizer.use_nm = True
    visualizer.center_coords = True
    visualizer.convert_to_eV = True
    visualizer.view_mode = '3D'  # Start with 3D view
    visualizer.plot_type = 'wavefunction'
    visualizer.azimuth = 45
    visualizer.elevation = 30
    
    # Show the interactive plot
    visualizer.show_interactive_plot()

def create_side_by_side_visualization(results):
    """
    Create a side-by-side visualization with wavefunction and potential.
    
    Args:
        results: Simulation results dictionary
    """
    # Extract results
    mesh = results["mesh"]
    eigenvectors = results["eigenvectors"]
    eigenvalues = results["eigenvalues"]
    potential = results["potential"]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot wavefunction
    qdsim.plot_wavefunction(
        ax=ax1,
        mesh=mesh,
        eigenvector=eigenvectors[0],
        use_nm=True,
        center_coords=True,
        title=f"Ground State (E = {eigenvalues[0]:.4f} eV)"
    )
    
    # Plot potential
    qdsim.plot_potential(
        ax=ax2,
        mesh=mesh,
        potential_values=potential,
        use_nm=True,
        center_coords=True,
        title="Potential",
        convert_to_eV=True
    )
    
    # Add state selection slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'State', 0, len(eigenvectors)-1, valinit=0, valstep=1)
    
    def update(val):
        state = int(slider.val)
        ax1.clear()
        qdsim.plot_wavefunction(
            ax=ax1,
            mesh=mesh,
            eigenvector=eigenvectors[state],
            use_nm=True,
            center_coords=True,
            title=f"State {state} (E = {eigenvalues[state]:.4f} eV)"
        )
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add tight layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def create_tkinter_visualization(results):
    """
    Create a custom Tkinter-based visualization.
    
    Args:
        results: Simulation results dictionary
    """
    # Extract results
    mesh = results["mesh"]
    eigenvectors = results["eigenvectors"]
    eigenvalues = results["eigenvalues"]
    potential = results["potential"]
    
    # Create Tkinter window
    root = tk.Tk()
    root.title("QDSim Custom Visualization")
    root.geometry("1000x600")
    
    # Create main frame
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create left frame for controls
    left_frame = ttk.Frame(main_frame, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
    
    # Create right frame for plots
    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create matplotlib figure
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    
    fig = Figure(figsize=(8, 6), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add toolbar
    toolbar = NavigationToolbar2Tk(canvas, right_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Create axes
    ax = fig.add_subplot(111)
    
    # Add controls
    ttk.Label(left_frame, text="Visualization Controls", font=("Arial", 12, "bold")).pack(pady=10)
    
    # State selection
    ttk.Label(left_frame, text="Quantum State:").pack(anchor=tk.W, pady=(10, 0))
    state_var = tk.IntVar(value=0)
    state_slider = ttk.Scale(left_frame, from_=0, to=len(eigenvectors)-1, variable=state_var, orient=tk.HORIZONTAL)
    state_slider.pack(fill=tk.X, pady=5)
    
    # State label
    state_label = ttk.Label(left_frame, text=f"State: 0")
    state_label.pack(anchor=tk.W)
    
    # Energy label
    energy_label = ttk.Label(left_frame, text=f"Energy: {eigenvalues[0]:.4f} eV")
    energy_label.pack(anchor=tk.W)
    
    # Plot type selection
    ttk.Label(left_frame, text="Plot Type:").pack(anchor=tk.W, pady=(10, 0))
    plot_type_var = tk.StringVar(value="wavefunction")
    plot_types = [("Wavefunction", "wavefunction"), ("Potential", "potential")]
    for text, value in plot_types:
        ttk.Radiobutton(left_frame, text=text, variable=plot_type_var, value=value).pack(anchor=tk.W)
    
    # Update function
    def update_plot():
        ax.clear()
        
        state = state_var.get()
        plot_type = plot_type_var.get()
        
        if plot_type == "wavefunction":
            qdsim.plot_wavefunction(
                ax=ax,
                mesh=mesh,
                eigenvector=eigenvectors[state],
                use_nm=True,
                center_coords=True,
                title=f"State {state} (E = {eigenvalues[state]:.4f} eV)"
            )
        else:
            qdsim.plot_potential(
                ax=ax,
                mesh=mesh,
                potential_values=potential,
                use_nm=True,
                center_coords=True,
                title="Potential",
                convert_to_eV=True
            )
        
        # Update labels
        state_label.config(text=f"State: {state}")
        energy_label.config(text=f"Energy: {eigenvalues[state]:.4f} eV")
        
        # Update canvas
        fig.tight_layout()
        canvas.draw()
    
    # Bind update function to controls
    state_slider.bind("<ButtonRelease-1>", lambda e: update_plot())
    for widget in left_frame.winfo_children():
        if isinstance(widget, ttk.Radiobutton):
            widget.config(command=update_plot)
    
    # Initial plot
    update_plot()
    
    # Start Tkinter event loop
    root.mainloop()

def main():
    """
    Main function to demonstrate advanced interactive visualization.
    """
    print("QDSim Advanced Interactive Visualization Demo")
    print("============================================")
    
    # Create a configuration
    config = qdsim.Config()
    config.mesh.Lx = 100e-9  # 100 nm
    config.mesh.Ly = 100e-9  # 100 nm
    config.mesh.nx = 50
    config.mesh.ny = 50
    config.mesh.element_order = 1
    
    config.quantum_dot.radius = 20e-9  # 20 nm
    config.quantum_dot.depth = 0.3  # 0.3 eV
    config.quantum_dot.type = "gaussian"
    
    config.simulation.num_eigenstates = 10
    
    # Create a simulator
    simulator = qdsim.Simulator(config)
    
    # Run the simulation
    results = simulator.run()
    
    # Show menu
    print("\nSelect visualization type:")
    print("1. Default Interactive Visualization")
    print("2. Custom Interactive Visualization")
    print("3. Side-by-Side Visualization")
    print("4. Tkinter-based Visualization")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        # Show default interactive visualization
        qdsim.show_interactive_visualization(
            mesh=results["mesh"],
            eigenvectors=results["eigenvectors"],
            eigenvalues=results["eigenvalues"],
            potential=results["potential"],
            poisson_solver=results["poisson_solver"]
        )
    elif choice == "2":
        # Show custom interactive visualization
        create_custom_visualization(results)
    elif choice == "3":
        # Show side-by-side visualization
        create_side_by_side_visualization(results)
    elif choice == "4":
        # Show Tkinter-based visualization
        create_tkinter_visualization(results)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
