#!/usr/bin/env python3
"""
Interactive Visualization Demo

This script demonstrates the interactive visualization capabilities of QDSim,
including the ability to zoom, pan, and rotate visualizations using mouse controls.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qdsim

def main():
    """
    Main function to demonstrate interactive visualization.
    """
    print("QDSim Interactive Visualization Demo")
    print("====================================")
    
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
    
    config.simulation.num_eigenstates = 5
    
    # Create a simulator
    simulator = qdsim.Simulator(config)
    
    # Run the simulation
    results = simulator.run()
    
    # Extract results
    mesh = results["mesh"]
    eigenvectors = results["eigenvectors"]
    eigenvalues = results["eigenvalues"]
    potential = results["potential"]
    poisson_solver = results["poisson_solver"]
    
    # Show interactive visualization
    qdsim.show_interactive_visualization(
        mesh=mesh,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        potential=potential,
        poisson_solver=poisson_solver
    )

if __name__ == "__main__":
    main()
