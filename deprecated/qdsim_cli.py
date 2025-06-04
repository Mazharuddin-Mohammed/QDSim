#!/usr/bin/env python3
"""
Command-line interface for QDSim.

This script provides a command-line interface for running QDSim simulations.
It allows users to specify simulation parameters via command-line arguments
or configuration files.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend'))

# Import the QDSim modules
from qdsim.config import Config
from qdsim.simulator import Simulator
from qdsim.visualization import (
    plot_potential_3d, plot_wavefunction_3d, 
    create_simulation_dashboard, save_simulation_results
)
from qdsim.analysis import (
    calculate_transition_energies, calculate_transition_probabilities,
    calculate_energy_level_statistics, calculate_wavefunction_localization,
    find_bound_states
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='QDSim: Quantum Dot Simulator')
    
    # Input/output options
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--output-dir', '-o', type=str, default='results', help='Output directory for results')
    parser.add_argument('--prefix', '-p', type=str, default='qdsim', help='Prefix for output files')
    
    # Simulation parameters
    parser.add_argument('--lx', type=float, help='Domain length in x direction (nm)')
    parser.add_argument('--ly', type=float, help='Domain length in y direction (nm)')
    parser.add_argument('--nx', type=int, help='Number of elements in x direction')
    parser.add_argument('--ny', type=int, help='Number of elements in y direction')
    parser.add_argument('--element-order', type=int, choices=[1, 2, 3], help='Element order (1, 2, or 3)')
    parser.add_argument('--qd-material', type=str, help='Quantum dot material')
    parser.add_argument('--matrix-material', type=str, help='Matrix material')
    parser.add_argument('--qd-radius', type=float, help='Quantum dot radius (nm)')
    parser.add_argument('--potential-depth', type=float, help='Quantum dot potential depth (eV)')
    parser.add_argument('--potential-type', type=str, choices=['gaussian', 'square'], help='Potential type')
    parser.add_argument('--num-states', type=int, default=5, help='Number of eigenstates to compute')
    parser.add_argument('--use-mpi', action='store_true', help='Use MPI for parallel computation')
    
    # Diode parameters
    parser.add_argument('--diode-p-material', type=str, help='P-type material for diode')
    parser.add_argument('--diode-n-material', type=str, help='N-type material for diode')
    parser.add_argument('--na', type=float, help='Acceptor concentration (m^-3)')
    parser.add_argument('--nd', type=float, help='Donor concentration (m^-3)')
    parser.add_argument('--reverse-bias', type=float, help='Reverse bias voltage (V)')
    
    # Visualization options
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    parser.add_argument('--plot-format', type=str, default='png', choices=['png', 'pdf', 'svg'], help='Plot file format')
    parser.add_argument('--plot-dpi', type=int, default=300, help='Plot resolution (DPI)')
    
    # Analysis options
    parser.add_argument('--analyze', action='store_true', help='Perform analysis of results')
    parser.add_argument('--save-analysis', action='store_true', help='Save analysis results to file')
    
    # Batch processing options
    parser.add_argument('--batch', type=str, help='Path to batch configuration file')
    
    return parser.parse_args()

def load_config_file(config_path):
    """Load configuration from a file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format based on extension
    ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if ext == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
    except Exception as e:
        raise ValueError(f"Error loading configuration file: {e}")
    
    return config_data

def create_config_from_args(args, config_data=None):
    """Create a Config object from command-line arguments and/or config file data."""
    config = Config()
    
    # Set values from config file if provided
    if config_data:
        for key, value in config_data.items():
            # Convert snake_case to camelCase
            key_parts = key.split('_')
            camel_key = key_parts[0] + ''.join(x.title() for x in key_parts[1:])
            
            # Set the attribute
            setattr(config, camel_key, value)
    
    # Override with command-line arguments if provided
    if args.lx:
        config.Lx = args.lx * 1e-9  # Convert from nm to m
    if args.ly:
        config.Ly = args.ly * 1e-9  # Convert from nm to m
    if args.nx:
        config.nx = args.nx
    if args.ny:
        config.ny = args.ny
    if args.element_order:
        config.element_order = args.element_order
    if args.qd_material:
        config.qd_material = args.qd_material
    if args.matrix_material:
        config.matrix_material = args.matrix_material
    if args.qd_radius:
        config.R = args.qd_radius * 1e-9  # Convert from nm to m
    if args.potential_depth:
        config.V_0 = args.potential_depth * config.e_charge  # Convert from eV to J
    if args.potential_type:
        config.potential_type = args.potential_type
    if args.diode_p_material:
        config.diode_p_material = args.diode_p_material
    if args.diode_n_material:
        config.diode_n_material = args.diode_n_material
    if args.na:
        config.N_A = args.na
    if args.nd:
        config.N_D = args.nd
    if args.reverse_bias:
        config.V_r = args.reverse_bias
    if args.use_mpi:
        config.use_mpi = args.use_mpi
    
    # Set default values for required parameters if not provided
    if not hasattr(config, 'Lx'):
        config.Lx = 200e-9  # 200 nm
    if not hasattr(config, 'Ly'):
        config.Ly = 100e-9  # 100 nm
    if not hasattr(config, 'nx'):
        config.nx = 101
    if not hasattr(config, 'ny'):
        config.ny = 51
    if not hasattr(config, 'element_order'):
        config.element_order = 1
    if not hasattr(config, 'qd_material'):
        config.qd_material = 'InAs'
    if not hasattr(config, 'matrix_material'):
        config.matrix_material = 'GaAs'
    if not hasattr(config, 'diode_p_material'):
        config.diode_p_material = 'GaAs'
    if not hasattr(config, 'diode_n_material'):
        config.diode_n_material = 'GaAs'
    if not hasattr(config, 'R'):
        config.R = 10e-9  # 10 nm
    if not hasattr(config, 'V_0'):
        config.V_0 = 0.5 * config.e_charge  # 0.5 eV
    if not hasattr(config, 'potential_type'):
        config.potential_type = 'gaussian'
    if not hasattr(config, 'N_A'):
        config.N_A = 1e24  # m^-3
    if not hasattr(config, 'N_D'):
        config.N_D = 1e24  # m^-3
    if not hasattr(config, 'V_r'):
        config.V_r = 0.0  # V
    if not hasattr(config, 'use_mpi'):
        config.use_mpi = False
    if not hasattr(config, 'e_charge'):
        config.e_charge = 1.602e-19  # C
    if not hasattr(config, 'm_e'):
        config.m_e = 9.109e-31  # kg
    if not hasattr(config, 'tolerance'):
        config.tolerance = 1e-6
    if not hasattr(config, 'max_iter'):
        config.max_iter = 100
    
    return config

def run_simulation(config, args):
    """Run a simulation with the given configuration."""
    # Create output directory if it doesn't exist
    if args.save_plots or args.save_analysis:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the simulator
    print("Creating simulator...")
    sim = Simulator(config)
    
    # Solve for eigenvalues
    print(f"\nSolving for {args.num_states} eigenvalues...")
    eigenvalues, eigenvectors = sim.solve(args.num_states)
    
    # Print the eigenvalues
    print("\nEigenvalues (eV):")
    for i, ev in enumerate(eigenvalues):
        print(f"  {i}: {ev/config.e_charge:.6f}")
    
    # Analyze the results if requested
    if args.analyze:
        analyze_results(sim, eigenvalues, eigenvectors, args)
    
    # Plot the results if requested
    if not args.no_plot:
        plot_results(sim, args)
    
    return sim, eigenvalues, eigenvectors

def analyze_results(sim, eigenvalues, eigenvectors, args):
    """Analyze the simulation results."""
    print("\nAnalyzing results...")
    
    # Calculate transition energies
    transitions = calculate_transition_energies(eigenvalues, sim.config.e_charge)
    print("\nTransition energies (eV):")
    for i in range(min(3, len(eigenvalues))):
        for j in range(i+1, min(3, len(eigenvalues))):
            print(f"  {i} -> {j}: {transitions[i, j]:.6f}")
    
    # Calculate transition probabilities
    if eigenvectors is not None and eigenvectors.shape[1] > 1:
        probs = calculate_transition_probabilities(eigenvectors)
        print("\nTransition probabilities:")
        for i in range(min(3, eigenvectors.shape[1])):
            for j in range(i+1, min(3, eigenvectors.shape[1])):
                print(f"  {i} -> {j}: {probs[i, j]:.6f}")
    
    # Calculate energy level statistics
    stats = calculate_energy_level_statistics(eigenvalues, sim.config.e_charge)
    print("\nEnergy level statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    
    # Calculate wavefunction localization
    if eigenvectors is not None and eigenvectors.shape[1] > 0:
        ipr = calculate_wavefunction_localization(eigenvectors)
        print("\nWavefunction localization (IPR):")
        for i in range(min(5, len(ipr))):
            print(f"  State {i}: {ipr[i]:.6f}")
    
    # Find bound states
    bound_states = find_bound_states(eigenvalues, sim.config.V_0 / sim.config.e_charge, sim.config.e_charge)
    print("\nBound states:")
    if bound_states:
        for i in bound_states:
            energy = eigenvalues[i] / sim.config.e_charge
            print(f"  State {i}: {energy:.6f} eV")
    else:
        print("  No bound states found")
    
    # Save analysis results to file if requested
    if args.save_analysis:
        analysis_file = os.path.join(args.output_dir, f"{args.prefix}_analysis.txt")
        with open(analysis_file, "w") as f:
            f.write(f"QDSim Analysis Results\n")
            f.write(f"=====================\n\n")
            f.write(f"Simulation parameters:\n")
            f.write(f"  Domain size: {sim.config.Lx*1e9:.1f} x {sim.config.Ly*1e9:.1f} nm\n")
            f.write(f"  Mesh: {sim.config.nx} x {sim.config.ny} elements, order {sim.config.element_order}\n")
            f.write(f"  Quantum dot: {sim.config.qd_material} in {sim.config.matrix_material}, R = {sim.config.R*1e9:.1f} nm\n")
            f.write(f"  Potential type: {sim.config.potential_type}, V_0 = {sim.config.V_0/sim.config.e_charge:.3f} eV\n")
            f.write(f"  P-N junction: {sim.config.diode_p_material}/{sim.config.diode_n_material}\n")
            f.write(f"  Doping: N_A = {sim.config.N_A:.3e} m^-3, N_D = {sim.config.N_D:.3e} m^-3\n")
            f.write(f"  Bias: V_r = {sim.config.V_r:.3f} V\n\n")
            
            f.write("Eigenvalues (eV):\n")
            for i, ev in enumerate(eigenvalues):
                f.write(f"  {i}: {ev/sim.config.e_charge:.6f}\n")
            
            f.write("\nTransition energies (eV):\n")
            for i in range(min(3, len(eigenvalues))):
                for j in range(i+1, min(3, len(eigenvalues))):
                    f.write(f"  {i} -> {j}: {transitions[i, j]:.6f}\n")
            
            if eigenvectors is not None and eigenvectors.shape[1] > 1:
                f.write("\nTransition probabilities:\n")
                for i in range(min(3, eigenvectors.shape[1])):
                    for j in range(i+1, min(3, eigenvectors.shape[1])):
                        f.write(f"  {i} -> {j}: {probs[i, j]:.6f}\n")
            
            f.write("\nEnergy level statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            if eigenvectors is not None and eigenvectors.shape[1] > 0:
                f.write("\nWavefunction localization (IPR):\n")
                for i in range(min(5, len(ipr))):
                    f.write(f"  State {i}: {ipr[i]:.6f}\n")
            
            f.write("\nBound states:\n")
            if bound_states:
                for i in bound_states:
                    energy = eigenvalues[i] / sim.config.e_charge
                    f.write(f"  State {i}: {energy:.6f} eV\n")
            else:
                f.write("  No bound states found\n")
        
        print(f"\nAnalysis results saved to {analysis_file}")

def plot_results(sim, args):
    """Plot the simulation results."""
    print("\nPlotting results...")
    
    # Create a dashboard of all results
    print("Creating simulation dashboard...")
    fig = create_simulation_dashboard(sim, num_states=min(3, len(sim.eigenvalues) if sim.eigenvalues is not None else 0), resolution=50)
    
    # Save plots if requested
    if args.save_plots:
        # Save the dashboard
        dashboard_file = os.path.join(args.output_dir, f"{args.prefix}_dashboard.{args.plot_format}")
        fig.savefig(dashboard_file, dpi=args.plot_dpi, bbox_inches='tight')
        print(f"Dashboard saved to {dashboard_file}")
        
        # Save all results using the save_simulation_results function
        print("Saving all plots...")
        filenames = save_simulation_results(
            sim, 
            filename_prefix=os.path.join(args.output_dir, args.prefix),
            format=args.plot_format,
            dpi=args.plot_dpi
        )
    
    # Show the dashboard
    plt.figure(fig.number)
    plt.show()

def run_batch(batch_config_path, args):
    """Run a batch of simulations with different parameters."""
    print(f"Running batch simulations from {batch_config_path}...")
    
    # Load batch configuration
    batch_config = load_config_file(batch_config_path)
    
    # Get base configuration
    base_config_data = batch_config.get('base_config', {})
    
    # Get parameter sweeps
    parameter_sweeps = batch_config.get('parameter_sweeps', [])
    
    if not parameter_sweeps:
        print("No parameter sweeps defined in batch configuration")
        return
    
    # Create output directory for batch results
    batch_output_dir = os.path.join(args.output_dir, 'batch')
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # Run simulations for each parameter sweep
    for i, sweep in enumerate(parameter_sweeps):
        print(f"\nRunning parameter sweep {i+1}/{len(parameter_sweeps)}: {sweep['name']}")
        
        # Get parameter to sweep
        param_name = sweep['parameter']
        param_values = sweep['values']
        
        # Create output directory for this sweep
        sweep_output_dir = os.path.join(batch_output_dir, f"sweep_{i+1}_{param_name}")
        os.makedirs(sweep_output_dir, exist_ok=True)
        
        # Initialize arrays to store results
        all_eigenvalues = []
        
        # Run simulation for each parameter value
        for j, value in enumerate(param_values):
            print(f"\nRunning simulation {j+1}/{len(param_values)}: {param_name} = {value}")
            
            # Create a copy of the base configuration
            config_data = base_config_data.copy()
            
            # Set the parameter value
            config_data[param_name] = value
            
            # Create a Config object
            config = create_config_from_args(args, config_data)
            
            # Update output directory and prefix for this simulation
            args_copy = argparse.Namespace(**vars(args))
            args_copy.output_dir = sweep_output_dir
            args_copy.prefix = f"{args.prefix}_{param_name}_{value}"
            args_copy.no_plot = True  # Disable plotting for batch simulations
            
            # Run the simulation
            sim, eigenvalues, eigenvectors = run_simulation(config, args_copy)
            
            # Store results
            all_eigenvalues.append((value, eigenvalues))
        
        # Plot sweep results
        plot_sweep_results(param_name, all_eigenvalues, sweep_output_dir, args)

def plot_sweep_results(param_name, all_eigenvalues, output_dir, args):
    """Plot the results of a parameter sweep."""
    print(f"\nPlotting sweep results for {param_name}...")
    
    # Extract parameter values and eigenvalues
    param_values = [x[0] for x in all_eigenvalues]
    eigenvalues_list = [x[1] for x in all_eigenvalues]
    
    # Determine how many states to plot
    num_states = min(5, min(len(ev) for ev in eigenvalues_list))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot eigenvalues vs parameter
    for i in range(num_states):
        energies = [ev[i].real / 1.602e-19 for ev in eigenvalues_list]  # Convert to eV
        ax.plot(param_values, energies, 'o-', label=f'State {i}')
    
    # Set labels and title
    ax.set_xlabel(f'{param_name}')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(f'Energy levels vs {param_name}')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    sweep_file = os.path.join(output_dir, f"{args.prefix}_sweep_{param_name}.{args.plot_format}")
    fig.savefig(sweep_file, dpi=args.plot_dpi, bbox_inches='tight')
    print(f"Sweep results saved to {sweep_file}")
    
    # Create a summary file
    summary_file = os.path.join(output_dir, f"{args.prefix}_sweep_{param_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Parameter sweep results for {param_name}\n")
        f.write("=====================================\n\n")
        
        f.write(f"Parameter values: {param_values}\n\n")
        
        f.write("Eigenvalues (eV):\n")
        for i, value in enumerate(param_values):
            f.write(f"\n{param_name} = {value}:\n")
            for j in range(num_states):
                energy = eigenvalues_list[i][j].real / 1.602e-19
                f.write(f"  State {j}: {energy:.6f}\n")
    
    print(f"Sweep summary saved to {summary_file}")

def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Check if batch mode is enabled
    if args.batch:
        run_batch(args.batch, args)
        return
    
    # Load configuration from file if provided
    config_data = None
    if args.config:
        config_data = load_config_file(args.config)
    
    # Create Config object
    config = create_config_from_args(args, config_data)
    
    # Run the simulation
    run_simulation(config, args)

if __name__ == "__main__":
    main()
