"""
Result analysis module for QDSim.

This module provides tools for analyzing quantum dot simulation results,
including energy level extraction, transition probability calculations,
and wavefunction analysis.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import pandas as pd

def extract_energy_levels(eigenvalues, convert_to_eV=True, sort=True):
    """
    Extract energy levels from eigenvalues.
    
    Args:
        eigenvalues: Array of eigenvalues
        convert_to_eV: If True, convert energies from J to eV
        sort: If True, sort energy levels in ascending order
    
    Returns:
        energies: Array of energy levels in eV or J
        indices: Array of indices corresponding to the sorted energies
    """
    # Convert to numpy array if not already
    eigenvalues = np.array(eigenvalues)
    
    # Convert to eV if requested
    if convert_to_eV:
        energies = eigenvalues / 1.602e-19  # Convert J to eV
    else:
        energies = eigenvalues
    
    # Sort if requested
    if sort:
        indices = np.argsort(energies)
        energies = energies[indices]
    else:
        indices = np.arange(len(energies))
    
    return energies, indices

def calculate_energy_spacing(eigenvalues, convert_to_eV=True):
    """
    Calculate energy spacing between adjacent energy levels.
    
    Args:
        eigenvalues: Array of eigenvalues
        convert_to_eV: If True, convert energies from J to eV
    
    Returns:
        energy_spacing: Array of energy spacings
        avg_spacing: Average energy spacing
        std_spacing: Standard deviation of energy spacing
    """
    # Extract sorted energy levels
    energies, _ = extract_energy_levels(eigenvalues, convert_to_eV=convert_to_eV, sort=True)
    
    # Calculate energy spacing
    energy_spacing = np.diff(energies)
    
    # Calculate statistics
    avg_spacing = np.mean(energy_spacing)
    std_spacing = np.std(energy_spacing)
    
    return energy_spacing, avg_spacing, std_spacing

def calculate_transition_matrix(eigenvectors, operator=None, normalize=True):
    """
    Calculate transition matrix elements between states.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        operator: Operator matrix for transitions (if None, use identity)
        normalize: If True, normalize the matrix elements
    
    Returns:
        transition_matrix: Matrix of transition probabilities
    """
    n_states = eigenvectors.shape[1]
    transition_matrix = np.zeros((n_states, n_states), dtype=complex)
    
    # If no operator is provided, use identity (overlap integral)
    if operator is None:
        operator = np.eye(eigenvectors.shape[0])
    
    # Calculate transition matrix elements
    for i in range(n_states):
        for j in range(n_states):
            # Calculate matrix element <ψi|O|ψj>
            if np.iscomplexobj(eigenvectors):
                matrix_element = np.vdot(eigenvectors[:, i], operator @ eigenvectors[:, j])
            else:
                matrix_element = eigenvectors[:, i].T @ operator @ eigenvectors[:, j]
            
            transition_matrix[i, j] = matrix_element
    
    # Normalize if requested
    if normalize:
        # Compute the maximum absolute value
        max_abs = np.max(np.abs(transition_matrix))
        if max_abs > 0:
            transition_matrix = transition_matrix / max_abs
    
    return transition_matrix

def calculate_dipole_matrix(eigenvectors, mesh, convert_to_debye=True):
    """
    Calculate dipole transition matrix elements between states.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        mesh: Mesh object containing node coordinates
        convert_to_debye: If True, convert dipole moments to Debye units
    
    Returns:
        dipole_matrix_x: Matrix of x-component dipole transition moments
        dipole_matrix_y: Matrix of y-component dipole transition moments
        dipole_matrix_total: Matrix of total dipole transition moments
    """
    n_states = eigenvectors.shape[1]
    n_nodes = mesh.get_num_nodes()
    
    # Get node coordinates
    nodes = np.array(mesh.get_nodes())
    
    # Create position operators
    x_operator = np.zeros((n_nodes, n_nodes))
    y_operator = np.zeros((n_nodes, n_nodes))
    
    # Fill diagonal elements with node coordinates
    for i in range(n_nodes):
        x_operator[i, i] = nodes[i, 0]
        y_operator[i, i] = nodes[i, 1]
    
    # Calculate dipole transition matrices
    dipole_matrix_x = calculate_transition_matrix(eigenvectors, x_operator, normalize=False)
    dipole_matrix_y = calculate_transition_matrix(eigenvectors, y_operator, normalize=False)
    
    # Calculate total dipole matrix
    dipole_matrix_total = np.sqrt(np.abs(dipole_matrix_x)**2 + np.abs(dipole_matrix_y)**2)
    
    # Convert to Debye if requested
    if convert_to_debye:
        # Conversion factor: 1 e*nm ≈ 4.8 Debye
        conversion_factor = 4.8
        dipole_matrix_x *= conversion_factor
        dipole_matrix_y *= conversion_factor
        dipole_matrix_total *= conversion_factor
    
    return dipole_matrix_x, dipole_matrix_y, dipole_matrix_total

def calculate_oscillator_strengths(eigenvalues, dipole_matrix_total):
    """
    Calculate oscillator strengths for transitions.
    
    Args:
        eigenvalues: Array of eigenvalues
        dipole_matrix_total: Matrix of total dipole transition moments
    
    Returns:
        oscillator_strengths: Matrix of oscillator strengths
    """
    n_states = len(eigenvalues)
    oscillator_strengths = np.zeros((n_states, n_states))
    
    # Calculate energy differences (in eV)
    energies, _ = extract_energy_levels(eigenvalues, convert_to_eV=True)
    
    # Calculate oscillator strengths
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                # Oscillator strength formula: f = (2/3) * ΔE * |<ψi|r|ψj>|^2
                # where ΔE is in atomic units and dipole moment is in atomic units
                # For simplicity, we use a proportionality constant
                energy_diff = np.abs(energies[j] - energies[i])
                oscillator_strengths[i, j] = (2/3) * energy_diff * dipole_matrix_total[i, j]**2
    
    return oscillator_strengths

def analyze_wavefunction_localization(eigenvectors, mesh, threshold=0.1):
    """
    Analyze the spatial localization of wavefunctions.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        mesh: Mesh object containing node coordinates
        threshold: Threshold for considering a node as part of the localization region
    
    Returns:
        localization_metrics: Dictionary with localization metrics for each state
    """
    n_states = eigenvectors.shape[1]
    n_nodes = mesh.get_num_nodes()
    
    # Get node coordinates
    nodes = np.array(mesh.get_nodes())
    
    # Initialize results dictionary
    localization_metrics = {
        'state_idx': [],
        'center_of_mass_x': [],
        'center_of_mass_y': [],
        'spread_x': [],
        'spread_y': [],
        'localization_area': [],
        'participation_ratio': []
    }
    
    # Analyze each state
    for state_idx in range(n_states):
        # Get wavefunction probability density
        psi = np.abs(eigenvectors[:, state_idx])**2
        
        # Normalize
        psi_sum = np.sum(psi)
        if psi_sum > 0:
            psi_norm = psi / psi_sum
        else:
            psi_norm = psi
        
        # Calculate center of mass
        com_x = np.sum(nodes[:, 0] * psi_norm)
        com_y = np.sum(nodes[:, 1] * psi_norm)
        
        # Calculate spread (standard deviation)
        spread_x = np.sqrt(np.sum(psi_norm * (nodes[:, 0] - com_x)**2))
        spread_y = np.sqrt(np.sum(psi_norm * (nodes[:, 1] - com_y)**2))
        
        # Calculate localization area (nodes where psi > threshold * max(psi))
        psi_max = np.max(psi_norm)
        localized_nodes = np.where(psi_norm > threshold * psi_max)[0]
        
        # Estimate area using the number of nodes and average node spacing
        if len(localized_nodes) > 0:
            # Estimate average node spacing
            if len(localized_nodes) > 1:
                # Use the average distance between adjacent nodes as an estimate
                localized_coords = nodes[localized_nodes]
                dists = []
                for i in range(len(localized_coords)):
                    for j in range(i+1, len(localized_coords)):
                        dists.append(np.linalg.norm(localized_coords[i] - localized_coords[j]))
                avg_spacing = np.mean(dists) if dists else 0
            else:
                # Default spacing if only one node
                avg_spacing = 1.0
            
            # Estimate area
            localization_area = len(localized_nodes) * avg_spacing**2
        else:
            localization_area = 0.0
        
        # Calculate participation ratio (measure of localization)
        # PR = 1 / Σ(|ψ|^4) - smaller values indicate more localization
        participation_ratio = 1.0 / np.sum(psi_norm**2) if np.sum(psi_norm**2) > 0 else 0.0
        
        # Store results
        localization_metrics['state_idx'].append(state_idx)
        localization_metrics['center_of_mass_x'].append(com_x)
        localization_metrics['center_of_mass_y'].append(com_y)
        localization_metrics['spread_x'].append(spread_x)
        localization_metrics['spread_y'].append(spread_y)
        localization_metrics['localization_area'].append(localization_area)
        localization_metrics['participation_ratio'].append(participation_ratio)
    
    return localization_metrics

def fit_energy_levels(voltage_values, energy_values, model='linear'):
    """
    Fit energy levels as a function of voltage.
    
    Args:
        voltage_values: Array of voltage values
        energy_values: Array of energy values
        model: Fitting model ('linear', 'quadratic', or 'exponential')
    
    Returns:
        params: Fitted parameters
        fit_function: Function that returns fitted values
        r_squared: R-squared value of the fit
    """
    # Define fitting functions
    def linear_func(x, a, b):
        return a * x + b
    
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c
    
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    # Select fitting function based on model
    if model == 'quadratic':
        func = quadratic_func
        p0 = [0.1, 0.1, energy_values[0]]  # Initial guess
    elif model == 'exponential':
        func = exponential_func
        p0 = [0.1, 0.1, energy_values[0]]  # Initial guess
    else:  # Default to linear
        func = linear_func
        p0 = [0.1, energy_values[0]]  # Initial guess
    
    # Fit the data
    try:
        params, pcov = curve_fit(func, voltage_values, energy_values, p0=p0)
        
        # Calculate R-squared
        residuals = energy_values - func(voltage_values, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((energy_values - np.mean(energy_values))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Create fit function
        def fit_function(x):
            return func(x, *params)
        
        return params, fit_function, r_squared
    
    except Exception as e:
        print(f"Fitting error: {e}")
        return None, lambda x: x, 0.0

def create_energy_level_report(eigenvalues, convert_to_eV=True):
    """
    Create a comprehensive report of energy levels.
    
    Args:
        eigenvalues: Array of eigenvalues
        convert_to_eV: If True, convert energies from J to eV
    
    Returns:
        report: Dictionary with energy level information
        fig: Matplotlib figure with energy level diagram
    """
    # Extract energy levels
    energies, indices = extract_energy_levels(eigenvalues, convert_to_eV=convert_to_eV)
    
    # Calculate energy spacing
    energy_spacing, avg_spacing, std_spacing = calculate_energy_spacing(eigenvalues, convert_to_eV=convert_to_eV)
    
    # Create report dictionary
    report = {
        'energies': energies,
        'indices': indices,
        'energy_spacing': energy_spacing,
        'avg_spacing': avg_spacing,
        'std_spacing': std_spacing,
        'ground_state_energy': energies[0],
        'excited_state_energies': energies[1:],
        'unit': 'eV' if convert_to_eV else 'J'
    }
    
    # Create energy level diagram
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Plot energy levels
    for i, energy in enumerate(energies):
        # Plot horizontal line for energy level
        ax.plot([-0.2, 0.2], [energy, energy], 'b-', linewidth=2)
        
        # Add label
        ax.text(0.25, energy, f'E{indices[i]} = {energy:.6f} {report["unit"]}', 
               va='center', ha='left')
    
    # Plot transitions
    for i in range(len(energies) - 1):
        # Plot arrow for transition
        ax.arrow(0, energies[i], 0, energies[i+1] - energies[i] - 0.02,
                head_width=0.05, head_length=0.02, fc='r', ec='r', length_includes_head=True)
        
        # Add transition energy label
        transition_energy = energies[i+1] - energies[i]
        ax.text(-0.3, energies[i] + transition_energy/2, 
               f'ΔE = {transition_energy:.6f} {report["unit"]}', 
               va='center', ha='right', color='r')
    
    # Set labels and title
    ax.set_ylabel(f'Energy ({report["unit"]})')
    ax.set_title('Energy Level Diagram')
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics as text
    stats_text = (
        f'Number of states: {len(energies)}\n'
        f'Ground state energy: {report["ground_state_energy"]:.6f} {report["unit"]}\n'
        f'Average level spacing: {report["avg_spacing"]:.6f} {report["unit"]}\n'
        f'Standard deviation: {report["std_spacing"]:.6f} {report["unit"]}'
    )
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
           ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    
    return report, fig

def create_transition_probability_report(eigenvectors, eigenvalues, mesh=None, convert_to_eV=True):
    """
    Create a comprehensive report of transition probabilities.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        eigenvalues: Array of eigenvalues
        mesh: Mesh object containing node coordinates (optional, for dipole moments)
        convert_to_eV: If True, convert energies from J to eV
    
    Returns:
        report: Dictionary with transition probability information
        figs: List of matplotlib figures with transition probability visualizations
    """
    # Calculate transition matrix (overlap integrals)
    transition_matrix = calculate_transition_matrix(eigenvectors)
    
    # Calculate dipole transition matrices if mesh is provided
    if mesh is not None:
        dipole_matrix_x, dipole_matrix_y, dipole_matrix_total = calculate_dipole_matrix(
            eigenvectors, mesh, convert_to_debye=True)
        
        # Calculate oscillator strengths
        oscillator_strengths = calculate_oscillator_strengths(eigenvalues, dipole_matrix_total)
    else:
        dipole_matrix_x = None
        dipole_matrix_y = None
        dipole_matrix_total = None
        oscillator_strengths = None
    
    # Extract energy levels
    energies, indices = extract_energy_levels(eigenvalues, convert_to_eV=convert_to_eV)
    
    # Create report dictionary
    report = {
        'transition_matrix': transition_matrix,
        'dipole_matrix_x': dipole_matrix_x,
        'dipole_matrix_y': dipole_matrix_y,
        'dipole_matrix_total': dipole_matrix_total,
        'oscillator_strengths': oscillator_strengths,
        'energies': energies,
        'indices': indices,
        'unit': 'eV' if convert_to_eV else 'J'
    }
    
    # Create figures
    figs = []
    
    # Figure 1: Transition probability matrix
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im1 = ax1.imshow(np.abs(transition_matrix)**2, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im1, ax=ax1, label='Transition Probability')
    
    # Add labels
    ax1.set_xticks(np.arange(len(energies)))
    ax1.set_yticks(np.arange(len(energies)))
    
    # Create labels with energy information
    labels = [f'E{i}\n({e:.4f} {report["unit"]})' for i, e in zip(indices, energies)]
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    
    # Rotate x labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values in the cells
    for i in range(len(energies)):
        for j in range(len(energies)):
            text_color = 'white' if np.abs(transition_matrix[i, j])**2 > 0.5 else 'black'
            ax1.text(j, i, f'{np.abs(transition_matrix[i, j])**2:.2f}', 
                   ha="center", va="center", color=text_color)
    
    ax1.set_title('Transition Probability Matrix')
    ax1.set_xlabel('Final State')
    ax1.set_ylabel('Initial State')
    
    fig1.tight_layout()
    figs.append(fig1)
    
    # Figure 2: Dipole transition matrix (if available)
    if dipole_matrix_total is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        im2 = ax2.imshow(dipole_matrix_total, cmap='plasma')
        plt.colorbar(im2, ax=ax2, label='Dipole Moment (Debye)')
        
        # Add labels
        ax2.set_xticks(np.arange(len(energies)))
        ax2.set_yticks(np.arange(len(energies)))
        ax2.set_xticklabels(labels)
        ax2.set_yticklabels(labels)
        
        # Rotate x labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values in the cells
        for i in range(len(energies)):
            for j in range(len(energies)):
                text_color = 'white' if dipole_matrix_total[i, j] > np.max(dipole_matrix_total)/2 else 'black'
                ax2.text(j, i, f'{dipole_matrix_total[i, j]:.2f}', 
                       ha="center", va="center", color=text_color)
        
        ax2.set_title('Dipole Transition Matrix')
        ax2.set_xlabel('Final State')
        ax2.set_ylabel('Initial State')
        
        fig2.tight_layout()
        figs.append(fig2)
    
    # Figure 3: Oscillator strengths (if available)
    if oscillator_strengths is not None:
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        im3 = ax3.imshow(oscillator_strengths, cmap='inferno')
        plt.colorbar(im3, ax=ax3, label='Oscillator Strength')
        
        # Add labels
        ax3.set_xticks(np.arange(len(energies)))
        ax3.set_yticks(np.arange(len(energies)))
        ax3.set_xticklabels(labels)
        ax3.set_yticklabels(labels)
        
        # Rotate x labels for better readability
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values in the cells
        for i in range(len(energies)):
            for j in range(len(energies)):
                if i != j:  # Only show non-diagonal elements
                    text_color = 'white' if oscillator_strengths[i, j] > np.max(oscillator_strengths)/2 else 'black'
                    ax3.text(j, i, f'{oscillator_strengths[i, j]:.4f}', 
                           ha="center", va="center", color=text_color)
        
        ax3.set_title('Oscillator Strengths')
        ax3.set_xlabel('Final State')
        ax3.set_ylabel('Initial State')
        
        fig3.tight_layout()
        figs.append(fig3)
    
    return report, figs

def create_wavefunction_localization_report(eigenvectors, mesh, eigenvalues=None, convert_to_eV=True):
    """
    Create a comprehensive report of wavefunction localization.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        mesh: Mesh object containing node coordinates
        eigenvalues: Array of eigenvalues (optional)
        convert_to_eV: If True, convert energies from J to eV
    
    Returns:
        report: Dictionary with localization information
        fig: Matplotlib figure with localization visualization
    """
    # Analyze wavefunction localization
    localization_metrics = analyze_wavefunction_localization(eigenvectors, mesh)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(localization_metrics)
    
    # Add energy information if available
    if eigenvalues is not None:
        energies, _ = extract_energy_levels(eigenvalues, convert_to_eV=convert_to_eV)
        df['energy'] = energies
        df['energy_unit'] = 'eV' if convert_to_eV else 'J'
    
    # Create report dictionary
    report = {
        'localization_metrics': localization_metrics,
        'dataframe': df
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot center of mass
    axes[0, 0].scatter(df['center_of_mass_x'], df['center_of_mass_y'], c=df['state_idx'], cmap='viridis')
    for i, row in df.iterrows():
        axes[0, 0].annotate(str(row['state_idx']), 
                          (row['center_of_mass_x'], row['center_of_mass_y']),
                          xytext=(5, 5), textcoords='offset points')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].set_title('Wavefunction Centers of Mass')
    axes[0, 0].grid(True)
    
    # Plot spread
    axes[0, 1].scatter(df['spread_x'], df['spread_y'], c=df['state_idx'], cmap='viridis')
    for i, row in df.iterrows():
        axes[0, 1].annotate(str(row['state_idx']), 
                          (row['spread_x'], row['spread_y']),
                          xytext=(5, 5), textcoords='offset points')
    axes[0, 1].set_xlabel('X Spread')
    axes[0, 1].set_ylabel('Y Spread')
    axes[0, 1].set_title('Wavefunction Spatial Spread')
    axes[0, 1].grid(True)
    
    # Plot localization area vs. state index
    axes[1, 0].bar(df['state_idx'], df['localization_area'])
    axes[1, 0].set_xlabel('State Index')
    axes[1, 0].set_ylabel('Localization Area')
    axes[1, 0].set_title('Wavefunction Localization Area')
    axes[1, 0].grid(True)
    
    # Plot participation ratio vs. state index
    axes[1, 1].bar(df['state_idx'], df['participation_ratio'])
    axes[1, 1].set_xlabel('State Index')
    axes[1, 1].set_ylabel('Participation Ratio')
    axes[1, 1].set_title('Wavefunction Participation Ratio')
    axes[1, 1].grid(True)
    
    # Add energy information if available
    if eigenvalues is not None:
        energy_unit = 'eV' if convert_to_eV else 'J'
        for i, ax in enumerate(axes.flat):
            ax2 = ax.twinx()
            ax2.plot(df['state_idx'], df['energy'], 'r-', marker='o')
            ax2.set_ylabel(f'Energy ({energy_unit})', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
    
    fig.tight_layout()
    
    return report, fig

def export_results_to_csv(report, filename):
    """
    Export analysis results to CSV file.
    
    Args:
        report: Dictionary with analysis results
        filename: Output filename
    
    Returns:
        success: True if export was successful, False otherwise
    """
    try:
        # Check if report contains a DataFrame
        if 'dataframe' in report:
            report['dataframe'].to_csv(filename, index=False)
            return True
        
        # Otherwise, create a DataFrame from the report
        data = {}
        
        # Add energy levels if available
        if 'energies' in report:
            data['state_idx'] = np.arange(len(report['energies']))
            data['energy'] = report['energies']
            data['energy_unit'] = report['unit']
        
        # Add transition probabilities if available
        if 'transition_matrix' in report and report['transition_matrix'] is not None:
            n_states = report['transition_matrix'].shape[0]
            for i in range(n_states):
                for j in range(n_states):
                    data[f'transition_prob_{i}_{j}'] = np.abs(report['transition_matrix'][i, j])**2
        
        # Add dipole moments if available
        if 'dipole_matrix_total' in report and report['dipole_matrix_total'] is not None:
            n_states = report['dipole_matrix_total'].shape[0]
            for i in range(n_states):
                for j in range(n_states):
                    data[f'dipole_moment_{i}_{j}'] = report['dipole_matrix_total'][i, j]
        
        # Add oscillator strengths if available
        if 'oscillator_strengths' in report and report['oscillator_strengths'] is not None:
            n_states = report['oscillator_strengths'].shape[0]
            for i in range(n_states):
                for j in range(n_states):
                    if i != j:  # Only include non-diagonal elements
                        data[f'oscillator_strength_{i}_{j}'] = report['oscillator_strengths'][i, j]
        
        # Create DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return True
    
    except Exception as e:
        print(f"Error exporting results: {e}")
        return False
