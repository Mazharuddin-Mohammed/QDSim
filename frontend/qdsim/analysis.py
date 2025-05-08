"""
Analysis module for QDSim.

This module provides functions for analyzing simulation results,
including energy level calculations, transition probabilities,
and other quantum dot properties.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import scipy.constants as const
from typing import List, Tuple, Dict, Optional, Union

def calculate_transition_energies(eigenvalues: np.ndarray, 
                                 e_charge: float = const.e) -> np.ndarray:
    """
    Calculate transition energies between energy levels.
    
    Args:
        eigenvalues: Array of eigenvalues in Joules
        e_charge: Elementary charge in Coulombs (for conversion to eV)
        
    Returns:
        Array of transition energies in eV
    """
    # Convert eigenvalues to real part only
    real_eigenvalues = np.real(eigenvalues)
    
    # Calculate all possible transitions
    n = len(real_eigenvalues)
    transitions = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Energy difference in Joules
                energy_diff = abs(real_eigenvalues[j] - real_eigenvalues[i])
                # Convert to eV
                transitions[i, j] = energy_diff / e_charge
    
    return transitions

def calculate_transition_probabilities(eigenvectors: np.ndarray, 
                                      dipole_operator: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate transition probabilities between states.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        dipole_operator: Dipole operator matrix (if None, use identity)
        
    Returns:
        Matrix of transition probabilities
    """
    n = eigenvectors.shape[1]
    probabilities = np.zeros((n, n))
    
    # If no dipole operator is provided, use identity (overlap integral)
    if dipole_operator is None:
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate overlap integral
                    overlap = np.abs(np.vdot(eigenvectors[:, i], eigenvectors[:, j]))**2
                    probabilities[i, j] = overlap
    else:
        # Use the provided dipole operator
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate matrix element
                    matrix_element = np.vdot(eigenvectors[:, i], 
                                           np.dot(dipole_operator, eigenvectors[:, j]))
                    probabilities[i, j] = np.abs(matrix_element)**2
    
    return probabilities

def calculate_oscillator_strengths(eigenvalues: np.ndarray, 
                                  eigenvectors: np.ndarray,
                                  position_operator: np.ndarray,
                                  mass: float,
                                  e_charge: float = const.e) -> np.ndarray:
    """
    Calculate oscillator strengths for transitions.
    
    Args:
        eigenvalues: Array of eigenvalues in Joules
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        position_operator: Position operator matrix
        mass: Effective mass in kg
        e_charge: Elementary charge in Coulombs
        
    Returns:
        Matrix of oscillator strengths
    """
    n = eigenvectors.shape[1]
    f = np.zeros((n, n))
    
    # Convert eigenvalues to real part only
    real_eigenvalues = np.real(eigenvalues)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Energy difference in Joules
                energy_diff = abs(real_eigenvalues[j] - real_eigenvalues[i])
                
                # Calculate dipole matrix element
                dipole = np.vdot(eigenvectors[:, i], 
                               np.dot(position_operator, eigenvectors[:, j]))
                
                # Calculate oscillator strength
                # f = (2m/ħ²) * (E_j - E_i) * |<i|x|j>|²
                f[i, j] = (2 * mass / const.hbar**2) * energy_diff * np.abs(dipole)**2
    
    return f

def calculate_density_of_states(eigenvalues: np.ndarray, 
                               energy_range: Tuple[float, float],
                               num_points: int = 1000,
                               broadening: float = 0.01,
                               e_charge: float = const.e) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the density of states.
    
    Args:
        eigenvalues: Array of eigenvalues in Joules
        energy_range: Tuple of (min_energy, max_energy) in eV
        num_points: Number of points in the energy grid
        broadening: Broadening parameter in eV
        e_charge: Elementary charge in Coulombs (for conversion to eV)
        
    Returns:
        Tuple of (energies, DOS) where energies is in eV and DOS is in arbitrary units
    """
    # Convert eigenvalues to eV
    eigenvalues_eV = np.real(eigenvalues) / e_charge
    
    # Create energy grid
    energies = np.linspace(energy_range[0], energy_range[1], num_points)
    
    # Calculate DOS using Gaussian broadening
    dos = np.zeros_like(energies)
    
    for e_val in eigenvalues_eV:
        # Add Gaussian centered at each eigenvalue
        dos += np.exp(-(energies - e_val)**2 / (2 * broadening**2)) / (broadening * np.sqrt(2 * np.pi))
    
    return energies, dos

def calculate_electron_density(eigenvectors: np.ndarray, 
                              occupation: np.ndarray,
                              mesh_nodes: np.ndarray) -> np.ndarray:
    """
    Calculate the electron density.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        occupation: Array of occupation numbers for each state
        mesh_nodes: Array of mesh node coordinates
        
    Returns:
        Array of electron density at each mesh node
    """
    n_nodes = eigenvectors.shape[0]
    n_states = eigenvectors.shape[1]
    
    # Initialize density
    density = np.zeros(n_nodes)
    
    # Sum over all states
    for i in range(n_states):
        if occupation[i] > 0:
            # Add contribution from this state
            density += occupation[i] * np.abs(eigenvectors[:, i])**2
    
    return density

def calculate_expectation_value(operator: np.ndarray, 
                               eigenvector: np.ndarray) -> complex:
    """
    Calculate the expectation value of an operator for a given state.
    
    Args:
        operator: Operator matrix
        eigenvector: State vector
        
    Returns:
        Expectation value
    """
    return np.vdot(eigenvector, np.dot(operator, eigenvector))

def calculate_energy_level_statistics(eigenvalues: np.ndarray, 
                                     e_charge: float = const.e) -> Dict[str, float]:
    """
    Calculate statistics of energy levels.
    
    Args:
        eigenvalues: Array of eigenvalues in Joules
        e_charge: Elementary charge in Coulombs (for conversion to eV)
        
    Returns:
        Dictionary of statistics
    """
    # Convert eigenvalues to eV
    eigenvalues_eV = np.real(eigenvalues) / e_charge
    
    # Calculate level spacings
    spacings = np.diff(eigenvalues_eV)
    
    # Calculate statistics
    stats = {
        'mean_spacing': np.mean(spacings),
        'std_spacing': np.std(spacings),
        'min_spacing': np.min(spacings),
        'max_spacing': np.max(spacings),
        'mean_energy': np.mean(eigenvalues_eV),
        'std_energy': np.std(eigenvalues_eV),
        'min_energy': np.min(eigenvalues_eV),
        'max_energy': np.max(eigenvalues_eV)
    }
    
    return stats

def calculate_wavefunction_localization(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse participation ratio (IPR) for each state.
    
    The IPR is a measure of localization. Higher values indicate more localized states.
    
    Args:
        eigenvectors: Matrix of eigenvectors (columns are eigenvectors)
        
    Returns:
        Array of IPR values for each state
    """
    n_states = eigenvectors.shape[1]
    ipr = np.zeros(n_states)
    
    for i in range(n_states):
        # Calculate probability density
        prob_density = np.abs(eigenvectors[:, i])**2
        
        # Calculate IPR
        ipr[i] = np.sum(prob_density**2) / np.sum(prob_density)**2
    
    return ipr

def find_bound_states(eigenvalues: np.ndarray, 
                     potential_depth: float,
                     e_charge: float = const.e) -> List[int]:
    """
    Find indices of bound states.
    
    Args:
        eigenvalues: Array of eigenvalues in Joules
        potential_depth: Depth of the potential well in eV
        e_charge: Elementary charge in Coulombs (for conversion to eV)
        
    Returns:
        List of indices of bound states
    """
    # Convert eigenvalues to eV
    eigenvalues_eV = np.real(eigenvalues) / e_charge
    
    # Find states with energy less than zero (bound states)
    bound_indices = [i for i, e in enumerate(eigenvalues_eV) if e < 0]
    
    return bound_indices

def calculate_confinement_energy(eigenvalues: np.ndarray, 
                                ground_state_bulk: float,
                                e_charge: float = const.e) -> float:
    """
    Calculate the confinement energy.
    
    Args:
        eigenvalues: Array of eigenvalues in Joules
        ground_state_bulk: Ground state energy in bulk material in eV
        e_charge: Elementary charge in Coulombs (for conversion to eV)
        
    Returns:
        Confinement energy in eV
    """
    # Convert ground state energy to eV
    ground_state_eV = np.real(eigenvalues[0]) / e_charge
    
    # Calculate confinement energy
    confinement_energy = ground_state_eV - ground_state_bulk
    
    return confinement_energy
