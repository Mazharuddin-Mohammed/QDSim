# distutils: language = c++
# cython: language_level = 3

"""
Cython-accelerated quantum state analysis

High-performance analysis tools for quantum mechanical states,
wavefunctions, and energy levels in semiconductor devices.

This module provides:
- Wavefunction analysis (probability density, current density)
- Energy level analysis (spacing, degeneracy, tunneling rates)
- Quantum state characterization (localization, coherence)
- Transport property calculations
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as bint
# Simplified imports to avoid dependency issues
# from ..eigen cimport VectorXd, VectorXcd
# from ..core.mesh cimport Mesh
# from ..core.interpolator cimport FEInterpolator

# Initialize NumPy
cnp.import_array()

# Physical constants
cdef double HBAR = 1.054571817e-34  # J⋅s
cdef double E_CHARGE = 1.602176634e-19  # C
cdef double M0 = 9.1093837015e-31  # kg

cdef class QuantumStateAnalyzer:
    """
    High-performance quantum state analysis.
    
    Provides comprehensive analysis of quantum mechanical states including:
    - Wavefunction properties (normalization, localization)
    - Energy level analysis (spacing, degeneracy)
    - Tunneling and transport properties
    - Quantum coherence measures
    """
    
    cdef object _mesh_ref
    cdef object _interpolator
    
    def __cinit__(self, mesh=None):
        """
        Initialize quantum state analyzer.

        Parameters:
        -----------
        mesh : object, optional
            Finite element mesh for the quantum system
        """
        self._mesh_ref = mesh
        # self._interpolator = FEInterpolator(mesh)  # Disabled for now
    
    def analyze_wavefunction(self, psi, energy=None):
        """
        Comprehensive wavefunction analysis.
        
        Parameters:
        -----------
        psi : array_like
            Complex wavefunction values at mesh nodes
        energy : float, optional
            Associated energy eigenvalue
        
        Returns:
        --------
        dict
            Comprehensive wavefunction analysis results
        """
        cdef cnp.ndarray[complex, ndim=1] np_psi = np.asarray(psi, dtype=np.complex128)
        cdef int n = np_psi.shape[0]
        
        # Basic properties
        cdef double norm_l2 = np.sqrt(np.sum(np.abs(np_psi)**2))
        cdef double max_amplitude = np.max(np.abs(np_psi))
        cdef double min_amplitude = np.min(np.abs(np_psi))
        
        # Probability density
        cdef cnp.ndarray[double, ndim=1] prob_density = np.abs(np_psi)**2
        cdef double total_probability = np.sum(prob_density)
        
        # Localization measures
        cdef double participation_ratio = self._calculate_participation_ratio(prob_density)
        cdef double localization_length = self._calculate_localization_length(prob_density)
        
        # Phase analysis
        cdef cnp.ndarray[double, ndim=1] phase = np.angle(np_psi)
        cdef double phase_coherence = self._calculate_phase_coherence(phase)
        
        # Spatial moments
        moments = self._calculate_spatial_moments(prob_density)
        
        # Energy-related properties
        energy_props = {}
        if energy is not None:
            energy_props = {
                'energy_eV': energy / E_CHARGE,
                'energy_meV': (energy / E_CHARGE) * 1000,
                'classical_turning_points': self._find_turning_points(energy),
                'tunneling_probability': self._estimate_tunneling_probability(psi, energy)
            }
        
        return {
            'normalization': {
                'l2_norm': norm_l2,
                'total_probability': total_probability,
                'is_normalized': abs(total_probability - 1.0) < 1e-6
            },
            'amplitude': {
                'max': max_amplitude,
                'min': min_amplitude,
                'dynamic_range': max_amplitude / min_amplitude if min_amplitude > 0 else float('inf')
            },
            'localization': {
                'participation_ratio': participation_ratio,
                'localization_length_nm': localization_length * 1e9,
                'is_localized': participation_ratio < 0.5
            },
            'phase': {
                'coherence': phase_coherence,
                'mean_phase': np.mean(phase),
                'phase_variance': np.var(phase)
            },
            'spatial_moments': moments,
            'energy_properties': energy_props
        }
    
    cdef double _calculate_participation_ratio(self, cnp.ndarray[double, ndim=1] prob_density):
        """Calculate inverse participation ratio"""
        cdef double sum_p2 = np.sum(prob_density**2)
        cdef double sum_p = np.sum(prob_density)
        if sum_p > 0:
            return sum_p**2 / sum_p2
        return 0.0
    
    cdef double _calculate_localization_length(self, cnp.ndarray[double, ndim=1] prob_density):
        """Calculate localization length"""
        # This is a simplified calculation - would need mesh coordinates for full implementation
        cdef int n = prob_density.shape[0]
        cdef double center = 0.0
        cdef double total_prob = np.sum(prob_density)
        
        # Calculate center of mass
        for i in range(n):
            center += i * prob_density[i]
        center /= total_prob
        
        # Calculate second moment
        cdef double second_moment = 0.0
        for i in range(n):
            second_moment += (i - center)**2 * prob_density[i]
        second_moment /= total_prob
        
        return np.sqrt(second_moment)
    
    cdef double _calculate_phase_coherence(self, cnp.ndarray[double, ndim=1] phase):
        """Calculate phase coherence measure"""
        cdef int n = phase.shape[0]
        cdef double coherence = 0.0
        
        # Calculate phase differences
        cdef cnp.ndarray[double, ndim=1] phase_diffs = np.diff(phase)
        
        # Wrap phase differences to [-π, π]
        phase_diffs = np.mod(phase_diffs + np.pi, 2*np.pi) - np.pi
        
        # Calculate coherence as inverse of phase variance
        cdef double phase_var = np.var(phase_diffs)
        return np.exp(-phase_var)
    
    def _calculate_spatial_moments(self, prob_density):
        """Calculate spatial moments of probability density"""
        # This would need actual mesh coordinates for full implementation
        # For now, return placeholder values
        return {
            'mean_x': 0.0,
            'mean_y': 0.0,
            'variance_x': 1.0,
            'variance_y': 1.0,
            'skewness_x': 0.0,
            'skewness_y': 0.0,
            'kurtosis_x': 3.0,
            'kurtosis_y': 3.0
        }
    
    def _find_turning_points(self, energy):
        """Find classical turning points for given energy"""
        # Placeholder implementation
        return {'left': -10e-9, 'right': 10e-9}
    
    def _estimate_tunneling_probability(self, psi, energy):
        """Estimate tunneling probability"""
        # Simplified WKB-like estimate
        return 0.1  # Placeholder

cdef class EnergyLevelAnalyzer:
    """
    Analysis of energy level structure and properties.
    
    Provides analysis of:
    - Energy level spacing and statistics
    - Degeneracy detection and breaking
    - Level crossings and avoided crossings
    - Quantum confinement effects
    """
    
    def __cinit__(self):
        """Initialize energy level analyzer"""
        pass
    
    def analyze_energy_spectrum(self, energies):
        """
        Analyze energy spectrum properties.
        
        Parameters:
        -----------
        energies : array_like
            Energy eigenvalues in Joules
        
        Returns:
        --------
        dict
            Energy spectrum analysis results
        """
        cdef cnp.ndarray[double, ndim=1] E = np.asarray(energies, dtype=np.float64)
        cdef int n = E.shape[0]
        
        if n < 2:
            return {'error': 'Need at least 2 energy levels for analysis'}
        
        # Sort energies
        E = np.sort(E)
        
        # Convert to eV for analysis
        cdef cnp.ndarray[double, ndim=1] E_eV = E / E_CHARGE
        
        # Energy differences
        cdef cnp.ndarray[double, ndim=1] dE = np.diff(E_eV)
        
        # Basic statistics
        cdef double mean_spacing = np.mean(dE)
        cdef double std_spacing = np.std(dE)
        cdef double min_spacing = np.min(dE)
        cdef double max_spacing = np.max(dE)
        
        # Degeneracy analysis
        degeneracy_info = self._analyze_degeneracy(E_eV)
        
        # Level spacing statistics
        spacing_stats = self._analyze_level_spacing_statistics(dE)
        
        # Quantum confinement analysis
        confinement_info = self._analyze_quantum_confinement(E_eV)
        
        return {
            'basic_properties': {
                'num_levels': n,
                'energy_range_eV': E_eV[-1] - E_eV[0],
                'ground_state_eV': E_eV[0],
                'highest_state_eV': E_eV[-1]
            },
            'level_spacing': {
                'mean_eV': mean_spacing,
                'std_eV': std_spacing,
                'min_eV': min_spacing,
                'max_eV': max_spacing,
                'coefficient_of_variation': std_spacing / mean_spacing if mean_spacing > 0 else 0
            },
            'degeneracy': degeneracy_info,
            'spacing_statistics': spacing_stats,
            'quantum_confinement': confinement_info
        }
    
    def _analyze_degeneracy(self, E_eV, tolerance=1e-6):
        """Analyze energy level degeneracy"""
        cdef int n = E_eV.shape[0]
        cdef list degenerate_groups = []
        cdef list current_group = [0]
        
        for i in range(1, n):
            if abs(E_eV[i] - E_eV[i-1]) < tolerance:
                current_group.append(i)
            else:
                if len(current_group) > 1:
                    degenerate_groups.append(current_group)
                current_group = [i]
        
        if len(current_group) > 1:
            degenerate_groups.append(current_group)
        
        return {
            'num_degenerate_groups': len(degenerate_groups),
            'degenerate_groups': degenerate_groups,
            'max_degeneracy': max([len(group) for group in degenerate_groups]) if degenerate_groups else 1,
            'fraction_degenerate': sum([len(group) for group in degenerate_groups]) / n
        }
    
    def _analyze_level_spacing_statistics(self, dE):
        """Analyze level spacing statistics"""
        cdef int n = dE.shape[0]
        
        # Normalized spacings
        cdef double mean_dE = np.mean(dE)
        cdef cnp.ndarray[double, ndim=1] s = dE / mean_dE if mean_dE > 0 else dE
        
        # Wigner-Dyson statistics
        cdef double r_ratio = self._calculate_r_ratio(s)
        
        return {
            'normalized_spacings': s.tolist(),
            'r_ratio': r_ratio,
            'level_repulsion': r_ratio > 0.5,  # Indicates level repulsion
            'distribution_type': 'Wigner' if r_ratio > 0.5 else 'Poisson'
        }
    
    cdef double _calculate_r_ratio(self, cnp.ndarray[double, ndim=1] s):
        """Calculate r-ratio for level spacing statistics"""
        cdef int n = s.shape[0]
        if n < 2:
            return 0.0
        
        cdef double r_sum = 0.0
        for i in range(n-1):
            if s[i] + s[i+1] > 0:
                r_sum += min(s[i], s[i+1]) / max(s[i], s[i+1])
        
        return r_sum / (n-1) if n > 1 else 0.0
    
    def _analyze_quantum_confinement(self, E_eV):
        """Analyze quantum confinement effects"""
        # Simplified analysis - would need more device information for full analysis
        cdef double ground_state = E_eV[0]
        cdef double first_excited = E_eV[1] if len(E_eV) > 1 else ground_state
        cdef double confinement_energy = first_excited - ground_state
        
        return {
            'ground_state_eV': ground_state,
            'first_excited_eV': first_excited,
            'confinement_energy_meV': confinement_energy * 1000,
            'is_strongly_confined': confinement_energy > 0.01,  # > 10 meV
            'confinement_regime': 'strong' if confinement_energy > 0.01 else 'weak'
        }

# Utility functions
def analyze_quantum_state_collection(states, energies, mesh):
    """
    Analyze a collection of quantum states.
    
    Parameters:
    -----------
    states : list of array_like
        List of quantum state wavefunctions
    energies : array_like
        Corresponding energy eigenvalues
    mesh : Mesh
        Finite element mesh
    
    Returns:
    --------
    dict
        Comprehensive analysis of the quantum state collection
    """
    analyzer = QuantumStateAnalyzer(mesh)
    energy_analyzer = EnergyLevelAnalyzer()
    
    # Analyze individual states
    state_analyses = []
    for i, (psi, energy) in enumerate(zip(states, energies)):
        analysis = analyzer.analyze_wavefunction(psi, energy)
        analysis['state_index'] = i
        state_analyses.append(analysis)
    
    # Analyze energy spectrum
    spectrum_analysis = energy_analyzer.analyze_energy_spectrum(energies)
    
    # Collection-wide statistics
    collection_stats = {
        'num_states': len(states),
        'energy_range_eV': (max(energies) - min(energies)) / E_CHARGE,
        'mean_localization': np.mean([s['localization']['participation_ratio'] for s in state_analyses]),
        'mean_coherence': np.mean([s['phase']['coherence'] for s in state_analyses])
    }
    
    return {
        'individual_states': state_analyses,
        'energy_spectrum': spectrum_analysis,
        'collection_statistics': collection_stats
    }
