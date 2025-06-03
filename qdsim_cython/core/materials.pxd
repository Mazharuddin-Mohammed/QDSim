# distutils: language = c++
# cython: language_level = 3

"""
Cython declaration file for Materials classes

This file declares the C++ Material struct and MaterialDatabase class
from the backend that will be wrapped by Cython.
"""

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

# Material struct declaration
cdef extern from "materials.h" namespace "Materials":
    cdef struct Material:
        # Effective masses (relative to electron mass)
        double m_e                    # Electron effective mass
        double m_h                    # Average hole effective mass
        double m_lh                   # Light hole effective mass
        double m_hh                   # Heavy hole effective mass
        double m_so                   # Split-off hole effective mass
        
        # Band structure properties (eV)
        double E_g                    # Bandgap at 300K
        double chi                    # Electron affinity
        double Delta_E_c              # Conduction band offset
        double Delta_E_v              # Valence band offset
        
        # Electrical properties
        double epsilon_r              # Dielectric constant
        double mu_n                   # Electron mobility (m²/V·s)
        double mu_p                   # Hole mobility (m²/V·s)
        double N_c                    # Effective DOS, conduction band (m⁻³)
        double N_v                    # Effective DOS, valence band (m⁻³)
        
        # Structural properties
        double lattice_constant       # Lattice constant (nm)
        double spin_orbit_splitting   # Spin-orbit splitting energy (eV)
        
        # Mechanical properties
        double deformation_potential_c # Deformation potential, conduction (eV)
        double deformation_potential_v # Deformation potential, valence (eV)
        double elastic_c11            # Elastic constant (GPa)
        double elastic_c12            # Elastic constant (GPa)
        double elastic_c44            # Elastic constant (GPa)
        
        # Temperature dependence (Varshni parameters)
        double varshni_alpha          # Varshni alpha parameter
        double varshni_beta           # Varshni beta parameter
        
        # k·p parameters
        double luttinger_gamma1       # Luttinger parameter γ₁
        double luttinger_gamma2       # Luttinger parameter γ₂
        double luttinger_gamma3       # Luttinger parameter γ₃
        double kane_parameter         # Kane parameter (eV·nm)
        
        # Alloy bowing parameters
        double bowing_bandgap         # Bandgap bowing parameter
        double bowing_effective_mass  # Effective mass bowing parameter
        double bowing_lattice_constant # Lattice constant bowing parameter

    cdef cppclass MaterialDatabase:
        # Constructor
        MaterialDatabase() except +
        
        # Material access methods
        const Material& get_material(const string& name) const except +
        Material get_material_at_temperature(const string& name, double temperature) const except +
        
        # Alloy creation
        Material create_alloy(const string& material1, const string& material2, 
                            double x, const string& name) const except +
        
        # Database management
        void add_material(const string& name, const Material& material) except +
        vector[string] get_available_materials() const
        
        # Static utility methods
        @staticmethod
        double calculate_bandgap_at_temperature(const Material& material, 
                                              double temperature) except +
        
        @staticmethod
        double calculate_effective_mass_under_strain(const Material& material,
                                                    double strain_xx, double strain_yy, 
                                                    double strain_zz, bint is_electron) except +
