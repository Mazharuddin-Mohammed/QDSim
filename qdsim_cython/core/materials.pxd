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

# Simplified Material struct for Cython compilation
cdef struct CppMaterial:
    # Basic material properties
    double m_e                    # Electron effective mass
    double m_h                    # Hole effective mass
    double E_g                    # Bandgap (eV)
    double epsilon_r              # Dielectric constant
    double mu_n                   # Electron mobility
    double mu_p                   # Hole mobility
    double lattice_constant       # Lattice constant (nm)
