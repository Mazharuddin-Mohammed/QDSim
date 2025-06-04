# distutils: language = c++
# cython: language_level = 3

"""
Cython wrapper for Materials classes

This module provides Python access to the C++ Material struct and 
MaterialDatabase class for semiconductor material properties.
"""

import numpy as np
cimport numpy as cnp
from libcpp.string cimport string
from libcpp.vector cimport vector

# Import C++ declarations
from .materials cimport CppMaterial

# Initialize NumPy
cnp.import_array()

cdef class Material:
    """
    Python wrapper for the C++ Material struct.
    
    Contains semiconductor material properties including effective masses,
    band structure, electrical, structural, and mechanical properties.
    """
    
    cdef CppMaterial _material
    
    def __cinit__(self):
        """Initialize with default values."""
        pass
        
    @staticmethod
    cdef Material from_cpp_material(CppMaterial cpp_material):
        """Create a Python Material from a C++ Material."""
        cdef Material material = Material()
        material._material = cpp_material
        return material
        
    # Basic material properties
    @property
    def m_e(self):
        """Electron effective mass (relative to m₀)."""
        return self._material.m_e

    @m_e.setter
    def m_e(self, double value):
        self._material.m_e = value

    @property
    def m_h(self):
        """Hole effective mass (relative to m₀)."""
        return self._material.m_h

    @m_h.setter
    def m_h(self, double value):
        self._material.m_h = value

    @property
    def E_g(self):
        """Bandgap (eV)."""
        return self._material.E_g

    @E_g.setter
    def E_g(self, double value):
        self._material.E_g = value

    @property
    def epsilon_r(self):
        """Dielectric constant."""
        return self._material.epsilon_r

    @epsilon_r.setter
    def epsilon_r(self, double value):
        self._material.epsilon_r = value

    @property
    def mu_n(self):
        """Electron mobility."""
        return self._material.mu_n

    @mu_n.setter
    def mu_n(self, double value):
        self._material.mu_n = value

    @property
    def mu_p(self):
        """Hole mobility."""
        return self._material.mu_p

    @mu_p.setter
    def mu_p(self, double value):
        self._material.mu_p = value

    @property
    def lattice_constant(self):
        """Lattice constant (nm)."""
        return self._material.lattice_constant

    @lattice_constant.setter
    def lattice_constant(self, double value):
        self._material.lattice_constant = value
        
    def __repr__(self):
        """String representation of the material."""
        return (f"Material(m_e={self.m_e:.3f}, m_h={self.m_h:.3f}, "
                f"E_g={self.E_g:.3f} eV, epsilon_r={self.epsilon_r:.1f})")

# Convenience function to create a material with default values
def create_material(name="GaAs"):
    """
    Create a Material with default properties.

    Parameters
    ----------
    name : str
        Material name (for future database lookup)

    Returns
    -------
    Material
        Material with default semiconductor properties
    """
    material = Material()

    # Set some default GaAs properties
    material.m_e = 0.067
    material.m_h = 0.45
    material.E_g = 1.424
    material.epsilon_r = 12.9
    material.mu_n = 8500e-4  # m²/V⋅s
    material.mu_p = 400e-4   # m²/V⋅s
    material.lattice_constant = 0.5653  # nm

    return material
