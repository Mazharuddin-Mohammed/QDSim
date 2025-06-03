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
from .materials cimport Material as CppMaterial
from .materials cimport MaterialDatabase as CppMaterialDatabase

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
    cdef Material from_cpp_material(const CppMaterial& cpp_material):
        """Create a Python Material from a C++ Material."""
        cdef Material material = Material()
        material._material = cpp_material
        return material
        
    # Effective mass properties
    @property
    def m_e(self):
        """Electron effective mass (relative to m₀)."""
        return self._material.m_e
        
    @m_e.setter
    def m_e(self, double value):
        self._material.m_e = value
        
    @property
    def m_h(self):
        """Average hole effective mass (relative to m₀)."""
        return self._material.m_h
        
    @m_h.setter
    def m_h(self, double value):
        self._material.m_h = value
        
    @property
    def m_lh(self):
        """Light hole effective mass (relative to m₀)."""
        return self._material.m_lh
        
    @m_lh.setter
    def m_lh(self, double value):
        self._material.m_lh = value
        
    @property
    def m_hh(self):
        """Heavy hole effective mass (relative to m₀)."""
        return self._material.m_hh
        
    @m_hh.setter
    def m_hh(self, double value):
        self._material.m_hh = value
        
    @property
    def m_so(self):
        """Split-off hole effective mass (relative to m₀)."""
        return self._material.m_so
        
    @m_so.setter
    def m_so(self, double value):
        self._material.m_so = value
        
    # Band structure properties
    @property
    def E_g(self):
        """Bandgap at 300K (eV)."""
        return self._material.E_g
        
    @E_g.setter
    def E_g(self, double value):
        self._material.E_g = value
        
    @property
    def chi(self):
        """Electron affinity (eV)."""
        return self._material.chi
        
    @chi.setter
    def chi(self, double value):
        self._material.chi = value
        
    @property
    def Delta_E_c(self):
        """Conduction band offset (eV)."""
        return self._material.Delta_E_c
        
    @Delta_E_c.setter
    def Delta_E_c(self, double value):
        self._material.Delta_E_c = value
        
    @property
    def Delta_E_v(self):
        """Valence band offset (eV)."""
        return self._material.Delta_E_v
        
    @Delta_E_v.setter
    def Delta_E_v(self, double value):
        self._material.Delta_E_v = value
        
    # Electrical properties
    @property
    def epsilon_r(self):
        """Relative dielectric constant."""
        return self._material.epsilon_r
        
    @epsilon_r.setter
    def epsilon_r(self, double value):
        self._material.epsilon_r = value
        
    @property
    def mu_n(self):
        """Electron mobility (m²/V·s)."""
        return self._material.mu_n
        
    @mu_n.setter
    def mu_n(self, double value):
        self._material.mu_n = value
        
    @property
    def mu_p(self):
        """Hole mobility (m²/V·s)."""
        return self._material.mu_p
        
    @mu_p.setter
    def mu_p(self, double value):
        self._material.mu_p = value
        
    @property
    def N_c(self):
        """Effective density of states, conduction band (m⁻³)."""
        return self._material.N_c
        
    @N_c.setter
    def N_c(self, double value):
        self._material.N_c = value
        
    @property
    def N_v(self):
        """Effective density of states, valence band (m⁻³)."""
        return self._material.N_v
        
    @N_v.setter
    def N_v(self, double value):
        self._material.N_v = value
        
    # Structural properties
    @property
    def lattice_constant(self):
        """Lattice constant (nm)."""
        return self._material.lattice_constant
        
    @lattice_constant.setter
    def lattice_constant(self, double value):
        self._material.lattice_constant = value
        
    @property
    def spin_orbit_splitting(self):
        """Spin-orbit splitting energy (eV)."""
        return self._material.spin_orbit_splitting
        
    @spin_orbit_splitting.setter
    def spin_orbit_splitting(self, double value):
        self._material.spin_orbit_splitting = value
        
    def __repr__(self):
        """String representation of the material."""
        return (f"Material(m_e={self.m_e:.3f}, m_h={self.m_h:.3f}, "
                f"E_g={self.E_g:.3f} eV, epsilon_r={self.epsilon_r:.1f})")

cdef class MaterialDatabase:
    """
    Python wrapper for the C++ MaterialDatabase class.
    
    Provides access to a database of semiconductor material properties
    with methods for querying materials, creating alloys, and calculating
    temperature-dependent properties.
    """
    
    cdef CppMaterialDatabase* _db
    cdef bool _owns_db
    
    def __cinit__(self):
        """Initialize the material database."""
        self._db = new CppMaterialDatabase()
        self._owns_db = True
        
    def __dealloc__(self):
        """Clean up the C++ database object."""
        if self._owns_db and self._db is not NULL:
            del self._db
            
    def get_material(self, str name):
        """
        Get material properties by name.
        
        Parameters
        ----------
        name : str
            Name of the material (e.g., "GaAs", "InAs", "AlGaAs")
            
        Returns
        -------
        Material
            Material object with properties
        """
        cdef string cpp_name = name.encode('utf-8')
        cdef const CppMaterial& cpp_material = self._db.get_material(cpp_name)
        return Material.from_cpp_material(cpp_material)
        
    def get_material_at_temperature(self, str name, double temperature):
        """
        Get material properties at a specific temperature.
        
        Parameters
        ----------
        name : str
            Name of the material
        temperature : float
            Temperature in Kelvin
            
        Returns
        -------
        Material
            Material object with temperature-dependent properties
        """
        cdef string cpp_name = name.encode('utf-8')
        cdef CppMaterial cpp_material = self._db.get_material_at_temperature(cpp_name, temperature)
        return Material.from_cpp_material(cpp_material)
        
    def create_alloy(self, str material1, str material2, double x, str name=""):
        """
        Create an alloy material with specified composition.
        
        Parameters
        ----------
        material1 : str
            Name of the first base material
        material2 : str
            Name of the second base material
        x : float
            Composition parameter (0 <= x <= 1)
        name : str, optional
            Name for the alloy (auto-generated if empty)
            
        Returns
        -------
        Material
            Alloy material with interpolated properties
        """
        cdef string cpp_mat1 = material1.encode('utf-8')
        cdef string cpp_mat2 = material2.encode('utf-8')
        cdef string cpp_name = name.encode('utf-8')
        
        cdef CppMaterial cpp_alloy = self._db.create_alloy(cpp_mat1, cpp_mat2, x, cpp_name)
        return Material.from_cpp_material(cpp_alloy)
        
    def add_material(self, str name, Material material):
        """
        Add a custom material to the database.
        
        Parameters
        ----------
        name : str
            Name of the material
        material : Material
            Material object with properties
        """
        cdef string cpp_name = name.encode('utf-8')
        self._db.add_material(cpp_name, material._material)
        
    def get_available_materials(self):
        """
        Get list of all available material names.
        
        Returns
        -------
        list of str
            List of material names in the database
        """
        cdef vector[string] cpp_names = self._db.get_available_materials()
        cdef list names = []
        
        cdef int i
        for i in range(cpp_names.size()):
            names.append(cpp_names[i].decode('utf-8'))
            
        return names
        
    @staticmethod
    def calculate_bandgap_at_temperature(Material material, double temperature):
        """
        Calculate bandgap at a specific temperature using Varshni equation.
        
        Parameters
        ----------
        material : Material
            Material object
        temperature : float
            Temperature in Kelvin
            
        Returns
        -------
        float
            Bandgap in eV at the specified temperature
        """
        return CppMaterialDatabase.calculate_bandgap_at_temperature(material._material, temperature)
        
    @staticmethod
    def calculate_effective_mass_under_strain(Material material, 
                                            double strain_xx, double strain_yy, double strain_zz,
                                            bool is_electron=True):
        """
        Calculate effective mass under strain.
        
        Parameters
        ----------
        material : Material
            Material object
        strain_xx : float
            xx component of strain tensor
        strain_yy : float
            yy component of strain tensor
        strain_zz : float
            zz component of strain tensor
        is_electron : bool, optional
            Whether to calculate electron (True) or hole (False) effective mass
            
        Returns
        -------
        float
            Effective mass under strain (relative to m₀)
        """
        return CppMaterialDatabase.calculate_effective_mass_under_strain(
            material._material, strain_xx, strain_yy, strain_zz, is_electron)

# Convenience function to create a default database
def create_material_database():
    """
    Create a MaterialDatabase with default materials.
    
    Returns
    -------
    MaterialDatabase
        Database initialized with common semiconductor materials
    """
    return MaterialDatabase()
