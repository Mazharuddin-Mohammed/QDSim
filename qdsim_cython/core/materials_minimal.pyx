# distutils: language = c++
# cython: language_level = 3

"""
Minimal Cython materials module for debugging
No NumPy dependencies, just basic functionality
"""

from libcpp.string cimport string

# Minimal Material struct
cdef struct CppMaterial:
    double m_e                    # Electron effective mass
    double m_h                    # Hole effective mass
    double E_g                    # Bandgap (eV)
    double epsilon_r              # Dielectric constant

cdef class Material:
    """
    Minimal Material class for debugging
    """
    
    cdef CppMaterial _material
    
    def __cinit__(self):
        """Initialize with default values."""
        self._material.m_e = 0.067
        self._material.m_h = 0.45
        self._material.E_g = 1.424
        self._material.epsilon_r = 12.9
    
    @property
    def m_e(self):
        """Electron effective mass."""
        return self._material.m_e
        
    @m_e.setter
    def m_e(self, double value):
        self._material.m_e = value
        
    @property
    def m_h(self):
        """Hole effective mass."""
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
        
    def __repr__(self):
        """String representation."""
        return (f"Material(m_e={self.m_e:.3f}, m_h={self.m_h:.3f}, "
                f"E_g={self.E_g:.3f} eV, epsilon_r={self.epsilon_r:.1f})")

def create_material(name="GaAs"):
    """
    Create a Material with default properties.
    """
    return Material()

def test_basic_functionality():
    """
    Test basic functionality
    """
    mat = create_material()
    print(f"Created: {mat}")
    
    # Test modification
    mat.m_e = 0.1
    mat.E_g = 1.5
    print(f"Modified: {mat}")
    
    return True
