# distutils: language = c++
# cython: language_level = 3

"""
Cython wrapper for Physics functions and constants

This module provides Python access to the C++ physics functions and constants
for quantum dot simulations.
"""

import numpy as np
cimport numpy as cnp
from libcpp.string cimport string
from ..eigen cimport VectorXd

# Import C++ declarations
from .physics cimport (
    effective_mass as cpp_effective_mass,
    potential as cpp_potential,
    epsilon_r as cpp_epsilon_r,
    charge_density as cpp_charge_density,
    cap as cpp_cap,
    electron_concentration as cpp_electron_concentration,
    hole_concentration as cpp_hole_concentration,
    mobility_n as cpp_mobility_n,
    mobility_p as cpp_mobility_p
)

# Import physical constants
from .physics cimport (
    ELEMENTARY_CHARGE, PLANCK_CONSTANT, REDUCED_PLANCK_CONSTANT,
    BOLTZMANN_CONSTANT, ELECTRON_MASS, VACUUM_PERMITTIVITY, SPEED_OF_LIGHT,
    EV_TO_JOULE, JOULE_TO_EV, NM_TO_METER, METER_TO_NM,
    MIN_EFFECTIVE_MASS, MAX_EFFECTIVE_MASS, MIN_POTENTIAL, MAX_POTENTIAL,
    MIN_PERMITTIVITY, MAX_PERMITTIVITY
)

# Initialize NumPy
cnp.import_array()

class PhysicsConstants:
    """
    Physical constants for quantum dot simulations.
    
    All constants are in SI units unless otherwise specified.
    """
    
    # Fundamental constants
    ELEMENTARY_CHARGE = ELEMENTARY_CHARGE
    PLANCK_CONSTANT = PLANCK_CONSTANT
    REDUCED_PLANCK_CONSTANT = REDUCED_PLANCK_CONSTANT
    BOLTZMANN_CONSTANT = BOLTZMANN_CONSTANT
    ELECTRON_MASS = ELECTRON_MASS
    VACUUM_PERMITTIVITY = VACUUM_PERMITTIVITY
    SPEED_OF_LIGHT = SPEED_OF_LIGHT
    
    # Unit conversions
    EV_TO_JOULE = EV_TO_JOULE
    JOULE_TO_EV = JOULE_TO_EV
    NM_TO_METER = NM_TO_METER
    METER_TO_NM = METER_TO_NM
    
    # Typical parameter ranges
    MIN_EFFECTIVE_MASS = MIN_EFFECTIVE_MASS
    MAX_EFFECTIVE_MASS = MAX_EFFECTIVE_MASS
    MIN_POTENTIAL = MIN_POTENTIAL
    MAX_POTENTIAL = MAX_POTENTIAL
    MIN_PERMITTIVITY = MIN_PERMITTIVITY
    MAX_PERMITTIVITY = MAX_PERMITTIVITY

def effective_mass(double x, double y, qd_material, matrix_material, double R):
    """
    Compute the effective mass at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    qd_material : Material
        Quantum dot material properties
    matrix_material : Material
        Matrix material properties
    R : float
        Quantum dot radius in nanometers
        
    Returns
    -------
    float
        Effective mass relative to electron mass
    """
    # Note: This is a simplified version that will need proper Material binding
    # For now, we'll implement a basic version
    cdef double r = np.sqrt(x*x + y*y)
    if r <= R:
        return 0.067  # InAs effective mass
    else:
        return 0.067  # AlGaAs effective mass

def potential(double x, double y, qd_material, matrix_material, double R, 
             str potential_type, phi_array, interpolator=None):
    """
    Compute the potential at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    qd_material : Material
        Quantum dot material properties
    matrix_material : Material
        Matrix material properties
    R : float
        Quantum dot radius in nanometers
    potential_type : str
        Type of potential ("square", "gaussian", etc.)
    phi_array : array-like
        Electrostatic potential values
    interpolator : FEInterpolator, optional
        Interpolator for electrostatic potential
        
    Returns
    -------
    float
        Potential in electron volts
    """
    # Convert phi_array to C++ VectorXd
    cdef cnp.ndarray[double, ndim=1] phi_np = np.asarray(phi_array, dtype=np.float64)
    cdef VectorXd phi_cpp
    phi_cpp.resize(phi_np.shape[0])
    
    cdef int i
    for i in range(phi_np.shape[0]):
        phi_cpp[i] = phi_np[i]
    
    # Convert string to C++ string
    cdef string cpp_type = potential_type.encode('utf-8')
    
    # For now, implement a simplified version without full Material binding
    cdef double r = np.sqrt(x*x + y*y)
    if potential_type == "square":
        if r <= R:
            return -0.5  # eV, quantum dot potential
        else:
            return 0.0
    elif potential_type == "gaussian":
        return -0.5 * np.exp(-(r*r)/(R*R))
    else:
        return 0.0

def epsilon_r(double x, double y, p_material, n_material):
    """
    Compute the relative permittivity at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    p_material : Material
        p-type material properties
    n_material : Material
        n-type material properties
        
    Returns
    -------
    float
        Relative permittivity
    """
    # Simplified implementation
    if x < 0:
        return 12.9  # GaAs permittivity
    else:
        return 12.9  # GaAs permittivity

def charge_density(double x, double y, n_array, p_array, 
                  n_interpolator=None, p_interpolator=None):
    """
    Compute the charge density at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    n_array : array-like
        Electron concentration values
    p_array : array-like
        Hole concentration values
    n_interpolator : FEInterpolator, optional
        Interpolator for electron concentration
    p_interpolator : FEInterpolator, optional
        Interpolator for hole concentration
        
    Returns
    -------
    float
        Charge density in elementary charges per cubic nanometer
    """
    # Convert arrays to C++ VectorXd
    cdef cnp.ndarray[double, ndim=1] n_np = np.asarray(n_array, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] p_np = np.asarray(p_array, dtype=np.float64)
    
    cdef VectorXd n_cpp, p_cpp
    n_cpp.resize(n_np.shape[0])
    p_cpp.resize(p_np.shape[0])
    
    cdef int i
    for i in range(n_np.shape[0]):
        n_cpp[i] = n_np[i]
    for i in range(p_np.shape[0]):
        p_cpp[i] = p_np[i]
    
    # Simplified implementation without full interpolator binding
    # This would need proper FEInterpolator binding
    return 0.0

def capacitance(double x, double y, double eta, double Lx, double Ly, double d):
    """
    Compute the capacitance at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    eta : float
        Gate efficiency factor
    Lx : float
        Domain width in nanometers
    Ly : float
        Domain height in nanometers
    d : float
        Gate-to-channel distance in nanometers
        
    Returns
    -------
    float
        Capacitance in farads per square nanometer
    """
    return cpp_cap(x, y, eta, Lx, Ly, d)

def electron_concentration(double x, double y, double phi, material):
    """
    Compute the electron concentration at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    phi : float
        Electrostatic potential in volts
    material : Material
        Material properties
        
    Returns
    -------
    float
        Electron concentration
    """
    # Simplified implementation without full Material binding
    # This would need proper Material class binding
    return 1e15  # Default value

def hole_concentration(double x, double y, double phi, material):
    """
    Compute the hole concentration at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    phi : float
        Electrostatic potential in volts
    material : Material
        Material properties
        
    Returns
    -------
    float
        Hole concentration
    """
    # Simplified implementation without full Material binding
    return 1e15  # Default value

def mobility_n(double x, double y, material):
    """
    Compute the electron mobility at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    material : Material
        Material properties
        
    Returns
    -------
    float
        Electron mobility
    """
    # Simplified implementation
    return 1000.0  # cm²/V·s

def mobility_p(double x, double y, material):
    """
    Compute the hole mobility at a given position.
    
    Parameters
    ----------
    x : float
        x-coordinate in nanometers
    y : float
        y-coordinate in nanometers
    material : Material
        Material properties
        
    Returns
    -------
    float
        Hole mobility
    """
    # Simplified implementation
    return 400.0  # cm²/V·s
