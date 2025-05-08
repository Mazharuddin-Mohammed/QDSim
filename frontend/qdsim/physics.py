"""
Physics module for QDSim.

This module provides physical constants and functions for quantum dot simulations,
including Fermi-Dirac statistics, density of states calculations, and carrier
concentration calculations.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np

# Physical constants
ELEMENTARY_CHARGE = 1.602176634e-19  # C
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
PLANCK_CONSTANT = 6.62607015e-34  # J.s
REDUCED_PLANCK_CONSTANT = PLANCK_CONSTANT / (2 * np.pi)  # J.s
ELECTRON_MASS = 9.1093837015e-31  # kg
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
TEMPERATURE = 300  # K
THERMAL_VOLTAGE = BOLTZMANN_CONSTANT * TEMPERATURE / ELEMENTARY_CHARGE  # V

def fermi_dirac(E, E_F, T=TEMPERATURE):
    """
    Fermi-Dirac distribution function.

    Args:
        E: Energy (eV)
        E_F: Fermi energy (eV)
        T: Temperature (K)

    Returns:
        Fermi-Dirac distribution value
    """
    kT = BOLTZMANN_CONSTANT * T / ELEMENTARY_CHARGE  # eV
    return 1.0 / (1.0 + np.exp((E - E_F) / kT))

def density_of_states_3d(E, E_c, m_eff):
    """
    3D density of states.

    Args:
        E: Energy (eV)
        E_c: Conduction band edge (eV)
        m_eff: Effective mass (m0)

    Returns:
        Density of states (1/eV/m^3)
    """
    if E < E_c:
        return 0.0

    m = m_eff * ELECTRON_MASS
    hbar = REDUCED_PLANCK_CONSTANT

    # Convert eV to J
    E_J = E * ELEMENTARY_CHARGE
    E_c_J = E_c * ELEMENTARY_CHARGE

    # Calculate DOS
    prefactor = 1.0 / (2.0 * np.pi**2) * (2.0 * m / hbar**2)**(3.0/2.0)
    dos = prefactor * np.sqrt(E_J - E_c_J)

    # Convert to 1/eV/m^3
    return dos / ELEMENTARY_CHARGE

def carrier_concentration(E_F, E_c, m_eff, T=TEMPERATURE):
    """
    Calculate carrier concentration using Fermi-Dirac statistics.

    Args:
        E_F: Fermi energy (eV)
        E_c: Conduction band edge (eV)
        m_eff: Effective mass (m0)
        T: Temperature (K)

    Returns:
        Carrier concentration (1/m^3)
    """
    kT = BOLTZMANN_CONSTANT * T / ELEMENTARY_CHARGE  # eV

    # For non-degenerate semiconductors, use Boltzmann approximation
    if E_F < E_c - 3 * kT:
        m = m_eff * ELECTRON_MASS
        hbar = REDUCED_PLANCK_CONSTANT

        # Effective density of states
        N_c = 2.0 * (m * kT * ELEMENTARY_CHARGE / (2.0 * np.pi * hbar**2))**(3.0/2.0)

        # Boltzmann approximation
        return N_c * np.exp((E_F - E_c) / kT)

    # For degenerate semiconductors, use numerical integration
    # (simplified for this example)
    return 1e24  # Placeholder value
