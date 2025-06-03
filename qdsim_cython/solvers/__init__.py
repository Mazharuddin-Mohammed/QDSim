"""
Solvers module for QDSim Cython bindings

This module contains numerical solvers for:
- Poisson equation (electrostatic potential)
- Schrödinger equation (quantum mechanics)
- Self-consistent Poisson-drift-diffusion
"""

from . import poisson, schrodinger, self_consistent

__all__ = ['poisson', 'schrodinger', 'self_consistent']
