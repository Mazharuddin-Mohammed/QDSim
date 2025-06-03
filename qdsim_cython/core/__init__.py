"""
Core module for QDSim Cython bindings

This module contains the core functionality including:
- Mesh generation and finite element methods
- Physics calculations and constants
- Material database and properties
- GPU acceleration interfaces
"""

from . import mesh, physics, materials

try:
    from . import gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = ['mesh', 'physics', 'materials']
if GPU_AVAILABLE:
    __all__.append('gpu')
