"""
QDSim Cython - High-performance quantum dot simulation library

This package provides Cython bindings for the QDSim C++ backend,
offering improved performance and memory management compared to
the previous pybind11-based implementation.

Modules:
    core: Core functionality (mesh, physics, materials)
    solvers: Numerical solvers (Poisson, Schrödinger, self-consistent)
    utils: Utility functions (interpolation, visualization helpers)
"""

__version__ = "2.0.0"
__author__ = "Dr. Mazharuddin Mohammed"
__email__ = "mazharuddin.mohammed.official@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/your-username/QDSim"
__docs__ = "https://qdsimx.readthedocs.io"

# Import core modules
try:
    from .core import mesh, physics, materials
    from .solvers import poisson, schrodinger, self_consistent
    from .utils import interpolation
    
    # Make key classes available at package level
    from .core.mesh import Mesh, FEMSolver
    from .core.physics import PhysicsConstants, MaterialProperties
    from .core.materials import MaterialDatabase, Material
    from .solvers.poisson import PoissonSolver
    from .solvers.schrodinger import SchrodingerSolver
    from .solvers.self_consistent import SelfConsistentSolver
    from .utils.interpolation import FEInterpolator
    
    __all__ = [
        'Mesh', 'FEMSolver', 'PhysicsConstants', 'MaterialProperties',
        'MaterialDatabase', 'Material', 'PoissonSolver', 'SchrodingerSolver',
        'SelfConsistentSolver', 'FEInterpolator'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import Cython modules: {e}. "
                  "Make sure the package is properly compiled.", ImportWarning)
    
    # Provide fallback imports if needed
    __all__ = []
