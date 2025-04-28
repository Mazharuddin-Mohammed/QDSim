import sys
import os

# Try to import the C++ extension
try:
    # Try to import from the build directory
    build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'build')
    if os.path.exists(build_dir):
        sys.path.append(build_dir)
        import qdsim_cpp
        # Make the C++ extension available to the rest of the package
        globals()['qdsim_cpp'] = qdsim_cpp
except ImportError as e:
    print(f"Warning: Could not import C++ extension: {e}. Make sure it's built and in the Python path.")
    qdsim_cpp = None

from .config import Config
from .simulator import Simulator
from .visualization import plot_wavefunction, plot_electric_field, plot_potential, plot_energy_shift