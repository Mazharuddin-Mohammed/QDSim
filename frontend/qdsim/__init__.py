import sys
import os

# Try to import the C++ extension
try:
    # Try to import from the build directory
    build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'build')
    if os.path.exists(build_dir):
        sys.path.append(build_dir)

    # Try to import the C++ module
    import qdsim_cpp

    # Make the C++ extension available to the rest of the package
    globals()['qdsim_cpp'] = qdsim_cpp

    # Import C++ classes
    Mesh = qdsim_cpp.Mesh
    PoissonSolver = qdsim_cpp.PoissonSolver
    FEMSolver = qdsim_cpp.FEMSolver
    EigenSolver = qdsim_cpp.EigenSolver
    # Try to import FullPoissonDriftDiffusionSolver if available
    try:
        FullPoissonDriftDiffusionSolver = qdsim_cpp.FullPoissonDriftDiffusionSolver
    except AttributeError:
        print("Warning: FullPoissonDriftDiffusionSolver not available in C++ module. Using Python fallback.")
        from .python_full_poisson_dd_solver import PythonFullPoissonDriftDiffusionSolver
        FullPoissonDriftDiffusionSolver = PythonFullPoissonDriftDiffusionSolver
    # AdaptiveMesh is imported from adaptive_mesh.py
    # SchrodingerSolver is accessed through create_schrodinger_solver
    try:
        SchrodingerSolver = qdsim_cpp.SchrodingerSolver
    except AttributeError:
        print("Warning: SchrodingerSolver not available in C++ module. Using Python fallback.")
        from .python_schrodinger_solver import PythonSchrodingerSolver
        SchrodingerSolver = PythonSchrodingerSolver

    Materials = qdsim_cpp

    # Create a function to create a SchrodingerSolver
    def create_schrodinger_solver(mesh, potential_function, m_eff=0.067, use_gpu=False):
        """
        Create a SchrodingerSolver.

        Args:
            mesh: The mesh to use for the simulation
            potential_function: Function that returns the potential at a given position
            m_eff: Effective mass (in units of m0)
            use_gpu: Whether to use GPU acceleration

        Returns:
            A SchrodingerSolver object
        """
        try:
            return qdsim_cpp.create_schrodinger_solver(mesh, lambda x, y: m_eff, potential_function, use_gpu)
        except (AttributeError, TypeError):
            # Fallback to a Python implementation
            print("Warning: SchrodingerSolver C++ implementation not available. Using Python fallback.")
            return SchrodingerSolver(mesh, potential_function)

except ImportError as e:
    print(f"Warning: Could not import C++ extension: {e}. Make sure it's built and in the Python path.")
    qdsim_cpp = None
    Mesh = None
    PoissonSolver = None
    FEMSolver = None
    EigenSolver = None
    FullPoissonDriftDiffusionSolver = None
    # AdaptiveMesh is imported from adaptive_mesh.py
    SchrodingerSolver = None
    Materials = None

from .config import Config
from .simulator import Simulator
from .visualization import plot_wavefunction, plot_electric_field, plot_potential, plot_energy_shift
from .interactive_viz import show_interactive_visualization
from .enhanced_visualization import (
    plot_enhanced_wavefunction_3d,
    plot_enhanced_potential_3d,
    plot_combined_visualization,
    create_energy_level_diagram,
    calculate_transition_probabilities,
    plot_transition_matrix
)
from .interactive_controls import show_enhanced_visualization
from .result_analysis import (
    extract_energy_levels,
    calculate_energy_spacing,
    calculate_transition_matrix,
    calculate_dipole_matrix,
    calculate_oscillator_strengths,
    analyze_wavefunction_localization,
    fit_energy_levels,
    create_energy_level_report,
    create_transition_probability_report,
    create_wavefunction_localization_report,
    export_results_to_csv
)