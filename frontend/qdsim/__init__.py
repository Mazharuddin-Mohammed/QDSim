import sys
import os

# Try to import the C++ extension
try:
    # First try to import from the backend build directory (correct location)
    backend_build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', 'build')
    local_dir = os.path.dirname(__file__)

    # Prioritize backend build directory over local directory
    if os.path.exists(backend_build_dir):
        if backend_build_dir not in sys.path:
            sys.path.insert(0, backend_build_dir)
        print(f"🔍 Added backend build directory to path: {backend_build_dir}")

    # Remove local directory from path temporarily to avoid conflicts
    if local_dir in sys.path:
        sys.path.remove(local_dir)

    # Add local directory at the end as fallback
    if local_dir not in sys.path:
        sys.path.append(local_dir)

    # Try to import the C++ module
    import qdsim_cpp
    print(f"✅ Successfully imported qdsim_cpp from: {qdsim_cpp.__file__ if hasattr(qdsim_cpp, '__file__') else 'built-in'}")

    # Make the C++ extension available to the rest of the package
    globals()['qdsim_cpp'] = qdsim_cpp

    # Import C++ classes (with fallbacks)
    print("📦 Mapping C++ classes to Python interface...")

    # Core classes that should be available
    try:
        Mesh = qdsim_cpp.Mesh
        PoissonSolver = qdsim_cpp.PoissonSolver
        FEMSolver = qdsim_cpp.FEMSolver
        EigenSolver = qdsim_cpp.EigenSolver
        print("✅ Core classes (Mesh, PoissonSolver, FEMSolver, EigenSolver) available")
    except AttributeError as e:
        print(f"❌ Error importing core classes: {e}")
        Mesh = None
        PoissonSolver = None
        FEMSolver = None
        EigenSolver = None

    # SchrodingerSolver with fallback
    try:
        SchrodingerSolver = qdsim_cpp.SchrodingerSolver
        print("✅ SchrodingerSolver available")
    except AttributeError:
        print("⚠️  SchrodingerSolver not available in C++ module. Using Python fallback.")
        try:
            from .python_schrodinger_solver import PythonSchrodingerSolver
            SchrodingerSolver = PythonSchrodingerSolver
            print("✅ Python SchrodingerSolver fallback loaded")
        except ImportError:
            print("❌ Python SchrodingerSolver fallback not available")
            SchrodingerSolver = None

    # Advanced solvers with fallbacks
    try:
        FullPoissonDriftDiffusionSolver = qdsim_cpp.FullPoissonDriftDiffusionSolver
        print("✅ FullPoissonDriftDiffusionSolver available")
    except AttributeError:
        print("⚠️  FullPoissonDriftDiffusionSolver not available in C++ module. Using Python fallback.")
        from .python_full_poisson_dd_solver import PythonFullPoissonDriftDiffusionSolver
        FullPoissonDriftDiffusionSolver = PythonFullPoissonDriftDiffusionSolver

    # Self-consistent solvers
    try:
        SelfConsistentSolver = qdsim_cpp.SelfConsistentSolver
        SimpleSelfConsistentSolver = qdsim_cpp.SimpleSelfConsistentSolver
        ImprovedSelfConsistentSolver = qdsim_cpp.ImprovedSelfConsistentSolver
        BasicSolver = qdsim_cpp.BasicSolver
        print("✅ Self-consistent solvers available")
    except AttributeError as e:
        print(f"⚠️  Some self-consistent solvers not available: {e}")
        SelfConsistentSolver = None
        SimpleSelfConsistentSolver = None
        ImprovedSelfConsistentSolver = None
        BasicSolver = None

    # GPU and advanced features
    try:
        GPUAccelerator = qdsim_cpp.GPUAccelerator
        AdaptiveMesh = qdsim_cpp.AdaptiveMesh
        print("✅ GPU and adaptive mesh features available")
    except AttributeError as e:
        print(f"⚠️  Advanced features not available: {e}")
        GPUAccelerator = None
        # AdaptiveMesh will be imported from adaptive_mesh.py as fallback

    # Materials and interpolation
    try:
        MaterialDatabase = qdsim_cpp.MaterialDatabase
        Material = qdsim_cpp.Material
        FEInterpolator = qdsim_cpp.FEInterpolator
        SimpleInterpolator = qdsim_cpp.SimpleInterpolator
        SimpleMesh = qdsim_cpp.SimpleMesh
        print("✅ Materials and interpolation classes available")
    except AttributeError as e:
        print(f"⚠️  Some materials/interpolation classes not available: {e}")
        MaterialDatabase = None
        Material = None
        FEInterpolator = None
        SimpleInterpolator = None
        SimpleMesh = None

    # P-N Junction modeling
    try:
        PNJunction = qdsim_cpp.PNJunction
        print("✅ P-N Junction modeling available")
    except AttributeError:
        print("⚠️  P-N Junction modeling not available")
        PNJunction = None

    # Make Materials namespace available
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
    print(f"❌ Warning: Could not import C++ extension: {e}")
    print("🔄 Falling back to Python implementations where available...")
    qdsim_cpp = None

    # Set all C++ classes to None - fallbacks will be imported later
    Mesh = None
    PoissonSolver = None
    FEMSolver = None
    EigenSolver = None
    SchrodingerSolver = None
    FullPoissonDriftDiffusionSolver = None
    SelfConsistentSolver = None
    SimpleSelfConsistentSolver = None
    ImprovedSelfConsistentSolver = None
    BasicSolver = None
    GPUAccelerator = None
    MaterialDatabase = None
    Material = None
    FEInterpolator = None
    SimpleInterpolator = None
    SimpleMesh = None
    PNJunction = None
    Materials = None

    # Import Python fallbacks
    try:
        from .python_full_poisson_dd_solver import PythonFullPoissonDriftDiffusionSolver
        FullPoissonDriftDiffusionSolver = PythonFullPoissonDriftDiffusionSolver
        print("✅ Python FullPoissonDriftDiffusionSolver fallback loaded")
    except ImportError:
        print("⚠️  Python FullPoissonDriftDiffusionSolver fallback not available")

    try:
        from .python_schrodinger_solver import PythonSchrodingerSolver
        SchrodingerSolver = PythonSchrodingerSolver
        print("✅ Python SchrodingerSolver fallback loaded")
    except ImportError:
        print("⚠️  Python SchrodingerSolver fallback not available")

    try:
        from .python_poisson_solver import PythonPoissonSolver
        PoissonSolver = PythonPoissonSolver
        print("✅ Python PoissonSolver fallback loaded")
    except ImportError:
        print("⚠️  Python PoissonSolver fallback not available")

from .config import Config
from .simulator import Simulator

# Optional visualization imports
try:
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
    VISUALIZATION_AVAILABLE = True
    print("✅ Visualization modules loaded")
except ImportError as e:
    print(f"⚠️  Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False
    # Create dummy functions
    def plot_wavefunction(*args, **kwargs):
        print("⚠️  Visualization not available - install matplotlib to enable plotting")
    def plot_electric_field(*args, **kwargs):
        print("⚠️  Visualization not available - install matplotlib to enable plotting")
    def plot_potential(*args, **kwargs):
        print("⚠️  Visualization not available - install matplotlib to enable plotting")
    def plot_energy_shift(*args, **kwargs):
        print("⚠️  Visualization not available - install matplotlib to enable plotting")
    def show_interactive_visualization(*args, **kwargs):
        print("⚠️  Interactive visualization not available - install matplotlib to enable plotting")
    def plot_enhanced_wavefunction_3d(*args, **kwargs):
        print("⚠️  Enhanced visualization not available - install matplotlib to enable plotting")
    def plot_enhanced_potential_3d(*args, **kwargs):
        print("⚠️  Enhanced visualization not available - install matplotlib to enable plotting")
    def plot_combined_visualization(*args, **kwargs):
        print("⚠️  Combined visualization not available - install matplotlib to enable plotting")
    def create_energy_level_diagram(*args, **kwargs):
        print("⚠️  Energy level diagram not available - install matplotlib to enable plotting")
    def calculate_transition_probabilities(*args, **kwargs):
        print("⚠️  Transition probabilities calculation not available - install matplotlib to enable plotting")
        return None
    def plot_transition_matrix(*args, **kwargs):
        print("⚠️  Transition matrix plot not available - install matplotlib to enable plotting")
    def show_enhanced_visualization(*args, **kwargs):
        print("⚠️  Enhanced visualization not available - install matplotlib to enable plotting")
# Optional result analysis imports
try:
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
    RESULT_ANALYSIS_AVAILABLE = True
    print("✅ Result analysis modules loaded")
except ImportError as e:
    print(f"⚠️  Result analysis modules not available: {e}")
    RESULT_ANALYSIS_AVAILABLE = False
    # Create dummy functions
    def extract_energy_levels(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return []
    def calculate_energy_spacing(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return []
    def calculate_transition_matrix(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return None
    def calculate_dipole_matrix(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return None
    def calculate_oscillator_strengths(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return []
    def analyze_wavefunction_localization(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return {}
    def fit_energy_levels(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return {}
    def create_energy_level_report(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return ""
    def create_transition_probability_report(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return ""
    def create_wavefunction_localization_report(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")
        return ""
    def export_results_to_csv(*args, **kwargs):
        print("⚠️  Result analysis not available - install matplotlib to enable analysis")