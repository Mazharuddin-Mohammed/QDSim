"""
QDSim Cython Solvers Package

This package contains high-performance Cython implementations of all core
solvers, replacing the C++ backend with optimized Cython code.

Available Solvers:
- PoissonSolver: Electrostatic potential calculations
- SchrodingerSolver: Quantum mechanical eigenvalue problems
- FEMSolver: General finite element method capabilities
- EigenSolver: Eigenvalue problem solvers
- SelfConsistentSolver: Coupled Poisson-Schrödinger solver
"""

# Import all solver modules when available
try:
    from .poisson_solver import CythonPoissonSolver, create_poisson_solver
    POISSON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Poisson solver not available: {e}")
    POISSON_AVAILABLE = False

try:
    from .schrodinger_solver import CythonSchrodingerSolver, create_schrodinger_solver
    SCHRODINGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Schrödinger solver not available: {e}")
    SCHRODINGER_AVAILABLE = False

try:
    from .fem_solver import CythonFEMSolver, create_fem_solver
    FEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FEM solver not available: {e}")
    FEM_AVAILABLE = False

try:
    from .eigen_solver import CythonEigenSolver, create_eigen_solver
    EIGEN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Eigen solver not available: {e}")
    EIGEN_AVAILABLE = False

try:
    from .self_consistent_solver import CythonSelfConsistentSolver, create_self_consistent_solver
    SELF_CONSISTENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Self-consistent solver not available: {e}")
    SELF_CONSISTENT_AVAILABLE = False

# Provide convenient aliases
if POISSON_AVAILABLE:
    PoissonSolver = CythonPoissonSolver

if SCHRODINGER_AVAILABLE:
    SchrodingerSolver = CythonSchrodingerSolver

if FEM_AVAILABLE:
    FEMSolver = CythonFEMSolver

if EIGEN_AVAILABLE:
    EigenSolver = CythonEigenSolver

if SELF_CONSISTENT_AVAILABLE:
    SelfConsistentSolver = CythonSelfConsistentSolver

# Define what's available for import
__all__ = []

if POISSON_AVAILABLE:
    __all__.extend(['CythonPoissonSolver', 'PoissonSolver', 'create_poisson_solver'])

if SCHRODINGER_AVAILABLE:
    __all__.extend(['CythonSchrodingerSolver', 'SchrodingerSolver', 'create_schrodinger_solver'])

if FEM_AVAILABLE:
    __all__.extend(['CythonFEMSolver', 'FEMSolver', 'create_fem_solver'])

if EIGEN_AVAILABLE:
    __all__.extend(['CythonEigenSolver', 'EigenSolver', 'create_eigen_solver'])

if SELF_CONSISTENT_AVAILABLE:
    __all__.extend(['CythonSelfConsistentSolver', 'SelfConsistentSolver', 'create_self_consistent_solver'])

def get_available_solvers():
    """
    Get list of available solvers.

    Returns:
    --------
    dict
        Dictionary of solver availability
    """
    return {
        'poisson': POISSON_AVAILABLE,
        'schrodinger': SCHRODINGER_AVAILABLE,
        'fem': FEM_AVAILABLE,
        'eigen': EIGEN_AVAILABLE,
        'self_consistent': SELF_CONSISTENT_AVAILABLE
    }

def test_all_solvers():
    """
    Test all available solvers.

    Returns:
    --------
    dict
        Test results for each solver
    """
    results = {}

    if POISSON_AVAILABLE:
        try:
            from .poisson_solver import test_poisson_solver
            results['poisson'] = test_poisson_solver()
        except Exception as e:
            results['poisson'] = False
            print(f"Poisson solver test failed: {e}")

    if SCHRODINGER_AVAILABLE:
        try:
            from .schrodinger_solver import test_schrodinger_solver
            results['schrodinger'] = test_schrodinger_solver()
        except Exception as e:
            results['schrodinger'] = False
            print(f"Schrödinger solver test failed: {e}")

    if FEM_AVAILABLE:
        try:
            from .fem_solver import test_fem_solver
            results['fem'] = test_fem_solver()
        except Exception as e:
            results['fem'] = False
            print(f"FEM solver test failed: {e}")

    if EIGEN_AVAILABLE:
        try:
            from .eigen_solver import test_eigen_solver
            results['eigen'] = test_eigen_solver()
        except Exception as e:
            results['eigen'] = False
            print(f"Eigen solver test failed: {e}")

    if SELF_CONSISTENT_AVAILABLE:
        try:
            from .self_consistent_solver import test_self_consistent_solver
            results['self_consistent'] = test_self_consistent_solver()
        except Exception as e:
            results['self_consistent'] = False
            print(f"Self-consistent solver test failed: {e}")

    return results

def print_solver_status():
    """Print status of all solvers"""
    print("🔧 QDSim Cython Solvers Status:")
    print("=" * 50)

    solvers = get_available_solvers()
    for name, available in solvers.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"   {name.title()} Solver: {status}")

    total_available = sum(solvers.values())
    total_solvers = len(solvers)
    print(f"\nTotal: {total_available}/{total_solvers} solvers available")

    if total_available == total_solvers:
        print("🎉 All Cython solvers are available!")
    elif total_available > 0:
        print("✅ Some Cython solvers are available")
    else:
        print("❌ No Cython solvers are available")
