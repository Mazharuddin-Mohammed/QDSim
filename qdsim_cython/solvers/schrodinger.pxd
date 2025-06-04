# distutils: language = c++
# cython: language_level = 3

"""
Cython declaration file for SchrodingerSolver class

This file declares the C++ SchrodingerSolver class from the backend
that will be wrapped by Cython for high-performance quantum calculations.
"""

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as bint
from ..eigen cimport VectorXd, VectorXcd, MatrixXcd, SparseMatrixXcd
from ..core.mesh cimport Mesh

# Function pointer types for callbacks
ctypedef double (*mass_func_t)(double x, double y)
ctypedef double (*potential_func_t)(double x, double y)

# C++ SchrodingerSolver class declaration
cdef extern from "schrodinger.h":
    cdef cppclass SchrodingerSolver:
        # Constructors
        SchrodingerSolver(const Mesh& mesh, mass_func_t mass_func, 
                         potential_func_t potential_func, bint use_dirichlet) except +
        
        # Core solving methods
        void solve(int num_eigenvalues) except +
        void solve_with_tolerance(int num_eigenvalues, double tolerance) except +
        
        # Results access
        VectorXd get_eigenvalues() except +
        vector[VectorXd] get_eigenvectors() except +
        int get_num_eigenvalues() except +
        
        # Advanced methods
        void apply_open_system_boundary_conditions() except +
        void apply_conservative_boundary_conditions() except +
        void apply_minimal_cap_boundaries() except +
        void apply_dirac_delta_normalization() except +
        void configure_device_specific_solver(const string& device_type, double bias_voltage) except +
        
        # Matrix access for advanced users
        SparseMatrixXcd get_hamiltonian_matrix() except +
        SparseMatrixXcd get_mass_matrix() except +
        
        # Solver configuration
        void set_solver_tolerance(double tolerance) except +
        void set_max_iterations(int max_iter) except +
        void enable_cap_boundaries(bint enable) except +
        void set_cap_parameters(double strength, double thickness) except +
        
        # Energy filtering and validation
        void set_energy_filter_range(double min_energy, double max_energy) except +
        void enable_degeneracy_breaking(bint enable) except +
        
        # Performance and debugging
        double get_last_solve_time() except +
        int get_last_iteration_count() except +
        string get_solver_status() except +
        
        # Cleanup
        void clear_results() except +

# Helper functions for callback management
cdef extern from "callback_system.h":
    void register_mass_callback(int callback_id, mass_func_t func) except +
    void register_potential_callback(int callback_id, potential_func_t func) except +
    void clear_callbacks() except +
    int get_next_callback_id() except +

# Advanced solver configuration structures
cdef extern from "schrodinger.h":
    cdef struct SolverConfig:
        double tolerance
        int max_iterations
        bint use_cap_boundaries
        double cap_strength
        double cap_thickness
        bint enable_degeneracy_breaking
        double energy_filter_min
        double energy_filter_max
        string device_type
        double bias_voltage
    
    cdef struct SolverResults:
        VectorXd eigenvalues
        vector[VectorXd] eigenvectors
        double solve_time
        int iteration_count
        string status
        bint converged

# Device-specific optimization parameters
cdef extern from "schrodinger.h" namespace "DeviceOptimization":
    cdef struct NanowireParams:
        double cap_layer_fraction
        double absorption_strength
        double profile_exponent
        double asymmetry_factor
    
    cdef struct QuantumDotParams:
        double cap_layer_fraction
        double absorption_strength
        double profile_exponent
        double asymmetry_factor
    
    cdef struct WideChannelParams:
        double cap_layer_fraction
        double absorption_strength
        double profile_exponent
        double asymmetry_factor

# Physical constants for quantum calculations
cdef extern from "physical_constants.h" namespace "QuantumConstants":
    const double HBAR                    # Reduced Planck constant (J⋅s)
    const double ELECTRON_MASS           # Electron rest mass (kg)
    const double ELEMENTARY_CHARGE       # Elementary charge (C)
    const double VACUUM_PERMITTIVITY     # Vacuum permittivity (F/m)
    
    # Energy conversion factors
    const double JOULE_TO_EV            # Convert J to eV
    const double EV_TO_JOULE            # Convert eV to J
    const double HARTREE_TO_EV          # Convert Hartree to eV
    const double EV_TO_HARTREE          # Convert eV to Hartree
    
    # Length conversion factors
    const double BOHR_TO_METER          # Convert Bohr radius to meters
    const double METER_TO_BOHR          # Convert meters to Bohr radius
    const double ANGSTROM_TO_METER      # Convert Angstrom to meters
    const double METER_TO_ANGSTROM      # Convert meters to Angstrom

# Error handling for quantum calculations
cdef extern from "quantum_errors.h":
    cdef cppclass QuantumSolverError:
        QuantumSolverError(const string& message) except +
        const char* what() except +
    
    cdef cppclass ConvergenceError:
        ConvergenceError(const string& message) except +
        const char* what() except +
    
    cdef cppclass EigenvalueError:
        EigenvalueError(const string& message) except +
        const char* what() except +
