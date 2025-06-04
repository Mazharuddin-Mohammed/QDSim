# distutils: language = c++
# cython: language_level = 3

"""
Cython declaration file for FEInterpolator class

High-performance finite element interpolation for quantum wavefunctions
and physical fields in semiconductor devices.
"""

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as bint
from ..eigen cimport VectorXd, VectorXcd
from .mesh cimport Mesh

# C++ FEInterpolator class declaration
cdef extern from "fe_interpolator.h":
    cdef cppclass FEInterpolator:
        # Constructors
        FEInterpolator(const Mesh& mesh) except +
        
        # Core interpolation methods
        double interpolate(double x, double y, const VectorXd& values) except +
        double interpolate_complex_real(double x, double y, const VectorXcd& values) except +
        double interpolate_complex_imag(double x, double y, const VectorXcd& values) except +
        double interpolate_complex_abs(double x, double y, const VectorXcd& values) except +
        
        # Vector interpolation
        vector[double] interpolate_vector(const vector[double]& x_points,
                                        const vector[double]& y_points,
                                        const VectorXd& values) except +
        
        # Gradient computation
        vector[double] interpolate_gradient(double x, double y, const VectorXd& values) except +
        vector[double] interpolate_gradient_x(double x, double y, const VectorXd& values) except +
        vector[double] interpolate_gradient_y(double x, double y, const VectorXd& values) except +
        
        # Advanced interpolation methods
        double interpolate_with_derivatives(double x, double y, const VectorXd& values,
                                          double& dx, double& dy) except +
        
        # Mesh queries
        bint is_inside_domain(double x, double y) except +
        int find_containing_element(double x, double y) except +
        vector[int] find_nearest_nodes(double x, double y, int num_nodes) except +
        
        # Interpolation quality assessment
        double estimate_interpolation_error(double x, double y, const VectorXd& values) except +
        double get_local_mesh_size(double x, double y) except +
        
        # Performance optimization
        void precompute_interpolation_data() except +
        void enable_caching(bint enable) except +
        void clear_cache() except +
        
        # Boundary handling
        void set_boundary_extrapolation(bint enable) except +
        void set_boundary_value(double value) except +
        
        # Statistics and debugging
        int get_interpolation_count() except +
        double get_average_interpolation_time() except +
        string get_interpolation_statistics() except +

# Advanced interpolation algorithms
cdef extern from "fe_interpolator.h" namespace "InterpolationMethod":
    cdef enum InterpolationMethod:
        LINEAR_LAGRANGE = 0
        QUADRATIC_LAGRANGE = 1
        CUBIC_HERMITE = 2
        SPLINE_INTERPOLATION = 3
        RADIAL_BASIS_FUNCTION = 4

# Interpolation configuration
cdef extern from "fe_interpolator.h":
    cdef struct InterpolationConfig:
        InterpolationMethod method
        bint enable_caching
        bint enable_extrapolation
        double boundary_value
        double error_tolerance
        int max_search_iterations

# Error handling for interpolation
cdef extern from "interpolation_errors.h":
    cdef cppclass InterpolationError:
        InterpolationError(const string& message) except +
        const char* what() except +
    
    cdef cppclass OutOfDomainError:
        OutOfDomainError(const string& message) except +
        const char* what() except +
    
    cdef cppclass ConvergenceError:
        ConvergenceError(const string& message) except +
        const char* what() except +

# Specialized interpolators for different data types
cdef extern from "specialized_interpolators.h":
    cdef cppclass WavefunctionInterpolator:
        WavefunctionInterpolator(const Mesh& mesh) except +
        double interpolate_probability_density(double x, double y, const VectorXcd& psi) except +
        double interpolate_current_density_x(double x, double y, const VectorXcd& psi) except +
        double interpolate_current_density_y(double x, double y, const VectorXcd& psi) except +
        vector[double] interpolate_current_density(double x, double y, const VectorXcd& psi) except +
    
    cdef cppclass PotentialInterpolator:
        PotentialInterpolator(const Mesh& mesh) except +
        double interpolate_electric_field_x(double x, double y, const VectorXd& potential) except +
        double interpolate_electric_field_y(double x, double y, const VectorXd& potential) except +
        vector[double] interpolate_electric_field(double x, double y, const VectorXd& potential) except +
        double interpolate_charge_density(double x, double y, const VectorXd& potential) except +
    
    cdef cppclass MaterialInterpolator:
        MaterialInterpolator(const Mesh& mesh) except +
        double interpolate_effective_mass(double x, double y, const VectorXd& mass_values) except +
        double interpolate_dielectric_constant(double x, double y, const VectorXd& epsilon_values) except +
        double interpolate_band_gap(double x, double y, const VectorXd& bandgap_values) except +

# Performance monitoring
cdef extern from "interpolation_profiler.h":
    cdef cppclass InterpolationProfiler:
        InterpolationProfiler() except +
        void start_timing(const string& operation_name) except +
        void end_timing(const string& operation_name) except +
        double get_timing(const string& operation_name) except +
        void print_timing_report() except +
        void reset_timings() except +
        
        # Memory usage tracking
        size_t get_memory_usage() except +
        size_t get_peak_memory_usage() except +
        void reset_memory_tracking() except +

# Parallel interpolation support
cdef extern from "parallel_interpolation.h":
    cdef cppclass ParallelInterpolator:
        ParallelInterpolator(const Mesh& mesh, int num_threads) except +
        vector[double] interpolate_parallel(const vector[double]& x_points,
                                          const vector[double]& y_points,
                                          const VectorXd& values) except +
        void set_num_threads(int num_threads) except +
        int get_num_threads() except +
