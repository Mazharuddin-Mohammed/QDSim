# distutils: language = c++
# cython: language_level = 3

"""
Cython declaration file for Physics functions and constants

This file declares the C++ physics functions and constants from the backend
that will be wrapped by Cython.
"""

from libcpp.string cimport string
from ..eigen cimport VectorXd

# Forward declaration for Materials
cdef extern from "materials.h" namespace "Materials":
    cdef cppclass Material:
        pass

# Forward declaration for FEInterpolator
cdef extern from "fe_interpolator.h":
    cdef cppclass FEInterpolator:
        pass

# Physics functions
cdef extern from "physics.h" namespace "Physics":
    # Effective mass function
    double effective_mass(double x, double y, 
                         const Material& qd_mat,
                         const Material& matrix_mat, 
                         double R) except +
    
    # Potential function
    double potential(double x, double y,
                    const Material& qd_mat,
                    const Material& matrix_mat,
                    double R, const string& type,
                    const VectorXd& phi,
                    const FEInterpolator* interpolator) except +
    
    # Relative permittivity function
    double epsilon_r(double x, double y,
                    const Material& p_mat,
                    const Material& n_mat) except +
    
    # Charge density function
    double charge_density(double x, double y,
                         const VectorXd& n, const VectorXd& p,
                         const FEInterpolator* n_interpolator,
                         const FEInterpolator* p_interpolator) except +
    
    # Capacitance function
    double cap(double x, double y, double eta, 
              double Lx, double Ly, double d) except +
    
    # Carrier concentration functions
    double electron_concentration(double x, double y, double phi,
                                const Material& mat) except +
    double hole_concentration(double x, double y, double phi,
                            const Material& mat) except +
    
    # Mobility functions
    double mobility_n(double x, double y, const Material& mat) except +
    double mobility_p(double x, double y, const Material& mat) except +

# Physical constants
cdef extern from "physical_constants.h" namespace "PhysicalConstants":
    # Fundamental constants
    const double ELEMENTARY_CHARGE
    const double PLANCK_CONSTANT
    const double REDUCED_PLANCK_CONSTANT
    const double BOLTZMANN_CONSTANT
    const double ELECTRON_MASS
    const double VACUUM_PERMITTIVITY
    const double SPEED_OF_LIGHT
    
    # Unit conversions
    const double EV_TO_JOULE
    const double JOULE_TO_EV
    const double NM_TO_METER
    const double METER_TO_NM
    
    # Typical parameter ranges
    const double MIN_EFFECTIVE_MASS
    const double MAX_EFFECTIVE_MASS
    const double MIN_POTENTIAL
    const double MAX_POTENTIAL
    const double MIN_PERMITTIVITY
    const double MAX_PERMITTIVITY
