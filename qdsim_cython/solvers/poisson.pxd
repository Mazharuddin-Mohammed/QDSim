# distutils: language = c++
# cython: language_level = 3

"""
Cython declaration file for PoissonSolver class

This file declares the C++ PoissonSolver class from the backend
that will be wrapped by Cython.
"""

from ..eigen cimport VectorXd, Vector2d, SparseMatrixXd
from ..core.mesh cimport Mesh

# Function pointer types for callbacks
ctypedef double (*epsilon_r_func_t)(double x, double y)
ctypedef double (*rho_func_t)(double x, double y, const VectorXd& n, const VectorXd& p)

# PoissonSolver class declaration
cdef extern from "poisson.h":
    cdef cppclass PoissonSolver:
        # Constructors
        PoissonSolver(Mesh& mesh, 
                     epsilon_r_func_t epsilon_r,
                     rho_func_t rho) except +
        
        # Solver methods
        void solve(double V_p, double V_n) except +
        void solve(double V_p, double V_n, 
                  const VectorXd& n, const VectorXd& p) except +
        
        # Potential management
        void set_potential(const VectorXd& potential) except +
        void update_and_solve(const VectorXd& potential, double V_p, double V_n,
                             const VectorXd& n, const VectorXd& p) except +
        
        # Initialization and configuration
        void initialize(Mesh& mesh, 
                       epsilon_r_func_t epsilon_r,
                       rho_func_t rho) except +
        void set_charge_density(const VectorXd& charge_density) except +
        
        # Results access
        const VectorXd& get_potential() const
        Vector2d get_electric_field(double x, double y) const except +
        
        # Public member (potential vector)
        VectorXd phi
