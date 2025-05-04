#pragma once
#include "mesh.h"
#include "poisson.h"
#include "materials.h"
#include <Eigen/Sparse>
#include <iostream>

// Make MPI optional
#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @class SelfConsistentSolver
 * @brief Solver for self-consistent Poisson-drift-diffusion equations.
 *
 * This class implements a solver for the self-consistent Poisson-drift-diffusion
 * equations used in semiconductor device simulations. It couples the Poisson equation
 * for the electrostatic potential with the drift-diffusion equations for carrier
 * transport.
 */
class SelfConsistentSolver {
public:
    /**
     * @brief Constructs a new SelfConsistentSolver object.
     *
     * @param mesh The mesh to use for the simulation
     * @param epsilon_r Function that returns the relative permittivity at a given position
     * @param rho Function that returns the charge density at a given position
     * @param n_conc Function that returns the electron concentration at a given position
     * @param p_conc Function that returns the hole concentration at a given position
     * @param mu_n Function that returns the electron mobility at a given position
     * @param mu_p Function that returns the hole mobility at a given position
     */
    SelfConsistentSolver(Mesh& mesh,
                         double (*epsilon_r)(double, double),
                         double (*rho)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&),
                         double (*n_conc)(double, double, double, const Materials::Material&),
                         double (*p_conc)(double, double, double, const Materials::Material&),
                         double (*mu_n)(double, double, const Materials::Material&),
                         double (*mu_p)(double, double, const Materials::Material&));

    /**
     * @brief Solves the self-consistent Poisson-drift-diffusion equations.
     *
     * This function solves the self-consistent Poisson-drift-diffusion equations
     * using an iterative approach. It alternates between solving the Poisson equation
     * and the drift-diffusion equations until convergence is reached or the maximum
     * number of iterations is exceeded.
     *
     * @param V_p The voltage applied to the p-contact
     * @param V_n The voltage applied to the n-contact
     * @param N_A The acceptor doping concentration
     * @param N_D The donor doping concentration
     * @param tolerance The convergence tolerance (default: 1e-6)
     * @param max_iter The maximum number of iterations (default: 100)
     */
    void solve(double V_p, double V_n, double N_A, double N_D, double tolerance = 1e-6, int max_iter = 100);

    /**
     * @brief Gets the computed electrostatic potential.
     *
     * @return The electrostatic potential vector
     */
    const Eigen::VectorXd& get_potential() const { return poisson.get_potential(); }

    /**
     * @brief Gets the computed electron concentration.
     *
     * @return The electron concentration vector
     */
    const Eigen::VectorXd& get_n() const { return n; }

    /**
     * @brief Gets the computed hole concentration.
     *
     * @return The hole concentration vector
     */
    const Eigen::VectorXd& get_p() const { return p; }

    /**
     * @brief Gets the electric field at a given position.
     *
     * @param x The x-coordinate of the position
     * @param y The y-coordinate of the position
     * @return The electric field vector at the given position
     */
    Eigen::Vector2d get_electric_field(double x, double y) const { return poisson.get_electric_field(x, y); }

public:
    // Convergence acceleration parameters
    double damping_factor;               ///< Damping factor for potential updates (0 < damping_factor <= 1)
    int anderson_history_size;           ///< Number of previous iterations to use for Anderson acceleration

private:
    Mesh& mesh;                          ///< Reference to the mesh used for the simulation
    PoissonSolver poisson;               ///< Poisson solver for the electrostatic potential
    Eigen::VectorXd n, p;                ///< Carrier concentrations (electrons and holes)
    Eigen::SparseMatrix<double> Kn, Kp;  ///< Drift-diffusion matrices for electrons and holes

    std::vector<Eigen::VectorXd> phi_history;  ///< History of potential vectors for Anderson acceleration
    std::vector<Eigen::VectorXd> res_history;  ///< History of residual vectors for Anderson acceleration

    // Function pointers for physical quantities
    double (*epsilon_r)(double, double);  ///< Function for relative permittivity
    double (*rho)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&);  ///< Function for charge density
    double (*n_conc)(double, double, double, const Materials::Material&);  ///< Function for electron concentration
    double (*p_conc)(double, double, double, const Materials::Material&);  ///< Function for hole concentration
    double (*mu_n)(double, double, const Materials::Material&);  ///< Function for electron mobility
    double (*mu_p)(double, double, const Materials::Material&);  ///< Function for hole mobility

    /**
     * @brief Apply damping to the potential update.
     *
     * This method applies damping to the potential update to improve convergence.
     * It uses a simple linear damping scheme: phi_new = phi_old + damping_factor * (phi_update - phi_old)
     *
     * @param phi_old The previous potential vector
     * @param phi_update The updated potential vector
     * @return The damped potential vector
     */
    Eigen::VectorXd apply_damping(const Eigen::VectorXd& phi_old, const Eigen::VectorXd& phi_update) const;

    /**
     * @brief Apply Anderson acceleration to the potential update.
     *
     * This method applies Anderson acceleration to the potential update to improve convergence.
     * It uses the history of potential and residual vectors to compute an optimal combination
     * of previous iterations.
     *
     * @param phi_old The previous potential vector
     * @param phi_update The updated potential vector
     * @return The accelerated potential vector
     */
    Eigen::VectorXd apply_anderson_acceleration(const Eigen::VectorXd& phi_old, const Eigen::VectorXd& phi_update);

    /**
     * @brief Initializes the carrier concentrations based on the doping concentrations.
     *
     * @param N_A The acceptor doping concentration
     * @param N_D The donor doping concentration
     */
    void initialize_carriers(double N_A, double N_D);

    /**
     * @brief Assembles the drift-diffusion matrices for electron and hole transport.
     */
    void assemble_drift_diffusion_matrices();

    /**
     * @brief Solves the drift-diffusion equations for electron and hole transport.
     */
    void solve_drift_diffusion();

    /**
     * @brief Applies boundary conditions to the carrier concentrations.
     *
     * @param V_p The voltage applied to the p-contact
     * @param V_n The voltage applied to the n-contact
     */
    void apply_boundary_conditions(double V_p, double V_n);
};