#pragma once
/**
 * @file self_consistent.h
 * @brief Defines the SelfConsistentSolver class for semiconductor device simulations.
 *
 * This file contains the declaration of the SelfConsistentSolver class, which implements
 * a solver for the self-consistent Poisson-drift-diffusion equations used in semiconductor
 * device simulations. It couples the Poisson equation for the electrostatic potential
 * with the drift-diffusion equations for carrier transport.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

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
    /**
     * @brief Default constructor for SelfConsistentSolver.
     *
     * @param mesh Reference to the mesh used for the simulation
     */
    SelfConsistentSolver(Mesh& mesh);

    /**
     * @brief Constructor for SelfConsistentSolver with callback functions.
     *
     * @param mesh Reference to the mesh used for the simulation
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
     * @brief Gets the computed quasi-Fermi potential for electrons.
     *
     * @return The quasi-Fermi potential vector for electrons
     */
    const Eigen::VectorXd& get_phi_n() const { return phi_n; }

    /**
     * @brief Gets the computed quasi-Fermi potential for holes.
     *
     * @return The quasi-Fermi potential vector for holes
     */
    const Eigen::VectorXd& get_phi_p() const { return phi_p; }

    /**
     * @brief Calculates the built-in potential of the P-N junction with temperature dependence.
     *
     * This method calculates the built-in potential of the P-N junction
     * based on the doping concentrations and material properties, with proper
     * temperature dependence.
     *
     * @param N_A The acceptor doping concentration
     * @param N_D The donor doping concentration
     * @param T The temperature in Kelvin (default: 300K)
     * @return The built-in potential in volts
     */
    double calculate_built_in_potential(double N_A, double N_D, double T = 300.0) const;

    /**
     * @brief Calculates the built-in potential of a heterojunction.
     *
     * This method calculates the built-in potential of a heterojunction
     * based on the doping concentrations, material properties, and band offsets.
     * It accounts for the difference in bandgaps, electron affinities, and
     * effective densities of states between the two materials.
     *
     * @param N_A The acceptor doping concentration in p-type material
     * @param N_D The donor doping concentration in n-type material
     * @param mat_p The material properties of the p-type region
     * @param mat_n The material properties of the n-type region
     * @param T The temperature in Kelvin (default: 300K)
     * @return The built-in potential in volts
     */
    double calculate_heterojunction_potential(double N_A, double N_D,
                                             const Materials::Material& mat_p,
                                             const Materials::Material& mat_n,
                                             double T = 300.0) const;

    /**
     * @brief Gets the electric field at a given position.
     *
     * @param x The x-coordinate of the position
     * @param y The y-coordinate of the position
     * @return The electric field vector at the given position
     */
    Eigen::Vector2d get_electric_field(double x, double y) const { return poisson.get_electric_field(x, y); }

    /**
     * @brief Sets the material properties for a heterojunction.
     *
     * This method sets the material properties for a heterojunction by defining
     * different materials in different regions of the device. It allows for
     * more complex device structures with multiple materials.
     *
     * @param materials Vector of materials
     * @param regions Vector of functions that define the regions for each material
     */
    void set_heterojunction(const std::vector<Materials::Material>& materials,
                           const std::vector<std::function<bool(double, double)>>& regions);

    /**
     * @brief Gets the material at a given position.
     *
     * This method returns the material at the given position based on the
     * heterojunction regions defined by set_heterojunction.
     *
     * @param x The x-coordinate of the position
     * @param y The y-coordinate of the position
     * @return The material at the given position
     */
    Materials::Material get_material_at(double x, double y) const;

public:
    // Convergence acceleration parameters
    double damping_factor;               ///< Damping factor for potential updates (0 < damping_factor <= 1)
    int anderson_history_size;           ///< Number of previous iterations to use for Anderson acceleration

private:
    Mesh& mesh;                          ///< Reference to the mesh used for the simulation
    PoissonSolver poisson;               ///< Poisson solver for the electrostatic potential
    Eigen::VectorXd n, p;                ///< Carrier concentrations (electrons and holes)
    Eigen::VectorXd phi_n, phi_p;        ///< Quasi-Fermi potentials for electrons and holes
    Eigen::SparseMatrix<double> Kn, Kp;  ///< Drift-diffusion matrices for electrons and holes
    double N_A, N_D;                     ///< Doping concentrations

    std::vector<Eigen::VectorXd> phi_history;  ///< History of potential vectors for Anderson acceleration
    std::vector<Eigen::VectorXd> res_history;  ///< History of residual vectors for Anderson acceleration

    // Function pointers for physical quantities
    double (*epsilon_r)(double, double);  ///< Function for relative permittivity
    double (*rho)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&);  ///< Function for charge density
    double (*n_conc)(double, double, double, const Materials::Material&);  ///< Function for electron concentration
    double (*p_conc)(double, double, double, const Materials::Material&);  ///< Function for hole concentration
    double (*mu_n)(double, double, const Materials::Material&);  ///< Function for electron mobility
    double (*mu_p)(double, double, const Materials::Material&);  ///< Function for hole mobility

    // Heterojunction properties
    std::vector<Materials::Material> materials;  ///< Vector of materials for heterojunction
    std::vector<std::function<bool(double, double)>> regions;  ///< Vector of region functions for heterojunction
    bool has_heterojunction;  ///< Flag indicating whether a heterojunction is defined

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
     * @brief Performs line search to find optimal step size.
     *
     * This function performs a line search to find the optimal step size
     * for the Anderson acceleration. It uses a backtracking line search
     * algorithm to find a step size that reduces the residual.
     *
     * @param phi_old The potential from the previous iteration
     * @param phi_update The updated potential from the current iteration
     * @param phi_accel The accelerated potential
     * @return The optimal step size
     */
    double perform_line_search(const Eigen::VectorXd& phi_old,
                              const Eigen::VectorXd& phi_update,
                              const Eigen::VectorXd& phi_accel);

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
     * @brief Updates the quasi-Fermi potentials based on carrier concentrations.
     *
     * This method updates the quasi-Fermi potentials for electrons and holes
     * based on the current carrier concentrations and material properties.
     * The quasi-Fermi potentials are used to calculate the current densities
     * and to ensure self-consistency in non-equilibrium conditions.
     */
    void update_quasi_fermi_potentials();

    /**
     * @brief Applies boundary conditions to the carrier concentrations.
     *
     * @param V_p The voltage applied to the p-contact
     * @param V_n The voltage applied to the n-contact
     */
    void apply_boundary_conditions(double V_p, double V_n);

    /**
     * @brief Refines the mesh adaptively based on error estimators.
     *
     * This method refines the mesh adaptively based on error estimators
     * for the potential and carrier concentrations. It identifies regions
     * with high gradients or errors and refines the mesh in those regions.
     *
     * @param refinement_threshold The threshold for mesh refinement
     * @param max_refinement_level The maximum refinement level
     * @return True if the mesh was refined, false otherwise
     */
    bool refine_mesh_adaptively(double refinement_threshold, int max_refinement_level);

    /**
     * @brief Computes error estimators for mesh refinement.
     *
     * This method computes error estimators for mesh refinement based on
     * the gradients of the potential and carrier concentrations. It identifies
     * regions with high gradients or errors that need mesh refinement.
     *
     * @return Vector of error estimators for each element
     */
    Eigen::VectorXd compute_error_estimators() const;

    /**
     * @brief Computes the quantum correction to the potential.
     *
     * This method computes the quantum correction to the potential using
     * the Bohm quantum potential approach. It accounts for quantum effects
     * like tunneling and quantum confinement.
     *
     * @param n The electron concentration
     * @param p The hole concentration
     * @return The quantum correction to the potential
     */
    Eigen::VectorXd compute_quantum_correction(const Eigen::VectorXd& n, const Eigen::VectorXd& p) const;

    /**
     * @brief Computes the tunneling current.
     *
     * This method computes the tunneling current using the WKB approximation.
     * It accounts for band-to-band tunneling and trap-assisted tunneling.
     *
     * @param E_field The electric field
     * @return The tunneling current
     */
    Eigen::VectorXd compute_tunneling_current(const std::vector<Eigen::Vector2d>& E_field) const;
};