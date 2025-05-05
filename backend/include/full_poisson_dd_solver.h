#pragma once
/**
 * @file full_poisson_dd_solver.h
 * @brief Defines the FullPoissonDriftDiffusionSolver class for semiconductor device simulations.
 *
 * This file contains the declaration of the FullPoissonDriftDiffusionSolver class, which implements
 * a comprehensive solver for the coupled Poisson-drift-diffusion equations used in semiconductor
 * device simulations. It properly accounts for carrier statistics, self-consistent iteration,
 * and non-equilibrium conditions.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include "poisson.h"
#include "materials.h"
#include <Eigen/Sparse>
#include <functional>
#include <vector>
#include <memory>

// Make MPI optional
#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @class FullPoissonDriftDiffusionSolver
 * @brief Comprehensive solver for coupled Poisson-drift-diffusion equations.
 *
 * This class implements a comprehensive solver for the coupled Poisson-drift-diffusion
 * equations used in semiconductor device simulations. It properly accounts for carrier
 * statistics, self-consistent iteration, and non-equilibrium conditions.
 *
 * Key features:
 * - Full Poisson solver that takes actual charge distribution as input
 * - Proper carrier statistics based on Fermi-Dirac or Boltzmann distributions
 * - Robust self-consistent iteration with acceleration techniques
 * - Drift-diffusion model for non-equilibrium conditions
 * - Support for heterojunctions and complex device structures
 * - Quantum corrections for nanoscale devices
 */
class FullPoissonDriftDiffusionSolver {
public:
    /**
     * @brief Constructs a new FullPoissonDriftDiffusionSolver object.
     *
     * @param mesh The mesh to use for the simulation
     * @param epsilon_r Function that returns the relative permittivity at a given position
     * @param doping_profile Function that returns the doping profile at a given position (positive for donors, negative for acceptors)
     */
    FullPoissonDriftDiffusionSolver(Mesh& mesh,
                                   std::function<double(double, double)> epsilon_r,
                                   std::function<double(double, double)> doping_profile);

    /**
     * @brief Solves the coupled Poisson-drift-diffusion equations.
     *
     * This function solves the coupled Poisson-drift-diffusion equations
     * using a self-consistent iterative approach. It alternates between solving
     * the Poisson equation and the drift-diffusion equations until convergence
     * is reached or the maximum number of iterations is exceeded.
     *
     * @param V_p The voltage applied to the p-contact
     * @param V_n The voltage applied to the n-contact
     * @param tolerance The convergence tolerance (default: 1e-6)
     * @param max_iter The maximum number of iterations (default: 100)
     */
    void solve(double V_p, double V_n, double tolerance = 1e-6, int max_iter = 100);

    /**
     * @brief Gets the computed electrostatic potential.
     *
     * @return The electrostatic potential vector
     */
    const Eigen::VectorXd& get_potential() const { return phi; }

    /**
     * @brief Gets the computed electron concentration.
     *
     * @return The electron concentration vector
     */
    const Eigen::VectorXd& get_electron_concentration() const { return n; }

    /**
     * @brief Gets the computed hole concentration.
     *
     * @return The hole concentration vector
     */
    const Eigen::VectorXd& get_hole_concentration() const { return p; }

    /**
     * @brief Gets the computed electron current density.
     *
     * @return The electron current density vector
     */
    const std::vector<Eigen::Vector2d>& get_electron_current_density() const { return J_n; }

    /**
     * @brief Gets the computed hole current density.
     *
     * @return The hole current density vector
     */
    const std::vector<Eigen::Vector2d>& get_hole_current_density() const { return J_p; }

    /**
     * @brief Gets the electric field at a given position.
     *
     * @param x The x-coordinate of the position
     * @param y The y-coordinate of the position
     * @return The electric field vector at the given position
     */
    Eigen::Vector2d get_electric_field(double x, double y) const;

    /**
     * @brief Gets the computed electric field at all mesh nodes.
     *
     * @return The electric field vectors at all mesh nodes
     */
    const std::vector<Eigen::Vector2d>& get_electric_field() const { return E_field; }

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
     * @brief Sets the generation-recombination model.
     *
     * This method sets the generation-recombination model to use in the simulation.
     * It allows for different recombination mechanisms like SRH, Auger, and radiative.
     *
     * @param g_r Function that computes the generation-recombination rate
     */
    void set_generation_recombination_model(
        std::function<double(double, double, double, double, const Materials::Material&)> g_r);

    /**
     * @brief Sets the mobility models for electrons and holes.
     *
     * This method sets the mobility models for electrons and holes. It allows for
     * field-dependent and concentration-dependent mobility models.
     *
     * @param mu_n Function that computes the electron mobility
     * @param mu_p Function that computes the hole mobility
     */
    void set_mobility_models(
        std::function<double(double, double, double, const Materials::Material&)> mu_n,
        std::function<double(double, double, double, const Materials::Material&)> mu_p);

    /**
     * @brief Sets the carrier statistics model.
     *
     * This method sets the carrier statistics model to use in the simulation.
     * It allows for different statistics like Boltzmann, Fermi-Dirac, or Maxwell-Boltzmann.
     *
     * @param use_fermi_dirac Whether to use Fermi-Dirac statistics (true) or Boltzmann statistics (false)
     */
    void set_carrier_statistics_model(bool use_fermi_dirac);

    /**
     * @brief Enables or disables quantum corrections.
     *
     * This method enables or disables quantum corrections in the simulation.
     * Quantum corrections are important for nanoscale devices where quantum
     * effects like tunneling and confinement become significant.
     *
     * @param enable Whether to enable quantum corrections
     */
    void enable_quantum_corrections(bool enable);

    /**
     * @brief Enables or disables adaptive mesh refinement.
     *
     * This method enables or disables adaptive mesh refinement in the simulation.
     * Adaptive mesh refinement is useful for improving accuracy in regions with
     * high gradients or rapid variations in the solution.
     *
     * @param enable Whether to enable adaptive mesh refinement
     * @param refinement_threshold The threshold for mesh refinement
     * @param max_refinement_level The maximum refinement level
     */
    void enable_adaptive_mesh_refinement(bool enable, double refinement_threshold = 0.1, int max_refinement_level = 3);

private:
    // Reference to the mesh
    Mesh& mesh;

    // Electrostatic potential
    Eigen::VectorXd phi;

    // Carrier concentrations
    Eigen::VectorXd n;  // Electron concentration
    Eigen::VectorXd p;  // Hole concentration

    // Current densities
    std::vector<Eigen::Vector2d> J_n;  // Electron current density
    std::vector<Eigen::Vector2d> J_p;  // Hole current density

    // Electric field
    std::vector<Eigen::Vector2d> E_field;

    // Quasi-Fermi potentials
    Eigen::VectorXd phi_n;  // Electron quasi-Fermi potential
    Eigen::VectorXd phi_p;  // Hole quasi-Fermi potential

    // Material properties
    std::function<double(double, double)> epsilon_r;  // Relative permittivity
    std::function<double(double, double)> doping_profile;  // Doping profile

    // Heterojunction properties
    std::vector<Materials::Material> materials;  // Vector of materials for heterojunction
    std::vector<std::function<bool(double, double)>> regions;  // Vector of region functions for heterojunction
    bool has_heterojunction;  // Flag indicating whether a heterojunction is defined

    // Generation-recombination model
    std::function<double(double, double, double, double, const Materials::Material&)> g_r_model;
    bool has_g_r_model;  // Flag indicating whether a generation-recombination model is defined

    // Mobility models
    std::function<double(double, double, double, const Materials::Material&)> mu_n_model;
    std::function<double(double, double, double, const Materials::Material&)> mu_p_model;
    bool has_mobility_models;  // Flag indicating whether mobility models are defined

    // Carrier statistics model
    bool use_fermi_dirac_statistics;  // Whether to use Fermi-Dirac statistics

    // Quantum corrections
    bool use_quantum_corrections;  // Whether to use quantum corrections

    // Adaptive mesh refinement
    bool use_adaptive_mesh_refinement;  // Whether to use adaptive mesh refinement
    double adaptive_mesh_refinement_threshold;  // Threshold for mesh refinement
    int adaptive_mesh_refinement_max_level;  // Maximum refinement level

    // Convergence acceleration parameters
    double damping_factor;  // Damping factor for potential updates
    int anderson_history_size;  // Number of previous iterations to use for Anderson acceleration
    std::vector<Eigen::VectorXd> phi_history;  // History of potential vectors for Anderson acceleration
    std::vector<Eigen::VectorXd> res_history;  // History of residual vectors for Anderson acceleration

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

    /**
     * @brief Initializes the carrier concentrations based on the doping profile.
     *
     * This method initializes the carrier concentrations based on the doping profile
     * and the carrier statistics model. It computes the equilibrium carrier concentrations
     * using either Boltzmann or Fermi-Dirac statistics.
     */
    void initialize_carrier_concentrations();

    /**
     * @brief Computes the charge density based on the carrier concentrations and doping profile.
     *
     * This method computes the charge density based on the carrier concentrations and
     * doping profile. The charge density is used in the Poisson equation to compute
     * the electrostatic potential.
     *
     * @return The charge density vector
     */
    Eigen::VectorXd compute_charge_density() const;

    /**
     * @brief Solves the Poisson equation.
     *
     * This method solves the Poisson equation using the finite element method.
     * It computes the electrostatic potential based on the charge density.
     *
     * @param V_p The voltage applied to the p-contact
     * @param V_n The voltage applied to the n-contact
     */
    void solve_poisson_equation(double V_p, double V_n);

    /**
     * @brief Computes the electric field from the electrostatic potential.
     *
     * This method computes the electric field from the electrostatic potential
     * using the finite element method. The electric field is the negative gradient
     * of the potential.
     */
    void compute_electric_field();

    /**
     * @brief Solves the drift-diffusion equations.
     *
     * This method solves the drift-diffusion equations using the finite element method.
     * It computes the carrier concentrations based on the electrostatic potential
     * and the quasi-Fermi potentials.
     *
     * @param V_p The voltage applied to the p-contact
     * @param V_n The voltage applied to the n-contact
     */
    void solve_drift_diffusion_equations(double V_p, double V_n);

    /**
     * @brief Computes the current densities.
     *
     * This method computes the electron and hole current densities based on
     * the carrier concentrations, electric field, and quasi-Fermi potentials.
     */
    void compute_current_densities();

    /**
     * @brief Applies boundary conditions to the carrier concentrations.
     *
     * This method applies boundary conditions to the carrier concentrations
     * based on the applied voltages and the position of the contacts.
     *
     * @param V_p The voltage applied to the p-contact
     * @param V_n The voltage applied to the n-contact
     */
    void apply_boundary_conditions(double V_p, double V_n);

    /**
     * @brief Computes the generation-recombination rate.
     *
     * This method computes the generation-recombination rate based on the
     * carrier concentrations and the generation-recombination model.
     *
     * @return The generation-recombination rate vector
     */
    Eigen::VectorXd compute_generation_recombination_rate() const;

    /**
     * @brief Computes the electron and hole mobilities.
     *
     * This method computes the electron and hole mobilities based on the
     * electric field, carrier concentrations, and mobility models.
     *
     * @param mu_n_out Output vector for electron mobility
     * @param mu_p_out Output vector for hole mobility
     */
    void compute_mobilities(Eigen::VectorXd& mu_n_out, Eigen::VectorXd& mu_p_out) const;

    /**
     * @brief Computes the carrier concentrations using Boltzmann statistics.
     *
     * This method computes the carrier concentrations using Boltzmann statistics
     * based on the electrostatic potential and the quasi-Fermi potentials.
     */
    void compute_carrier_concentrations_boltzmann();

    /**
     * @brief Computes the carrier concentrations using Fermi-Dirac statistics.
     *
     * This method computes the carrier concentrations using Fermi-Dirac statistics
     * based on the electrostatic potential and the quasi-Fermi potentials.
     */
    void compute_carrier_concentrations_fermi_dirac();

    /**
     * @brief Computes the quantum correction to the potential.
     *
     * This method computes the quantum correction to the potential using
     * the Bohm quantum potential approach. It accounts for quantum effects
     * like tunneling and quantum confinement.
     *
     * @return The quantum correction to the potential
     */
    Eigen::VectorXd compute_quantum_correction() const;

    /**
     * @brief Refines the mesh adaptively based on error estimators.
     *
     * This method refines the mesh adaptively based on error estimators
     * for the potential and carrier concentrations. It identifies regions
     * with high gradients or errors and refines the mesh in those regions.
     *
     * @return True if the mesh was refined, false otherwise
     */
    bool refine_mesh_adaptively();

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
};
