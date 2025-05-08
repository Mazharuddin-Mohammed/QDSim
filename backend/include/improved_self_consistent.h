#ifndef IMPROVED_SELF_CONSISTENT_H
#define IMPROVED_SELF_CONSISTENT_H

/**
 * @file improved_self_consistent.h
 * @brief Defines an improved self-consistent Poisson-drift-diffusion solver.
 *
 * This file contains the declaration of the ImprovedSelfConsistentSolver class,
 * which builds on the BasicSolver to implement a more robust self-consistent
 * Poisson-drift-diffusion solver with proper error handling and convergence
 * acceleration techniques.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Dense>
#include <functional>
#include "mesh.h"
#include "basic_solver.h"

/**
 * @brief An improved self-consistent Poisson-drift-diffusion solver.
 *
 * This class builds on the BasicSolver to implement a more robust
 * self-consistent Poisson-drift-diffusion solver with proper error handling.
 */
class ImprovedSelfConsistentSolver {
public:
    /**
     * @brief Construct a new ImprovedSelfConsistentSolver object.
     *
     * @param mesh The mesh on which to solve the equations.
     * @param epsilon_r Function that returns the relative permittivity at a point (x, y).
     * @param rho Function that returns the charge density at a point (x, y) given the electron and hole concentrations.
     */
    ImprovedSelfConsistentSolver(
        Mesh& mesh,
        std::function<double(double, double)> epsilon_r,
        std::function<double(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)> rho
    );

    /**
     * @brief Solve the self-consistent Poisson-drift-diffusion equations.
     *
     * @param V_p Voltage at the p-contact.
     * @param V_n Voltage at the n-contact.
     * @param N_A Acceptor doping concentration.
     * @param N_D Donor doping concentration.
     * @param tolerance Convergence tolerance.
     * @param max_iter Maximum number of iterations.
     */
    void solve(double V_p, double V_n, double N_A, double N_D, double tolerance = 1e-6, int max_iter = 100);

    /**
     * @brief Get the electrostatic potential.
     *
     * @return const Eigen::VectorXd& The electrostatic potential at each node.
     */
    const Eigen::VectorXd& get_potential() const { return potential; }

    /**
     * @brief Get the electron concentration.
     *
     * @return const Eigen::VectorXd& The electron concentration at each node.
     */
    const Eigen::VectorXd& get_n() const { return n; }

    /**
     * @brief Get the hole concentration.
     *
     * @return const Eigen::VectorXd& The hole concentration at each node.
     */
    const Eigen::VectorXd& get_p() const { return p; }

private:
    Mesh& mesh;                                  ///< The mesh on which to solve the equations.
    std::function<double(double, double)> epsilon_r; ///< Function that returns the relative permittivity at a point (x, y).
    std::function<double(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)> rho; ///< Function that returns the charge density at a point (x, y) given the electron and hole concentrations.
    Eigen::VectorXd potential;                   ///< Electrostatic potential at each node.
    Eigen::VectorXd n;                           ///< Electron concentration at each node.
    Eigen::VectorXd p;                           ///< Hole concentration at each node.

    /**
     * @brief Calculate the charge density at each node.
     *
     * @return Eigen::VectorXd The charge density at each node.
     */
    Eigen::VectorXd calculate_charge_density();

    /**
     * @brief Solve the Poisson equation.
     *
     * @param V_p Voltage at the p-contact.
     * @param V_n Voltage at the n-contact.
     * @param charge_density The charge density at each node.
     */
    void solve_poisson(double V_p, double V_n, const Eigen::VectorXd& charge_density);

    /**
     * @brief Update the carrier concentrations based on the current potential.
     *
     * @param N_A Acceptor doping concentration.
     * @param N_D Donor doping concentration.
     */
    void update_carriers(double N_A, double N_D);
};

#endif // IMPROVED_SELF_CONSISTENT_H
