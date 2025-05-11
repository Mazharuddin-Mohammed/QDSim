#ifndef SIMPLE_SELF_CONSISTENT_H
#define SIMPLE_SELF_CONSISTENT_H

/**
 * @file simple_self_consistent.h
 * @brief Defines a simplified self-consistent Poisson-drift-diffusion solver.
 *
 * This file contains the declaration of the SimpleSelfConsistentSolver class,
 * which implements a simplified approach to solving the coupled Poisson-drift-diffusion
 * equations self-consistently to model semiconductor devices.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Dense>
#include <functional>
#include "mesh.h"
#include "poisson.h"

/**
 * @brief A simplified self-consistent Poisson-drift-diffusion solver.
 *
 * This class solves the coupled Poisson-drift-diffusion equations self-consistently
 * to model semiconductor devices. It uses a simplified approach that doesn't rely
 * on the Materials::Material type.
 */
class SimpleSelfConsistentSolver {
public:
    /**
     * @brief Default constructor for SimpleSelfConsistentSolver.
     *
     * @param mesh The mesh on which to solve the equations.
     */
    SimpleSelfConsistentSolver(Mesh& mesh);

    /**
     * @brief Construct a new SimpleSelfConsistentSolver object.
     *
     * @param mesh The mesh on which to solve the equations.
     * @param epsilon_r Function that returns the relative permittivity at a point (x, y).
     * @param rho Function that returns the charge density at a point (x, y) given the electron and hole concentrations.
     */
    SimpleSelfConsistentSolver(
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
    const Eigen::VectorXd& get_potential() const { return poisson.get_potential(); }

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
    PoissonSolver poisson;                       ///< The Poisson solver.
    std::function<double(double, double)> epsilon_r; ///< Function that returns the relative permittivity at a point (x, y).
    std::function<double(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)> rho; ///< Function that returns the charge density at a point (x, y) given the electron and hole concentrations.
    Eigen::VectorXd n;                           ///< Electron concentration at each node.
    Eigen::VectorXd p;                           ///< Hole concentration at each node.
};

#endif // SIMPLE_SELF_CONSISTENT_H
