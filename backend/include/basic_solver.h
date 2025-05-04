#ifndef BASIC_SOLVER_H
#define BASIC_SOLVER_H

/**
 * @file basic_solver.h
 * @brief Defines a basic solver for demonstration purposes.
 *
 * This file contains the declaration of the BasicSolver class, which implements
 * a very simple solver that sets up basic fields without doing any actual solving.
 * It's meant to be used as a test for the Python bindings and as a template for
 * more complex solvers.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Dense>
#include "mesh.h"

/**
 * @brief A basic solver for demonstration purposes.
 *
 * This class implements a very simple solver that just sets up some
 * basic fields without doing any actual solving. It's meant to be
 * used as a test for the Python bindings.
 */
class BasicSolver {
public:
    /**
     * @brief Construct a new BasicSolver object.
     *
     * @param mesh The mesh on which to solve.
     */
    BasicSolver(Mesh& mesh);

    /**
     * @brief Solve the equations.
     *
     * @param V_p Voltage at the p-contact.
     * @param V_n Voltage at the n-contact.
     * @param N_A Acceptor doping concentration.
     * @param N_D Donor doping concentration.
     */
    void solve(double V_p, double V_n, double N_A, double N_D);

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
    Mesh& mesh;                  ///< The mesh on which to solve.
    Eigen::VectorXd potential;   ///< Electrostatic potential at each node.
    Eigen::VectorXd n;           ///< Electron concentration at each node.
    Eigen::VectorXd p;           ///< Hole concentration at each node.
};

#endif // BASIC_SOLVER_H
