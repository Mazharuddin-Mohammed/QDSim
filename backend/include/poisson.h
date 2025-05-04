#pragma once
/**
 * @file poisson.h
 * @brief Defines the PoissonSolver class for electrostatic calculations.
 *
 * This file contains the declaration of the PoissonSolver class, which implements
 * the finite element method for solving the Poisson equation in electrostatic
 * calculations. The solver computes the electrostatic potential and electric field
 * for a given charge distribution and boundary conditions.
 *
 * The Poisson equation is:
 * \f[ \nabla \cdot (\epsilon_r \nabla \phi) = -\rho / \epsilon_0 \f]
 *
 * where \f$\phi\f$ is the electrostatic potential, \f$\epsilon_r\f$ is the relative
 * permittivity, \f$\rho\f$ is the charge density, and \f$\epsilon_0\f$ is the
 * vacuum permittivity.
 *
 * Physical units:
 * - Coordinates: nanometers (nm)
 * - Potential: volts (V)
 * - Electric field: volts per nanometer (V/nm)
 * - Charge density: elementary charges per cubic nanometer (e/nm^3)
 *
 * Assumptions and limitations:
 * - The solver uses linear (P1) finite elements
 * - The solver assumes Dirichlet boundary conditions
 * - The solver is designed for 2D simulations
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include <Eigen/Sparse>
#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @class PoissonSolver
 * @brief Solves the Poisson equation for electrostatic calculations.
 *
 * The PoissonSolver class implements the finite element method for solving
 * the Poisson equation in electrostatic calculations. It assembles the stiffness
 * matrix and right-hand side vector, applies boundary conditions, and solves
 * the resulting linear system to obtain the electrostatic potential.
 *
 * The class also provides methods for computing the electric field from the
 * potential gradient.
 */
class PoissonSolver {
public:
    /**
     * @brief Constructs a new PoissonSolver object.
     *
     * @param mesh The mesh to use for the simulation
     * @param epsilon_r Function that returns the relative permittivity at a given position
     * @param rho Function that returns the charge density at a given position
     *
     * @throws std::invalid_argument If the input parameters are invalid
     */
    PoissonSolver(Mesh& mesh, double (*epsilon_r)(double, double), 
                  double (*rho)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&));
    /**
     * @brief Solves the Poisson equation.
     *
     * This method assembles the stiffness matrix and right-hand side vector,
     * applies boundary conditions, and solves the resulting linear system to
     * obtain the electrostatic potential.
     *
     * @param V_p Potential at the p-type boundary in volts (V)
     * @param V_n Potential at the n-type boundary in volts (V)
     *
     * @throws std::runtime_error If the solver fails to converge
     */
    void solve(double V_p, double V_n, const Eigen::VectorXd& n, const Eigen::VectorXd& p);

    /**
     * @brief Gets the computed electrostatic potential.
     *
     * @return A reference to the vector of potential values at mesh nodes in volts (V)
     */
    const Eigen::VectorXd& get_potential() const { return phi; }

    /**
     * @brief Computes the electric field at a given position.
     *
     * This method computes the electric field at a given position by taking
     * the negative gradient of the electrostatic potential.
     *
     * @param x The x-coordinate of the position in nanometers (nm)
     * @param y The y-coordinate of the position in nanometers (nm)
     * @return The electric field vector in volts per nanometer (V/nm)
     *
     * @throws std::runtime_error If the position is outside the mesh
     */
    Eigen::Vector2d get_electric_field(double x, double y) const;

private:
    /** @brief Reference to the mesh used for the simulation */
    Mesh& mesh;

    /** @brief Stiffness matrix for the Poisson equation */
    Eigen::SparseMatrix<double> K;

public:
    /** @brief Electrostatic potential at mesh nodes in volts (V) */
    Eigen::VectorXd phi;

private:
    /** @brief Right-hand side vector for the Poisson equation */
    Eigen::VectorXd f;

    /** @brief Function that returns the relative permittivity at a given position */
    double (*epsilon_r)(double, double);

    /** @brief Function that returns the charge density at a given position */
    double (*rho)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&);

    /**
     * @brief Assembles the stiffness matrix.
     *
     * This private method assembles the stiffness matrix for the Poisson equation
     * using the finite element method.
     */
    void assemble_matrix();

    /**
     * @brief Assembles the right-hand side vector.
     *
     * This private method assembles the right-hand side vector for the Poisson equation
     * using the finite element method.
     */
    void assemble_rhs(const Eigen::VectorXd& n, const Eigen::VectorXd& p);

    /**
     * @brief Applies boundary conditions.
     *
     * This private method applies Dirichlet boundary conditions to the stiffness
     * matrix and right-hand side vector.
     *
     * @param V_p Potential at the p-type boundary in volts (V)
     * @param V_n Potential at the n-type boundary in volts (V)
     */
    void apply_boundary_conditions(double V_p, double V_n);
};