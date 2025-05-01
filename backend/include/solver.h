#pragma once
/**
 * @file solver.h
 * @brief Defines the EigenSolver class for solving eigenvalue problems.
 *
 * This file contains the declaration of the EigenSolver class, which implements
 * methods for solving the generalized eigenvalue problem arising from the
 * finite element discretization of the Schrödinger equation. The solver uses
 * the ARPACK library through Eigen's SparseGenRealShift solver.
 *
 * Physical units:
 * - Eigenvalues: electron volts (eV)
 * - Eigenvectors: dimensionless (normalized)
 *
 * Assumptions and limitations:
 * - The solver assumes that the matrices are already assembled
 * - The solver computes the lowest eigenvalues and corresponding eigenvectors
 * - The solver uses the shift-and-invert mode for better convergence
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "fem.h"
#include <Eigen/Sparse>
#include <vector>
#include <complex>

/**
 * @class EigenSolver
 * @brief Solves the generalized eigenvalue problem for quantum simulations.
 *
 * The EigenSolver class implements methods for solving the generalized eigenvalue
 * problem arising from the finite element discretization of the Schrödinger equation:
 *
 * \f[ H \psi = E M \psi \f]
 *
 * where \f$H\f$ is the Hamiltonian matrix, \f$M\f$ is the mass matrix,
 * \f$\psi\f$ is the eigenvector (wavefunction), and \f$E\f$ is the eigenvalue (energy).
 *
 * The solver computes the lowest eigenvalues and corresponding eigenvectors,
 * which represent the energy levels and wavefunctions of the quantum system.
 */
class EigenSolver {
public:
    /**
     * @brief Constructs a new EigenSolver object.
     *
     * @param fem The FEMSolver object containing the assembled matrices
     *
     * @throws std::runtime_error If the matrices are not assembled
     */
    EigenSolver(FEMSolver& fem);
    /**
     * @brief Solves the generalized eigenvalue problem.
     *
     * This method solves the generalized eigenvalue problem using the ARPACK
     * library through Eigen's SparseGenRealShift solver. It computes the
     * lowest eigenvalues and corresponding eigenvectors.
     *
     * @param num_eigenvalues The number of eigenvalues to compute
     *
     * @throws std::runtime_error If the solver fails to converge
     * @throws std::invalid_argument If num_eigenvalues is invalid
     */
    void solve(int num_eigenvalues);

    /**
     * @brief Gets the computed eigenvalues.
     *
     * @return A reference to the vector of eigenvalues in electron volts (eV)
     */
    const std::vector<std::complex<double>>& get_eigenvalues() const { return eigenvalues; }

    /**
     * @brief Gets the computed eigenvectors.
     *
     * @return A reference to the vector of eigenvectors (wavefunctions)
     */
    const std::vector<Eigen::VectorXd>& get_eigenvectors() const { return eigenvectors; }

private:
    /** @brief Reference to the FEMSolver object containing the assembled matrices */
    FEMSolver& fem;

    /** @brief Vector of computed eigenvalues in electron volts (eV) */
    std::vector<std::complex<double>> eigenvalues;

    /** @brief Vector of computed eigenvectors (wavefunctions) */
    std::vector<Eigen::VectorXd> eigenvectors;
};