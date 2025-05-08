#pragma once
/**
 * @file solver.h
 * @brief Defines the EigenSolver class for solving eigenvalue problems in quantum simulations.
 *
 * This file contains the declaration of the EigenSolver class, which implements
 * methods for solving the generalized eigenvalue problem arising from the
 * finite element discretization of the Schrödinger equation. The solver uses
 * the ARPACK library through Eigen's SparseGenRealShift solver for efficient
 * computation of eigenvalues and eigenvectors.
 *
 * The generalized eigenvalue problem is formulated as:
 * \f[ H \psi = E M \psi \f]
 *
 * where:
 * - \f$H\f$ is the Hamiltonian matrix representing the total energy operator
 * - \f$M\f$ is the mass matrix arising from the finite element discretization
 * - \f$\psi\f$ is the eigenvector representing the wavefunction
 * - \f$E\f$ is the eigenvalue representing the energy level
 *
 * Physical units:
 * - Eigenvalues: electron volts (eV)
 * - Eigenvectors: dimensionless (normalized)
 * - Spatial coordinates: nanometers (nm)
 * - Effective mass: relative to electron mass (m_0)
 * - Potential: electron volts (eV)
 *
 * Assumptions and limitations:
 * - The solver assumes that the matrices are already assembled by the FEMSolver
 * - The solver computes the lowest eigenvalues and corresponding eigenvectors
 * - The solver uses the shift-and-invert mode for better convergence
 * - The solver handles complex Hamiltonians for systems with complex potentials
 * - The eigenvectors are normalized with respect to the mass matrix (M-orthogonal)
 *
 * Performance considerations:
 * - For large systems, the solver may require significant memory and computation time
 * - The performance scales with the number of mesh nodes and the number of eigenvalues requested
 * - For systems with many degrees of freedom, consider using MPI parallelization
 *
 * @see FEMSolver for matrix assembly
 * @see Mesh for the underlying spatial discretization
 *
 * @author Dr. Mazharuddin Mohammed
 * @date 2023-07-15
 */

#include "fem.h"
#include "error_handling.h"
#include "recovery_manager.h"
#include "input_validation.h"
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
 *
 * @details
 * The solver uses the following algorithm to solve the generalized eigenvalue problem:
 * 1. Convert the sparse matrices to dense format for better numerical stability
 * 2. Ensure the matrices are Hermitian by averaging with their adjoints
 * 3. Compute the Cholesky decomposition of the mass matrix M = L·L†
 * 4. Transform the generalized eigenvalue problem to a standard eigenvalue problem:
 *    A·x = λ·x, where A = L⁻¹·H·L⁻† and x = L†·ψ
 * 5. Solve the standard eigenvalue problem using Eigen's SelfAdjointEigenSolver
 * 6. Transform the eigenvectors back to the original problem: ψ = L⁻†·x
 * 7. Normalize the eigenvectors with respect to the mass matrix: ψ†·M·ψ = 1
 *
 * The solver includes robust error handling and recovery mechanisms:
 * - Validation of input parameters
 * - Detection and handling of non-positive-definite mass matrices
 * - Regularization of ill-conditioned matrices
 * - Fallback algorithms for numerical instabilities
 * - Comprehensive error reporting and logging
 *
 * @note The eigenvalues are sorted in ascending order, with the ground state
 * (lowest energy state) at index 0.
 *
 * @warning For very large systems or highly irregular meshes, the solver may
 * encounter numerical instabilities. In such cases, consider refining the mesh,
 * adjusting the potential, or using a different solver algorithm.
 */
class EigenSolver {
public:
    /**
     * @brief Constructs a new EigenSolver object.
     *
     * This constructor initializes the EigenSolver with a reference to a FEMSolver
     * object that contains the assembled Hamiltonian and mass matrices. The matrices
     * must be properly assembled before creating the EigenSolver.
     *
     * @param fem The FEMSolver object containing the assembled matrices
     *
     * @throws ErrorHandling::QDSimException with code SOLVER_INITIALIZATION_FAILED
     *         if the matrices are not assembled or are invalid
     *
     * @see FEMSolver::assemble_matrices() for matrix assembly
     *
     * @code{.cpp}
     * // Example usage:
     * Mesh mesh(100.0, 100.0, 101, 101);
     * FEMSolver fem_solver(mesh);
     * fem_solver.set_potential([](double x, double y) { return 0.5 * (x*x + y*y); });
     * fem_solver.set_effective_mass(0.067);
     * fem_solver.assemble_matrices();
     * EigenSolver eigen_solver(fem_solver);
     * @endcode
     */
    EigenSolver(FEMSolver& fem);

    /**
     * @brief Solves the generalized eigenvalue problem.
     *
     * This method solves the generalized eigenvalue problem H·ψ = E·M·ψ using
     * the algorithm described in the class documentation. It computes the
     * lowest `num_eigenvalues` eigenvalues and corresponding eigenvectors.
     *
     * The method includes robust error handling and recovery mechanisms for
     * numerical instabilities, such as non-positive-definite mass matrices
     * and convergence failures.
     *
     * @param num_eigenvalues The number of eigenvalues to compute (must be positive
     *                        and less than the size of the matrices)
     *
     * @throws ErrorHandling::QDSimException with code INVALID_ARGUMENT
     *         if num_eigenvalues is invalid
     * @throws ErrorHandling::QDSimException with code SOLVER_INITIALIZATION_FAILED
     *         if the matrices are not properly initialized
     * @throws ErrorHandling::QDSimException with code MATRIX_DIMENSION_MISMATCH
     *         if the matrices have incompatible dimensions
     * @throws ErrorHandling::QDSimException with code MATRIX_NOT_POSITIVE_DEFINITE
     *         if the mass matrix is not positive definite
     * @throws ErrorHandling::QDSimException with code EIGENVALUE_COMPUTATION_FAILED
     *         if the eigenvalue computation fails
     *
     * @note The method attempts to recover from certain errors, such as
     *       non-positive-definite mass matrices, by applying regularization.
     *       If recovery fails, dummy values are returned.
     *
     * @see get_eigenvalues() to retrieve the computed eigenvalues
     * @see get_eigenvectors() to retrieve the computed eigenvectors
     *
     * @code{.cpp}
     * // Example usage:
     * EigenSolver eigen_solver(fem_solver);
     * try {
     *     eigen_solver.solve(10);  // Compute the 10 lowest eigenvalues
     *     auto eigenvalues = eigen_solver.get_eigenvalues();
     *     auto eigenvectors = eigen_solver.get_eigenvectors();
     * } catch (const ErrorHandling::QDSimException& e) {
     *     std::cerr << "Error solving eigenvalue problem: " << e.what() << std::endl;
     * }
     * @endcode
     */
    void solve(int num_eigenvalues);

    /**
     * @brief Gets the computed eigenvalues.
     *
     * This method returns a reference to the vector of eigenvalues computed by
     * the solve() method. The eigenvalues are sorted in ascending order, with
     * the ground state (lowest energy state) at index 0.
     *
     * @return A reference to the vector of eigenvalues in electron volts (eV)
     *
     * @note The eigenvalues are stored as complex numbers, but for Hermitian
     *       Hamiltonians, the imaginary part should be zero or negligibly small.
     *       Non-zero imaginary parts may indicate numerical instabilities or
     *       non-Hermitian Hamiltonians.
     *
     * @warning The returned vector may be empty if solve() has not been called
     *          or if it failed to compute any eigenvalues.
     *
     * @see solve() to compute the eigenvalues
     */
    const std::vector<std::complex<double>>& get_eigenvalues() const { return eigenvalues; }

    /**
     * @brief Gets the computed eigenvectors.
     *
     * This method returns a reference to the vector of eigenvectors (wavefunctions)
     * computed by the solve() method. The eigenvectors correspond to the eigenvalues
     * returned by get_eigenvalues() and are sorted in the same order.
     *
     * The eigenvectors are normalized with respect to the mass matrix M, such that
     * ψ†·M·ψ = 1 for each eigenvector ψ.
     *
     * @return A reference to the vector of eigenvectors (wavefunctions)
     *
     * @note Each eigenvector is represented as an Eigen::VectorXd, where each
     *       element corresponds to the wavefunction value at a mesh node.
     *
     * @warning The returned vector may be empty if solve() has not been called
     *          or if it failed to compute any eigenvectors.
     *
     * @see solve() to compute the eigenvectors
     * @see get_eigenvalues() to get the corresponding eigenvalues
     */
    const std::vector<Eigen::VectorXd>& get_eigenvectors() const { return eigenvectors; }

private:
    /**
     * @brief Reference to the FEMSolver object containing the assembled matrices.
     *
     * This reference provides access to the Hamiltonian and mass matrices
     * assembled by the FEMSolver. The matrices must be properly assembled
     * before calling the solve() method.
     *
     * @see FEMSolver::get_H() to get the Hamiltonian matrix
     * @see FEMSolver::get_M() to get the mass matrix
     */
    FEMSolver& fem;

    /**
     * @brief Vector of computed eigenvalues in electron volts (eV).
     *
     * This vector stores the eigenvalues computed by the solve() method.
     * The eigenvalues are sorted in ascending order, with the ground state
     * (lowest energy state) at index 0.
     *
     * @note The eigenvalues are stored as complex numbers, but for Hermitian
     *       Hamiltonians, the imaginary part should be zero or negligibly small.
     */
    std::vector<std::complex<double>> eigenvalues;

    /**
     * @brief Vector of computed eigenvectors (wavefunctions).
     *
     * This vector stores the eigenvectors computed by the solve() method.
     * Each eigenvector corresponds to an eigenvalue in the eigenvalues vector
     * and is stored at the same index.
     *
     * @note Each eigenvector is represented as an Eigen::VectorXd, where each
     *       element corresponds to the wavefunction value at a mesh node.
     */
    std::vector<Eigen::VectorXd> eigenvectors;
};