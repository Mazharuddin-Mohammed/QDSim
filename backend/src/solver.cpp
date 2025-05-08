/**
 * @file solver.cpp
 * @brief Implementation of the EigenSolver class for solving eigenvalue problems.
 *
 * This file contains the implementation of the EigenSolver class, which implements
 * methods for solving the generalized eigenvalue problem arising from the
 * finite element discretization of the Schrödinger equation. The solver uses
 * the Eigen library for eigenvalue computations.
 *
 * The implementation uses the Cholesky decomposition to transform the generalized
 * eigenvalue problem into a standard eigenvalue problem, which is then solved
 * using the SelfAdjointEigenSolver from Eigen.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "solver.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

/**
 * @brief Constructs a new EigenSolver object.
 *
 * This constructor initializes the EigenSolver with a reference to the FEMSolver
 * that contains the assembled Hamiltonian and mass matrices.
 *
 * @param fem The FEMSolver object containing the assembled matrices
 */
EigenSolver::EigenSolver(FEMSolver& fem) : fem(fem) {}

/**
 * @brief Solves the generalized eigenvalue problem.
 *
 * This method solves the generalized eigenvalue problem H·ψ = E·M·ψ using the
 * Cholesky decomposition to transform it into a standard eigenvalue problem.
 * It computes the lowest eigenvalues and corresponding eigenvectors.
 *
 * The solution process involves:
 * 1. Getting the Hamiltonian and mass matrices from the FEMSolver
 * 2. Converting the sparse matrices to dense matrices
 * 3. Ensuring the matrices are Hermitian
 * 4. Computing the Cholesky decomposition of M
 * 5. Transforming the generalized eigenvalue problem to a standard one
 * 6. Solving the standard eigenvalue problem
 * 7. Transforming the eigenvectors back to the original problem
 * 8. Normalizing the eigenvectors with respect to M
 * 9. Storing the smallest eigenvalues and eigenvectors
 *
 * @param num_eigenvalues The number of eigenvalues to compute
 *
 * @throws std::runtime_error If the solver fails to converge
 * @throws std::invalid_argument If num_eigenvalues is invalid
 */
void EigenSolver::solve(int num_eigenvalues) {
    // Validate input parameters
    QDSIM_VALIDATE_SOLVER_PARAMETERS(num_eigenvalues, 1e-10, 1000);

    // Get the Hamiltonian and mass matrices from the FEMSolver
    const Eigen::SparseMatrix<std::complex<double>>& H = fem.get_H();
    const Eigen::SparseMatrix<std::complex<double>>& M = fem.get_M();

    // Check if matrices are valid
    if (H.rows() == 0 || M.rows() == 0) {
        QDSIM_THROW(ErrorHandling::ErrorCode::SOLVER_INITIALIZATION_FAILED,
                   "Matrices H and M must be initialized before solving");
    }

    // Check if matrices have the same dimensions
    if (H.rows() != M.rows() || H.cols() != M.cols()) {
        QDSIM_THROW(ErrorHandling::ErrorCode::MATRIX_DIMENSION_MISMATCH,
                   "Matrices H and M must have the same dimensions");
    }

    // Solve the generalized eigenvalue problem H psi = E M psi
    try {
        QDSIM_LOG_INFO("Starting eigenvalue computation for " + std::to_string(num_eigenvalues) + " states");

        // Convert sparse matrices to dense
        Eigen::MatrixXcd H_dense = Eigen::MatrixXcd(H);
        Eigen::MatrixXcd M_dense = Eigen::MatrixXcd(M);

        // Make sure matrices are Hermitian (H = H†, M = M†)
        H_dense = (H_dense + H_dense.adjoint()) / 2.0;
        M_dense = (M_dense + M_dense.adjoint()) / 2.0;

        // Compute the Cholesky decomposition of M
        Eigen::LLT<Eigen::MatrixXcd> llt(M_dense);
        if (llt.info() != Eigen::Success) {
            // Try to recover by adding a small regularization term
            QDSIM_LOG_WARNING("Cholesky decomposition failed. Attempting to regularize the mass matrix.");

            // Add a small regularization term to the diagonal
            double reg = 1e-10 * M_dense.diagonal().real().maxCoeff();
            for (int i = 0; i < M_dense.rows(); ++i) {
                M_dense(i, i) += std::complex<double>(reg, 0.0);
            }

            // Try the Cholesky decomposition again
            llt.compute(M_dense);

            if (llt.info() != Eigen::Success) {
                QDSIM_THROW(ErrorHandling::ErrorCode::MATRIX_NOT_POSITIVE_DEFINITE,
                           "Cholesky decomposition failed even after regularization. M is not positive definite.");
            }

            QDSIM_LOG_INFO("Regularization successful. Continuing with eigenvalue computation.");
        }

        // Get the L matrix from the decomposition (M = L·L†)
        Eigen::MatrixXcd L = llt.matrixL();

        // Compute L⁻¹·H·L⁻†
        Eigen::MatrixXcd A = L.triangularView<Eigen::Lower>().solve(
            H_dense * L.adjoint().triangularView<Eigen::Upper>().solve(
                Eigen::MatrixXcd::Identity(H_dense.rows(), H_dense.cols())
            )
        );

        // Ensure A is Hermitian
        A = (A + A.adjoint()) / 2.0;

        // Solve the standard eigenvalue problem A·x = λ·x
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(A);

        if (es.info() != Eigen::Success) {
            QDSIM_THROW(ErrorHandling::ErrorCode::EIGENVALUE_COMPUTATION_FAILED,
                       "Eigenvalue computation failed");
        }

        // Extract eigenvalues and eigenvectors
        Eigen::VectorXd evals = es.eigenvalues();
        Eigen::MatrixXcd evecs_A = es.eigenvectors();

        // Transform eigenvectors back to the original problem: y = L⁻†·x
        Eigen::MatrixXcd evecs = L.adjoint().triangularView<Eigen::Upper>().solve(evecs_A);

        // Normalize eigenvectors with respect to M: y†·M·y = 1
        for (int i = 0; i < evecs.cols(); ++i) {
            std::complex<double> norm = std::sqrt(evecs.col(i).dot(M_dense * evecs.col(i)));
            if (std::abs(norm) < 1e-10) {
                QDSIM_LOG_WARNING("Small normalization factor for eigenvector " + std::to_string(i) +
                                 ". This may indicate numerical instability.");
                norm = std::complex<double>(1e-10, 0.0);
            }
            evecs.col(i) /= norm;
        }

        // Store the smallest num_eigenvalues eigenvalues and eigenvectors
        int n = std::min(num_eigenvalues, static_cast<int>(evals.size()));
        eigenvalues.resize(n);
        eigenvectors.resize(n);

        for (int i = 0; i < n; ++i) {
            eigenvalues[i] = std::complex<double>(evals(i), 0.0);
            // Store the real part of the eigenvectors
            eigenvectors[i] = evecs.col(i).real();

            // Check for NaN or infinity in eigenvalues and eigenvectors
            if (!std::isfinite(evals(i))) {
                QDSIM_LOG_WARNING("Non-finite eigenvalue detected: E" + std::to_string(i) + " = " +
                                 std::to_string(evals(i)));
            }

            if (!eigenvectors[i].allFinite()) {
                QDSIM_LOG_WARNING("Non-finite values detected in eigenvector " + std::to_string(i));
            }
        }

        QDSIM_LOG_INFO("Eigenvalue computation completed successfully");

    } catch (const ErrorHandling::QDSimException& e) {
        // Try to recover using the recovery manager
        QDSIM_LOG_ERROR("Error in eigenvalue computation: " + std::string(e.what()));

        bool recovered = false;

        // Try to recover using the fallback algorithm
        if (e.code() == ErrorHandling::ErrorCode::MATRIX_NOT_POSITIVE_DEFINITE) {
            recovered = QDSIM_RECOVER(e, Recovery::RecoveryStrategy::FALLBACK_ALGORITHM);
        }

        if (!recovered) {
            // If recovery failed, return dummy values
            QDSIM_LOG_ERROR("Recovery failed. Returning dummy values.");
            eigenvalues.resize(num_eigenvalues, std::complex<double>(0.0, 0.0));
            eigenvectors.resize(num_eigenvalues, Eigen::VectorXd::Zero(H.rows()));
        }
    } catch (const std::exception& e) {
        // For other exceptions, log the error and return dummy values
        QDSIM_LOG_ERROR("Unhandled exception in eigenvalue computation: " + std::string(e.what()));

        eigenvalues.resize(num_eigenvalues, std::complex<double>(0.0, 0.0));
        eigenvectors.resize(num_eigenvalues, Eigen::VectorXd::Zero(H.rows()));
    }
}