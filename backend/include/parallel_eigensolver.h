#pragma once
/**
 * @file parallel_eigensolver.h
 * @brief Defines the ParallelEigensolver class for parallel eigenvalue solving.
 *
 * This file contains the declaration of the ParallelEigensolver class, which provides
 * parallel implementations of eigenvalue solvers for large-scale quantum simulations.
 * It supports both shared-memory parallelism using OpenMP and distributed-memory
 * parallelism using MPI.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <string>
#include <memory>

// Conditional compilation for MPI support
#ifdef USE_MPI
#include <mpi.h>
#endif

// Conditional compilation for SLEPc support
#ifdef USE_SLEPC
#include <slepceps.h>
#endif

/**
 * @enum EigensolverType
 * @brief Enumeration of eigensolver types.
 */
enum class EigensolverType {
    ARPACK,     ///< ARPACK eigensolver (default)
    SLEPC,      ///< SLEPc eigensolver (requires USE_SLEPC)
    FEAST,      ///< FEAST eigensolver
    CUDA        ///< CUDA-accelerated eigensolver (requires USE_CUDA)
};

/**
 * @class ParallelEigensolver
 * @brief Provides parallel implementations of eigenvalue solvers.
 *
 * The ParallelEigensolver class provides parallel implementations of eigenvalue
 * solvers for large-scale quantum simulations. It supports both shared-memory
 * parallelism using OpenMP and distributed-memory parallelism using MPI.
 */
class ParallelEigensolver {
public:
    /**
     * @brief Constructs a new ParallelEigensolver object.
     *
     * @param solver_type The type of eigensolver to use
     * @param use_mpi Whether to use MPI for distributed-memory parallelism
     * @param num_threads The number of threads to use for shared-memory parallelism
     *
     * @throws std::runtime_error If the requested solver is not available
     */
    ParallelEigensolver(EigensolverType solver_type = EigensolverType::ARPACK,
                       bool use_mpi = false,
                       int num_threads = 0);

    /**
     * @brief Destructor for the ParallelEigensolver object.
     *
     * Cleans up resources used by the eigensolver.
     */
    ~ParallelEigensolver();

    /**
     * @brief Solves the generalized eigenvalue problem.
     *
     * This method solves the generalized eigenvalue problem H·ψ = E·M·ψ using
     * parallel algorithms. It computes the lowest eigenvalues and corresponding
     * eigenvectors.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The computed eigenvalues (output)
     * @param eigenvectors The computed eigenvectors (output)
     * @param tolerance The convergence tolerance
     * @param max_iterations The maximum number of iterations
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve(const Eigen::SparseMatrix<std::complex<double>>& H,
              const Eigen::SparseMatrix<std::complex<double>>& M,
              int num_eigenvalues,
              std::vector<std::complex<double>>& eigenvalues,
              std::vector<Eigen::VectorXd>& eigenvectors,
              double tolerance = 1e-10,
              int max_iterations = 1000);

    /**
     * @brief Gets the solver type.
     *
     * @return The solver type
     */
    EigensolverType get_solver_type() const;

    /**
     * @brief Gets the number of threads used for shared-memory parallelism.
     *
     * @return The number of threads
     */
    int get_num_threads() const;

    /**
     * @brief Gets whether MPI is used for distributed-memory parallelism.
     *
     * @return True if MPI is used, false otherwise
     */
    bool is_mpi_enabled() const;

    /**
     * @brief Gets the number of iterations used in the last solve.
     *
     * @return The number of iterations
     */
    int get_num_iterations() const;

    /**
     * @brief Gets the convergence error in the last solve.
     *
     * @return The convergence error
     */
    double get_error() const;

private:
    EigensolverType solver_type_;  ///< The type of eigensolver to use
    bool use_mpi_;                 ///< Whether to use MPI for distributed-memory parallelism
    int num_threads_;              ///< The number of threads to use for shared-memory parallelism
    int num_iterations_;           ///< The number of iterations used in the last solve
    double error_;                 ///< The convergence error in the last solve

#ifdef USE_SLEPC
    // SLEPc-specific members
    EPS eps_;                      ///< SLEPc eigensolver context
    bool slepc_initialized_;       ///< Whether SLEPc has been initialized

    /**
     * @brief Initializes SLEPc.
     *
     * @throws std::runtime_error If SLEPc initialization fails
     */
    void initialize_slepc();

    /**
     * @brief Finalizes SLEPc.
     */
    void finalize_slepc();

    /**
     * @brief Solves the generalized eigenvalue problem using SLEPc.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The computed eigenvalues (output)
     * @param eigenvectors The computed eigenvectors (output)
     * @param tolerance The convergence tolerance
     * @param max_iterations The maximum number of iterations
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve_slepc(const Eigen::SparseMatrix<std::complex<double>>& H,
                    const Eigen::SparseMatrix<std::complex<double>>& M,
                    int num_eigenvalues,
                    std::vector<std::complex<double>>& eigenvalues,
                    std::vector<Eigen::VectorXd>& eigenvectors,
                    double tolerance,
                    int max_iterations);
#endif

    /**
     * @brief Solves the generalized eigenvalue problem using ARPACK.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The computed eigenvalues (output)
     * @param eigenvectors The computed eigenvectors (output)
     * @param tolerance The convergence tolerance
     * @param max_iterations The maximum number of iterations
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve_arpack(const Eigen::SparseMatrix<std::complex<double>>& H,
                     const Eigen::SparseMatrix<std::complex<double>>& M,
                     int num_eigenvalues,
                     std::vector<std::complex<double>>& eigenvalues,
                     std::vector<Eigen::VectorXd>& eigenvectors,
                     double tolerance,
                     int max_iterations);

    /**
     * @brief Solves the generalized eigenvalue problem using FEAST.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The computed eigenvalues (output)
     * @param eigenvectors The computed eigenvectors (output)
     * @param tolerance The convergence tolerance
     * @param max_iterations The maximum number of iterations
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve_feast(const Eigen::SparseMatrix<std::complex<double>>& H,
                    const Eigen::SparseMatrix<std::complex<double>>& M,
                    int num_eigenvalues,
                    std::vector<std::complex<double>>& eigenvalues,
                    std::vector<Eigen::VectorXd>& eigenvectors,
                    double tolerance,
                    int max_iterations);

    /**
     * @brief Solves the generalized eigenvalue problem using CUDA.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param num_eigenvalues The number of eigenvalues to compute
     * @param eigenvalues The computed eigenvalues (output)
     * @param eigenvectors The computed eigenvectors (output)
     * @param tolerance The convergence tolerance
     * @param max_iterations The maximum number of iterations
     *
     * @throws std::runtime_error If the solver fails
     */
    void solve_cuda(const Eigen::SparseMatrix<std::complex<double>>& H,
                   const Eigen::SparseMatrix<std::complex<double>>& M,
                   int num_eigenvalues,
                   std::vector<std::complex<double>>& eigenvalues,
                   std::vector<Eigen::VectorXd>& eigenvectors,
                   double tolerance,
                   int max_iterations);
};
