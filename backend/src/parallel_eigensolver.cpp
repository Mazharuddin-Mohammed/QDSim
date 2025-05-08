/**
 * @file parallel_eigensolver.cpp
 * @brief Implementation of the ParallelEigensolver class for parallel eigenvalue solving.
 *
 * This file contains the implementation of the ParallelEigensolver class, which provides
 * parallel implementations of eigenvalue solvers for large-scale quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "parallel_eigensolver.h"
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <Eigen/Eigenvalues>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>

// Constructor
ParallelEigensolver::ParallelEigensolver(EigensolverType solver_type, bool use_mpi, int num_threads)
    : solver_type_(solver_type), use_mpi_(use_mpi), num_threads_(num_threads),
      num_iterations_(0), error_(0.0) {
    
    // Set number of threads for OpenMP
    if (num_threads_ > 0) {
        omp_set_num_threads(num_threads_);
    } else {
        // Use all available threads by default
        num_threads_ = omp_get_max_threads();
    }
    
    // Check if requested solver is available
    if (solver_type_ == EigensolverType::SLEPC) {
#ifndef USE_SLEPC
        std::cerr << "Warning: SLEPc solver requested but not available. Falling back to ARPACK." << std::endl;
        solver_type_ = EigensolverType::ARPACK;
#else
        // Initialize SLEPc
        initialize_slepc();
#endif
    } else if (solver_type_ == EigensolverType::CUDA) {
#ifndef USE_CUDA
        std::cerr << "Warning: CUDA solver requested but not available. Falling back to ARPACK." << std::endl;
        solver_type_ = EigensolverType::ARPACK;
#endif
    }
    
    // Check if MPI is available
    if (use_mpi_) {
#ifndef USE_MPI
        std::cerr << "Warning: MPI requested but not available. Falling back to shared-memory parallelism." << std::endl;
        use_mpi_ = false;
#endif
    }
}

// Destructor
ParallelEigensolver::~ParallelEigensolver() {
#ifdef USE_SLEPC
    if (solver_type_ == EigensolverType::SLEPC && slepc_initialized_) {
        finalize_slepc();
    }
#endif
}

// Solve the generalized eigenvalue problem
void ParallelEigensolver::solve(const Eigen::SparseMatrix<std::complex<double>>& H,
                              const Eigen::SparseMatrix<std::complex<double>>& M,
                              int num_eigenvalues,
                              std::vector<std::complex<double>>& eigenvalues,
                              std::vector<Eigen::VectorXd>& eigenvectors,
                              double tolerance,
                              int max_iterations) {
    // Check if matrices are valid
    if (H.rows() != H.cols() || M.rows() != M.cols() || H.rows() != M.rows()) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    
    // Check if number of eigenvalues is valid
    if (num_eigenvalues <= 0 || num_eigenvalues > H.rows()) {
        throw std::invalid_argument("Invalid number of eigenvalues");
    }
    
    // Solve using the selected solver
    switch (solver_type_) {
        case EigensolverType::ARPACK:
            solve_arpack(H, M, num_eigenvalues, eigenvalues, eigenvectors, tolerance, max_iterations);
            break;
        case EigensolverType::SLEPC:
#ifdef USE_SLEPC
            solve_slepc(H, M, num_eigenvalues, eigenvalues, eigenvectors, tolerance, max_iterations);
#else
            throw std::runtime_error("SLEPc solver not available");
#endif
            break;
        case EigensolverType::FEAST:
            solve_feast(H, M, num_eigenvalues, eigenvalues, eigenvectors, tolerance, max_iterations);
            break;
        case EigensolverType::CUDA:
            solve_cuda(H, M, num_eigenvalues, eigenvalues, eigenvectors, tolerance, max_iterations);
            break;
        default:
            throw std::runtime_error("Unknown solver type");
    }
}

// Get solver type
EigensolverType ParallelEigensolver::get_solver_type() const {
    return solver_type_;
}

// Get number of threads
int ParallelEigensolver::get_num_threads() const {
    return num_threads_;
}

// Check if MPI is enabled
bool ParallelEigensolver::is_mpi_enabled() const {
    return use_mpi_;
}

// Get number of iterations
int ParallelEigensolver::get_num_iterations() const {
    return num_iterations_;
}

// Get error
double ParallelEigensolver::get_error() const {
    return error_;
}

#ifdef USE_SLEPC
// Initialize SLEPc
void ParallelEigensolver::initialize_slepc() {
    // Initialize SLEPc
    SlepcInitialize(nullptr, nullptr, nullptr, nullptr);
    
    // Create eigensolver context
    EPSCreate(PETSC_COMM_WORLD, &eps_);
    
    slepc_initialized_ = true;
}

// Finalize SLEPc
void ParallelEigensolver::finalize_slepc() {
    // Destroy eigensolver context
    EPSDestroy(&eps_);
    
    // Finalize SLEPc
    SlepcFinalize();
    
    slepc_initialized_ = false;
}

// Solve using SLEPc
void ParallelEigensolver::solve_slepc(const Eigen::SparseMatrix<std::complex<double>>& H,
                                    const Eigen::SparseMatrix<std::complex<double>>& M,
                                    int num_eigenvalues,
                                    std::vector<std::complex<double>>& eigenvalues,
                                    std::vector<Eigen::VectorXd>& eigenvectors,
                                    double tolerance,
                                    int max_iterations) {
    // TODO: Implement SLEPc solver
    throw std::runtime_error("SLEPc solver not implemented yet");
}
#endif

// Solve using ARPACK
void ParallelEigensolver::solve_arpack(const Eigen::SparseMatrix<std::complex<double>>& H,
                                     const Eigen::SparseMatrix<std::complex<double>>& M,
                                     int num_eigenvalues,
                                     std::vector<std::complex<double>>& eigenvalues,
                                     std::vector<Eigen::VectorXd>& eigenvectors,
                                     double tolerance,
                                     int max_iterations) {
    // Check if matrices are Hermitian
    bool is_hermitian = true;
    for (int k = 0; k < H.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(H, k); it; ++it) {
            if (std::abs(it.value() - std::conj(H.coeff(it.col(), it.row()))) > 1e-10) {
                is_hermitian = false;
                break;
            }
        }
        if (!is_hermitian) break;
    }
    
    if (!is_hermitian) {
        throw std::runtime_error("Hamiltonian matrix is not Hermitian");
    }
    
    // Convert complex matrices to real matrices
    // For Hermitian matrices, we can use the real part only
    Eigen::SparseMatrix<double> H_real(H.rows(), H.cols());
    Eigen::SparseMatrix<double> M_real(M.rows(), M.cols());
    
    // Copy real parts
    for (int k = 0; k < H.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(H, k); it; ++it) {
            H_real.insert(it.row(), it.col()) = it.value().real();
        }
    }
    
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, k); it; ++it) {
            M_real.insert(it.row(), it.col()) = it.value().real();
        }
    }
    
    // Compress matrices
    H_real.makeCompressed();
    M_real.makeCompressed();
    
    // Define the matrix operations
    Spectra::SparseSymMatProd<double> op(H_real);
    Spectra::SparseCholesky<double> Bop(M_real);
    
    // Construct the generalized eigensolver
    // ncv is the number of Ritz values to use (usually 2*num_eigenvalues is a good choice)
    int ncv = std::min(2 * num_eigenvalues, static_cast<int>(H_real.rows()));
    Spectra::SymGEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> geigs(op, Bop, num_eigenvalues, ncv);
    
    // Set parameters
    geigs.init();
    
    // Compute eigenvalues and eigenvectors
    int nconv = geigs.compute(max_iterations, tolerance, Spectra::SMALLEST_ALGE);
    
    // Check for convergence
    if (geigs.info() != Spectra::SUCCESSFUL) {
        throw std::runtime_error("ARPACK solver failed to converge");
    }
    
    // Get eigenvalues and eigenvectors
    Eigen::VectorXd evals = geigs.eigenvalues();
    Eigen::MatrixXd evecs = geigs.eigenvectors();
    
    // Store results
    eigenvalues.resize(num_eigenvalues);
    eigenvectors.resize(num_eigenvalues);
    
    for (int i = 0; i < num_eigenvalues; ++i) {
        eigenvalues[i] = std::complex<double>(evals(i), 0.0);
        eigenvectors[i] = evecs.col(i);
    }
    
    // Store number of iterations and error
    num_iterations_ = geigs.num_iterations();
    error_ = geigs.error();
}

// Solve using FEAST
void ParallelEigensolver::solve_feast(const Eigen::SparseMatrix<std::complex<double>>& H,
                                    const Eigen::SparseMatrix<std::complex<double>>& M,
                                    int num_eigenvalues,
                                    std::vector<std::complex<double>>& eigenvalues,
                                    std::vector<Eigen::VectorXd>& eigenvectors,
                                    double tolerance,
                                    int max_iterations) {
    // TODO: Implement FEAST solver
    throw std::runtime_error("FEAST solver not implemented yet");
}

// Solve using CUDA
void ParallelEigensolver::solve_cuda(const Eigen::SparseMatrix<std::complex<double>>& H,
                                   const Eigen::SparseMatrix<std::complex<double>>& M,
                                   int num_eigenvalues,
                                   std::vector<std::complex<double>>& eigenvalues,
                                   std::vector<Eigen::VectorXd>& eigenvectors,
                                   double tolerance,
                                   int max_iterations) {
#ifdef USE_CUDA
    // TODO: Implement CUDA solver
    throw std::runtime_error("CUDA solver not implemented yet");
#else
    throw std::runtime_error("CUDA solver not available");
#endif
}
