/**
 * @file schrodinger.cpp
 * @brief Implementation of the SchrodingerSolver class for quantum simulations.
 *
 * This file contains the implementation of the SchrodingerSolver class, which implements
 * methods for solving the Schrödinger equation in quantum dot simulations. The solver
 * supports GPU acceleration for higher-order elements.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "schrodinger.h"
#include "physics.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <stdexcept>

/**
 * @brief Constructs a new SchrodingerSolver object.
 *
 * @param mesh The mesh on which to solve the Schrödinger equation
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @param use_gpu Whether to use GPU acceleration (if available)
 */
SchrodingerSolver::SchrodingerSolver(Mesh& mesh,
                                 std::function<double(double, double)> m_star,
                                 std::function<double(double, double)> V,
                                 bool use_gpu)
    : mesh(mesh), m_star(m_star), V(V), use_gpu(use_gpu), gpu_accelerator(use_gpu) {
    // Assemble the matrices
    assemble_matrices();
}

/**
 * @brief Solves the Schrödinger equation.
 *
 * This method solves the generalized eigenvalue problem arising from the
 * finite element discretization of the Schrödinger equation. It computes
 * the lowest `num_eigenvalues` eigenvalues and corresponding eigenvectors.
 *
 * @param num_eigenvalues The number of eigenvalues to compute
 * @return A pair containing the eigenvalues and eigenvectors
 */
std::pair<std::vector<double>, std::vector<Eigen::VectorXd>> SchrodingerSolver::solve(int num_eigenvalues) {
    // Check if the matrices are assembled
    if (H.rows() == 0 || M.rows() == 0) {
        throw std::runtime_error("Matrices not assembled");
    }

    // Check if the number of eigenvalues is valid
    if (num_eigenvalues <= 0 || num_eigenvalues > H.rows()) {
        throw std::invalid_argument("Invalid number of eigenvalues");
    }

    // Solve the eigenvalue problem
    if (use_gpu && gpu_accelerator.is_gpu_enabled()) {
        solve_eigen_gpu(num_eigenvalues);
    } else {
        solve_eigen_cpu(num_eigenvalues);
    }

    return std::make_pair(eigenvalues, eigenvectors);
}

/**
 * @brief Assembles the Hamiltonian and mass matrices.
 *
 * This method assembles the Hamiltonian and mass matrices for the
 * finite element discretization of the Schrödinger equation.
 */
void SchrodingerSolver::assemble_matrices() {
    // Check if GPU acceleration is enabled
    if (use_gpu && gpu_accelerator.is_gpu_enabled()) {
        assemble_matrices_gpu();
    } else {
        assemble_matrices_cpu();
    }
}

/**
 * @brief Assembles the Hamiltonian and mass matrices on the CPU.
 *
 * This method assembles the Hamiltonian and mass matrices on the CPU.
 */
void SchrodingerSolver::assemble_matrices_cpu() {
    // Get mesh data
    int num_nodes = mesh.getNumNodes();
    int num_elements = mesh.getNumElements();
    int element_order = mesh.getElementOrder();
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();

    // Initialize matrices
    H.resize(num_nodes, num_nodes);
    M.resize(num_nodes, num_nodes);

    // Create triplet lists for sparse matrix assembly
    std::vector<Eigen::Triplet<std::complex<double>>> H_triplets;
    std::vector<Eigen::Triplet<std::complex<double>>> M_triplets;

    // Reserve space for triplets
    H_triplets.reserve(num_elements * 9);  // Assuming 3 nodes per element
    M_triplets.reserve(num_elements * 9);  // Assuming 3 nodes per element

    // Assemble matrices element by element
    for (int e = 0; e < num_elements; ++e) {
        // Get element nodes
        std::array<int, 3> element = elements[e];
        int n1 = element[0];
        int n2 = element[1];
        int n3 = element[2];

        // Get node coordinates
        double x1 = nodes[n1][0];
        double y1 = nodes[n1][1];
        double x2 = nodes[n2][0];
        double y2 = nodes[n2][1];
        double x3 = nodes[n3][0];
        double y3 = nodes[n3][1];

        // Calculate element area
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Calculate shape function gradients
        double b1 = (y2 - y3) / (2.0 * area);
        double c1 = (x3 - x2) / (2.0 * area);
        double b2 = (y3 - y1) / (2.0 * area);
        double c2 = (x1 - x3) / (2.0 * area);
        double b3 = (y1 - y2) / (2.0 * area);
        double c3 = (x2 - x1) / (2.0 * area);

        // Calculate element centroid
        double xc = (x1 + x2 + x3) / 3.0;
        double yc = (y1 + y2 + y3) / 3.0;

        // Get effective mass and potential at centroid
        double m = m_star(xc, yc);
        double V_val = V(xc, yc);

        // Assemble element matrices
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Get global node indices
                int ni = element[i];
                int nj = element[j];

                // Calculate gradients of shape functions
                double dNi_dx, dNi_dy, dNj_dx, dNj_dy;

                if (i == 0) { dNi_dx = b1; dNi_dy = c1; }
                else if (i == 1) { dNi_dx = b2; dNi_dy = c2; }
                else { dNi_dx = b3; dNi_dy = c3; }

                if (j == 0) { dNj_dx = b1; dNj_dy = c1; }
                else if (j == 1) { dNj_dx = b2; dNj_dy = c2; }
                else { dNj_dx = b3; dNj_dy = c3; }

                // Calculate shape functions at centroid
                double Ni, Nj;

                if (i == 0) Ni = 1.0/3.0;
                else if (i == 1) Ni = 1.0/3.0;
                else Ni = 1.0/3.0;

                if (j == 0) Nj = 1.0/3.0;
                else if (j == 1) Nj = 1.0/3.0;
                else Nj = 1.0/3.0;

                // Calculate Hamiltonian matrix element
                // Use the reduced Planck constant in eV·s
                const double hbar = 6.582119569e-16; // eV·s
                double kinetic_term = (hbar * hbar / (2.0 * m)) * (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * area;
                double potential_term = V_val * Ni * Nj * area;
                double H_ij = kinetic_term + potential_term;

                // Calculate mass matrix element
                double M_ij = Ni * Nj * area;

                // Add to triplet lists
                H_triplets.emplace_back(ni, nj, std::complex<double>(H_ij, 0.0));
                M_triplets.emplace_back(ni, nj, std::complex<double>(M_ij, 0.0));
            }
        }
    }

    // Set matrices from triplets
    H.setFromTriplets(H_triplets.begin(), H_triplets.end());
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());

    // Compress matrices
    H.makeCompressed();
    M.makeCompressed();
}

/**
 * @brief Assembles the Hamiltonian and mass matrices on the GPU.
 *
 * This method assembles the Hamiltonian and mass matrices on the GPU.
 */
// Static wrapper functions to convert std::function to function pointers
static double m_star_static(double x, double y) {
    // This is a placeholder that will be replaced at runtime
    return 0.0;
}

static double V_static(double x, double y) {
    // This is a placeholder that will be replaced at runtime
    return 0.0;
}

// Global variables to store the std::function objects
static std::function<double(double, double)> g_m_star;
static std::function<double(double, double)> g_V;

// Static wrapper functions that call the global std::function objects
static double m_star_wrapper(double x, double y) {
    return g_m_star(x, y);
}

static double V_wrapper(double x, double y) {
    return g_V(x, y);
}

void SchrodingerSolver::assemble_matrices_gpu() {
    // Get mesh data
    int element_order = mesh.getElementOrder();

    // Initialize matrices
    H.resize(mesh.getNumNodes(), mesh.getNumNodes());
    M.resize(mesh.getNumNodes(), mesh.getNumNodes());

    // Set the global std::function objects
    g_m_star = m_star;
    g_V = V;

    // Use the GPU accelerator to assemble the matrices
    try {
        if (element_order > 1) {
            // Use specialized implementation for higher-order elements
            gpu_accelerator.assemble_higher_order_matrices(mesh, m_star_wrapper, V_wrapper, element_order, H, M);
        } else {
            // Use standard implementation for linear elements
            gpu_accelerator.assemble_matrices(mesh, m_star_wrapper, V_wrapper, element_order, H, M);
        }
    } catch (const std::exception& e) {
        std::cerr << "GPU acceleration failed: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU implementation." << std::endl;
        assemble_matrices_cpu();
    }
}

/**
 * @brief Solves the generalized eigenvalue problem on the CPU.
 *
 * This method solves the generalized eigenvalue problem on the CPU.
 *
 * @param num_eigenvalues The number of eigenvalues to compute
 */
void SchrodingerSolver::solve_eigen_cpu(int num_eigenvalues) {
    // Convert sparse matrices to dense
    Eigen::MatrixXcd H_dense = Eigen::MatrixXcd(H);
    Eigen::MatrixXcd M_dense = Eigen::MatrixXcd(M);

    // Make sure matrices are Hermitian
    H_dense = (H_dense + H_dense.adjoint()) / 2.0;
    M_dense = (M_dense + M_dense.adjoint()) / 2.0;

    // Compute the Cholesky decomposition of M
    Eigen::LLT<Eigen::MatrixXcd> llt(M_dense);
    if (llt.info() != Eigen::Success) {
        // M is not positive definite, try to regularize it
        double reg = 1e-10;
        while (reg < 1.0) {
            Eigen::MatrixXcd M_reg = M_dense + reg * Eigen::MatrixXcd::Identity(M_dense.rows(), M_dense.cols());
            llt.compute(M_reg);
            if (llt.info() == Eigen::Success) {
                M_dense = M_reg;
                break;
            }
            reg *= 10.0;
        }

        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("Failed to compute Cholesky decomposition of mass matrix");
        }
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
        throw std::runtime_error("Eigenvalue computation failed");
    }

    // Get the eigenvalues and eigenvectors
    Eigen::VectorXd evals = es.eigenvalues();
    Eigen::MatrixXcd evecs = es.eigenvectors();

    // Transform the eigenvectors back to the original problem: ψ = L⁻†·x
    Eigen::MatrixXcd psi = L.adjoint().triangularView<Eigen::Upper>().solve(evecs);

    // Store the eigenvalues and eigenvectors
    eigenvalues.resize(num_eigenvalues);
    eigenvectors.resize(num_eigenvalues);

    for (int i = 0; i < num_eigenvalues; ++i) {
        eigenvalues[i] = evals(i);
        eigenvectors[i] = psi.col(i).real();
    }
}

/**
 * @brief Solves the generalized eigenvalue problem on the GPU.
 *
 * This method solves the generalized eigenvalue problem on the GPU.
 *
 * @param num_eigenvalues The number of eigenvalues to compute
 */
void SchrodingerSolver::solve_eigen_gpu(int num_eigenvalues) {
    try {
        // Check if the matrices are sparse
        if (H.nonZeros() < 0.1 * H.rows() * H.cols() && M.nonZeros() < 0.1 * M.rows() * M.cols()) {
            // Use sparse eigensolver for sparse matrices
            std::vector<std::complex<double>> complex_eigenvalues;
            gpu_accelerator.solve_eigen_sparse(H, M, num_eigenvalues, complex_eigenvalues, eigenvectors);

            // Convert complex eigenvalues to real
            eigenvalues.resize(complex_eigenvalues.size());
            for (size_t i = 0; i < complex_eigenvalues.size(); ++i) {
                eigenvalues[i] = complex_eigenvalues[i].real();
            }
        } else {
            // Convert sparse matrices to dense
            Eigen::MatrixXcd H_dense = Eigen::MatrixXcd(H);
            Eigen::MatrixXcd M_dense = Eigen::MatrixXcd(M);

            // Make sure matrices are Hermitian
            H_dense = (H_dense + H_dense.adjoint()) / 2.0;
            M_dense = (M_dense + M_dense.adjoint()) / 2.0;

            // Use dense eigensolver for dense matrices
            std::vector<std::complex<double>> complex_eigenvalues;
            gpu_accelerator.solve_eigen_cusolver(H_dense, M_dense, num_eigenvalues, complex_eigenvalues, eigenvectors);

            // Convert complex eigenvalues to real
            eigenvalues.resize(complex_eigenvalues.size());
            for (size_t i = 0; i < complex_eigenvalues.size(); ++i) {
                eigenvalues[i] = complex_eigenvalues[i].real();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "GPU acceleration failed: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU implementation." << std::endl;
        solve_eigen_cpu(num_eigenvalues);
    }
}
