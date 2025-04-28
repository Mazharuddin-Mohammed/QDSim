#pragma once
#include "mesh.h"
#include "poisson.h"
#include <Eigen/Sparse>
#ifdef USE_MPI
#include <mpi.h>
#endif

class FEMSolver {
public:
    FEMSolver(Mesh& mesh, double (*m_star)(double, double), double (*V)(double, double),
              double (*cap)(double, double), PoissonSolver& poisson, int order, bool use_mpi = true);
    void assemble_matrices();
    void adapt_mesh(const Eigen::VectorXd& eigenvector, double threshold, const std::string& cache_dir);
    const Eigen::SparseMatrix<std::complex<double>>& get_H() const { return H; }
    const Eigen::SparseMatrix<std::complex<double>>& get_M() const { return M; }
    bool is_using_mpi() const { return use_mpi; }

private:
    Mesh& mesh;
    PoissonSolver& poisson;
    Eigen::SparseMatrix<std::complex<double>> H; // Hamiltonian
    Eigen::SparseMatrix<std::complex<double>> M; // Mass matrix
    double (*m_star)(double, double);
    double (*V)(double, double);
    double (*cap)(double, double);
    int order; // 1 (P1), 2 (P2), 3 (P3)
    bool use_mpi; // Flag to enable/disable MPI
    void assemble_element_matrix(size_t e, Eigen::MatrixXcd& H_e, Eigen::MatrixXcd& M_e);
    void assemble_matrices_serial();
#ifdef USE_MPI
    void assemble_matrices_parallel();
#endif
};

