#include "poisson.h"
#include <Eigen/SparseCholesky>

PoissonSolver::PoissonSolver(Mesh& mesh, double (*epsilon_r)(double, double), double (*rho)(double, double))
    : mesh(mesh), epsilon_r(epsilon_r), rho(rho) {
    K.resize(mesh.get_num_nodes(), mesh.get_num_nodes());
    phi.resize(mesh.get_num_nodes());
    f.resize(mesh.get_num_nodes());
}

void PoissonSolver::assemble_matrix() {
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t e = 0; e < mesh.get_num_elements(); ++e) {
        auto element = mesh.get_element(e);
        auto nodes = mesh.get_element_nodes(e);
        // 3-point Gaussian quadrature for P1 elements
        for (int q = 0; q < 3; ++q) {
            double x = 0.0, y = 0.0; // Compute quadrature point
            double eps = epsilon_r(x, y) * 8.854e-12; // F/m
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    // Gradient of basis functions
                    double grad_i_dot_grad_j = 1.0; // Compute using mesh geometry
                    triplets.emplace_back(element[i], element[j], eps * grad_i_dot_grad_j);
                }
            }
        }
    }
    K.setFromTriplets(triplets.begin(), triplets.end());
}

void PoissonSolver::assemble_rhs() {
    f.setZero();
    for (size_t e = 0; e < mesh.get_num_elements(); ++e) {
        auto nodes = mesh.get_element_nodes(e);
        for (int q = 0; q < 3; ++q) {
            double x = 0.0, y = 0.0; // Compute quadrature point
            double charge = rho(x, y);
            for (int i = 0; i < 3; ++i) {
                f[mesh.get_element(e)[i]] += charge * 1.0; // Basis function value
            }
        }
    }
}

void PoissonSolver::apply_boundary_conditions(double V_p, double V_n) {
    // Apply Dirichlet BC: phi = V_p at x = -Lx/2, phi = V_n at x = Lx/2
    for (size_t i = 0; i < mesh.get_num_nodes(); ++i) {
        auto [x, y] = mesh.get_node(i);
        if (std::abs(x - (-mesh.get_Lx() / 2)) < 1e-10) {
            phi[i] = V_p;
            K.row(i).setZero();
            K.coeffRef(i, i) = 1.0;
            f[i] = V_p;
        } else if (std::abs(x - (mesh.get_Lx() / 2)) < 1e-10) {
            phi[i] = V_n;
            K.row(i).setZero();
            K.coeffRef(i, i) = 1.0;
            f[i] = V_n;
        }
    }
}

void PoissonSolver::solve(double V_p, double V_n) {
    assemble_matrix();
    assemble_rhs();
    apply_boundary_conditions(V_p, V_n);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);
    phi = solver.solve(f);
}

Eigen::Vector2d PoissonSolver::get_electric_field(double x, double y) const {
    // Approximate E = -grad phi using finite differences or basis function gradients
    Eigen::Vector2d E(0.0, 0.0);
    // Implementation TBD based on mesh and phi
    return E;
}