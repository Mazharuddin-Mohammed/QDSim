#pragma once
#include "mesh.h"
#include <Eigen/Sparse>
#include <mpi.h>

class PoissonSolver {
public:
    PoissonSolver(Mesh& mesh, double (*epsilon_r)(double, double), double (*rho)(double, double));
    void solve(double V_p, double V_n); // V_p, V_n: potentials at p/n boundaries
    const Eigen::VectorXd& get_potential() const { return phi; }
    Eigen::Vector2d get_electric_field(double x, double y) const;

private:
    Mesh& mesh;
    Eigen::SparseMatrix<double> K; // Stiffness matrix
public:
    Eigen::VectorXd phi;           // Electrostatic potential
private:
    Eigen::VectorXd f;             // Right-hand side
    double (*epsilon_r)(double, double); // Dielectric constant
    double (*rho)(double, double);       // Charge density
    void assemble_matrix();
    void assemble_rhs();
    void apply_boundary_conditions(double V_p, double V_n);
};