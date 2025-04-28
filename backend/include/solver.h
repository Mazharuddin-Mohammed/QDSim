#pragma once
#include "fem.h"
#include <Eigen/Sparse>
#include <vector>
#include <complex>

class EigenSolver {
public:
    EigenSolver(FEMSolver& fem);
    void solve(int num_eigenvalues);
    const std::vector<std::complex<double>>& get_eigenvalues() const { return eigenvalues; }
    const std::vector<Eigen::VectorXd>& get_eigenvectors() const { return eigenvectors; }

private:
    FEMSolver& fem;
    std::vector<std::complex<double>> eigenvalues;
    std::vector<Eigen::VectorXd> eigenvectors;
};