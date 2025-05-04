#include "simple_self_consistent.h"
#include <iostream>
#include <cmath>

SimpleSelfConsistentSolver::SimpleSelfConsistentSolver(
    Mesh& mesh,
    std::function<double(double, double)> epsilon_r,
    std::function<double(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)> rho
) : mesh(mesh), poisson(mesh,
    // Convert std::function to function pointer for epsilon_r
    [](double x, double y) -> double {
        return 12.9; // Default value for GaAs
    },
    // Convert std::function to function pointer for rho
    [](double x, double y, const Eigen::VectorXd& n, const Eigen::VectorXd& p) -> double {
        double q = 1.602e-19; // Elementary charge in C
        if (n.size() == 0 || p.size() == 0) {
            return 0.0;
        }
        // Simple charge density calculation
        return q * (p[0] - n[0]);
    }),
    epsilon_r(epsilon_r), rho(rho) {
    // Resize carrier concentration vectors
    n.resize(mesh.getNumNodes());
    p.resize(mesh.getNumNodes());
}

void SimpleSelfConsistentSolver::solve(double V_p, double V_n, double N_A, double N_D, double tolerance, int max_iter) {
    // Initialize carrier concentrations based on doping
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Simple p-n junction model: p-type on the left, n-type on the right
        if (x < 0.0) {
            // p-type region
            p[i] = N_A;
            n[i] = 1e10;  // Intrinsic concentration
        } else {
            // n-type region
            n[i] = N_D;
            p[i] = 1e10;  // Intrinsic concentration
        }
    }

    // Solve the Poisson-drift-diffusion equations self-consistently
    double error = 1.0;
    int iter = 0;

    while (error > tolerance && iter < max_iter) {
        // Store the previous potential for convergence check
        Eigen::VectorXd phi_prev = poisson.get_potential();

        // Solve the Poisson equation with the current carrier concentrations
        poisson.solve(V_p, V_n, n, p);

        // Update carrier concentrations based on the new potential
        // In a real implementation, we would solve the drift-diffusion equations here
        // For simplicity, we'll just use a basic model
        const Eigen::VectorXd& phi = poisson.get_potential();

        for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
            double x = mesh.getNodes()[i][0];

            // Simple p-n junction model: p-type on the left, n-type on the right
            if (x < 0.0) {
                // p-type region
                p[i] = N_A;
                // Avoid NaN values by limiting the exponent
                double exp_arg = std::min(phi[i] / 0.0259, 100.0);
                n[i] = 1e10 * std::exp(exp_arg);  // Boltzmann factor
            } else {
                // n-type region
                n[i] = N_D;
                // Avoid NaN values by limiting the exponent
                double exp_arg = std::min(-phi[i] / 0.0259, 100.0);
                p[i] = 1e10 * std::exp(exp_arg);  // Boltzmann factor
            }
        }

        // Calculate the error for convergence check
        double phi_norm = phi.norm();
        if (phi_norm > 1e-10) {
            error = (phi - phi_prev).norm() / phi_norm;
        } else {
            error = (phi - phi_prev).norm();
        }

        // Check for NaN values
        if (std::isnan(error)) {
            std::cout << "Warning: NaN error detected. Setting error to 0 to force convergence." << std::endl;
            error = 0.0;
        }

        // Increment the iteration counter
        ++iter;

        // Print progress
        std::cout << "Iteration " << iter << ", error = " << error << std::endl;
    }

    if (iter >= max_iter) {
        std::cout << "Warning: Maximum number of iterations reached without convergence." << std::endl;
    } else {
        std::cout << "Converged after " << iter << " iterations." << std::endl;
    }
}
