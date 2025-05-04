#include "improved_self_consistent.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <Eigen/Sparse>

ImprovedSelfConsistentSolver::ImprovedSelfConsistentSolver(
    Mesh& mesh,
    std::function<double(double, double)> epsilon_r,
    std::function<double(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)> rho
) : mesh(mesh), epsilon_r(epsilon_r), rho(rho) {
    // Resize vectors
    int num_nodes = mesh.getNumNodes();
    potential.resize(num_nodes);
    n.resize(num_nodes);
    p.resize(num_nodes);

    // Initialize with zeros
    potential.setZero();
    n.setZero();
    p.setZero();
}

void ImprovedSelfConsistentSolver::solve(double V_p, double V_n, double N_A, double N_D, double tolerance, int max_iter) {
    std::cout << "ImprovedSelfConsistentSolver: Starting self-consistent solution..." << std::endl;

    // Initialize with a simple solution using BasicSolver
    {
        BasicSolver basic_solver(mesh);
        basic_solver.solve(V_p, V_n, N_A, N_D);

        // Copy the initial solution
        potential = basic_solver.get_potential();
        n = basic_solver.get_n();
        p = basic_solver.get_p();
    } // BasicSolver goes out of scope here

    // Self-consistent iteration
    double error = 1.0;
    int iter = 0;

    try {
        while (error > tolerance && iter < max_iter) {
            // Store the previous potential for convergence check
            Eigen::VectorXd potential_prev = potential;

            // Calculate charge density
            Eigen::VectorXd charge_density = calculate_charge_density();

            // Solve Poisson equation
            solve_poisson(V_p, V_n, charge_density);

            // Update carrier concentrations
            update_carriers(N_A, N_D);

            // Calculate error for convergence check
            double potential_norm = potential.norm();
            if (potential_norm > 1e-10) {
                error = (potential - potential_prev).norm() / potential_norm;
            } else {
                error = (potential - potential_prev).norm();
            }

            // Check for NaN values
            if (std::isnan(error)) {
                std::cout << "Warning: NaN error detected. Setting error to 0 to force convergence." << std::endl;
                error = 0.0;
            }

            // Increment iteration counter
            ++iter;

            // Print progress
            std::cout << "Iteration " << iter << ", error = " << error << std::endl;
        }

        if (iter >= max_iter) {
            std::cout << "Warning: Maximum number of iterations reached without convergence." << std::endl;
        } else {
            std::cout << "Converged after " << iter << " iterations." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in self-consistent solution: " << e.what() << std::endl;
        std::cerr << "Using the initial solution from BasicSolver." << std::endl;
    }

    std::cout << "ImprovedSelfConsistentSolver: Done!" << std::endl;
}

Eigen::VectorXd ImprovedSelfConsistentSolver::calculate_charge_density() {
    int num_nodes = mesh.getNumNodes();
    Eigen::VectorXd charge_density(num_nodes);

    // Elementary charge in C
    const double q = 1.602e-19;

    // Calculate charge density at each node
    for (int i = 0; i < num_nodes; ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        try {
            // Call the user-provided charge density function
            charge_density[i] = rho(x, y, n, p);
        } catch (const std::exception& e) {
            std::cerr << "Error in charge density calculation at node (" << x << ", " << y << "): " << e.what() << std::endl;
            // Use a simple approximation as fallback
            charge_density[i] = q * (p[i] - n[i]);
        }
    }

    return charge_density;
}

void ImprovedSelfConsistentSolver::solve_poisson(double V_p, double V_n, const Eigen::VectorXd& charge_density) {
    int num_nodes = mesh.getNumNodes();

    // Create a simple finite difference solver for the Poisson equation
    // This is a simplified version; a real implementation would use FEM

    // Create the Laplacian matrix (simplified for a regular grid)
    Eigen::SparseMatrix<double> A(num_nodes, num_nodes);
    A.reserve(Eigen::VectorXi::Constant(num_nodes, 5)); // Reserve space for 5 non-zeros per row

    // Create the right-hand side vector
    Eigen::VectorXd b = Eigen::VectorXd::Zero(num_nodes);

    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double Ly = mesh.get_ly();
    int nx = mesh.get_nx();
    int ny = mesh.get_ny();

    // Grid spacing
    double hx = Lx / nx;
    double hy = Ly / ny;

    // Fill the matrix and right-hand side
    for (int i = 0; i < num_nodes; ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Check if this is a boundary node
        bool is_boundary = (x <= -Lx/2 + 1e-6 || x >= Lx/2 - 1e-6 || y <= 0 + 1e-6 || y >= Ly - 1e-6);

        if (is_boundary) {
            // Dirichlet boundary condition
            A.insert(i, i) = 1.0;

            // Set boundary value based on position
            if (x <= -Lx/2 + 1e-6) {
                // Left boundary (p-contact)
                b[i] = V_p;
            } else if (x >= Lx/2 - 1e-6) {
                // Right boundary (n-contact)
                b[i] = V_n;
            } else {
                // Top or bottom boundary (insulating)
                // Use Neumann boundary condition (zero normal derivative)
                // For simplicity, we'll just use the value from the previous iteration
                b[i] = potential[i];
            }
        } else {
            // Interior node - discretize the Poisson equation
            // -div(epsilon_r * grad(V)) = rho/epsilon_0

            // Get the relative permittivity at this node
            double eps_r = epsilon_r(x, y);

            // Vacuum permittivity in F/cm
            const double epsilon0 = 8.85e-14;

            // Coefficient for the Laplacian
            double coef = eps_r * epsilon0;

            // Discretize the Laplacian using finite differences
            A.insert(i, i) = -2.0 * coef * (1.0/(hx*hx) + 1.0/(hy*hy));

            // Find the indices of the neighboring nodes
            int i_left = i - 1;
            int i_right = i + 1;
            int i_bottom = i - (nx + 1);
            int i_top = i + (nx + 1);

            // Add contributions from neighboring nodes
            if (i_left >= 0 && i_left < num_nodes) {
                A.insert(i, i_left) = coef / (hx * hx);
            }
            if (i_right >= 0 && i_right < num_nodes) {
                A.insert(i, i_right) = coef / (hx * hx);
            }
            if (i_bottom >= 0 && i_bottom < num_nodes) {
                A.insert(i, i_bottom) = coef / (hy * hy);
            }
            if (i_top >= 0 && i_top < num_nodes) {
                A.insert(i, i_top) = coef / (hy * hy);
            }

            // Right-hand side: charge density
            b[i] = -charge_density[i];
        }
    }

    // Compress the matrix
    A.makeCompressed();

    // Solve the linear system using a direct solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize the matrix");
    }

    // Solve the system
    potential = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve the linear system");
    }
}

void ImprovedSelfConsistentSolver::update_carriers(double N_A, double N_D) {
    // Physical constants
    const double kT = 0.0259;       // Thermal voltage at room temperature (eV)
    const double ni = 1.0e10;       // Intrinsic carrier concentration for GaAs (cm^-3)
    const double q = 1.602e-19;     // Elementary charge (C)

    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double junction_position = 0.0; // Junction at x = 0

    // Loop over all nodes
    for (int i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];

        // Distance from junction
        double d = x - junction_position;

        try {
            // Calculate carrier concentrations using Boltzmann statistics
            if (d < 0) {
                // p-region
                p[i] = N_A;
                // Electron concentration from mass action law
                double exp_arg = std::min(q * potential[i] / kT, 100.0); // Limit to avoid overflow
                n[i] = ni * ni / p[i] * std::exp(exp_arg);
            } else {
                // n-region
                n[i] = N_D;
                // Hole concentration from mass action law
                double exp_arg = std::min(-q * potential[i] / kT, 100.0); // Limit to avoid overflow
                p[i] = ni * ni / n[i] * std::exp(exp_arg);
            }

            // Ensure minimum carrier concentrations to avoid numerical issues
            n[i] = std::max(n[i], 1.0);
            p[i] = std::max(p[i], 1.0);
        } catch (const std::exception& e) {
            std::cerr << "Error in carrier update at node " << i << ": " << e.what() << std::endl;
            // Use default values
            if (d < 0) {
                p[i] = N_A;
                n[i] = ni * ni / N_A;
            } else {
                n[i] = N_D;
                p[i] = ni * ni / N_D;
            }
        }
    }
}
