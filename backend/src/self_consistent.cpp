#include "self_consistent.h"
#include <Eigen/SparseCholesky>
#include <cmath>
#include <iostream>
#include <stdexcept>

/**
 * @brief Constructs a new SelfConsistentSolver object.
 *
 * This constructor initializes the SelfConsistentSolver with the given mesh and callback functions
 * for computing physical quantities. It also initializes the PoissonSolver and resizes the carrier
 * concentration vectors and drift-diffusion matrices.
 *
 * @param mesh The mesh to use for the simulation
 * @param epsilon_r Function that returns the relative permittivity at a given position
 * @param rho Function that returns the charge density at a given position
 * @param n_conc Function that returns the electron concentration at a given position
 * @param p_conc Function that returns the hole concentration at a given position
 * @param mu_n Function that returns the electron mobility at a given position
 * @param mu_p Function that returns the hole mobility at a given position
 */
SelfConsistentSolver::SelfConsistentSolver(
    Mesh& mesh,
    double (*epsilon_r)(double, double),
    double (*rho)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&),
    double (*n_conc)(double, double, double, const Materials::Material&),
    double (*p_conc)(double, double, double, const Materials::Material&),
    double (*mu_n)(double, double, const Materials::Material&),
    double (*mu_p)(double, double, const Materials::Material&))
    : mesh(mesh), poisson(mesh, epsilon_r, rho),
      epsilon_r(epsilon_r), rho(rho), n_conc(n_conc), p_conc(p_conc), mu_n(mu_n), mu_p(mu_p) {

    // Resize carrier concentration vectors and drift-diffusion matrices
    n.resize(mesh.getNumNodes());
    p.resize(mesh.getNumNodes());
    Kn.resize(mesh.getNumNodes(), mesh.getNumNodes());
    Kp.resize(mesh.getNumNodes(), mesh.getNumNodes());

    // Initialize with zeros
    n.setZero();
    p.setZero();
}

/**
 * @brief Initializes the carrier concentrations based on the doping concentrations.
 *
 * This function initializes the electron and hole concentrations based on the doping
 * concentrations and the depletion approximation. It assumes a p-n junction with the
 * p-side at x < 0 and the n-side at x > 0.
 *
 * @param N_A The acceptor doping concentration
 * @param N_D The donor doping concentration
 */
void SelfConsistentSolver::initialize_carriers(double N_A, double N_D) {
    // Constants
    const double kT = 0.0259; // eV at 300K
    const double ni = 1e10; // Intrinsic carrier concentration (simplified)

    // Get the potential from the Poisson solver
    const Eigen::VectorXd& phi = poisson.get_potential();

    // Initialize carrier concentrations based on doping and potential
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties based on position
        Materials::Material mat;
        if (x < 0) {
            // p-side: use material properties for p-type region
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value
        } else {
            // n-side: use material properties for n-type region
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value
        }

        // Calculate carrier concentrations using the Boltzmann approximation
        if (i < phi.size()) {
            n[i] = n_conc(x, y, phi[i], mat);
            p[i] = p_conc(x, y, phi[i], mat);
        } else {
            // Fallback to depletion approximation if potential is not available
            if (x < 0) {
                n[i] = ni * ni / N_A; // Low electron concentration in p-side
                p[i] = N_A;           // High hole concentration in p-side
            } else {
                n[i] = N_D;           // High electron concentration in n-side
                p[i] = ni * ni / N_D; // Low hole concentration in n-side
            }
        }

        // Ensure minimum carrier concentrations for numerical stability
        n[i] = std::max(n[i], 1e5);
        p[i] = std::max(p[i], 1e5);
    }
}

/**
 * @brief Assembles the drift-diffusion matrices for electron and hole transport.
 *
 * This function assembles the drift-diffusion matrices for electron and hole transport
 * using the finite element method. It computes the drift and diffusion terms for each
 * element and assembles them into the global matrices.
 */
void SelfConsistentSolver::assemble_drift_diffusion_matrices() {
    // Clear the matrices
    Kn.setZero();
    Kp.setZero();

    // Get the potential from the Poisson solver
    const Eigen::VectorXd& phi = poisson.get_potential();

    // Triplet lists for sparse matrix assembly
    std::vector<Eigen::Triplet<double>> Kn_triplets, Kp_triplets;

    // Constants
    const double kT = 0.0259; // eV at 300K
    const double q = 1.602e-19; // Elementary charge in C

    // Loop over all elements
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        // Get element nodes and coordinates
        auto element = mesh.getElements()[e];
        std::vector<std::pair<double, double>> nodes;
        for (int i = 0; i < 3; ++i) {
            const Eigen::Vector2d& node = mesh.getNodes()[element[i]];
            nodes.push_back(std::make_pair(node[0], node[1]));
        }

        // Calculate element area
        double x1 = nodes[0].first, y1 = nodes[0].second;
        double x2 = nodes[1].first, y2 = nodes[1].second;
        double x3 = nodes[2].first, y3 = nodes[2].second;
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Calculate shape function gradients
        double b1 = (y2 - y3) / (2.0 * area);
        double b2 = (y3 - y1) / (2.0 * area);
        double b3 = (y1 - y2) / (2.0 * area);
        double c1 = (x3 - x2) / (2.0 * area);
        double c2 = (x1 - x3) / (2.0 * area);
        double c3 = (x2 - x1) / (2.0 * area);

        // Calculate element centroid
        double xc = (x1 + x2 + x3) / 3.0;
        double yc = (y1 + y2 + y3) / 3.0;

        // Get material properties at the centroid
        Materials::Material mat;
        if (xc < 0) {
            // p-side: use material properties for p-type region
            mat.mu_n = 0.3;  // Example value
            mat.mu_p = 0.02; // Example value
        } else {
            // n-side: use material properties for n-type region
            mat.mu_n = 0.3;  // Example value
            mat.mu_p = 0.02; // Example value
        }

        // Get mobilities at the centroid
        double mu_n_val = mu_n(xc, yc, mat);
        double mu_p_val = mu_p(xc, yc, mat);

        // Calculate diffusion coefficients using Einstein relation
        double D_n = kT * mu_n_val / q;
        double D_p = kT * mu_p_val / q;

        // Get electric field at the centroid
        Eigen::Vector2d E = poisson.get_electric_field(xc, yc);

        // Assemble element matrices
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Calculate drift term
                double drift_n = mu_n_val * (E[0] * b1 + E[1] * c1) * area / 3.0;
                double drift_p = mu_p_val * (E[0] * b1 + E[1] * c1) * area / 3.0;

                // Calculate diffusion term
                double diff_n = D_n * (b1 * b1 + c1 * c1) * area;
                double diff_p = D_p * (b1 * b1 + c1 * c1) * area;

                // Add to triplet lists
                Kn_triplets.emplace_back(element[i], element[j], drift_n + diff_n);
                Kp_triplets.emplace_back(element[i], element[j], -drift_p + diff_p); // Note the sign change for holes
            }
        }
    }

    // Set matrices from triplets
    Kn.setFromTriplets(Kn_triplets.begin(), Kn_triplets.end());
    Kp.setFromTriplets(Kp_triplets.begin(), Kp_triplets.end());
}

/**
 * @brief Solves the drift-diffusion equations for electron and hole transport.
 *
 * This function solves the drift-diffusion equations for electron and hole transport
 * using the assembled matrices and the current potential. It updates the carrier
 * concentrations based on the solution.
 */
void SelfConsistentSolver::solve_drift_diffusion() {
    // Assemble the drift-diffusion matrices
    assemble_drift_diffusion_matrices();

    // Create solvers for the linear systems
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_n, solver_p;

    // Compute the factorization of the matrices
    solver_n.compute(Kn);
    solver_p.compute(Kp);

    // Check if the factorization was successful
    if (solver_n.info() != Eigen::Success || solver_p.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize drift-diffusion matrices");
    }

    // Create right-hand side vectors
    Eigen::VectorXd f_n(mesh.getNumNodes()), f_p(mesh.getNumNodes());

    // Get the potential from the Poisson solver
    const Eigen::VectorXd& phi = poisson.get_potential();

    // Calculate the right-hand side vectors
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties based on position
        Materials::Material mat;
        if (x < 0) {
            // p-side: use material properties for p-type region
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value
        } else {
            // n-side: use material properties for n-type region
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value
        }

        // Calculate generation-recombination term (simplified)
        double G = 0.0; // Generation rate
        double R = 0.0; // Recombination rate

        // Set right-hand side values
        f_n[i] = G - R;
        f_p[i] = G - R;
    }

    // Solve the linear systems
    Eigen::VectorXd delta_n = solver_n.solve(f_n);
    Eigen::VectorXd delta_p = solver_p.solve(f_p);

    // Check if the solve was successful
    if (solver_n.info() != Eigen::Success || solver_p.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve drift-diffusion equations");
    }

    // Update carrier concentrations
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        n[i] += delta_n[i];
        p[i] += delta_p[i];

        // Ensure positive carrier concentrations
        n[i] = std::max(n[i], 1e5);
        p[i] = std::max(p[i], 1e5);
    }
}

/**
 * @brief Applies boundary conditions to the carrier concentrations.
 *
 * This function applies boundary conditions to the carrier concentrations based on
 * the applied voltages and the position of the contacts. It sets the carrier
 * concentrations at the contacts to their equilibrium values.
 *
 * @param V_p The voltage applied to the p-contact
 * @param V_n The voltage applied to the n-contact
 */
void SelfConsistentSolver::apply_boundary_conditions(double V_p, double V_n) {
    // Constants
    const double kT = 0.0259; // eV at 300K
    const double ni = 1e10; // Intrinsic carrier concentration (simplified)

    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double Ly = mesh.get_ly();

    // Loop over all nodes
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Check if the node is at the p-contact (left boundary)
        if (std::abs(x - (-Lx / 2)) < 1e-10) {
            // Get material properties for p-type region
            Materials::Material mat;
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value

            // Calculate equilibrium carrier concentrations
            n[i] = n_conc(x, y, V_p, mat);
            p[i] = p_conc(x, y, V_p, mat);
        }
        // Check if the node is at the n-contact (right boundary)
        else if (std::abs(x - (Lx / 2)) < 1e-10) {
            // Get material properties for n-type region
            Materials::Material mat;
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value

            // Calculate equilibrium carrier concentrations
            n[i] = n_conc(x, y, V_n, mat);
            p[i] = p_conc(x, y, V_n, mat);
        }
    }
}

/**
 * @brief Solves the self-consistent Poisson-drift-diffusion equations.
 *
 * This function solves the self-consistent Poisson-drift-diffusion equations
 * using an iterative approach. It alternates between solving the Poisson equation
 * and the drift-diffusion equations until convergence is reached or the maximum
 * number of iterations is exceeded.
 *
 * @param V_p The voltage applied to the p-contact
 * @param V_n The voltage applied to the n-contact
 * @param N_A The acceptor doping concentration
 * @param N_D The donor doping concentration
 * @param tolerance The convergence tolerance
 * @param max_iter The maximum number of iterations
 */
void SelfConsistentSolver::solve(double V_p, double V_n, double N_A, double N_D,
                                double tolerance, int max_iter) {
    try {
        // Initialize carrier concentrations
        initialize_carriers(N_A, N_D);

        // Get initial potential
        Eigen::VectorXd phi_old = get_potential();

        // Iterative solution
        for (int iter = 0; iter < max_iter; ++iter) {
            // Solve Poisson equation with current carrier concentrations
            poisson.solve(V_p, V_n, n, p);

            // Solve drift-diffusion equations with current potential
            solve_drift_diffusion();

            // Apply boundary conditions
            apply_boundary_conditions(V_p, V_n);

            // Get updated potential
            Eigen::VectorXd phi_new = get_potential();

            // Check convergence
            double error = (phi_new - phi_old).norm() / phi_new.norm();
            if (error < tolerance) {
                std::cout << "Self-consistent solution converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }

            // Update old potential
            phi_old = phi_new;

            // Print progress
            if (iter % 10 == 0) {
                std::cout << "Iteration " << iter << ": error = " << error << std::endl;
            }

            // Check for divergence
            if (!std::isfinite(error)) {
                throw std::runtime_error("Self-consistent solution diverged");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in SelfConsistentSolver::solve: " << e.what() << std::endl;
        // Initialize with a simple solution to avoid crashing
        initialize_carriers(N_A, N_D);
        poisson.solve(V_p, V_n, n, p);
    }
}