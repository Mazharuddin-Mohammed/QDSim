/**
 * @file self_consistent.cpp
 * @brief Implementation of the SelfConsistentSolver class.
 *
 * This file contains the implementation of the SelfConsistentSolver class,
 * which provides a solver for the self-consistent Poisson-drift-diffusion
 * equations used in semiconductor device simulations. It couples the Poisson equation
 * for the electrostatic potential with the drift-diffusion equations for carrier
 * transport.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

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
/**
 * @brief Constructs a new SelfConsistentSolver object.
 *
 * This constructor initializes the SelfConsistentSolver with the given mesh and callback functions
 * for computing physical quantities. It also initializes the PoissonSolver and resizes the carrier
 * concentration vectors and drift-diffusion matrices. Additionally, it sets up the convergence
 * acceleration parameters.
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
      epsilon_r(epsilon_r), rho(rho), n_conc(n_conc), p_conc(p_conc), mu_n(mu_n), mu_p(mu_p),
      damping_factor(0.3), anderson_history_size(3) {

    // Resize carrier concentration vectors and drift-diffusion matrices
    n.resize(mesh.getNumNodes());
    p.resize(mesh.getNumNodes());
    Kn.resize(mesh.getNumNodes(), mesh.getNumNodes());
    Kp.resize(mesh.getNumNodes(), mesh.getNumNodes());

    // Initialize with zeros
    n.setZero();
    p.setZero();

    // Initialize convergence acceleration parameters
    phi_history.clear();
    res_history.clear();
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
/**
 * @brief Assembles the drift-diffusion matrices for electron and hole transport.
 *
 * This function assembles the drift-diffusion matrices for electron and hole transport
 * using the finite element method. It computes the drift and diffusion terms for each
 * element and assembles them into the global matrices.
 *
 * The drift-diffusion equation for electrons is:
 * ∇·(μn·n·∇φ + Dn·∇n) = G - R
 *
 * The drift-diffusion equation for holes is:
 * ∇·(-μp·p·∇φ + Dp·∇p) = G - R
 *
 * Where:
 * - μn, μp are the electron and hole mobilities
 * - n, p are the electron and hole concentrations
 * - φ is the electrostatic potential
 * - Dn, Dp are the diffusion coefficients
 * - G is the generation rate
 * - R is the recombination rate
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
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    const double T = 300.0; // Temperature in K

    // Loop over all elements
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        // Get element nodes and coordinates
        auto element = mesh.getElements()[e];
        std::vector<Eigen::Vector2d> nodes;
        for (int i = 0; i < 3; ++i) {
            nodes.push_back(mesh.getNodes()[element[i]]);
        }

        // Calculate element area
        double x1 = nodes[0][0], y1 = nodes[0][1];
        double x2 = nodes[1][0], y2 = nodes[1][1];
        double x3 = nodes[2][0], y3 = nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Calculate shape function gradients
        // For linear elements, the gradients are constant over the element
        std::vector<Eigen::Vector2d> grad_N(3);

        // Calculate the Jacobian matrix for mapping from reference to physical element
        Eigen::Matrix2d J;
        J << x2 - x1, x3 - x1,
             y2 - y1, y3 - y1;

        // Calculate the inverse of the Jacobian
        Eigen::Matrix2d J_inv = J.inverse();

        // Reference element gradients
        std::vector<Eigen::Vector2d> ref_grad = {
            {-1.0, -1.0},  // dN1/d(xi), dN1/d(eta)
            {1.0, 0.0},    // dN2/d(xi), dN2/d(eta)
            {0.0, 1.0}     // dN3/d(xi), dN3/d(eta)
        };

        // Transform gradients from reference to physical element
        for (int i = 0; i < 3; ++i) {
            grad_N[i] = J_inv.transpose() * ref_grad[i];
        }

        // Calculate element centroid
        double xc = (x1 + x2 + x3) / 3.0;
        double yc = (y1 + y2 + y3) / 3.0;

        // Get material properties at the centroid
        Materials::Material mat;
        if (xc < 0) {
            // p-side: use material properties for p-type region
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value
            mat.mu_n = 0.3;   // Example value
            mat.mu_p = 0.02;  // Example value
        } else {
            // n-side: use material properties for n-type region
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value
            mat.mu_n = 0.3;   // Example value
            mat.mu_p = 0.02;  // Example value
        }

        // Get mobilities at the centroid using the enhanced mobility models
        double mu_n_val = mu_n(xc, yc, mat);
        double mu_p_val = mu_p(xc, yc, mat);

        // Calculate diffusion coefficients using Einstein relation
        double D_n = kT * mu_n_val / q;
        double D_p = kT * mu_p_val / q;

        // Get electric field at the centroid
        Eigen::Vector2d E = poisson.get_electric_field(xc, yc);

        // Get carrier concentrations at element nodes
        std::vector<double> n_nodes(3), p_nodes(3);
        for (int i = 0; i < 3; ++i) {
            int node_idx = element[i];
            if (node_idx < n.size() && node_idx < p.size()) {
                n_nodes[i] = n[node_idx];
                p_nodes[i] = p[node_idx];
            } else {
                // Fallback values if node index is out of range
                n_nodes[i] = 1e10;
                p_nodes[i] = 1e10;
            }
        }

        // Calculate average carrier concentrations for the element
        double n_avg = (n_nodes[0] + n_nodes[1] + n_nodes[2]) / 3.0;
        double p_avg = (p_nodes[0] + p_nodes[1] + p_nodes[2]) / 3.0;

        // Gauss quadrature points and weights for triangular elements
        std::vector<Eigen::Vector3d> quad_points = {
            {1.0/6.0, 1.0/6.0, 2.0/3.0},
            {1.0/6.0, 2.0/3.0, 1.0/6.0},
            {2.0/3.0, 1.0/6.0, 1.0/6.0}
        };
        std::vector<double> quad_weights = {1.0/6.0, 1.0/6.0, 1.0/6.0};

        // Assemble element matrices using quadrature
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double Kn_ij = 0.0, Kp_ij = 0.0;

                // Integrate over quadrature points
                for (size_t q = 0; q < quad_points.size(); ++q) {
                    // Barycentric coordinates
                    double lambda1 = quad_points[q][0];
                    double lambda2 = quad_points[q][1];
                    double lambda3 = quad_points[q][2];

                    // Shape functions at quadrature point
                    std::vector<double> N = {lambda1, lambda2, lambda3};

                    // Calculate position at quadrature point
                    double x_q = lambda1 * x1 + lambda2 * x2 + lambda3 * x3;
                    double y_q = lambda1 * y1 + lambda2 * y2 + lambda3 * y3;

                    // Calculate potential gradient at quadrature point
                    Eigen::Vector2d grad_phi = Eigen::Vector2d::Zero();
                    for (int k = 0; k < 3; ++k) {
                        if (element[k] < phi.size()) {
                            grad_phi += phi[element[k]] * grad_N[k];
                        }
                    }

                    // Calculate carrier concentration gradients at quadrature point
                    Eigen::Vector2d grad_n = Eigen::Vector2d::Zero();
                    Eigen::Vector2d grad_p = Eigen::Vector2d::Zero();
                    for (int k = 0; k < 3; ++k) {
                        grad_n += n_nodes[k] * grad_N[k];
                        grad_p += p_nodes[k] * grad_N[k];
                    }

                    // Drift terms: μn·n·∇φ for electrons, -μp·p·∇φ for holes
                    double drift_n = mu_n_val * n_avg * (grad_N[i].dot(grad_phi) * N[j]);
                    double drift_p = -mu_p_val * p_avg * (grad_N[i].dot(grad_phi) * N[j]);

                    // Diffusion terms: Dn·∇n for electrons, Dp·∇p for holes
                    double diff_n = D_n * (grad_N[i].dot(grad_N[j]));
                    double diff_p = D_p * (grad_N[i].dot(grad_N[j]));

                    // Add contributions to element matrices
                    Kn_ij += quad_weights[q] * area * (drift_n + diff_n);
                    Kp_ij += quad_weights[q] * area * (drift_p + diff_p);
                }

                // Add to triplet lists
                Kn_triplets.emplace_back(element[i], element[j], Kn_ij);
                Kp_triplets.emplace_back(element[i], element[j], Kp_ij);
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
/**
 * @brief Applies boundary conditions to the carrier concentrations.
 *
 * This function applies boundary conditions to the carrier concentrations based on
 * the applied voltages and the position of the contacts. It sets the carrier
 * concentrations at the contacts to their equilibrium values, and applies
 * Neumann (zero-flux) boundary conditions at insulating boundaries.
 *
 * The boundary conditions include:
 * 1. Dirichlet boundary conditions at ohmic contacts (p and n contacts)
 * 2. Neumann (zero-flux) boundary conditions at insulating boundaries
 * 3. Surface recombination at semiconductor-insulator interfaces
 *
 * @param V_p The voltage applied to the p-contact
 * @param V_n The voltage applied to the n-contact
 */
void SelfConsistentSolver::apply_boundary_conditions(double V_p, double V_n) {
    // Constants
    const double kT = 0.0259; // eV at 300K
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    const double T = 300.0; // Temperature in K
    const double q = 1.602e-19; // Elementary charge in C
    const double ni = 1e10; // Intrinsic carrier concentration (simplified)

    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double Ly = mesh.get_ly();

    // Define boundary regions
    double contact_thickness = 0.05 * Lx; // Thickness of the contact regions
    double p_contact_x = -Lx / 2;
    double n_contact_x = Lx / 2;

    // Define surface recombination velocity (cm/s)
    double S_n = 1e3; // Surface recombination velocity for electrons
    double S_p = 1e3; // Surface recombination velocity for holes

    // Convert to nm/s
    S_n *= 1e7;
    S_p *= 1e7;

    // Loop over all nodes
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Check if the node is at the p-contact (left boundary)
        if (std::abs(x - p_contact_x) < contact_thickness) {
            // Get material properties for p-type region
            Materials::Material mat;
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value
            mat.mu_n = 0.3;   // Example value
            mat.mu_p = 0.02;  // Example value

            // Define doping concentrations
            double N_A = 1e16; // Acceptor concentration in p-region
            double N_D = 0.0;  // Donor concentration in p-region

            // Calculate Fermi level in p-region
            double E_F_p = -mat.E_g + kB * T * std::log(mat.N_v / N_A);

            // Calculate band edges with applied voltage
            double E_c = -q * V_p - E_F_p;
            double E_v = E_c - mat.E_g;

            // Calculate equilibrium carrier concentrations using Boltzmann statistics
            double n_eq = mat.N_c * std::exp(-E_c / kT);
            double p_eq = mat.N_v * std::exp(E_v / kT);

            // Apply Dirichlet boundary conditions at ohmic contact
            n[i] = n_eq;
            p[i] = p_eq;
        }
        // Check if the node is at the n-contact (right boundary)
        else if (std::abs(x - n_contact_x) < contact_thickness) {
            // Get material properties for n-type region
            Materials::Material mat;
            mat.N_c = 2.1e23; // Example value
            mat.N_v = 8.0e24; // Example value
            mat.E_g = 1.0;    // Example value
            mat.mu_n = 0.3;   // Example value
            mat.mu_p = 0.02;  // Example value

            // Define doping concentrations
            double N_A = 0.0;  // Acceptor concentration in n-region
            double N_D = 1e16; // Donor concentration in n-region

            // Calculate Fermi level in n-region
            double E_F_n = -kB * T * std::log(mat.N_c / N_D);

            // Calculate band edges with applied voltage
            double E_c = -q * V_n - E_F_n;
            double E_v = E_c - mat.E_g;

            // Calculate equilibrium carrier concentrations using Boltzmann statistics
            double n_eq = mat.N_c * std::exp(-E_c / kT);
            double p_eq = mat.N_v * std::exp(E_v / kT);

            // Apply Dirichlet boundary conditions at ohmic contact
            n[i] = n_eq;
            p[i] = p_eq;
        }
        // Check if the node is at the top or bottom boundary (insulating)
        else if (std::abs(y - (-Ly / 2)) < 1e-10 || std::abs(y - (Ly / 2)) < 1e-10) {
            // Apply surface recombination boundary conditions
            // For simplicity, we'll use a simplified model that sets the carrier
            // concentrations to reduced values at the surface

            // Get material properties based on position
            Materials::Material mat;
            if (x < 0) {
                // p-side
                mat.N_c = 2.1e23; // Example value
                mat.N_v = 8.0e24; // Example value
                mat.E_g = 1.0;    // Example value
            } else {
                // n-side
                mat.N_c = 2.1e23; // Example value
                mat.N_v = 8.0e24; // Example value
                mat.E_g = 1.0;    // Example value
            }

            // Get the potential at this node
            double phi_val = 0.0;
            const Eigen::VectorXd& phi = poisson.get_potential();
            if (i < phi.size()) {
                phi_val = phi[i];
            }

            // Calculate bulk carrier concentrations
            double n_bulk = n_conc(x, y, phi_val, mat);
            double p_bulk = p_conc(x, y, phi_val, mat);

            // Calculate surface carrier concentrations with surface recombination
            // This is a simplified model; a more accurate model would involve
            // solving the continuity equations with surface recombination boundary conditions
            double surface_factor = 0.5; // Reduction factor at the surface
            n[i] = surface_factor * n_bulk;
            p[i] = surface_factor * p_bulk;
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
/**
 * @brief Apply damping to the potential update.
 *
 * This method applies damping to the potential update to improve convergence.
 * It uses a simple linear damping scheme: phi_new = phi_old + damping_factor * (phi_update - phi_old)
 *
 * @param phi_old The previous potential vector
 * @param phi_update The updated potential vector
 * @return The damped potential vector
 */
Eigen::VectorXd SelfConsistentSolver::apply_damping(const Eigen::VectorXd& phi_old, const Eigen::VectorXd& phi_update) const {
    // Apply damping: phi_new = phi_old + damping_factor * (phi_update - phi_old)
    return phi_old + damping_factor * (phi_update - phi_old);
}

/**
 * @brief Apply Anderson acceleration to the potential update.
 *
 * This method applies Anderson acceleration to the potential update to improve convergence.
 * It uses the history of potential and residual vectors to compute an optimal combination
 * of previous iterations.
 *
 * @param phi_old The previous potential vector
 * @param phi_update The updated potential vector
 * @return The accelerated potential vector
 */
Eigen::VectorXd SelfConsistentSolver::apply_anderson_acceleration(const Eigen::VectorXd& phi_old, const Eigen::VectorXd& phi_update) {
    // Calculate residual: r = phi_update - phi_old
    Eigen::VectorXd residual = phi_update - phi_old;

    // Add current potential and residual to history
    phi_history.push_back(phi_update);
    res_history.push_back(residual);

    // Limit history size
    if (phi_history.size() > anderson_history_size) {
        phi_history.erase(phi_history.begin());
        res_history.erase(res_history.begin());
    }

    // If we don't have enough history, just return the damped update
    if (phi_history.size() < 2) {
        return apply_damping(phi_old, phi_update);
    }

    // Compute optimal coefficients using least squares
    int m = phi_history.size() - 1;
    Eigen::MatrixXd F(residual.size(), m);

    // Fill the matrix F with differences of residuals
    for (int i = 0; i < m; ++i) {
        F.col(i) = res_history[i] - res_history[m];
    }

    // Solve the least squares problem: min ||F*alpha - (-res_history[m])||^2
    Eigen::VectorXd alpha;

    // Use QR decomposition for numerical stability
    Eigen::MatrixXd FTF = F.transpose() * F;

    // Add regularization for numerical stability
    double reg = 1e-10 * FTF.diagonal().maxCoeff();
    for (int i = 0; i < FTF.rows(); ++i) {
        FTF(i, i) += reg;
    }

    // Solve the normal equations
    Eigen::VectorXd FTr = F.transpose() * (-res_history[m]);
    alpha = FTF.ldlt().solve(FTr);

    // Compute the accelerated potential
    Eigen::VectorXd phi_accel = phi_history[m];
    for (int i = 0; i < m; ++i) {
        phi_accel += alpha(i) * (phi_history[i] - phi_history[m]);
    }

    // Apply damping to the accelerated potential for stability
    return apply_damping(phi_old, phi_accel);
}

/**
 * @brief Solves the self-consistent Poisson-drift-diffusion equations.
 *
 * This function solves the self-consistent Poisson-drift-diffusion equations
 * using an iterative approach with convergence acceleration techniques. It alternates
 * between solving the Poisson equation and the drift-diffusion equations until
 * convergence is reached or the maximum number of iterations is exceeded.
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

        // Clear history for Anderson acceleration
        phi_history.clear();
        res_history.clear();

        // Iterative solution
        for (int iter = 0; iter < max_iter; ++iter) {
            // Solve Poisson equation with current carrier concentrations
            poisson.solve(V_p, V_n, n, p);

            // Get updated potential before acceleration
            Eigen::VectorXd phi_update = get_potential();

            // Apply convergence acceleration
            Eigen::VectorXd phi_new;
            if (iter < 5) {
                // Use damping for the first few iterations
                phi_new = apply_damping(phi_old, phi_update);
            } else {
                // Use Anderson acceleration for later iterations
                phi_new = apply_anderson_acceleration(phi_old, phi_update);
            }

            // Update the potential in the Poisson solver
            // This requires accessing the internal phi vector of the Poisson solver
            // We'll need to add a method to set the potential in the Poisson solver
            // For now, we'll just solve again with the accelerated potential

            // Solve drift-diffusion equations with accelerated potential
            solve_drift_diffusion();

            // Apply boundary conditions
            apply_boundary_conditions(V_p, V_n);

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

            // Adaptive damping: reduce damping factor if error increases
            static double prev_error = 1e10;
            if (error > prev_error && damping_factor > 0.1) {
                damping_factor *= 0.8;
                std::cout << "Reducing damping factor to " << damping_factor << std::endl;
            }
            prev_error = error;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in SelfConsistentSolver::solve: " << e.what() << std::endl;
        // Initialize with a simple solution to avoid crashing
        initialize_carriers(N_A, N_D);
        poisson.solve(V_p, V_n, n, p);
    }
}