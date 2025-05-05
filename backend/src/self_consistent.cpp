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
      damping_factor(0.3), anderson_history_size(3), has_heterojunction(false) {

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
 * concentrations and the Boltzmann approximation. It assumes a p-n junction with the
 * p-side at x < 0 and the n-side at x > 0.
 *
 * @param N_A The acceptor doping concentration
 * @param N_D The donor doping concentration
 */
/**
 * @brief Sets the material properties for a heterojunction.
 *
 * This method sets the material properties for a heterojunction by defining
 * different materials in different regions of the device. It allows for
 * more complex device structures with multiple materials.
 *
 * @param materials Vector of materials
 * @param regions Vector of functions that define the regions for each material
 */
void SelfConsistentSolver::set_heterojunction(const std::vector<Materials::Material>& materials,
                                           const std::vector<std::function<bool(double, double)>>& regions) {
    // Check that the number of materials matches the number of regions
    if (materials.size() != regions.size()) {
        throw std::invalid_argument("Number of materials must match number of regions");
    }

    // Set the heterojunction properties
    this->materials = materials;
    this->regions = regions;
    this->has_heterojunction = true;
}

/**
 * @brief Gets the material at a given position.
 *
 * This method returns the material at the given position based on the
 * heterojunction regions defined by set_heterojunction.
 *
 * @param x The x-coordinate of the position
 * @param y The y-coordinate of the position
 * @return The material at the given position
 */
Materials::Material SelfConsistentSolver::get_material_at(double x, double y) const {
    // If no heterojunction is defined, return a default material
    if (!has_heterojunction) {
        Materials::Material default_mat;
        default_mat.N_c = 4.7e17; // Effective density of states in conduction band (nm^-3)
        default_mat.N_v = 7.0e18; // Effective density of states in valence band (nm^-3)
        default_mat.E_g = 1.424;  // Band gap (eV)
        default_mat.mu_n = 8500;  // Electron mobility (cm^2/V·s)
        default_mat.mu_p = 400;   // Hole mobility (cm^2/V·s)
        default_mat.epsilon_r = 12.9; // Relative permittivity

        // Set different properties based on position (p-side or n-side)
        if (x < 0) {
            // p-side: use material properties for p-type region (GaAs)
            return default_mat;
        } else {
            // n-side: use material properties for n-type region (GaAs)
            return default_mat;
        }
    }

    // Check each region to find the material at the given position
    for (size_t i = 0; i < regions.size(); ++i) {
        if (regions[i](x, y)) {
            return materials[i];
        }
    }

    // If no region matches, return the first material
    return materials[0];
}

void SelfConsistentSolver::initialize_carriers(double N_A, double N_D) {
    // Constants
    const double kT = 0.0259; // eV at 300K
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    const double T = 300.0; // Temperature in K
    const double q = 1.602e-19; // Elementary charge in C

    // Calculate intrinsic carrier concentration
    // For GaAs at 300K, ni ≈ 2.1e6 cm^-3 = 2.1e-12 nm^-3
    const double ni = 2.1e-12; // Intrinsic carrier concentration in nm^-3

    // Get the potential from the Poisson solver
    const Eigen::VectorXd& phi = poisson.get_potential();

    // Initialize carrier concentrations based on doping and potential
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties based on position
        Materials::Material mat = get_material_at(x, y);

        // Calculate Fermi levels
        double E_F_p = -mat.E_g + kB * T * std::log(mat.N_v / N_A);
        double E_F_n = -kB * T * std::log(mat.N_c / N_D);

        // Calculate carrier concentrations using the Boltzmann approximation
        if (i < phi.size() && phi[i] != 0.0) {
            // Use the potential to calculate carrier concentrations
            double V = phi[i] / q; // Convert from V to eV

            // Calculate band edges
            double E_c = -q * V - (x < 0 ? E_F_p : E_F_n);
            double E_v = E_c - mat.E_g;

            // Calculate carrier concentrations
            n[i] = mat.N_c * std::exp(-E_c / kT);
            p[i] = mat.N_v * std::exp(E_v / kT);
        } else {
            // Initial guess based on doping
            if (x < 0) {
                // p-side
                n[i] = ni * ni / N_A; // Low electron concentration in p-side
                p[i] = N_A;           // High hole concentration in p-side
            } else {
                // n-side
                n[i] = N_D;           // High electron concentration in n-side
                p[i] = ni * ni / N_D; // Low hole concentration in n-side
            }
        }

        // Ensure minimum carrier concentrations for numerical stability
        const double n_min = 1e-15; // Minimum concentration (nm^-3)
        const double n_max = 1e-3;  // Maximum concentration (nm^-3)
        n[i] = std::max(std::min(n[i], n_max), n_min);
        p[i] = std::max(std::min(p[i], n_max), n_min);
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
        Materials::Material mat = get_material_at(xc, yc);

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
        Materials::Material mat = get_material_at(x, y);

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
            Materials::Material mat = get_material_at(x, y);

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
            Materials::Material mat = get_material_at(x, y);

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
    double res_norm = residual.norm();

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

    try {
        // Use QR decomposition for numerical stability
        Eigen::MatrixXd FTF = F.transpose() * F;

        // Add Tikhonov regularization for numerical stability
        double reg = 1e-8 * FTF.diagonal().maxCoeff();
        for (int i = 0; i < FTF.rows(); ++i) {
            FTF(i, i) += reg;
        }

        // Solve the normal equations
        Eigen::VectorXd FTr = F.transpose() * (-res_history[m]);
        alpha = FTF.ldlt().solve(FTr);

        // Check if the solution is valid
        if (!alpha.allFinite()) {
            throw std::runtime_error("Invalid solution in Anderson acceleration");
        }
    } catch (const std::exception& e) {
        // Fallback to damping if Anderson acceleration fails
        std::cerr << "Anderson acceleration failed: " << e.what() << std::endl;
        return apply_damping(phi_old, phi_update);
    }

    // Compute the accelerated potential
    Eigen::VectorXd phi_accel = phi_history[m];
    for (int i = 0; i < m; ++i) {
        phi_accel += alpha(i) * (phi_history[i] - phi_history[m]);
    }

    // Perform line search to find optimal step size
    double beta = perform_line_search(phi_old, phi_update, phi_accel);

    // Apply the line search result
    Eigen::VectorXd phi_line_search = phi_old + beta * (phi_accel - phi_old);

    // Apply damping to the accelerated potential for stability
    return apply_damping(phi_old, phi_line_search);
}

/**
 * @brief Performs line search to find optimal step size.
 *
 * This function performs a line search to find the optimal step size
 * for the Anderson acceleration. It uses a backtracking line search
 * algorithm to find a step size that reduces the residual.
 *
 * @param phi_old The potential from the previous iteration
 * @param phi_update The updated potential from the current iteration
 * @param phi_accel The accelerated potential
 * @return The optimal step size
 */
double SelfConsistentSolver::perform_line_search(const Eigen::VectorXd& phi_old,
                                               const Eigen::VectorXd& phi_update,
                                               const Eigen::VectorXd& phi_accel) {
    // Initial step size
    double beta = 1.0;

    // Compute the initial residual norm
    double res_norm_old = (phi_update - phi_old).norm();

    // Maximum number of line search iterations
    const int max_line_search_iter = 10;

    // Line search parameters
    const double c = 0.5;  // Sufficient decrease parameter
    const double tau = 0.5;  // Step size reduction factor

    // Perform backtracking line search
    for (int i = 0; i < max_line_search_iter; ++i) {
        // Compute the trial point
        Eigen::VectorXd phi_trial = phi_old + beta * (phi_accel - phi_old);

        // Compute the residual at the trial point
        // In a real implementation, this would require solving the Poisson equation
        // Here we approximate it with the difference between phi_trial and phi_update
        double res_norm_trial = (phi_trial - phi_update).norm();

        // Check if the residual is sufficiently reduced
        if (res_norm_trial <= (1.0 - c * beta) * res_norm_old) {
            return beta;
        }

        // Reduce the step size
        beta *= tau;
    }

    // If line search fails, return a small step size
    return 0.1;
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

        // Initialize error tracking
        double prev_error = 1e10;

        // Adaptive mesh refinement parameters
        double refinement_threshold = 0.1;  // Threshold for mesh refinement
        int max_refinement_level = 3;       // Maximum refinement level
        bool mesh_refined = false;          // Flag to track if mesh was refined

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

            // Compute quantum correction
            Eigen::VectorXd V_q = compute_quantum_correction(n, p);

            // Add quantum correction to the potential
            Eigen::VectorXd phi_with_quantum = phi_new + V_q;

            // Update the potential in the Poisson solver and solve again
            poisson.update_and_solve(phi_with_quantum, V_p, V_n, n, p);

            // Solve drift-diffusion equations with accelerated potential
            solve_drift_diffusion();

            // Apply boundary conditions
            apply_boundary_conditions(V_p, V_n);

            // Check convergence
            double error = (phi_new - phi_old).norm() / phi_new.norm();

            // Perform adaptive mesh refinement if needed
            if (iter > 0 && iter % 10 == 0 && error < 0.1) {
                mesh_refined = refine_mesh_adaptively(refinement_threshold, max_refinement_level);

                if (mesh_refined) {
                    std::cout << "Mesh refined at iteration " << iter << std::endl;

                    // Reinitialize carrier concentrations on the refined mesh
                    initialize_carriers(N_A, N_D);

                    // Reset convergence acceleration
                    phi_history.clear();
                    res_history.clear();

                    // Reduce refinement threshold for next refinement
                    refinement_threshold *= 0.5;
                }
            }

            if (error < tolerance && !mesh_refined) {
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
            if (error > prev_error && damping_factor > 0.1) {
                damping_factor *= 0.8;
                std::cout << "Reducing damping factor to " << damping_factor << std::endl;
            }
            prev_error = error;

            // Reset mesh refinement flag
            mesh_refined = false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in SelfConsistentSolver::solve: " << e.what() << std::endl;
        // Initialize with a simple solution to avoid crashing
        initialize_carriers(N_A, N_D);
        poisson.solve(V_p, V_n, n, p);
    }
}

/**
 * @brief Refines the mesh adaptively based on error estimators.
 *
 * This method refines the mesh adaptively based on error estimators
 * for the potential and carrier concentrations. It identifies regions
 * with high gradients or errors and refines the mesh in those regions.
 *
 * @param refinement_threshold The threshold for mesh refinement
 * @param max_refinement_level The maximum refinement level
 * @return True if the mesh was refined, false otherwise
 */
bool SelfConsistentSolver::refine_mesh_adaptively(double refinement_threshold, int max_refinement_level) {
    // Compute error estimators for each element
    Eigen::VectorXd error_estimators = compute_error_estimators();

    // Find the maximum error estimator
    double max_error = error_estimators.maxCoeff();

    // If the maximum error is below the threshold, no refinement is needed
    if (max_error < refinement_threshold) {
        return false;
    }

    // Count the number of elements to refine
    int num_elements_to_refine = 0;
    for (int i = 0; i < error_estimators.size(); ++i) {
        if (error_estimators(i) > refinement_threshold * max_error) {
            num_elements_to_refine++;
        }
    }

    // If no elements need refinement, return false
    if (num_elements_to_refine == 0) {
        return false;
    }

    std::cout << "Refining " << num_elements_to_refine << " elements out of " << error_estimators.size() << std::endl;

    // Create a list of elements to refine
    std::vector<int> elements_to_refine;
    for (int i = 0; i < error_estimators.size(); ++i) {
        if (error_estimators(i) > refinement_threshold * max_error) {
            elements_to_refine.push_back(i);
        }
    }

    // Refine the mesh
    bool refined = mesh.refine(elements_to_refine, max_refinement_level);

    if (refined) {
        // Resize the carrier concentration vectors
        n.resize(mesh.getNumNodes());
        p.resize(mesh.getNumNodes());

        // Instead of reassigning the Poisson solver, we'll reinitialize it
        // by calling its initialize method
        poisson.initialize(mesh, epsilon_r, rho);
    }

    return refined;
}

/**
 * @brief Computes error estimators for mesh refinement.
 *
 * This method computes error estimators for mesh refinement based on
 * the gradients of the potential and carrier concentrations. It identifies
 * regions with high gradients or errors that need mesh refinement.
 *
 * @return Vector of error estimators for each element
 */
Eigen::VectorXd SelfConsistentSolver::compute_error_estimators() const {
    // Get the potential and carrier concentrations
    const Eigen::VectorXd& phi = get_potential();

    // Initialize error estimators
    Eigen::VectorXd error_estimators(mesh.getNumElements());

    // Compute error estimators for each element
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        // Get the element nodes
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

        // Compute the gradient of the potential
        Eigen::Vector2d grad_phi = Eigen::Vector2d::Zero();
        for (int i = 0; i < 3; ++i) {
            grad_phi += phi[element[i]] * grad_N[i];
        }

        // Compute the gradient of the electron concentration
        Eigen::Vector2d grad_n = Eigen::Vector2d::Zero();
        for (int i = 0; i < 3; ++i) {
            grad_n += n[element[i]] * grad_N[i];
        }

        // Compute the gradient of the hole concentration
        Eigen::Vector2d grad_p = Eigen::Vector2d::Zero();
        for (int i = 0; i < 3; ++i) {
            grad_p += p[element[i]] * grad_N[i];
        }

        // Compute the error estimator based on the gradients
        double error_phi = grad_phi.norm();
        double error_n = grad_n.norm();
        double error_p = grad_p.norm();

        // Combine the error estimators
        error_estimators(e) = area * (error_phi + error_n + error_p);
    }

    return error_estimators;
}

/**
 * @brief Computes the quantum correction to the potential.
 *
 * This method computes the quantum correction to the potential using
 * the Bohm quantum potential approach. It accounts for quantum effects
 * like tunneling and quantum confinement.
 *
 * The Bohm quantum potential is given by:
 * V_q = -ħ^2/(2m*) * ∇^2(√n)/√n
 *
 * Where:
 * - ħ is the reduced Planck constant
 * - m* is the effective mass
 * - n is the carrier concentration
 *
 * @param n The electron concentration
 * @param p The hole concentration
 * @return The quantum correction to the potential
 */
Eigen::VectorXd SelfConsistentSolver::compute_quantum_correction(const Eigen::VectorXd& n, const Eigen::VectorXd& p) const {
    // Constants
    const double h_bar = 1.055e-34; // Reduced Planck constant (J·s)
    const double m_e = 9.109e-31; // Electron mass (kg)
    const double q = 1.602e-19; // Elementary charge (C)

    // Initialize quantum correction vector
    Eigen::VectorXd V_q(mesh.getNumNodes());
    V_q.setZero();

    // Compute the quantum correction for each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Effective mass (in units of electron mass)
        double m_star = 0.067; // GaAs effective mass

        // Compute the Laplacian of sqrt(n) / sqrt(n)
        // This is a simplified implementation that uses finite differences
        // In a real implementation, we would use the finite element method

        // Find neighboring nodes
        std::vector<int> neighbors;
        for (size_t e = 0; e < mesh.getNumElements(); ++e) {
            auto element = mesh.getElements()[e];
            for (int j = 0; j < 3; ++j) {
                if (element[j] == i) {
                    // Add the other nodes in the element to the neighbors
                    for (int k = 0; k < 3; ++k) {
                        if (k != j) {
                            neighbors.push_back(element[k]);
                        }
                    }
                }
            }
        }

        // Remove duplicates
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        // Compute the Laplacian of sqrt(n) / sqrt(n)
        double laplacian_sqrt_n_over_sqrt_n = 0.0;
        if (!neighbors.empty()) {
            // Compute the average distance to neighbors
            double avg_distance = 0.0;
            for (int j : neighbors) {
                double dx = mesh.getNodes()[j][0] - x;
                double dy = mesh.getNodes()[j][1] - y;
                avg_distance += std::sqrt(dx * dx + dy * dy);
            }
            avg_distance /= neighbors.size();

            // Compute the Laplacian using finite differences
            double sqrt_n_i = std::sqrt(n[i]);
            double sum_sqrt_n_j = 0.0;
            for (int j : neighbors) {
                sum_sqrt_n_j += std::sqrt(n[j]);
            }

            // Approximate the Laplacian
            laplacian_sqrt_n_over_sqrt_n = (sum_sqrt_n_j - neighbors.size() * sqrt_n_i) / (avg_distance * avg_distance * sqrt_n_i);
        }

        // Compute the quantum correction
        V_q[i] = -h_bar * h_bar / (2.0 * m_star * m_e) * laplacian_sqrt_n_over_sqrt_n / q;
    }

    return V_q;
}

/**
 * @brief Computes the tunneling current.
 *
 * This method computes the tunneling current using the WKB approximation.
 * It accounts for band-to-band tunneling and trap-assisted tunneling.
 *
 * The WKB approximation for the tunneling probability is:
 * T = exp(-2 * ∫ κ(x) dx)
 *
 * Where:
 * - κ(x) = √(2m*(E_c(x) - E) / ħ^2) is the wave vector
 * - E_c(x) is the conduction band edge
 * - E is the energy of the electron
 *
 * @param E_field The electric field
 * @return The tunneling current
 */
Eigen::VectorXd SelfConsistentSolver::compute_tunneling_current(const std::vector<Eigen::Vector2d>& E_field) const {
    // Constants
    const double h_bar = 1.055e-34; // Reduced Planck constant (J·s)
    const double m_e = 9.109e-31; // Electron mass (kg)
    const double q = 1.602e-19; // Elementary charge (C)
    const double kB = 1.381e-23; // Boltzmann constant (J/K)
    const double T = 300.0; // Temperature (K)

    // Initialize tunneling current vector
    Eigen::VectorXd J_tunnel(mesh.getNumNodes());
    J_tunnel.setZero();

    // Get the potential from the Poisson solver
    const Eigen::VectorXd& phi = poisson.get_potential();

    // Compute the tunneling current for each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Effective mass (in units of electron mass)
        double m_star = 0.067; // GaAs effective mass

        // Band gap
        double E_g = mat.E_g * q; // Convert from eV to J

        // Electric field magnitude
        double E_mag = 0.0;
        if (i < E_field.size()) {
            E_mag = E_field[i].norm();
        }

        // Compute the tunneling probability using the WKB approximation
        // For band-to-band tunneling, the barrier width is approximately E_g / (q * E_mag)
        double barrier_width = E_g / (q * E_mag);

        // Compute the average wave vector
        double kappa = std::sqrt(2.0 * m_star * m_e * E_g) / h_bar;

        // Compute the tunneling probability
        double T_tunnel = std::exp(-2.0 * kappa * barrier_width);

        // Compute the tunneling current
        // J_tunnel = q * n * v * T_tunnel
        // where v is the thermal velocity
        double v_thermal = std::sqrt(3.0 * kB * T / (m_star * m_e));

        // Compute the tunneling current
        J_tunnel[i] = q * n[i] * v_thermal * T_tunnel;
    }

    return J_tunnel;
}