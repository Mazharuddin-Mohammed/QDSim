/**
 * @file full_poisson_dd_solver.cpp
 * @brief Implementation of the FullPoissonDriftDiffusionSolver class.
 *
 * This file contains the implementation of the FullPoissonDriftDiffusionSolver class,
 * which provides a comprehensive solver for the coupled Poisson-drift-diffusion
 * equations used in semiconductor device simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "full_poisson_dd_solver.h"
#include <Eigen/SparseCholesky>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>

/**
 * @brief Constructs a new FullPoissonDriftDiffusionSolver object.
 *
 * @param mesh The mesh to use for the simulation
 * @param epsilon_r Function that returns the relative permittivity at a given position
 * @param doping_profile Function that returns the doping profile at a given position
 */
FullPoissonDriftDiffusionSolver::FullPoissonDriftDiffusionSolver(
    Mesh& mesh,
    std::function<double(double, double)> epsilon_r,
    std::function<double(double, double)> doping_profile)
    : mesh(mesh), epsilon_r(epsilon_r), doping_profile(doping_profile),
      has_heterojunction(false), has_g_r_model(false), has_mobility_models(false),
      use_fermi_dirac_statistics(false), use_quantum_corrections(false),
      use_adaptive_mesh_refinement(false), damping_factor(0.3), anderson_history_size(3) {

    // Initialize vectors
    phi.resize(mesh.getNumNodes());
    n.resize(mesh.getNumNodes());
    p.resize(mesh.getNumNodes());
    phi_n.resize(mesh.getNumNodes());
    phi_p.resize(mesh.getNumNodes());
    J_n.resize(mesh.getNumNodes());
    J_p.resize(mesh.getNumNodes());
    E_field.resize(mesh.getNumNodes());

    // Initialize with zeros
    phi.setZero();
    n.setZero();
    p.setZero();
    phi_n.setZero();
    phi_p.setZero();
    for (auto& j : J_n) j.setZero();
    for (auto& j : J_p) j.setZero();
    for (auto& e : E_field) e.setZero();

    // Initialize convergence acceleration parameters
    phi_history.clear();
    res_history.clear();

    // Set default mobility models
    mu_n_model = [](double x, double y, double E, const Materials::Material& mat) {
        return mat.mu_n; // Default constant mobility
    };

    mu_p_model = [](double x, double y, double E, const Materials::Material& mat) {
        return mat.mu_p; // Default constant mobility
    };

    has_mobility_models = true;

    // Set default generation-recombination model (SRH)
    g_r_model = [](double x, double y, double n, double p, const Materials::Material& mat) {
        // Constants
        const double kT = 0.0259; // eV at 300K
        const double ni = 1.5e6; // Intrinsic carrier concentration for GaAs at 300K (cm^-3)
        const double tau_n = 1e-9; // Electron lifetime (s)
        const double tau_p = 1e-9; // Hole lifetime (s)

        // SRH recombination rate
        double R_SRH = (n * p - ni * ni) / (tau_p * (n + ni) + tau_n * (p + ni));

        return R_SRH;
    };

    has_g_r_model = true;

    // Initialize carrier concentrations based on doping profile
    initialize_carrier_concentrations();
}

/**
 * @brief Sets the material properties for a heterojunction.
 *
 * @param materials Vector of materials
 * @param regions Vector of functions that define the regions for each material
 */
void FullPoissonDriftDiffusionSolver::set_heterojunction(
    const std::vector<Materials::Material>& materials,
    const std::vector<std::function<bool(double, double)>>& regions) {

    // Check that the number of materials matches the number of regions
    if (materials.size() != regions.size()) {
        throw std::invalid_argument("Number of materials must match number of regions");
    }

    // Set the heterojunction properties
    this->materials = materials;
    this->regions = regions;
    this->has_heterojunction = true;

    // Reinitialize carrier concentrations
    initialize_carrier_concentrations();
}

/**
 * @brief Sets the generation-recombination model.
 *
 * @param g_r Function that computes the generation-recombination rate
 */
void FullPoissonDriftDiffusionSolver::set_generation_recombination_model(
    std::function<double(double, double, double, double, const Materials::Material&)> g_r) {

    this->g_r_model = g_r;
    this->has_g_r_model = true;
}

/**
 * @brief Sets the mobility models for electrons and holes.
 *
 * @param mu_n Function that computes the electron mobility
 * @param mu_p Function that computes the hole mobility
 */
void FullPoissonDriftDiffusionSolver::set_mobility_models(
    std::function<double(double, double, double, const Materials::Material&)> mu_n,
    std::function<double(double, double, double, const Materials::Material&)> mu_p) {

    this->mu_n_model = mu_n;
    this->mu_p_model = mu_p;
    this->has_mobility_models = true;
}

/**
 * @brief Sets the carrier statistics model.
 *
 * @param use_fermi_dirac Whether to use Fermi-Dirac statistics (true) or Boltzmann statistics (false)
 */
void FullPoissonDriftDiffusionSolver::set_carrier_statistics_model(bool use_fermi_dirac) {
    this->use_fermi_dirac_statistics = use_fermi_dirac;

    // Reinitialize carrier concentrations
    initialize_carrier_concentrations();
}

/**
 * @brief Enables or disables quantum corrections.
 *
 * @param enable Whether to enable quantum corrections
 */
void FullPoissonDriftDiffusionSolver::enable_quantum_corrections(bool enable) {
    this->use_quantum_corrections = enable;
}

/**
 * @brief Enables or disables adaptive mesh refinement.
 *
 * @param enable Whether to enable adaptive mesh refinement
 * @param refinement_threshold The threshold for mesh refinement
 * @param max_refinement_level The maximum refinement level
 */
void FullPoissonDriftDiffusionSolver::enable_adaptive_mesh_refinement(
    bool enable, double refinement_threshold, int max_refinement_level) {

    this->use_adaptive_mesh_refinement = enable;
    this->adaptive_mesh_refinement_threshold = refinement_threshold;
    this->adaptive_mesh_refinement_max_level = max_refinement_level;
}

/**
 * @brief Gets the material at a given position.
 *
 * @param x The x-coordinate of the position
 * @param y The y-coordinate of the position
 * @return The material at the given position
 */
Materials::Material FullPoissonDriftDiffusionSolver::get_material_at(double x, double y) const {
    // If no heterojunction is defined, return a default material (GaAs)
    if (!has_heterojunction) {
        Materials::Material default_mat;
        default_mat.N_c = 4.7e17; // Effective density of states in conduction band (cm^-3)
        default_mat.N_v = 7.0e18; // Effective density of states in valence band (cm^-3)
        default_mat.E_g = 1.424;  // Band gap (eV)
        default_mat.mu_n = 8500;  // Electron mobility (cm^2/V·s)
        default_mat.mu_p = 400;   // Hole mobility (cm^2/V·s)
        default_mat.epsilon_r = 12.9; // Relative permittivity
        default_mat.m_e = 0.067;  // Effective electron mass (m0)
        default_mat.m_h = 0.45;   // Effective hole mass (m0)
        // Electron affinity is not part of the Material struct

        return default_mat;
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

/**
 * @brief Initializes the carrier concentrations based on the doping profile.
 */
void FullPoissonDriftDiffusionSolver::initialize_carrier_concentrations() {
    // Constants
    const double kT = 0.0259; // eV at 300K
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    const double T = 300.0; // Temperature in K

    // Initialize carrier concentrations based on doping profile
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get doping at this position (positive for donors, negative for acceptors)
        double doping = doping_profile(x, y);

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Calculate intrinsic carrier concentration
        double ni = std::sqrt(mat.N_c * mat.N_v) * std::exp(-mat.E_g / (2.0 * kT));

        // Initialize carrier concentrations based on doping
        if (doping > 0) {
            // n-type region
            n[i] = doping; // Electron concentration equals donor concentration
            p[i] = ni * ni / doping; // Hole concentration from mass action law
        } else if (doping < 0) {
            // p-type region
            p[i] = -doping; // Hole concentration equals acceptor concentration
            n[i] = ni * ni / p[i]; // Electron concentration from mass action law
        } else {
            // Intrinsic region
            n[i] = ni;
            p[i] = ni;
        }

        // Ensure minimum carrier concentrations for numerical stability
        const double n_min = 1e5; // Minimum concentration (cm^-3)
        n[i] = std::max(n[i], n_min);
        p[i] = std::max(p[i], n_min);

        // Initialize quasi-Fermi potentials
        phi_n[i] = kT * std::log(n[i] / mat.N_c);
        phi_p[i] = -kT * std::log(p[i] / mat.N_v) - mat.E_g;
    }
}

/**
 * @brief Computes the charge density based on the carrier concentrations and doping profile.
 *
 * @return The charge density vector
 */
Eigen::VectorXd FullPoissonDriftDiffusionSolver::compute_charge_density() const {
    // Constants
    const double q = 1.602e-19; // Elementary charge (C)

    // Initialize charge density vector
    Eigen::VectorXd rho(mesh.getNumNodes());

    // Compute charge density at each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get doping at this position (positive for donors, negative for acceptors)
        double doping = doping_profile(x, y);

        // Compute charge density: rho = q * (p - n + N_D - N_A)
        // where N_D - N_A is the net doping (positive for donors, negative for acceptors)
        rho[i] = q * (p[i] - n[i] + doping);
    }

    return rho;
}

/**
 * @brief Solves the Poisson equation.
 *
 * @param V_p The voltage applied to the p-contact
 * @param V_n The voltage applied to the n-contact
 */
void FullPoissonDriftDiffusionSolver::solve_poisson_equation(double V_p, double V_n) {
    // Constants
    const double epsilon_0 = 8.85418782e-14; // Vacuum permittivity (F/cm)

    // Get the number of nodes
    int num_nodes = mesh.getNumNodes();

    // Create stiffness matrix and right-hand side vector
    Eigen::SparseMatrix<double> K(num_nodes, num_nodes);
    Eigen::VectorXd f(num_nodes);
    f.setZero();

    // Create triplet list for sparse matrix assembly
    std::vector<Eigen::Triplet<double>> triplets;

    // Compute the charge density
    Eigen::VectorXd rho = compute_charge_density();

    // Assemble the stiffness matrix and right-hand side vector
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

        // Calculate element centroid
        double xc = (x1 + x2 + x3) / 3.0;
        double yc = (y1 + y2 + y3) / 3.0;

        // Get material properties at the centroid
        Materials::Material mat = get_material_at(xc, yc);

        // Get the permittivity at the centroid
        double eps = mat.epsilon_r * epsilon_0;

        // Compute the element matrix entries
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Gradient of basis functions dot product
                double grad_i_dot_grad_j = grad_N[i].dot(grad_N[j]);

                // Add the element matrix entry to the triplet list
                triplets.emplace_back(element[i], element[j], eps * grad_i_dot_grad_j * area);
            }
        }

        // Compute the element right-hand side vector
        // Gauss quadrature points and weights for triangular elements
        std::vector<std::array<double, 3>> quad_points = {
            {1.0/6.0, 1.0/6.0, 2.0/3.0},
            {1.0/6.0, 2.0/3.0, 1.0/6.0},
            {2.0/3.0, 1.0/6.0, 1.0/6.0}
        };
        std::vector<double> quad_weights = {1.0/6.0, 1.0/6.0, 1.0/6.0};

        // Integrate over quadrature points
        for (size_t q = 0; q < quad_points.size(); ++q) {
            // Barycentric coordinates
            double lambda1 = quad_points[q][0];
            double lambda2 = quad_points[q][1];
            double lambda3 = quad_points[q][2];

            // Calculate position at quadrature point
            double x_q = lambda1 * x1 + lambda2 * x2 + lambda3 * x3;
            double y_q = lambda1 * y1 + lambda2 * y2 + lambda3 * y3;

            // Interpolate charge density at quadrature point
            double rho_q = 0.0;
            for (int i = 0; i < 3; ++i) {
                double basis_value;
                if (i == 0) basis_value = lambda1;
                else if (i == 1) basis_value = lambda2;
                else basis_value = lambda3;

                // Get charge density at node
                rho_q += rho[element[i]] * basis_value;
            }

            // Add the element right-hand side vector entries
            for (int i = 0; i < 3; ++i) {
                // Basis function value at quadrature point
                double basis_value;
                if (i == 0) basis_value = lambda1;
                else if (i == 1) basis_value = lambda2;
                else basis_value = lambda3;

                // Add the contribution to the right-hand side vector
                f[element[i]] -= rho_q * basis_value * quad_weights[q] * area;
            }
        }
    }

    // Set the global matrix from the triplets
    K.setFromTriplets(triplets.begin(), triplets.end());

    // Apply Dirichlet boundary conditions
    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double Ly = mesh.get_ly();
    double tolerance = 1e-6; // Tolerance for boundary detection (increased for robustness)

    // Create a vector to mark boundary nodes
    std::vector<bool> is_boundary(num_nodes, false);
    std::vector<double> boundary_values(num_nodes, 0.0);

    // Identify boundary nodes and set their values
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.getNodes()[i];
        double x = node[0];
        double y = node[1];

        // Left boundary (p-type)
        if (x < tolerance) {
            is_boundary[i] = true;
            boundary_values[i] = V_p;
            phi[i] = V_p;
        }
        // Right boundary (n-type)
        else if (x > Lx - tolerance) {
            is_boundary[i] = true;
            boundary_values[i] = V_n;
            phi[i] = V_n;
        }
    }

    // Print the number of boundary nodes
    std::cout << "Number of boundary nodes: " << std::count(is_boundary.begin(), is_boundary.end(), true) << std::endl;

    // Apply Dirichlet boundary conditions to the stiffness matrix and right-hand side vector
    for (size_t i = 0; i < num_nodes; ++i) {
        if (is_boundary[i]) {
            // Zero out the row
            for (int k = 0; k < K.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it) {
                    if (it.row() == i) {
                        it.valueRef() = 0.0;
                    }
                }
            }

            // Set the diagonal entry to 1
            K.coeffRef(i, i) = 1.0;

            // Set the right-hand side value
            f[i] = boundary_values[i];
        }
    }

    // Solve the linear system using the SimplicialLDLT solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);

    // Check if the factorization was successful
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize the stiffness matrix");
    }

    // Solve the linear system
    Eigen::VectorXd phi_new = solver.solve(f);

    // Check if the solve was successful
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve the linear system");
    }

    // Check for NaN or Inf values
    bool has_nan_or_inf = false;
    for (int i = 0; i < phi_new.size(); ++i) {
        if (std::isnan(phi_new[i]) || std::isinf(phi_new[i])) {
            has_nan_or_inf = true;
            break;
        }
    }

    if (has_nan_or_inf) {
        std::cerr << "Warning: NaN or Inf values detected in the Poisson solution" << std::endl;
        // Use a simple initial guess instead
        for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
            const auto& node = mesh.getNodes()[i];
            double x = node[0];
            double Lx = mesh.get_lx();

            // Linear interpolation between V_p and V_n
            phi_new[i] = V_p + (V_n - V_p) * x / Lx;
        }
    }

    // Update the potential
    phi = phi_new;

    // Compute the electric field
    compute_electric_field();
}

/**
 * @brief Computes the electric field from the electrostatic potential.
 */
void FullPoissonDriftDiffusionSolver::compute_electric_field() {
    // Compute the electric field at each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        // Find all elements containing this node
        std::vector<size_t> node_elements;
        for (size_t e = 0; e < mesh.getNumElements(); ++e) {
            auto element = mesh.getElements()[e];
            for (int j = 0; j < 3; ++j) {
                if (element[j] == i) {
                    node_elements.push_back(e);
                    break;
                }
            }
        }

        // Compute the average electric field from all elements containing this node
        Eigen::Vector2d E_avg = Eigen::Vector2d::Zero();

        for (size_t e : node_elements) {
            auto element = mesh.getElements()[e];
            std::vector<Eigen::Vector2d> nodes;
            for (int j = 0; j < 3; ++j) {
                nodes.push_back(mesh.getNodes()[element[j]]);
            }

            // Calculate element area
            double x1 = nodes[0][0], y1 = nodes[0][1];
            double x2 = nodes[1][0], y2 = nodes[1][1];
            double x3 = nodes[2][0], y3 = nodes[2][1];

            // Compute the derivatives of the shape functions
            double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

            double dN1_dx = (y2 - y3) / (2.0 * area);
            double dN1_dy = (x3 - x2) / (2.0 * area);
            double dN2_dx = (y3 - y1) / (2.0 * area);
            double dN2_dy = (x1 - x3) / (2.0 * area);
            double dN3_dx = (y1 - y2) / (2.0 * area);
            double dN3_dy = (x2 - x1) / (2.0 * area);

            // Compute the gradient of the potential
            double dphi_dx = phi[element[0]] * dN1_dx + phi[element[1]] * dN2_dx + phi[element[2]] * dN3_dx;
            double dphi_dy = phi[element[0]] * dN1_dy + phi[element[1]] * dN2_dy + phi[element[2]] * dN3_dy;

            // The electric field is the negative gradient of the potential
            Eigen::Vector2d E_elem(-dphi_dx, -dphi_dy);

            // Add to the average
            E_avg += E_elem;
        }

        // Compute the average
        if (!node_elements.empty()) {
            E_avg /= node_elements.size();
        }

        // Store the electric field
        E_field[i] = E_avg;
    }
}

/**
 * @brief Gets the electric field at a given position.
 *
 * @param x The x-coordinate of the position
 * @param y The y-coordinate of the position
 * @return The electric field vector at the given position
 */
Eigen::Vector2d FullPoissonDriftDiffusionSolver::get_electric_field(double x, double y) const {
    // Find the element containing the point (x, y)
    int elem_idx = -1;
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        auto element = mesh.getElements()[e];
        std::vector<Eigen::Vector2d> nodes;
        for (int i = 0; i < 3; ++i) {
            nodes.push_back(mesh.getNodes()[element[i]]);
        }

        // Check if the point is inside the element using barycentric coordinates
        double x1 = nodes[0][0], y1 = nodes[0][1];
        double x2 = nodes[1][0], y2 = nodes[1][1];
        double x3 = nodes[2][0], y3 = nodes[2][1];

        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        double lambda1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / (2.0 * area);
        double lambda2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / (2.0 * area);
        double lambda3 = 1.0 - lambda1 - lambda2;

        if (lambda1 >= 0.0 && lambda2 >= 0.0 && lambda3 >= 0.0) {
            elem_idx = e;
            break;
        }
    }

    if (elem_idx >= 0) {
        auto element = mesh.getElements()[elem_idx];
        std::vector<Eigen::Vector2d> nodes;
        for (int i = 0; i < 3; ++i) {
            nodes.push_back(mesh.getNodes()[element[i]]);
        }

        // Calculate element area
        double x1 = nodes[0][0], y1 = nodes[0][1];
        double x2 = nodes[1][0], y2 = nodes[1][1];
        double x3 = nodes[2][0], y3 = nodes[2][1];

        // Compute the derivatives of the shape functions
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        double dN1_dx = (y2 - y3) / (2.0 * area);
        double dN1_dy = (x3 - x2) / (2.0 * area);
        double dN2_dx = (y3 - y1) / (2.0 * area);
        double dN2_dy = (x1 - x3) / (2.0 * area);
        double dN3_dx = (y1 - y2) / (2.0 * area);
        double dN3_dy = (x2 - x1) / (2.0 * area);

        // Compute the gradient of the potential
        double dphi_dx = phi[element[0]] * dN1_dx + phi[element[1]] * dN2_dx + phi[element[2]] * dN3_dx;
        double dphi_dy = phi[element[0]] * dN1_dy + phi[element[1]] * dN2_dy + phi[element[2]] * dN3_dy;

        // The electric field is the negative gradient of the potential
        return Eigen::Vector2d(-dphi_dx, -dphi_dy);
    } else {
        // If the point is outside the mesh, return zero electric field
        return Eigen::Vector2d::Zero();
    }
}

/**
 * @brief Computes the carrier concentrations using Boltzmann statistics.
 */
void FullPoissonDriftDiffusionSolver::compute_carrier_concentrations_boltzmann() {
    // Constants
    const double kT = 0.0259; // eV at 300K

    // Compute carrier concentrations at each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Compute carrier concentrations using Boltzmann statistics
        n[i] = mat.N_c * std::exp((phi_n[i] - phi[i]) / kT);
        p[i] = mat.N_v * std::exp((phi[i] - phi_p[i] - mat.E_g) / kT);

        // Ensure minimum carrier concentrations for numerical stability
        const double n_min = 1e5; // Minimum concentration (cm^-3)
        n[i] = std::max(n[i], n_min);
        p[i] = std::max(p[i], n_min);
    }
}

/**
 * @brief Computes the carrier concentrations using Fermi-Dirac statistics.
 */
void FullPoissonDriftDiffusionSolver::compute_carrier_concentrations_fermi_dirac() {
    // Constants
    const double kT = 0.0259; // eV at 300K

    // Compute carrier concentrations at each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Compute the Fermi-Dirac integral of order 1/2
        auto fermi_dirac_half = [](double eta) {
            // Approximation of the Fermi-Dirac integral of order 1/2
            // Based on the approximation by Bednarczyk and Bednarczyk
            if (eta <= 0) {
                return std::exp(eta) / (1.0 + 0.27 * std::exp(eta));
            } else {
                return std::pow(eta, 1.5) / (1.5 + 0.27 * std::pow(eta, 1.5));
            }
        };

        // Compute the reduced Fermi levels
        double eta_n = (phi_n[i] - phi[i]) / kT;
        double eta_p = (phi[i] - phi_p[i] - mat.E_g) / kT;

        // Compute carrier concentrations using Fermi-Dirac statistics
        n[i] = mat.N_c * fermi_dirac_half(eta_n);
        p[i] = mat.N_v * fermi_dirac_half(eta_p);

        // Ensure minimum carrier concentrations for numerical stability
        const double n_min = 1e5; // Minimum concentration (cm^-3)
        n[i] = std::max(n[i], n_min);
        p[i] = std::max(p[i], n_min);
    }
}

/**
 * @brief Computes the mobilities for electrons and holes.
 *
 * @param mu_n_out Output vector for electron mobility
 * @param mu_p_out Output vector for hole mobility
 */
void FullPoissonDriftDiffusionSolver::compute_mobilities(Eigen::VectorXd& mu_n_out, Eigen::VectorXd& mu_p_out) const {
    // Resize output vectors
    mu_n_out.resize(mesh.getNumNodes());
    mu_p_out.resize(mesh.getNumNodes());

    // Compute mobilities at each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Get the electric field magnitude at this node
        double E_mag = E_field[i].norm();

        // Compute mobilities using the mobility models
        mu_n_out[i] = mu_n_model(x, y, E_mag, mat);
        mu_p_out[i] = mu_p_model(x, y, E_mag, mat);
    }
}

/**
 * @brief Computes the generation-recombination rate.
 *
 * @return The generation-recombination rate vector
 */
Eigen::VectorXd FullPoissonDriftDiffusionSolver::compute_generation_recombination_rate() const {
    // Initialize the generation-recombination rate vector
    Eigen::VectorXd G_R(mesh.getNumNodes());

    // Compute the generation-recombination rate at each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Compute the generation-recombination rate using the model
        G_R[i] = g_r_model(x, y, n[i], p[i], mat);
    }

    return G_R;
}

/**
 * @brief Applies boundary conditions to the carrier concentrations.
 *
 * @param V_p The voltage applied to the p-contact
 * @param V_n The voltage applied to the n-contact
 */
void FullPoissonDriftDiffusionSolver::apply_boundary_conditions(double V_p, double V_n) {
    // Constants
    const double kT = 0.0259; // eV at 300K

    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double Ly = mesh.get_ly();
    double tolerance = 1e-6; // Tolerance for boundary detection (increased for robustness)

    // Count boundary nodes
    int left_boundary_count = 0;
    int right_boundary_count = 0;

    // Apply boundary conditions at each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        const auto& node = mesh.getNodes()[i];
        double x = node[0];
        double y = node[1];

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Calculate intrinsic carrier concentration
        double ni = std::sqrt(mat.N_c * mat.N_v) * std::exp(-mat.E_g / (2.0 * kT));

        // Left boundary (p-type)
        if (x < tolerance) {
            // Set the potential
            phi[i] = V_p;

            // Set the quasi-Fermi potentials
            double N_a = 1e17; // Acceptor concentration (cm^-3)
            phi_n[i] = V_p + kT * std::log(ni * ni / N_a / mat.N_c);
            phi_p[i] = V_p - mat.E_g;

            // Set the carrier concentrations
            n[i] = ni * ni / N_a;
            p[i] = N_a;

            left_boundary_count++;
        }
        // Right boundary (n-type)
        else if (x > Lx - tolerance) {
            // Set the potential
            phi[i] = V_n;

            // Set the quasi-Fermi potentials
            double N_d = 1e17; // Donor concentration (cm^-3)
            phi_n[i] = V_n;
            phi_p[i] = V_n - mat.E_g - kT * std::log(N_d / mat.N_v);

            // Set the carrier concentrations
            n[i] = N_d;
            p[i] = ni * ni / N_d;

            right_boundary_count++;
        }
    }

    std::cout << "Applied boundary conditions: " << left_boundary_count << " left nodes, "
              << right_boundary_count << " right nodes" << std::endl;
}

/**
 * @brief Solves the drift-diffusion equations.
 *
 * @param V_p The voltage applied to the p-contact
 * @param V_n The voltage applied to the n-contact
 */
void FullPoissonDriftDiffusionSolver::solve_drift_diffusion_equations(double V_p, double V_n) {
    // Constants
    const double q = 1.602e-19; // Elementary charge (C)
    const double kT = 0.0259; // eV at 300K
    const double epsilon_0 = 8.85418782e-14; // Vacuum permittivity (F/cm)

    // Get the number of nodes
    int num_nodes = mesh.getNumNodes();

    // Compute mobilities
    Eigen::VectorXd mu_n_vec, mu_p_vec;
    compute_mobilities(mu_n_vec, mu_p_vec);

    // Compute generation-recombination rate
    Eigen::VectorXd G_R = compute_generation_recombination_rate();

    // Create stiffness matrices and right-hand side vectors for electrons and holes
    Eigen::SparseMatrix<double> K_n(num_nodes, num_nodes);
    Eigen::SparseMatrix<double> K_p(num_nodes, num_nodes);
    Eigen::VectorXd f_n(num_nodes);
    Eigen::VectorXd f_p(num_nodes);
    f_n.setZero();
    f_p.setZero();

    // Create triplet lists for sparse matrix assembly
    std::vector<Eigen::Triplet<double>> triplets_n;
    std::vector<Eigen::Triplet<double>> triplets_p;

    // Assemble the stiffness matrices and right-hand side vectors
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

        // Calculate element centroid
        double xc = (x1 + x2 + x3) / 3.0;
        double yc = (y1 + y2 + y3) / 3.0;

        // Get material properties at the centroid
        Materials::Material mat = get_material_at(xc, yc);

        // Interpolate electric field, carrier concentrations, and mobilities at the centroid
        Eigen::Vector2d E_c = Eigen::Vector2d::Zero();
        double n_c = 0.0, p_c = 0.0;
        double mu_n_c = 0.0, mu_p_c = 0.0;
        double G_R_c = 0.0;

        for (int i = 0; i < 3; ++i) {
            double lambda = 1.0 / 3.0; // Equal weight for each node in the centroid

            E_c += lambda * E_field[element[i]];
            n_c += lambda * n[element[i]];
            p_c += lambda * p[element[i]];
            mu_n_c += lambda * mu_n_vec[element[i]];
            mu_p_c += lambda * mu_p_vec[element[i]];
            G_R_c += lambda * G_R[element[i]];
        }

        // Compute the diffusion coefficients
        double D_n = mu_n_c * kT;
        double D_p = mu_p_c * kT;

        // Compute the element matrices for electrons and holes
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Diffusion term
                double diff_term = D_n * grad_N[i].dot(grad_N[j]) * area;

                // Drift term for electrons
                double drift_term_n = mu_n_c * E_c.dot(grad_N[j]) * area / 3.0; // Lumped mass approximation

                // Add the element matrix entry to the triplet list for electrons
                triplets_n.emplace_back(element[i], element[j], diff_term + drift_term_n);

                // Diffusion term for holes
                diff_term = D_p * grad_N[i].dot(grad_N[j]) * area;

                // Drift term for holes (note the sign change)
                double drift_term_p = -mu_p_c * E_c.dot(grad_N[j]) * area / 3.0; // Lumped mass approximation

                // Add the element matrix entry to the triplet list for holes
                triplets_p.emplace_back(element[i], element[j], diff_term + drift_term_p);
            }

            // Add the generation-recombination term to the right-hand side vectors
            f_n[element[i]] += G_R_c * area / 3.0; // Lumped mass approximation
            f_p[element[i]] += G_R_c * area / 3.0; // Lumped mass approximation
        }
    }

    // Set the global matrices from the triplets
    K_n.setFromTriplets(triplets_n.begin(), triplets_n.end());
    K_p.setFromTriplets(triplets_p.begin(), triplets_p.end());

    // Apply boundary conditions
    apply_boundary_conditions(V_p, V_n);

    // Create a vector to mark boundary nodes
    std::vector<bool> is_boundary(num_nodes, false);

    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double Ly = mesh.get_ly();
    double tolerance = 1e-6; // Tolerance for boundary detection (increased for robustness)

    // Identify boundary nodes
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.getNodes()[i];
        double x = node[0];
        double y = node[1];

        // Left or right boundary
        if (x < tolerance || x > Lx - tolerance) {
            is_boundary[i] = true;
        }
    }

    // Print the number of boundary nodes
    std::cout << "Number of boundary nodes for drift-diffusion: " << std::count(is_boundary.begin(), is_boundary.end(), true) << std::endl;

    // Apply Dirichlet boundary conditions to the stiffness matrices and right-hand side vectors
    for (size_t i = 0; i < num_nodes; ++i) {
        if (is_boundary[i]) {
            // Zero out the rows
            for (int k = 0; k < K_n.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(K_n, k); it; ++it) {
                    if (it.row() == i) {
                        it.valueRef() = 0.0;
                    }
                }
            }

            for (int k = 0; k < K_p.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(K_p, k); it; ++it) {
                    if (it.row() == i) {
                        it.valueRef() = 0.0;
                    }
                }
            }

            // Set the diagonal entries to 1
            K_n.coeffRef(i, i) = 1.0;
            K_p.coeffRef(i, i) = 1.0;

            // Set the right-hand side values
            f_n[i] = phi_n[i];
            f_p[i] = phi_p[i];
        }
    }

    // Solve the linear systems using the SimplicialLDLT solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_n;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_p;

    solver_n.compute(K_n);
    solver_p.compute(K_p);

    // Check if the factorizations were successful
    if (solver_n.info() != Eigen::Success || solver_p.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize the stiffness matrices");
    }

    // Solve the linear systems
    Eigen::VectorXd phi_n_new = solver_n.solve(f_n);
    Eigen::VectorXd phi_p_new = solver_p.solve(f_p);

    // Check if the solves were successful
    if (solver_n.info() != Eigen::Success || solver_p.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve the linear systems");
    }

    // Update the quasi-Fermi potentials
    phi_n = phi_n_new;
    phi_p = phi_p_new;

    // Update the carrier concentrations
    if (use_fermi_dirac_statistics) {
        compute_carrier_concentrations_fermi_dirac();
    } else {
        compute_carrier_concentrations_boltzmann();
    }

    // Compute the current densities
    compute_current_densities();
}

/**
 * @brief Computes the current densities.
 */
void FullPoissonDriftDiffusionSolver::compute_current_densities() {
    // Constants
    const double q = 1.602e-19; // Elementary charge (C)
    const double kT = 0.0259; // eV at 300K

    // Compute mobilities
    Eigen::VectorXd mu_n_vec, mu_p_vec;
    compute_mobilities(mu_n_vec, mu_p_vec);

    // Compute current densities at each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        // Find all elements containing this node
        std::vector<size_t> node_elements;
        for (size_t e = 0; e < mesh.getNumElements(); ++e) {
            auto element = mesh.getElements()[e];
            for (int j = 0; j < 3; ++j) {
                if (element[j] == i) {
                    node_elements.push_back(e);
                    break;
                }
            }
        }

        // Compute the average current densities from all elements containing this node
        Eigen::Vector2d J_n_avg = Eigen::Vector2d::Zero();
        Eigen::Vector2d J_p_avg = Eigen::Vector2d::Zero();

        for (size_t e : node_elements) {
            auto element = mesh.getElements()[e];
            std::vector<Eigen::Vector2d> nodes;
            for (int j = 0; j < 3; ++j) {
                nodes.push_back(mesh.getNodes()[element[j]]);
            }

            // Calculate element area
            double x1 = nodes[0][0], y1 = nodes[0][1];
            double x2 = nodes[1][0], y2 = nodes[1][1];
            double x3 = nodes[2][0], y3 = nodes[2][1];

            // Compute the derivatives of the shape functions
            double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

            double dN1_dx = (y2 - y3) / (2.0 * area);
            double dN1_dy = (x3 - x2) / (2.0 * area);
            double dN2_dx = (y3 - y1) / (2.0 * area);
            double dN2_dy = (x1 - x3) / (2.0 * area);
            double dN3_dx = (y1 - y2) / (2.0 * area);
            double dN3_dy = (x2 - x1) / (2.0 * area);

            // Compute the gradients of the quasi-Fermi potentials
            double dphi_n_dx = phi_n[element[0]] * dN1_dx + phi_n[element[1]] * dN2_dx + phi_n[element[2]] * dN3_dx;
            double dphi_n_dy = phi_n[element[0]] * dN1_dy + phi_n[element[1]] * dN2_dy + phi_n[element[2]] * dN3_dy;
            double dphi_p_dx = phi_p[element[0]] * dN1_dx + phi_p[element[1]] * dN2_dx + phi_p[element[2]] * dN3_dx;
            double dphi_p_dy = phi_p[element[0]] * dN1_dy + phi_p[element[1]] * dN2_dy + phi_p[element[2]] * dN3_dy;

            // Interpolate carrier concentrations and mobilities at the element centroid
            double n_c = 0.0, p_c = 0.0;
            double mu_n_c = 0.0, mu_p_c = 0.0;

            for (int j = 0; j < 3; ++j) {
                double lambda = 1.0 / 3.0; // Equal weight for each node in the centroid

                n_c += lambda * n[element[j]];
                p_c += lambda * p[element[j]];
                mu_n_c += lambda * mu_n_vec[element[j]];
                mu_p_c += lambda * mu_p_vec[element[j]];
            }

            // Compute the diffusion coefficients
            double D_n = mu_n_c * kT;
            double D_p = mu_p_c * kT;

            // Compute the current densities
            Eigen::Vector2d J_n_elem = q * mu_n_c * n_c * Eigen::Vector2d(dphi_n_dx, dphi_n_dy);
            Eigen::Vector2d J_p_elem = q * mu_p_c * p_c * Eigen::Vector2d(dphi_p_dx, dphi_p_dy);

            // Add to the average
            J_n_avg += J_n_elem;
            J_p_avg += J_p_elem;
        }

        // Compute the average
        if (!node_elements.empty()) {
            J_n_avg /= node_elements.size();
            J_p_avg /= node_elements.size();
        }

        // Store the current densities
        J_n[i] = J_n_avg;
        J_p[i] = J_p_avg;
    }
}

/**
 * @brief Apply damping to the potential update.
 *
 * @param phi_old The previous potential vector
 * @param phi_update The updated potential vector
 * @return The damped potential vector
 */
Eigen::VectorXd FullPoissonDriftDiffusionSolver::apply_damping(
    const Eigen::VectorXd& phi_old, const Eigen::VectorXd& phi_update) const {

    // Apply damping: phi_new = phi_old + damping_factor * (phi_update - phi_old)
    return phi_old + damping_factor * (phi_update - phi_old);
}

/**
 * @brief Apply Anderson acceleration to the potential update.
 *
 * @param phi_old The previous potential vector
 * @param phi_update The updated potential vector
 * @return The accelerated potential vector
 */
Eigen::VectorXd FullPoissonDriftDiffusionSolver::apply_anderson_acceleration(
    const Eigen::VectorXd& phi_old, const Eigen::VectorXd& phi_update) {

    // Compute the residual
    Eigen::VectorXd residual = phi_update - phi_old;

    // Add the current potential and residual to the history
    phi_history.push_back(phi_old);
    res_history.push_back(residual);

    // Keep only the most recent iterations
    if (phi_history.size() > anderson_history_size) {
        phi_history.erase(phi_history.begin());
        res_history.erase(res_history.begin());
    }

    // If we don't have enough history, just apply damping
    if (phi_history.size() < 2) {
        return apply_damping(phi_old, phi_update);
    }

    // Compute the differences between consecutive residuals
    std::vector<Eigen::VectorXd> dF;
    for (size_t i = 1; i < res_history.size(); ++i) {
        dF.push_back(res_history[i] - res_history[i - 1]);
    }

    // Compute the Gram matrix
    Eigen::MatrixXd G(dF.size(), dF.size());
    for (size_t i = 0; i < dF.size(); ++i) {
        for (size_t j = 0; j < dF.size(); ++j) {
            G(i, j) = dF[i].dot(dF[j]);
        }
    }

    // Compute the right-hand side
    Eigen::VectorXd b(dF.size());
    for (size_t i = 0; i < dF.size(); ++i) {
        b(i) = dF[i].dot(res_history.back());
    }

    // Solve for the coefficients
    Eigen::VectorXd alpha = G.fullPivLu().solve(b);

    // Compute the accelerated potential
    Eigen::VectorXd phi_accel = phi_update;
    for (size_t i = 0; i < dF.size(); ++i) {
        phi_accel -= alpha(i) * (phi_history[i + 1] - phi_history[i] + res_history[i + 1] - res_history[i]);
    }

    // Perform line search to find optimal step size
    double beta = perform_line_search(phi_old, phi_update, phi_accel);

    // Apply the optimal step size
    return phi_old + beta * (phi_accel - phi_old);
}

/**
 * @brief Performs line search to find optimal step size.
 *
 * @param phi_old The potential from the previous iteration
 * @param phi_update The updated potential from the current iteration
 * @param phi_accel The accelerated potential
 * @return The optimal step size
 */
double FullPoissonDriftDiffusionSolver::perform_line_search(
    const Eigen::VectorXd& phi_old,
    const Eigen::VectorXd& phi_update,
    const Eigen::VectorXd& phi_accel) {

    // Compute the residual norm for the damped update
    double res_norm_damped = (phi_update - phi_old).norm();

    // Compute the residual norm for the accelerated update
    double res_norm_accel = (phi_accel - phi_old).norm();

    // If the accelerated update has a smaller residual, use it
    if (res_norm_accel < res_norm_damped) {
        return 1.0;
    }

    // Otherwise, use backtracking line search
    double beta = 1.0;
    const double beta_min = 0.1;
    const double beta_factor = 0.5;

    while (beta > beta_min) {
        // Compute the trial potential
        Eigen::VectorXd phi_trial = phi_old + beta * (phi_accel - phi_old);

        // Compute the residual norm for the trial potential
        double res_norm_trial = (phi_trial - phi_old).norm();

        // If the trial potential has a smaller residual, use it
        if (res_norm_trial < res_norm_damped) {
            return beta;
        }

        // Otherwise, reduce the step size
        beta *= beta_factor;
    }

    // If no good step size is found, use damping
    return damping_factor;
}

/**
 * @brief Solves the coupled Poisson-drift-diffusion equations.
 *
 * @param V_p The voltage applied to the p-contact
 * @param V_n The voltage applied to the n-contact
 * @param tolerance The convergence tolerance
 * @param max_iter The maximum number of iterations
 */
void FullPoissonDriftDiffusionSolver::solve(double V_p, double V_n, double tolerance, int max_iter) {
    // Initialize the solution
    initialize_carrier_concentrations();

    // Apply boundary conditions
    apply_boundary_conditions(V_p, V_n);

    // Initialize the potential with a linear profile between V_p and V_n
    std::cout << "Initializing potential with linear profile" << std::endl;
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        const auto& node = mesh.getNodes()[i];
        double x = node[0];
        double Lx = mesh.get_lx();

        // Linear interpolation between V_p and V_n
        phi[i] = V_p + (V_n - V_p) * x / Lx;
    }

    // Compute the electric field from the initial potential
    compute_electric_field();

    // Compute the carrier concentrations
    if (use_fermi_dirac_statistics) {
        compute_carrier_concentrations_fermi_dirac();
    } else {
        compute_carrier_concentrations_boltzmann();
    }

    // Compute the electric field
    compute_electric_field();

    // Compute the current densities
    compute_current_densities();

    // Self-consistent iteration
    double error = 1.0;
    int iter = 0;

    // Clear the history for Anderson acceleration
    phi_history.clear();
    res_history.clear();

    // Iterate until convergence or maximum iterations
    while (error > tolerance && iter < max_iter) {
        // Save the current potential
        Eigen::VectorXd phi_old = phi;

        // First iteration: use the initial linear profile
        // Subsequent iterations: solve the Poisson equation
        if (iter > 0) {
            // Solve the Poisson equation
            try {
                solve_poisson_equation(V_p, V_n);
            } catch (const std::exception& e) {
                std::cerr << "Error solving Poisson equation: " << e.what() << std::endl;
                std::cerr << "Using linear profile instead" << std::endl;

                // Fall back to linear profile
                for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
                    const auto& node = mesh.getNodes()[i];
                    double x = node[0];
                    double Lx = mesh.get_lx();

                    // Linear interpolation between V_p and V_n
                    phi[i] = V_p + (V_n - V_p) * x / Lx;
                }
            }

            // Apply convergence acceleration
            if (iter > 1) {
                // Apply damping instead of Anderson acceleration for stability
                phi = apply_damping(phi_old, phi);
            }
        }

        // Compute the electric field
        compute_electric_field();

        // Solve the drift-diffusion equations
        try {
            solve_drift_diffusion_equations(V_p, V_n);
        } catch (const std::exception& e) {
            std::cerr << "Error solving drift-diffusion equations: " << e.what() << std::endl;
            // Continue with the current carrier concentrations
        }

        // Compute the error
        double phi_norm = phi.norm();
        double diff_norm = (phi - phi_old).norm();

        // Check for NaN or Inf values
        bool has_nan_or_inf = false;
        for (int i = 0; i < phi.size(); ++i) {
            if (std::isnan(phi[i]) || std::isinf(phi[i])) {
                has_nan_or_inf = true;
                phi[i] = phi_old[i]; // Revert to previous value
            }
        }

        if (has_nan_or_inf) {
            std::cerr << "Warning: NaN or Inf values detected in the solution" << std::endl;
            error = 1.0; // Force another iteration or exit if max_iter is reached
        } else if (phi_norm > 1e-10) {
            error = diff_norm / phi_norm;
        } else {
            error = diff_norm;
        }

        // Ensure error is a valid number
        if (std::isnan(error) || std::isinf(error)) {
            std::cerr << "Warning: NaN or Inf error detected" << std::endl;
            error = 1.0; // Force another iteration or exit if max_iter is reached
        }

        // Increment the iteration counter
        ++iter;

        // Print the error
        std::cout << "Iteration " << iter << ": error = " << error << std::endl;
    }

    // Check if the solution converged
    if (iter >= max_iter && error > tolerance) {
        std::cout << "Warning: Solution did not converge after " << max_iter << " iterations" << std::endl;
    } else {
        std::cout << "Solution converged after " << iter << " iterations" << std::endl;
    }

    // Compute the final electric field and current densities
    compute_electric_field();
    compute_current_densities();
}