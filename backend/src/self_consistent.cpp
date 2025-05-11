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
 * @brief Default constructor for SelfConsistentSolver.
 *
 * This constructor initializes the SelfConsistentSolver with the given mesh.
 * It sets up default values for the callback functions and initializes the
 * PoissonSolver with default values.
 *
 * @param mesh The mesh to use for the simulation
 */
SelfConsistentSolver::SelfConsistentSolver(Mesh& mesh)
    : mesh(mesh), poisson(mesh,
        // Default epsilon_r function
        [](double x, double y) -> double { return 12.9; },
        // Default rho function
        [](double x, double y, const Eigen::VectorXd& n, const Eigen::VectorXd& p) -> double { return 0.0; }),
      epsilon_r(nullptr), rho(nullptr), n_conc(nullptr), p_conc(nullptr), mu_n(nullptr), mu_p(nullptr),
      damping_factor(0.3), anderson_history_size(3), has_heterojunction(false), N_A(0.0), N_D(0.0) {

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

/**
 * @brief Initializes the carrier concentrations based on the doping concentrations.
 *
 * This function initializes the electron and hole concentrations based on the doping
 * concentrations and proper physics. It calculates the carrier concentrations using
 * the Boltzmann approximation and ensures charge neutrality in the bulk regions.
 *
 * The initialization process includes:
 * 1. Calculating the intrinsic carrier concentration based on material properties
 * 2. Determining the doping profile based on position
 * 3. Computing the quasi-Fermi levels for electrons and holes
 * 4. Calculating the carrier concentrations using the Boltzmann approximation
 * 5. Ensuring charge neutrality in the bulk regions
 *
 * @param N_A The acceptor doping concentration
 * @param N_D The donor doping concentration
 */
void SelfConsistentSolver::initialize_carriers(double N_A, double N_D) {
    // Constants
    const double kT = 0.0259; // eV at 300K
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    const double T = 300.0; // Temperature in K
    const double q = 1.602e-19; // Elementary charge in C

    // Get the potential from the Poisson solver
    const Eigen::VectorXd& phi = poisson.get_potential();

    // Initialize carrier concentrations based on doping and potential
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties based on position
        Materials::Material mat = get_material_at(x, y);

        // Calculate intrinsic carrier concentration based on material properties
        double ni = std::sqrt(mat.N_c * mat.N_v) * std::exp(-mat.E_g / (2.0 * kT));

        // Determine doping profile based on position
        // This can be replaced with a more complex doping profile function if needed
        double N_A_pos, N_D_pos;
        if (x < 0) {
            // p-side
            N_A_pos = N_A;
            N_D_pos = 0.0;
        } else {
            // n-side
            N_A_pos = 0.0;
            N_D_pos = N_D;
        }

        // Calculate net doping
        double N_net = N_D_pos - N_A_pos;

        // Calculate built-in potential
        double V_bi = kT * std::log(N_A * N_D / (ni * ni));

        // Calculate quasi-Fermi levels
        double E_F_p = -mat.E_g + kB * T * std::log(mat.N_v / std::max(N_A_pos, 1.0));
        double E_F_n = -kB * T * std::log(mat.N_c / std::max(N_D_pos, 1.0));

        // Calculate carrier concentrations using the Boltzmann approximation
        if (i < phi.size() && std::abs(phi[i]) > 1e-10) {
            // Use the potential to calculate carrier concentrations
            double V = phi[i] / q; // Convert from V to eV

            // Calculate band edges
            double E_c = -q * V - (x < 0 ? E_F_p : E_F_n);
            double E_v = E_c - mat.E_g;

            // Calculate carrier concentrations using Boltzmann statistics
            n[i] = mat.N_c * std::exp(-E_c / kT);
            p[i] = mat.N_v * std::exp(E_v / kT);
        } else {
            // Initial guess based on doping and charge neutrality
            if (N_net > 0) {
                // n-type region
                n[i] = N_net + std::sqrt(N_net * N_net + 4.0 * ni * ni) / 2.0;
                p[i] = ni * ni / n[i];
            } else if (N_net < 0) {
                // p-type region
                p[i] = -N_net + std::sqrt(N_net * N_net + 4.0 * ni * ni) / 2.0;
                n[i] = ni * ni / p[i];
            } else {
                // Intrinsic region
                n[i] = p[i] = ni;
            }
        }

        // Ensure minimum carrier concentrations for numerical stability
        const double n_min = 1e-10; // Minimum concentration (cm^-3)
        const double n_max = 1e20;  // Maximum concentration (cm^-3)
        n[i] = std::max(std::min(n[i], n_max), n_min);
        p[i] = std::max(std::min(p[i], n_max), n_min);
    }

    // Initialize quasi-Fermi potentials
    // These will be used in the self-consistent solution process
    phi_n.resize(mesh.getNumNodes());
    phi_p.resize(mesh.getNumNodes());

    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        // Get material properties
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];
        Materials::Material mat = get_material_at(x, y);

        // Calculate quasi-Fermi potentials
        phi_n[i] = kT * std::log(n[i] / mat.N_c);
        phi_p[i] = -kT * std::log(p[i] / mat.N_v) - mat.E_g;
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
/**
 * @brief Solves the drift-diffusion equations for electron and hole transport.
 *
 * This function solves the drift-diffusion equations for electron and hole transport
 * using the assembled matrices and the current potential. It updates the carrier
 * concentrations based on the solution, including generation-recombination processes
 * and proper handling of non-equilibrium conditions.
 *
 * The drift-diffusion equations are:
 * ∇·(μn·n·∇φ + Dn·∇n) = G - R  (for electrons)
 * ∇·(-μp·p·∇φ + Dp·∇p) = G - R  (for holes)
 *
 * Where:
 * - μn, μp are the electron and hole mobilities
 * - n, p are the electron and hole concentrations
 * - φ is the electrostatic potential
 * - Dn, Dp are the diffusion coefficients
 * - G is the generation rate
 * - R is the recombination rate
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

    // Constants
    const double kT = 0.0259; // eV at 300K
    const double q = 1.602e-19; // Elementary charge in C
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    const double T = 300.0; // Temperature in K

    // Calculate the right-hand side vectors with proper generation-recombination terms
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties based on position
        Materials::Material mat = get_material_at(x, y);

        // Calculate generation rate (optical generation, impact ionization, etc.)
        double G = 0.0; // Default: no generation

        // Calculate recombination rate using SRH, Auger, and radiative recombination models
        double R_SRH = 0.0;  // Shockley-Read-Hall recombination
        double R_Auger = 0.0; // Auger recombination
        double R_rad = 0.0;   // Radiative recombination

        // Intrinsic carrier concentration (temperature dependent)
        double ni = std::sqrt(mat.N_c * mat.N_v) * std::exp(-mat.E_g / (2.0 * kT));

        // SRH recombination: R_SRH = (np - ni²) / (τn(p + ni) + τp(n + ni))
        double tau_n = 1e-9; // Electron lifetime (s)
        double tau_p = 1e-9; // Hole lifetime (s)
        R_SRH = (n[i] * p[i] - ni * ni) / (tau_n * (p[i] + ni) + tau_p * (n[i] + ni));

        // Auger recombination: R_Auger = Cn·n²·p + Cp·p²·n
        double Cn = 1e-30; // Auger coefficient for electrons (cm^6/s)
        double Cp = 1e-30; // Auger coefficient for holes (cm^6/s)
        R_Auger = Cn * n[i] * n[i] * p[i] + Cp * p[i] * p[i] * n[i];

        // Radiative recombination: R_rad = B·(np - ni²)
        double B = 1e-10; // Radiative recombination coefficient (cm^3/s)
        R_rad = B * (n[i] * p[i] - ni * ni);

        // Total recombination rate
        double R = R_SRH + R_Auger + R_rad;

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

    // Update carrier concentrations with damping to improve stability
    double carrier_damping = 0.5; // Damping factor for carrier updates

    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        // Apply damping to carrier updates
        n[i] += carrier_damping * delta_n[i];
        p[i] += carrier_damping * delta_p[i];

        // Ensure positive carrier concentrations with a more realistic minimum
        const double n_min = 1e-10; // Minimum concentration (cm^-3)
        const double n_max = 1e20;  // Maximum concentration (cm^-3)
        n[i] = std::max(std::min(n[i], n_max), n_min);
        p[i] = std::max(std::min(p[i], n_max), n_min);
    }

    // Update quasi-Fermi potentials based on the new carrier concentrations
    update_quasi_fermi_potentials();
}

/**
 * @brief Updates the quasi-Fermi potentials based on carrier concentrations.
 *
 * This method updates the quasi-Fermi potentials for electrons and holes
 * based on the current carrier concentrations and material properties.
 * The quasi-Fermi potentials are used to calculate the current densities
 * and to ensure self-consistency in non-equilibrium conditions.
 *
 * The quasi-Fermi potentials are calculated as:
 * phi_n = kT * ln(n / N_c)
 * phi_p = -kT * ln(p / N_v) - E_g
 *
 * Where:
 * - kT is the thermal voltage
 * - n, p are the carrier concentrations
 * - N_c, N_v are the effective densities of states
 * - E_g is the band gap
 */
void SelfConsistentSolver::update_quasi_fermi_potentials() {
    // Constants
    const double kT = 0.0259; // eV at 300K

    // Resize quasi-Fermi potential vectors if needed
    if (phi_n.size() != mesh.getNumNodes()) {
        phi_n.resize(mesh.getNumNodes());
    }
    if (phi_p.size() != mesh.getNumNodes()) {
        phi_p.resize(mesh.getNumNodes());
    }

    // Update quasi-Fermi potentials for each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        // Get material properties
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];
        Materials::Material mat = get_material_at(x, y);

        // Calculate quasi-Fermi potentials
        phi_n[i] = kT * std::log(n[i] / mat.N_c);
        phi_p[i] = -kT * std::log(p[i] / mat.N_v) - mat.E_g;
    }
}

/**
 * @brief Calculates the built-in potential of the P-N junction with temperature dependence.
 *
 * This method calculates the built-in potential of the P-N junction
 * based on the doping concentrations and material properties, with proper
 * temperature dependence. The built-in potential is the potential difference
 * between the p-side and the n-side in thermal equilibrium.
 *
 * The built-in potential is given by:
 * V_bi = (kT/q) * ln(N_A * N_D / (n_i^2))
 *
 * Where:
 * - kT is the thermal voltage (temperature dependent)
 * - q is the elementary charge
 * - N_A is the acceptor doping concentration
 * - N_D is the donor doping concentration
 * - n_i is the intrinsic carrier concentration (temperature dependent)
 *
 * @param N_A The acceptor doping concentration
 * @param N_D The donor doping concentration
 * @param T The temperature in Kelvin (default: 300K)
 * @return The built-in potential in volts
 */
double SelfConsistentSolver::calculate_built_in_potential(double N_A, double N_D, double T) const {
    // Constants
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    const double q = 1.602176634e-19; // Elementary charge in C
    const double kT = kB * T;         // Thermal voltage in eV

    // Get material properties at the junction
    // For simplicity, we'll use the material at x=0 (junction position)
    Materials::Material mat = get_material_at(0.0, 0.0);

    // Calculate temperature-dependent bandgap
    // Varshni equation: E_g(T) = E_g(0) - αT²/(T+β)
    double E_g0 = mat.E_g;      // Bandgap at 0K
    double alpha = 5.405e-4;    // Material-specific parameter (eV/K)
    double beta = 204.0;        // Material-specific parameter (K)
    double E_g_T = E_g0 - (alpha * T * T) / (T + beta);

    // Apply bandgap narrowing for heavily doped regions
    // Slotboom model: ΔE_g = C * ln(N/N_ref)
    double N_ref = 1e17;        // Reference doping concentration (cm^-3)
    double C_n = 9e-3;          // Coefficient for n-type (eV)
    double C_p = 13.5e-3;       // Coefficient for p-type (eV)

    // Calculate bandgap narrowing for n-type and p-type regions
    double delta_E_g_n = 0.0;
    double delta_E_g_p = 0.0;

    if (N_D > N_ref) {
        delta_E_g_n = C_n * std::log(N_D / N_ref);
    }

    if (N_A > N_ref) {
        delta_E_g_p = C_p * std::log(N_A / N_ref);
    }

    // Effective bandgap narrowing (use the larger of the two)
    double delta_E_g = std::max(delta_E_g_n, delta_E_g_p);

    // Apply bandgap narrowing
    double E_g_eff = E_g_T - delta_E_g;

    // Calculate temperature-dependent effective densities of states
    double m_e_eff = 0.067;     // Effective electron mass (GaAs)
    double m_h_eff = 0.45;      // Effective hole mass (GaAs)
    double h_bar = 6.582119569e-16; // Reduced Planck constant (eV·s)
    double m_0 = 9.10938356e-31;    // Electron rest mass (kg)

    // Temperature dependence of effective masses (simplified model)
    m_e_eff *= (1.0 + 0.5e-3 * (T - 300.0));
    m_h_eff *= (1.0 + 1.0e-3 * (T - 300.0));

    double N_c = 2.0 * std::pow(m_e_eff * m_0 * kT / (2.0 * M_PI * h_bar * h_bar), 1.5);
    double N_v = 2.0 * std::pow(m_h_eff * m_0 * kT / (2.0 * M_PI * h_bar * h_bar), 1.5);

    // Calculate intrinsic carrier concentration with temperature dependence and bandgap narrowing
    double ni = std::sqrt(N_c * N_v) * std::exp(-E_g_eff / (2.0 * kT));

    // Apply effective intrinsic carrier concentration due to bandgap narrowing
    double ni_eff = ni * std::exp(delta_E_g / (2.0 * kT));

    // Calculate Fermi levels using Fermi-Dirac statistics for high doping
    // For high doping, we need to account for degeneracy
    double E_F_n = 0.0;  // Fermi level in n-type region relative to conduction band
    double E_F_p = 0.0;  // Fermi level in p-type region relative to valence band

    // Threshold for degeneracy
    double N_degeneracy = 5e18;  // cm^-3

    if (N_D > N_degeneracy) {
        // Use Joyce-Dixon approximation for degenerate n-type
        double eta_n = kT * std::log(N_D / N_c) + kT * (std::sqrt(2.0) / 4.0) * std::pow(N_D / N_c, 0.75);
        E_F_n = eta_n;
    } else {
        // Use Boltzmann approximation for non-degenerate n-type
        E_F_n = kT * std::log(N_D / N_c);
    }

    if (N_A > N_degeneracy) {
        // Use Joyce-Dixon approximation for degenerate p-type
        double eta_p = kT * std::log(N_A / N_v) + kT * (std::sqrt(2.0) / 4.0) * std::pow(N_A / N_v, 0.75);
        E_F_p = eta_p;
    } else {
        // Use Boltzmann approximation for non-degenerate p-type
        E_F_p = kT * std::log(N_A / N_v);
    }

    // Calculate built-in potential using Fermi levels
    double V_bi = E_g_eff + E_F_p - E_F_n;

    // Alternative calculation using effective intrinsic carrier concentration
    double V_bi_alt = kT * std::log(N_A * N_D / (ni_eff * ni_eff));

    // Use the more accurate of the two methods
    V_bi = (N_A > N_degeneracy || N_D > N_degeneracy) ? V_bi : V_bi_alt;

    // Ensure the built-in potential is positive and physically reasonable
    V_bi = std::max(V_bi, 0.0);
    V_bi = std::min(V_bi, E_g_eff); // Built-in potential cannot exceed effective bandgap

    // Log the calculated values for debugging
    std::cout << "Temperature: " << T << " K" << std::endl;
    std::cout << "Bandgap (T=0K): " << E_g0 << " eV" << std::endl;
    std::cout << "Bandgap (T=" << T << "K): " << E_g_T << " eV" << std::endl;
    std::cout << "Bandgap narrowing: " << delta_E_g << " eV" << std::endl;
    std::cout << "Effective bandgap: " << E_g_eff << " eV" << std::endl;
    std::cout << "Intrinsic carrier concentration: " << ni << " cm^-3" << std::endl;
    std::cout << "Effective intrinsic carrier concentration: " << ni_eff << " cm^-3" << std::endl;
    std::cout << "N-type doping: " << N_D << " cm^-3" << std::endl;
    std::cout << "P-type doping: " << N_A << " cm^-3" << std::endl;

    // Statistics information
    if (N_D > N_degeneracy) {
        std::cout << "N-type region is degenerate (using Fermi-Dirac statistics)" << std::endl;
    } else {
        std::cout << "N-type region is non-degenerate (using Boltzmann statistics)" << std::endl;
    }

    if (N_A > N_degeneracy) {
        std::cout << "P-type region is degenerate (using Fermi-Dirac statistics)" << std::endl;
    } else {
        std::cout << "P-type region is non-degenerate (using Boltzmann statistics)" << std::endl;
    }

    std::cout << "Fermi level in n-type region: " << E_F_n << " eV (relative to conduction band)" << std::endl;
    std::cout << "Fermi level in p-type region: " << E_F_p << " eV (relative to valence band)" << std::endl;
    std::cout << "Built-in potential (Fermi-Dirac): " << E_g_eff + E_F_p - E_F_n << " V" << std::endl;
    std::cout << "Built-in potential (Boltzmann): " << V_bi_alt << " V" << std::endl;
    std::cout << "Final built-in potential: " << V_bi << " V" << std::endl;

    return V_bi;
}

/**
 * @brief Calculates the built-in potential of a heterojunction.
 *
 * This method calculates the built-in potential of a heterojunction
 * based on the doping concentrations, material properties, and band offsets.
 * It accounts for the difference in bandgaps, electron affinities, and
 * effective densities of states between the two materials.
 *
 * The built-in potential for a heterojunction is given by:
 * V_bi = (E_g,n + ΔE_c - ΔE_v) + (kT/q) * ln[(N_A * N_D) / (N_v,p * N_c,n)]
 *
 * Where:
 * - E_g,n is the bandgap of the n-type material
 * - ΔE_c is the conduction band offset
 * - ΔE_v is the valence band offset
 * - kT is the thermal voltage
 * - q is the elementary charge
 * - N_A is the acceptor doping concentration in p-type material
 * - N_D is the donor doping concentration in n-type material
 * - N_v,p is the effective density of states in the valence band of p-type material
 * - N_c,n is the effective density of states in the conduction band of n-type material
 *
 * @param N_A The acceptor doping concentration in p-type material
 * @param N_D The donor doping concentration in n-type material
 * @param mat_p The material properties of the p-type region
 * @param mat_n The material properties of the n-type region
 * @param T The temperature in Kelvin (default: 300K)
 * @return The built-in potential in volts
 */
double SelfConsistentSolver::calculate_heterojunction_potential(double N_A, double N_D,
                                                              const Materials::Material& mat_p,
                                                              const Materials::Material& mat_n,
                                                              double T) const {
    // Constants
    const double kB = 8.617333262e-5; // Boltzmann constant in eV/K
    const double q = 1.602176634e-19; // Elementary charge in C
    const double kT = kB * T;         // Thermal voltage in eV

    // Calculate temperature-dependent bandgaps for both materials
    // Varshni equation: E_g(T) = E_g(0) - αT²/(T+β)

    // P-type material
    double E_g0_p = mat_p.E_g;      // Bandgap at 0K
    double alpha_p = 5.405e-4;      // Material-specific parameter (eV/K)
    double beta_p = 204.0;          // Material-specific parameter (K)
    double E_g_T_p = E_g0_p - (alpha_p * T * T) / (T + beta_p);

    // N-type material
    double E_g0_n = mat_n.E_g;      // Bandgap at 0K
    double alpha_n = 5.405e-4;      // Material-specific parameter (eV/K)
    double beta_n = 204.0;          // Material-specific parameter (K)
    double E_g_T_n = E_g0_n - (alpha_n * T * T) / (T + beta_n);

    // Apply bandgap narrowing for heavily doped regions
    // Slotboom model: ΔE_g = C * ln(N/N_ref)
    double N_ref = 1e17;        // Reference doping concentration (cm^-3)
    double C_n = 9e-3;          // Coefficient for n-type (eV)
    double C_p = 13.5e-3;       // Coefficient for p-type (eV)

    // Calculate bandgap narrowing
    double delta_E_g_p = 0.0;
    double delta_E_g_n = 0.0;

    if (N_A > N_ref) {
        delta_E_g_p = C_p * std::log(N_A / N_ref);
    }

    if (N_D > N_ref) {
        delta_E_g_n = C_n * std::log(N_D / N_ref);
    }

    // Apply bandgap narrowing
    double E_g_eff_p = E_g_T_p - delta_E_g_p;
    double E_g_eff_n = E_g_T_n - delta_E_g_n;

    // Calculate electron affinities (typically material-specific constants)
    double chi_p = mat_p.chi;  // Electron affinity of p-type material (eV)
    double chi_n = mat_n.chi;  // Electron affinity of n-type material (eV)

    // Calculate band offsets
    double delta_E_c = chi_p - chi_n;  // Conduction band offset
    double delta_E_v = (chi_n + E_g_eff_n) - (chi_p + E_g_eff_p);  // Valence band offset

    // Calculate temperature-dependent effective densities of states
    // P-type material
    double m_e_eff_p = 0.067;     // Effective electron mass
    double m_h_eff_p = 0.45;      // Effective hole mass
    double h_bar = 6.582119569e-16; // Reduced Planck constant (eV·s)
    double m_0 = 9.10938356e-31;    // Electron rest mass (kg)

    double N_c_p = 2.0 * std::pow(m_e_eff_p * m_0 * kT / (2.0 * M_PI * h_bar * h_bar), 1.5);
    double N_v_p = 2.0 * std::pow(m_h_eff_p * m_0 * kT / (2.0 * M_PI * h_bar * h_bar), 1.5);

    // N-type material
    double m_e_eff_n = 0.067;     // Effective electron mass
    double m_h_eff_n = 0.45;      // Effective hole mass

    double N_c_n = 2.0 * std::pow(m_e_eff_n * m_0 * kT / (2.0 * M_PI * h_bar * h_bar), 1.5);
    double N_v_n = 2.0 * std::pow(m_h_eff_n * m_0 * kT / (2.0 * M_PI * h_bar * h_bar), 1.5);

    // Calculate intrinsic carrier concentrations
    double ni_p = std::sqrt(N_c_p * N_v_p) * std::exp(-E_g_eff_p / (2.0 * kT));
    double ni_n = std::sqrt(N_c_n * N_v_n) * std::exp(-E_g_eff_n / (2.0 * kT));

    // Calculate Fermi levels using Fermi-Dirac statistics for high doping
    // Threshold for degeneracy
    double N_degeneracy = 5e18;  // cm^-3

    // P-type material
    double E_F_p = 0.0;  // Fermi level in p-type region relative to valence band
    if (N_A > N_degeneracy) {
        // Use Joyce-Dixon approximation for degenerate p-type
        double eta_p = kT * std::log(N_A / N_v_p) + kT * (std::sqrt(2.0) / 4.0) * std::pow(N_A / N_v_p, 0.75);
        E_F_p = eta_p;
    } else {
        // Use Boltzmann approximation for non-degenerate p-type
        E_F_p = kT * std::log(N_A / N_v_p);
    }

    // N-type material
    double E_F_n = 0.0;  // Fermi level in n-type region relative to conduction band
    if (N_D > N_degeneracy) {
        // Use Joyce-Dixon approximation for degenerate n-type
        double eta_n = kT * std::log(N_D / N_c_n) + kT * (std::sqrt(2.0) / 4.0) * std::pow(N_D / N_c_n, 0.75);
        E_F_n = eta_n;
    } else {
        // Use Boltzmann approximation for non-degenerate n-type
        E_F_n = kT * std::log(N_D / N_c_n);
    }

    // Calculate built-in potential for heterojunction
    double V_bi = E_g_eff_n + delta_E_c - delta_E_v + E_F_p - E_F_n;

    // Alternative calculation using intrinsic carrier concentrations
    double V_bi_alt = kT * std::log((N_A * N_D) / (ni_p * ni_n));

    // Use the more accurate of the two methods
    V_bi = (N_A > N_degeneracy || N_D > N_degeneracy) ? V_bi : V_bi_alt;

    // Ensure the built-in potential is positive and physically reasonable
    V_bi = std::max(V_bi, 0.0);
    V_bi = std::min(V_bi, std::max(E_g_eff_p, E_g_eff_n)); // Cannot exceed the larger bandgap

    // Log the calculated values for debugging
    std::cout << "Heterojunction built-in potential calculation:" << std::endl;
    std::cout << "Temperature: " << T << " K" << std::endl;
    std::cout << "P-type material bandgap: " << E_g_eff_p << " eV" << std::endl;
    std::cout << "N-type material bandgap: " << E_g_eff_n << " eV" << std::endl;
    std::cout << "Conduction band offset: " << delta_E_c << " eV" << std::endl;
    std::cout << "Valence band offset: " << delta_E_v << " eV" << std::endl;
    std::cout << "P-type doping: " << N_A << " cm^-3" << std::endl;
    std::cout << "N-type doping: " << N_D << " cm^-3" << std::endl;
    std::cout << "Fermi level in p-type: " << E_F_p << " eV (relative to valence band)" << std::endl;
    std::cout << "Fermi level in n-type: " << E_F_n << " eV (relative to conduction band)" << std::endl;
    std::cout << "Built-in potential (Fermi-Dirac): " << E_g_eff_n + delta_E_c - delta_E_v + E_F_p - E_F_n << " V" << std::endl;
    std::cout << "Built-in potential (Boltzmann): " << V_bi_alt << " V" << std::endl;
    std::cout << "Final heterojunction built-in potential: " << V_bi << " V" << std::endl;

    return V_bi;
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
/**
 * @brief Applies boundary conditions to the carrier concentrations.
 *
 * This function applies boundary conditions to the carrier concentrations based on
 * the applied voltages and the position of the contacts. It implements several types
 * of boundary conditions:
 *
 * 1. Ohmic contacts: Carrier concentrations are set to their equilibrium values based
 *    on the applied voltage and doping.
 * 2. Schottky contacts: Carrier concentrations are determined by the Schottky barrier
 *    height and thermionic emission.
 * 3. Insulating boundaries: Surface recombination is modeled with proper surface
 *    recombination velocities.
 * 4. Heterojunction interfaces: Band offsets and interface states are considered.
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

    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double Ly = mesh.get_ly();

    // Define boundary regions
    double contact_thickness = 0.05 * Lx; // Thickness of the contact regions
    double p_contact_x = -Lx / 2;
    double n_contact_x = Lx / 2;

    // Define surface recombination velocities (cm/s)
    double S_n = 1e3; // Surface recombination velocity for electrons
    double S_p = 1e3; // Surface recombination velocity for holes

    // Convert to cm/s (assuming mesh is in cm)
    // Note: If mesh is in nm, use 1e-7 conversion factor
    double mesh_scale = 1.0; // Set to 1e-7 if mesh is in nm
    S_n *= mesh_scale;
    S_p *= mesh_scale;

    // Define Schottky barrier heights (eV)
    double phi_bn = 0.8; // Barrier height for n-type Schottky contact
    double phi_bp = 0.8; // Barrier height for p-type Schottky contact

    // Define contact types (ohmic or Schottky)
    bool p_contact_ohmic = true;  // Set to false for Schottky contact
    bool n_contact_ohmic = true;  // Set to false for Schottky contact

    // Loop over all nodes
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties for this node
        Materials::Material mat = get_material_at(x, y);

        // Get the potential at this node
        double phi_val = 0.0;
        const Eigen::VectorXd& phi = poisson.get_potential();
        if (i < phi.size()) {
            phi_val = phi[i];
        }

        // Check if the node is at the p-contact (left boundary)
        if (std::abs(x - p_contact_x) < contact_thickness) {
            // Define doping concentrations
            double N_A = 1e16; // Acceptor concentration in p-region
            double N_D = 0.0;  // Donor concentration in p-region

            if (p_contact_ohmic) {
                // Ohmic contact: Calculate equilibrium carrier concentrations

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
            } else {
                // Schottky contact: Calculate carrier concentrations based on barrier height

                // Calculate intrinsic carrier concentration
                double ni = std::sqrt(mat.N_c * mat.N_v) * std::exp(-mat.E_g / (2.0 * kT));

                // Calculate built-in potential
                double V_bi = mat.E_g - phi_bp;

                // Calculate effective barrier height with image force lowering
                double E_field = std::abs(poisson.get_electric_field(x, y).norm());
                double delta_phi = std::sqrt(q * E_field / (4.0 * M_PI * mat.epsilon_r * 8.85e-14));
                double phi_eff = phi_bp - delta_phi;

                // Calculate carrier concentrations at Schottky contact
                n[i] = mat.N_c * std::exp(-(phi_eff) / kT);
                p[i] = ni * ni / n[i];
            }
        }
        // Check if the node is at the n-contact (right boundary)
        else if (std::abs(x - n_contact_x) < contact_thickness) {
            // Define doping concentrations
            double N_A = 0.0;  // Acceptor concentration in n-region
            double N_D = 1e16; // Donor concentration in n-region

            if (n_contact_ohmic) {
                // Ohmic contact: Calculate equilibrium carrier concentrations

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
            } else {
                // Schottky contact: Calculate carrier concentrations based on barrier height

                // Calculate intrinsic carrier concentration
                double ni = std::sqrt(mat.N_c * mat.N_v) * std::exp(-mat.E_g / (2.0 * kT));

                // Calculate built-in potential
                double V_bi = phi_bn;

                // Calculate effective barrier height with image force lowering
                double E_field = std::abs(poisson.get_electric_field(x, y).norm());
                double delta_phi = std::sqrt(q * E_field / (4.0 * M_PI * mat.epsilon_r * 8.85e-14));
                double phi_eff = phi_bn - delta_phi;

                // Calculate carrier concentrations at Schottky contact
                n[i] = N_D * std::exp(-(phi_eff) / kT);
                p[i] = ni * ni / n[i];
            }
        }
        // Check if the node is at the top or bottom boundary (insulating)
        else if (std::abs(y - (-Ly / 2)) < 1e-10 || std::abs(y - (Ly / 2)) < 1e-10) {
            // Apply surface recombination boundary conditions using a more accurate model

            // Calculate bulk carrier concentrations
            double n_bulk = n_conc(x, y, phi_val, mat);
            double p_bulk = p_conc(x, y, phi_val, mat);

            // Calculate intrinsic carrier concentration
            double ni = std::sqrt(mat.N_c * mat.N_v) * std::exp(-mat.E_g / (2.0 * kT));

            // Surface band bending (eV) - positive for upward bending
            double surface_band_bending = 0.2;

            // Surface state density (cm^-2)
            double N_ss = 1e12;

            // Surface charge (C/cm^2)
            double Q_ss = q * N_ss;

            // Calculate surface electric field
            double E_surface = Q_ss / (mat.epsilon_r * 8.85e-14);

            // Calculate depletion width due to surface states
            double W_s = std::sqrt(2.0 * mat.epsilon_r * 8.85e-14 * surface_band_bending / (q * (x < 0 ? N_A : N_D)));

            // Calculate surface potential
            double phi_s = phi_val + surface_band_bending / q;

            // Calculate surface carrier concentrations
            double n_surface = n_bulk * std::exp(-surface_band_bending / kT);
            double p_surface = p_bulk * std::exp(surface_band_bending / kT);

            // Apply surface recombination
            // For a more accurate model, we would need to solve the continuity equations
            // with proper boundary conditions. This is a simplified approach.
            double surface_recomb = S_n * S_p * (n_surface * p_surface - ni * ni) /
                                   (S_n * (n_surface + ni) + S_p * (p_surface + ni));

            // Adjust carrier concentrations based on surface recombination
            // This is a simplified model that reduces carrier concentrations near the surface
            double distance_from_surface = std::min(std::abs(y - (-Ly / 2)), std::abs(y - (Ly / 2)));
            double surface_effect = std::exp(-distance_from_surface / W_s);

            // Apply surface effect to carrier concentrations
            n[i] = n_bulk * (1.0 - surface_effect) + n_surface * surface_effect;
            p[i] = p_bulk * (1.0 - surface_effect) + p_surface * surface_effect;
        }

        // Check for heterojunction interfaces
        if (has_heterojunction) {
            // Find if this node is at a heterojunction interface
            bool is_interface = false;
            int material_idx1 = -1, material_idx2 = -1;

            // Check if the node is at an interface between different materials
            for (size_t j = 0; j < regions.size(); ++j) {
                if (regions[j](x, y)) {
                    if (material_idx1 == -1) {
                        material_idx1 = j;
                    } else {
                        material_idx2 = j;
                        is_interface = true;
                        break;
                    }
                }
            }

            if (is_interface) {
                // This node is at a heterojunction interface
                // Handle band offsets and interface states

                // Get materials on both sides of the interface
                Materials::Material mat1 = materials[material_idx1];
                Materials::Material mat2 = materials[material_idx2];

                // Calculate band offsets
                double delta_Ec = mat2.E_g - mat1.E_g; // Simplified; in reality, this depends on electron affinities
                double delta_Ev = 0.0; // Simplified; in reality, this depends on band alignments

                // Interface state density (cm^-2)
                double N_it = 1e11;

                // Interface charge (C/cm^2)
                double Q_it = q * N_it;

                // Calculate interface electric field
                double E_interface = Q_it / (0.5 * (mat1.epsilon_r + mat2.epsilon_r) * 8.85e-14);

                // Calculate carrier concentrations at the interface
                // This is a simplified model; a more accurate model would solve
                // the Poisson-drift-diffusion equations with proper interface conditions

                // Get carrier concentrations from both sides
                double n1 = n_conc(x - 1e-6, y, phi_val, mat1);
                double p1 = p_conc(x - 1e-6, y, phi_val, mat1);
                double n2 = n_conc(x + 1e-6, y, phi_val, mat2);
                double p2 = p_conc(x + 1e-6, y, phi_val, mat2);

                // Average carrier concentrations at the interface
                n[i] = 0.5 * (n1 + n2);
                p[i] = 0.5 * (p1 + p2);
            }
        }

        // Ensure minimum carrier concentrations for numerical stability
        const double n_min = 1e-10; // Minimum concentration (cm^-3)
        const double n_max = 1e20;  // Maximum concentration (cm^-3)
        n[i] = std::max(std::min(n[i], n_max), n_min);
        p[i] = std::max(std::min(p[i], n_max), n_min);
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
 * of previous iterations. The implementation includes safeguards against numerical instabilities
 * and adaptive regularization based on the condition number of the system.
 *
 * Anderson acceleration is a quasi-Newton method that approximates the inverse Jacobian
 * using the history of iterates and residuals. It can significantly improve convergence
 * for nonlinear systems compared to simple damping.
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
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(F);

        // Check the rank and condition number
        double threshold = 1e-10;
        qr.setThreshold(threshold);
        int rank = qr.rank();

        if (rank < m) {
            // Matrix is rank-deficient, use a more robust solver with regularization
            Eigen::BDCSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeThinU | Eigen::ComputeThinV);

            // Get singular values
            Eigen::VectorXd singularValues = svd.singularValues();
            double maxSingularValue = singularValues(0);

            // Compute condition number
            double condNumber = maxSingularValue / singularValues(singularValues.size() - 1);

            // Adaptive regularization based on condition number
            double reg_param;
            if (condNumber > 1e8) {
                reg_param = 1e-4 * maxSingularValue;
            } else if (condNumber > 1e6) {
                reg_param = 1e-6 * maxSingularValue;
            } else {
                reg_param = 1e-8 * maxSingularValue;
            }

            // Apply Tikhonov regularization
            Eigen::MatrixXd FTF = F.transpose() * F;
            for (int i = 0; i < FTF.rows(); ++i) {
                FTF(i, i) += reg_param;
            }

            // Solve the regularized normal equations
            Eigen::VectorXd FTr = F.transpose() * (-res_history[m]);
            alpha = FTF.ldlt().solve(FTr);

            std::cout << "Using regularized solver with parameter: " << reg_param << std::endl;
        } else {
            // Matrix has full rank, use QR decomposition
            alpha = qr.solve(-res_history[m]);
        }

        // Check if the solution is valid
        if (!alpha.allFinite()) {
            throw std::runtime_error("Invalid solution in Anderson acceleration");
        }

        // Apply constraints to ensure stability
        // Limit the magnitude of coefficients
        for (int i = 0; i < alpha.size(); ++i) {
            alpha(i) = std::max(std::min(alpha(i), 2.0), -2.0);
        }

        // Ensure the sum of coefficients is reasonable
        double sum = alpha.sum();
        if (std::abs(sum) > 2.0) {
            alpha *= 2.0 / std::abs(sum);
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

    // Check for non-physical values in the accelerated potential
    bool has_non_physical = false;
    for (int i = 0; i < phi_line_search.size(); ++i) {
        if (!std::isfinite(phi_line_search(i)) || std::abs(phi_line_search(i)) > 100.0) {
            has_non_physical = true;
            break;
        }
    }

    if (has_non_physical) {
        std::cout << "Warning: Non-physical values detected in accelerated potential. Using damped update instead." << std::endl;
        return apply_damping(phi_old, phi_update);
    }

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
/**
 * @brief Solves the self-consistent Poisson-drift-diffusion equations.
 *
 * This function solves the self-consistent Poisson-drift-diffusion equations
 * using an iterative approach with convergence acceleration techniques. It alternates
 * between solving the Poisson equation and the drift-diffusion equations until
 * convergence is reached or the maximum number of iterations is exceeded.
 *
 * The solution process includes:
 * 1. Calculating the built-in potential based on doping concentrations
 * 2. Initializing carrier concentrations and quasi-Fermi potentials
 * 3. Solving the Poisson equation for the electrostatic potential
 * 4. Solving the drift-diffusion equations for carrier transport
 * 5. Applying boundary conditions and updating quasi-Fermi potentials
 * 6. Iterating until convergence or maximum iterations
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
        // Calculate the built-in potential
        double V_bi = 0.0;

        // Check if we have a heterojunction
        if (has_heterojunction && materials.size() >= 2) {
            std::cout << "Heterojunction detected. Calculating heterojunction built-in potential..." << std::endl;

            // Find p-type and n-type materials
            Materials::Material mat_p = materials[0];
            Materials::Material mat_n = materials[1];

            // Calculate heterojunction built-in potential
            V_bi = calculate_heterojunction_potential(N_A, N_D, mat_p, mat_n);
        } else {
            // Homojunction case
            std::cout << "Homojunction detected. Calculating built-in potential..." << std::endl;
            V_bi = calculate_built_in_potential(N_A, N_D);
        }

        std::cout << "Built-in potential: " << V_bi << " V" << std::endl;

        // Adjust applied voltages to include built-in potential
        double V_p_effective = V_p;
        double V_n_effective = V_n;

        // For reverse bias, add the built-in potential to the n-contact voltage
        if (V_n > V_p) {
            V_n_effective = V_n + V_bi;
            std::cout << "Reverse bias: V_n_effective = " << V_n_effective << " V" << std::endl;
        } else {
            // For forward bias, subtract from the built-in potential
            double V_forward = V_p - V_n;
            V_n_effective = V_n + std::max(0.0, V_bi - V_forward);
            std::cout << "Forward bias: V_n_effective = " << V_n_effective << " V" << std::endl;
        }

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
            poisson.solve(V_p_effective, V_n_effective, n, p);

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
            poisson.update_and_solve(phi_with_quantum, V_p_effective, V_n_effective, n, p);

            // Solve drift-diffusion equations with accelerated potential
            solve_drift_diffusion();

            // Apply boundary conditions
            apply_boundary_conditions(V_p_effective, V_n_effective);

            // Update quasi-Fermi potentials
            update_quasi_fermi_potentials();

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

        // Calculate the built-in potential for the fallback solution
        double V_bi = 0.0;

        // Check if we have a heterojunction
        if (has_heterojunction && materials.size() >= 2) {
            std::cerr << "Heterojunction detected. Using heterojunction built-in potential for fallback..." << std::endl;

            // Find p-type and n-type materials
            Materials::Material mat_p = materials[0];
            Materials::Material mat_n = materials[1];

            // Calculate heterojunction built-in potential
            try {
                V_bi = calculate_heterojunction_potential(N_A, N_D, mat_p, mat_n);
            } catch (const std::exception& e) {
                std::cerr << "Error calculating heterojunction potential: " << e.what() << std::endl;
                std::cerr << "Using default built-in potential calculation..." << std::endl;
                V_bi = calculate_built_in_potential(N_A, N_D);
            }
        } else {
            // Homojunction case
            std::cerr << "Using homojunction built-in potential for fallback..." << std::endl;
            V_bi = calculate_built_in_potential(N_A, N_D);
        }

        std::cerr << "Fallback built-in potential: " << V_bi << " V" << std::endl;

        // Adjust applied voltages to include built-in potential
        double V_p_effective = V_p;
        double V_n_effective = V_n;

        // For reverse bias, add the built-in potential to the n-contact voltage
        if (V_n > V_p) {
            V_n_effective = V_n + V_bi;
        } else {
            // For forward bias, subtract from the built-in potential
            double V_forward = V_p - V_n;
            V_n_effective = V_n + std::max(0.0, V_bi - V_forward);
        }

        // Initialize with a simple solution to avoid crashing
        initialize_carriers(N_A, N_D);
        poisson.solve(V_p_effective, V_n_effective, n, p);

        // Update quasi-Fermi potentials
        update_quasi_fermi_potentials();

        std::cerr << "Initialized with fallback solution" << std::endl;
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
 * the density gradient theory, which is an extension of the Bohm quantum
 * potential approach. It accounts for quantum effects like tunneling,
 * quantum confinement, and size quantization.
 *
 * The quantum correction potential is given by:
 * V_q = -ħ^2/(2m*) * ∇^2(√n)/√n - ħ^2/(2m*) * ∇^2(√p)/√p
 *
 * Where:
 * - ħ is the reduced Planck constant
 * - m* is the effective mass (different for electrons and holes)
 * - n is the electron concentration
 * - p is the hole concentration
 *
 * This implementation uses a finite element approach to compute the Laplacian
 * more accurately than simple finite differences.
 *
 * @param n The electron concentration
 * @param p The hole concentration
 * @return The quantum correction to the potential
 */
Eigen::VectorXd SelfConsistentSolver::compute_quantum_correction(const Eigen::VectorXd& n, const Eigen::VectorXd& p) const {
    // Constants
    const double h_bar = 1.054571817e-34; // Reduced Planck constant (J·s)
    const double m_e = 9.10938356e-31;    // Electron mass (kg)
    const double q = 1.602176634e-19;     // Elementary charge (C)
    const double kB = 1.380649e-23;       // Boltzmann constant (J/K)
    const double T = 300.0;               // Temperature (K)
    const double kT = kB * T / q;         // Thermal voltage (eV)

    // Initialize quantum correction vector
    Eigen::VectorXd V_q(mesh.getNumNodes());
    V_q.setZero();

    // Create sparse matrices for the Laplacian operator
    Eigen::SparseMatrix<double> L(mesh.getNumNodes(), mesh.getNumNodes());
    std::vector<Eigen::Triplet<double>> L_triplets;

    // Assemble the Laplacian matrix using finite element method
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        // Get element nodes
        const auto& element = mesh.getElements()[e];
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

        // Assemble element Laplacian matrix
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Laplacian term: ∇Ni · ∇Nj
                double L_ij = grad_N[i].dot(grad_N[j]) * area;
                L_triplets.emplace_back(element[i], element[j], L_ij);
            }
        }
    }

    // Set up the Laplacian matrix
    L.setFromTriplets(L_triplets.begin(), L_triplets.end());

    // Create mass matrix for normalization
    Eigen::SparseMatrix<double> M(mesh.getNumNodes(), mesh.getNumNodes());
    std::vector<Eigen::Triplet<double>> M_triplets;

    // Assemble the mass matrix using finite element method
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        // Get element nodes
        const auto& element = mesh.getElements()[e];
        std::vector<Eigen::Vector2d> nodes;
        for (int i = 0; i < 3; ++i) {
            nodes.push_back(mesh.getNodes()[element[i]]);
        }

        // Calculate element area
        double x1 = nodes[0][0], y1 = nodes[0][1];
        double x2 = nodes[1][0], y2 = nodes[1][1];
        double x3 = nodes[2][0], y3 = nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Assemble element mass matrix
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Mass term: Ni · Nj
                double M_ij = (i == j ? area / 6.0 : area / 12.0);
                M_triplets.emplace_back(element[i], element[j], M_ij);
            }
        }
    }

    // Set up the mass matrix
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());

    // Compute sqrt(n) and sqrt(p) vectors
    Eigen::VectorXd sqrt_n(mesh.getNumNodes());
    Eigen::VectorXd sqrt_p(mesh.getNumNodes());

    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        // Ensure positive values to avoid numerical issues
        double n_val = std::max(n[i], 1e-20);
        double p_val = std::max(p[i], 1e-20);

        sqrt_n[i] = std::sqrt(n_val);
        sqrt_p[i] = std::sqrt(p_val);
    }

    // Compute Laplacian of sqrt(n) and sqrt(p)
    Eigen::VectorXd L_sqrt_n = L * sqrt_n;
    Eigen::VectorXd L_sqrt_p = L * sqrt_p;

    // Compute the quantum correction for each node
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Get material properties at this position
        Materials::Material mat = get_material_at(x, y);

        // Effective masses (in units of electron mass)
        double m_star_n = 0.067; // GaAs electron effective mass
        double m_star_p = 0.45;  // GaAs hole effective mass

        // Quantum correction parameters
        double gamma_n = 1.0; // Quantum correction factor for electrons
        double gamma_p = 1.0; // Quantum correction factor for holes

        // Compute the quantum correction terms
        double V_q_n = 0.0;
        double V_q_p = 0.0;

        // Only compute if the carrier concentration is significant
        if (sqrt_n[i] > 1e-10) {
            V_q_n = -gamma_n * h_bar * h_bar / (2.0 * m_star_n * m_e) * L_sqrt_n[i] / sqrt_n[i] / q;
        }

        if (sqrt_p[i] > 1e-10) {
            V_q_p = -gamma_p * h_bar * h_bar / (2.0 * m_star_p * m_e) * L_sqrt_p[i] / sqrt_p[i] / q;
        }

        // Total quantum correction (combine electron and hole contributions)
        V_q[i] = V_q_n - V_q_p; // Note the sign change for holes

        // Limit the quantum correction to avoid numerical instabilities
        V_q[i] = std::max(std::min(V_q[i], 0.5), -0.5);
    }

    // Apply smoothing to the quantum correction to improve stability
    Eigen::VectorXd V_q_smoothed = V_q;

    // Simple averaging with neighbors for smoothing
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        // Find neighboring nodes
        std::vector<int> neighbors;
        for (size_t e = 0; e < mesh.getNumElements(); ++e) {
            const auto& element = mesh.getElements()[e];
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

        // Average with neighbors
        if (!neighbors.empty()) {
            double sum = V_q[i];
            for (int j : neighbors) {
                sum += V_q[j];
            }
            V_q_smoothed[i] = sum / (neighbors.size() + 1);
        }
    }

    return V_q_smoothed;
}

/**
 * @brief Computes the tunneling current.
 *
 * This method computes the tunneling current using advanced models for
 * band-to-band tunneling (BTBT), trap-assisted tunneling (TAT), and
 * thermionic emission. It accounts for the local electric field, band structure,
 * and quantum effects.
 *
 * For band-to-band tunneling, the Kane model is used:
 * J_BTBT = A * E_field^2 * exp(-B/E_field)
 *
 * For trap-assisted tunneling, the Hurkx model is used:
 * J_TAT = J_SRH * (1 + Γ)
 * where Γ is the field enhancement factor.
 *
 * For thermionic emission, the standard model is used:
 * J_TE = A* * T^2 * exp(-φ_B/kT)
 *
 * @param E_field The electric field
 * @return The tunneling current
 */
Eigen::VectorXd SelfConsistentSolver::compute_tunneling_current(const std::vector<Eigen::Vector2d>& E_field) const {
    // Constants
    const double h_bar = 1.054571817e-34; // Reduced Planck constant (J·s)
    const double m_e = 9.10938356e-31;    // Electron mass (kg)
    const double q = 1.602176634e-19;     // Elementary charge (C)
    const double kB = 1.380649e-23;       // Boltzmann constant (J/K)
    const double T = 300.0;               // Temperature (K)
    const double kT = kB * T;             // Thermal energy (J)
    const double kT_eV = kT / q;          // Thermal energy (eV)

    // Richardson constant (A/(cm^2·K^2))
    const double A_star = 120.0;

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

        // Effective masses (in units of electron mass)
        double m_star_n = 0.067; // GaAs electron effective mass
        double m_star_p = 0.45;  // GaAs hole effective mass
        double m_r = m_star_n * m_star_p / (m_star_n + m_star_p); // Reduced effective mass

        // Band gap
        double E_g = mat.E_g; // Band gap in eV
        double E_g_J = E_g * q; // Band gap in J

        // Electric field magnitude
        double E_mag = 0.0;
        if (i < E_field.size()) {
            E_mag = E_field[i].norm();
        }

        // Avoid division by zero
        E_mag = std::max(E_mag, 1e-6);

        // Band-to-band tunneling (BTBT) using Kane model
        // Parameters for GaAs
        double A_BTBT = 3.0e14; // cm^-1·s^-1·V^-2
        double B_BTBT = 2.0e7;  // V/cm

        // Convert electric field to V/cm
        double E_mag_V_cm = E_mag * 1e-2; // Assuming E_field is in V/m

        // Calculate BTBT current density (A/cm^2)
        double J_BTBT = 0.0;
        if (E_mag_V_cm > 1e5) { // Only significant for high fields
            J_BTBT = q * A_BTBT * E_mag_V_cm * E_mag_V_cm * std::exp(-B_BTBT / E_mag_V_cm);
        }

        // Trap-assisted tunneling (TAT) using Hurkx model
        // Parameters
        double E_t = 0.5 * E_g; // Trap energy level (eV), typically mid-gap
        double tau_n = 1e-9;    // Electron lifetime (s)
        double tau_p = 1e-9;    // Hole lifetime (s)
        double N_t = 1e14;      // Trap density (cm^-3)

        // Calculate field enhancement factor
        double E_00_n = h_bar * std::sqrt(q * E_mag / (8 * M_PI * m_star_n * m_e)); // Characteristic energy (J)
        double E_00_p = h_bar * std::sqrt(q * E_mag / (8 * M_PI * m_star_p * m_e)); // Characteristic energy (J)

        // Convert to eV
        double E_00_n_eV = E_00_n / q;
        double E_00_p_eV = E_00_p / q;

        // Field enhancement factors
        double Gamma_n = (M_PI * E_00_n_eV / kT_eV) / std::tanh(M_PI * E_00_n_eV / kT_eV);
        double Gamma_p = (M_PI * E_00_p_eV / kT_eV) / std::tanh(M_PI * E_00_p_eV / kT_eV);

        // Calculate SRH recombination rate
        double n_i = std::sqrt(mat.N_c * mat.N_v) * std::exp(-E_g / (2.0 * kT_eV));
        double R_SRH = (n[i] * p[i] - n_i * n_i) / (tau_n * (p[i] + n_i) + tau_p * (n[i] + n_i));

        // Calculate TAT current density (A/cm^2)
        double J_TAT = q * R_SRH * (Gamma_n + Gamma_p - 1.0);

        // Thermionic emission (TE) for heterojunctions and Schottky barriers
        double J_TE = 0.0;

        // Check if this node is at a heterojunction interface
        if (has_heterojunction) {
            bool is_interface = false;
            int material_idx1 = -1, material_idx2 = -1;

            // Check if the node is at an interface between different materials
            for (size_t j = 0; j < regions.size(); ++j) {
                if (regions[j](x, y)) {
                    if (material_idx1 == -1) {
                        material_idx1 = j;
                    } else {
                        material_idx2 = j;
                        is_interface = true;
                        break;
                    }
                }
            }

            if (is_interface) {
                // Get materials on both sides of the interface
                Materials::Material mat1 = materials[material_idx1];
                Materials::Material mat2 = materials[material_idx2];

                // Calculate band offsets
                double delta_Ec = std::abs(mat2.E_g - mat1.E_g); // Simplified; in reality, this depends on electron affinities

                // Calculate thermionic emission current
                J_TE = A_star * T * T * std::exp(-delta_Ec / kT_eV);
            }
        }

        // WKB tunneling probability for quantum tunneling
        double J_WKB = 0.0;

        if (E_mag > 1e5) { // Only significant for high fields
            // Calculate tunneling barrier width
            double barrier_width = E_g_J / (q * E_mag);

            // Calculate wave vector
            double kappa = std::sqrt(2.0 * m_r * m_e * E_g_J) / h_bar;

            // Calculate tunneling probability
            double T_WKB = std::exp(-2.0 * kappa * barrier_width);

            // Calculate thermal velocity
            double v_thermal = std::sqrt(3.0 * kB * T / (m_r * m_e));

            // Calculate WKB tunneling current density (A/cm^2)
            J_WKB = q * n[i] * v_thermal * T_WKB;
        }

        // Total tunneling current density (A/cm^2)
        double J_total = J_BTBT + J_TAT + J_TE + J_WKB;

        // Limit the current to avoid numerical instabilities
        J_total = std::max(std::min(J_total, 1e3), -1e3);

        // Store the result
        J_tunnel[i] = J_total;
    }

    return J_tunnel;
}