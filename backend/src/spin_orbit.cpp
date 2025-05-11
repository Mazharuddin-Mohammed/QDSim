/**
 * @file spin_orbit.cpp
 * @brief Implementation of the SpinOrbitCoupling class for including spin-orbit effects in quantum simulations.
 *
 * This file contains the implementation of the SpinOrbitCoupling class, which provides
 * methods for including spin-orbit coupling effects in quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "spin_orbit.h"
#include "physical_constants.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

// Constructor
SpinOrbitCoupling::SpinOrbitCoupling(const Mesh& mesh, const Materials::Material& material, SpinOrbitType type,
                                   double rashba_parameter, double dresselhaus_parameter)
    : mesh_(mesh), material_(material), type_(type),
      rashba_parameter_(rashba_parameter), dresselhaus_parameter_(dresselhaus_parameter) {

    // Validate parameters
    if (rashba_parameter_ < 0.0) {
        std::cerr << "Warning: Negative Rashba parameter. Setting to 0." << std::endl;
        rashba_parameter_ = 0.0;
    }

    if (dresselhaus_parameter_ < 0.0) {
        std::cerr << "Warning: Negative Dresselhaus parameter. Setting to 0." << std::endl;
        dresselhaus_parameter_ = 0.0;
    }
}

// Assemble spin-orbit coupling Hamiltonian
void SpinOrbitCoupling::assemble_spin_orbit_hamiltonian(Eigen::SparseMatrix<std::complex<double>>& H,
                                                      std::function<double(double, double)> electric_field_x,
                                                      std::function<double(double, double)> electric_field_y,
                                                      std::function<double(double, double)> electric_field_z) {
    // Check if spin-orbit coupling is enabled
    if (type_ == SpinOrbitType::NONE) {
        return;
    }

    // Check if the Hamiltonian matrix has the correct size
    int n = mesh_.getNumNodes();
    if (H.rows() != 2 * n || H.cols() != 2 * n) {
        throw std::runtime_error("Hamiltonian matrix has incorrect size for spin-orbit coupling");
    }

    // Assemble the appropriate spin-orbit coupling Hamiltonian
    if (type_ == SpinOrbitType::RASHBA || type_ == SpinOrbitType::BOTH) {
        assemble_rashba_hamiltonian(H, electric_field_z);
    }

    if (type_ == SpinOrbitType::DRESSELHAUS || type_ == SpinOrbitType::BOTH) {
        assemble_dresselhaus_hamiltonian(H);
    }
}

// Set spin-orbit coupling type
void SpinOrbitCoupling::set_spin_orbit_type(SpinOrbitType type) {
    type_ = type;
}

// Set Rashba parameter
void SpinOrbitCoupling::set_rashba_parameter(double parameter) {
    if (parameter < 0.0) {
        std::cerr << "Warning: Negative Rashba parameter. Setting to 0." << std::endl;
        rashba_parameter_ = 0.0;
    } else {
        rashba_parameter_ = parameter;
    }
}

// Set Dresselhaus parameter
void SpinOrbitCoupling::set_dresselhaus_parameter(double parameter) {
    if (parameter < 0.0) {
        std::cerr << "Warning: Negative Dresselhaus parameter. Setting to 0." << std::endl;
        dresselhaus_parameter_ = 0.0;
    } else {
        dresselhaus_parameter_ = parameter;
    }
}

// Get spin-orbit coupling type
SpinOrbitType SpinOrbitCoupling::get_spin_orbit_type() const {
    return type_;
}

// Get Rashba parameter
double SpinOrbitCoupling::get_rashba_parameter() const {
    return rashba_parameter_;
}

// Get Dresselhaus parameter
double SpinOrbitCoupling::get_dresselhaus_parameter() const {
    return dresselhaus_parameter_;
}

// Calculate Rashba parameter from electric field
double SpinOrbitCoupling::calculate_rashba_parameter(const Materials::Material& material, double electric_field_z) {
    // Rashba coefficient (in eV·nm²/(V/m))
    double alpha_0 = 5.0e-20; // Typical value for III-V semiconductors

    // Calculate Rashba parameter
    double alpha = alpha_0 * electric_field_z;

    return alpha;
}

// Calculate Dresselhaus parameter from material properties
double SpinOrbitCoupling::calculate_dresselhaus_parameter(const Materials::Material& material) {
    // Dresselhaus coefficient (in eV·nm³)
    double gamma_0 = 27.0; // Typical value for GaAs

    // Calculate Dresselhaus parameter
    double gamma = gamma_0 * std::pow(M_PI / (material.lattice_constant * 1.0e-9), 2);

    return gamma;
}

// Assemble Rashba spin-orbit coupling Hamiltonian
void SpinOrbitCoupling::assemble_rashba_hamiltonian(Eigen::SparseMatrix<std::complex<double>>& H,
                                                 std::function<double(double, double)> electric_field_z) {
    // Get mesh data
    int n = mesh_.getNumNodes();
    const auto& nodes = mesh_.getNodes();
    const auto& elements = mesh_.getElements();

    // Compute Pauli matrices
    Eigen::Matrix2cd sigma_x, sigma_y, sigma_z;
    compute_pauli_matrices(sigma_x, sigma_y, sigma_z);

    // Convert Rashba parameter from eV·nm to J·m
    double alpha = rashba_parameter_ * PhysicalConstants::EV_TO_J * 1.0e-9;

    // Create triplet list for sparse matrix assembly
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;

    // Loop over elements
    for (int e = 0; e < elements.size(); ++e) {
        // Get element nodes
        int n1 = elements[e][0];
        int n2 = elements[e][1];
        int n3 = elements[e][2];

        // Get node coordinates
        double x1 = nodes[n1][0];
        double y1 = nodes[n1][1];
        double x2 = nodes[n2][0];
        double y2 = nodes[n2][1];
        double x3 = nodes[n3][0];
        double y3 = nodes[n3][1];

        // Calculate element area
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

        // Get electric field at centroid
        double Ez = 1.0e6; // Default value in V/m
        if (electric_field_z) {
            Ez = electric_field_z(xc, yc);
        }

        // Calculate local Rashba parameter
        double alpha_local = alpha;
        if (electric_field_z) {
            // If electric field function is provided, recalculate alpha
            alpha_local = calculate_rashba_parameter(material_, Ez) * PhysicalConstants::EV_TO_J * 1.0e-9;
        }

        // Calculate element matrices
        for (int i = 0; i < 3; ++i) {
            int ni = elements[e][i];
            double bi, ci;
            if (i == 0) { bi = b1; ci = c1; }
            else if (i == 1) { bi = b2; ci = c2; }
            else { bi = b3; ci = c3; }

            for (int j = 0; j < 3; ++j) {
                int nj = elements[e][j];
                double bj, cj;
                if (j == 0) { bj = b1; cj = c1; }
                else if (j == 1) { bj = b2; cj = c2; }
                else { bj = b3; cj = c3; }

                // Calculate Rashba Hamiltonian element
                // H_R = α (σ_x k_y - σ_y k_x)
                Eigen::Matrix2cd H_R = alpha_local * (sigma_x * bj - sigma_y * cj) * area / 3.0;

                // Add to global Hamiltonian
                triplets.push_back(Eigen::Triplet<std::complex<double>>(ni, nj, H_R(0, 0)));
                triplets.push_back(Eigen::Triplet<std::complex<double>>(ni, nj + n, H_R(0, 1)));
                triplets.push_back(Eigen::Triplet<std::complex<double>>(ni + n, nj, H_R(1, 0)));
                triplets.push_back(Eigen::Triplet<std::complex<double>>(ni + n, nj + n, H_R(1, 1)));
            }
        }
    }

    // Add triplets to Hamiltonian
    H.setFromTriplets(triplets.begin(), triplets.end());
}

// Assemble Dresselhaus spin-orbit coupling Hamiltonian
void SpinOrbitCoupling::assemble_dresselhaus_hamiltonian(Eigen::SparseMatrix<std::complex<double>>& H) {
    // Get mesh data
    int n = mesh_.getNumNodes();
    const auto& nodes = mesh_.getNodes();
    const auto& elements = mesh_.getElements();

    // Compute Pauli matrices
    Eigen::Matrix2cd sigma_x, sigma_y, sigma_z;
    compute_pauli_matrices(sigma_x, sigma_y, sigma_z);

    // Convert Dresselhaus parameter from eV·nm to J·m
    double gamma = dresselhaus_parameter_ * PhysicalConstants::EV_TO_J * 1.0e-9;

    // Create triplet list for sparse matrix assembly
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;

    // Loop over elements
    for (int e = 0; e < elements.size(); ++e) {
        // Get element nodes
        int n1 = elements[e][0];
        int n2 = elements[e][1];
        int n3 = elements[e][2];

        // Get node coordinates
        double x1 = nodes[n1][0];
        double y1 = nodes[n1][1];
        double x2 = nodes[n2][0];
        double y2 = nodes[n2][1];
        double x3 = nodes[n3][0];
        double y3 = nodes[n3][1];

        // Calculate element area
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Calculate shape function gradients
        double b1 = (y2 - y3) / (2.0 * area);
        double b2 = (y3 - y1) / (2.0 * area);
        double b3 = (y1 - y2) / (2.0 * area);
        double c1 = (x3 - x2) / (2.0 * area);
        double c2 = (x1 - x3) / (2.0 * area);
        double c3 = (x2 - x1) / (2.0 * area);

        // Calculate element matrices
        for (int i = 0; i < 3; ++i) {
            int ni = elements[e][i];
            double bi, ci;
            if (i == 0) { bi = b1; ci = c1; }
            else if (i == 1) { bi = b2; ci = c2; }
            else { bi = b3; ci = c3; }

            for (int j = 0; j < 3; ++j) {
                int nj = elements[e][j];
                double bj, cj;
                if (j == 0) { bj = b1; cj = c1; }
                else if (j == 1) { bj = b2; cj = c2; }
                else { bj = b3; cj = c3; }

                // Calculate Dresselhaus Hamiltonian element
                // H_D = γ (σ_x k_x - σ_y k_y)
                Eigen::Matrix2cd H_D = gamma * (sigma_x * cj - sigma_y * bj) * area / 3.0;

                // Add to global Hamiltonian
                triplets.push_back(Eigen::Triplet<std::complex<double>>(ni, nj, H_D(0, 0)));
                triplets.push_back(Eigen::Triplet<std::complex<double>>(ni, nj + n, H_D(0, 1)));
                triplets.push_back(Eigen::Triplet<std::complex<double>>(ni + n, nj, H_D(1, 0)));
                triplets.push_back(Eigen::Triplet<std::complex<double>>(ni + n, nj + n, H_D(1, 1)));
            }
        }
    }

    // Add triplets to Hamiltonian
    H.setFromTriplets(triplets.begin(), triplets.end());
}

// Compute Pauli matrices
void SpinOrbitCoupling::compute_pauli_matrices(Eigen::Matrix2cd& sigma_x, Eigen::Matrix2cd& sigma_y, Eigen::Matrix2cd& sigma_z) const {
    // Initialize Pauli matrices
    sigma_x = Eigen::Matrix2cd::Zero();
    sigma_y = Eigen::Matrix2cd::Zero();
    sigma_z = Eigen::Matrix2cd::Zero();

    // Define complex unit
    std::complex<double> i(0.0, 1.0);

    // σ_x = [[0, 1], [1, 0]]
    sigma_x(0, 1) = 1.0;
    sigma_x(1, 0) = 1.0;

    // σ_y = [[0, -i], [i, 0]]
    sigma_y(0, 1) = -i;
    sigma_y(1, 0) = i;

    // σ_z = [[1, 0], [0, -1]]
    sigma_z(0, 0) = 1.0;
    sigma_z(1, 1) = -1.0;
}
