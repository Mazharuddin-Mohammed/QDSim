/**
 * @file strain_effects.cpp
 * @brief Implementation of strain effects for semiconductor simulations.
 *
 * This file contains the implementation of strain effects used in
 * semiconductor simulations, including deformation potentials, band structure
 * modifications, and effective mass changes due to strain.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "strain_effects.h"
#include "physical_constants.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>

namespace StrainEffects {

// Compute strain tensor for pseudomorphic growth
Eigen::Matrix3d compute_strain_tensor_pseudomorphic(double a_substrate, double a_layer, bool is_cubic) {
    // Initialize strain tensor
    Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();
    
    // Compute in-plane strain
    double epsilon_par = (a_substrate - a_layer) / a_layer;
    
    // Set in-plane strain components
    strain(0, 0) = epsilon_par;
    strain(1, 1) = epsilon_par;
    
    if (is_cubic) {
        // For cubic materials, compute out-of-plane strain using Poisson ratio
        // Assuming Poisson ratio of 0.3 for typical semiconductors
        double poisson_ratio = 0.3;
        strain(2, 2) = -2.0 * poisson_ratio * epsilon_par / (1.0 - poisson_ratio);
    } else {
        // For wurtzite materials, use different elastic constants
        // Typical values for GaN
        double C13 = 106.0; // GPa
        double C33 = 398.0; // GPa
        strain(2, 2) = -2.0 * C13 / C33 * epsilon_par;
    }
    
    return strain;
}

// Compute strain tensor from elastic constants and stress
Eigen::Matrix3d compute_strain_tensor_from_stress(const Eigen::Matrix3d& stress, const Eigen::Matrix6d& compliance) {
    // Convert stress tensor to Voigt notation
    Eigen::VectorXd stress_voigt(6);
    stress_voigt << stress(0, 0), stress(1, 1), stress(2, 2), stress(1, 2), stress(0, 2), stress(0, 1);
    
    // Compute strain in Voigt notation
    Eigen::VectorXd strain_voigt = compliance * stress_voigt;
    
    // Convert strain from Voigt notation to tensor
    Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();
    strain(0, 0) = strain_voigt(0);
    strain(1, 1) = strain_voigt(1);
    strain(2, 2) = strain_voigt(2);
    strain(1, 2) = strain_voigt(3) / 2.0;
    strain(2, 1) = strain_voigt(3) / 2.0;
    strain(0, 2) = strain_voigt(4) / 2.0;
    strain(2, 0) = strain_voigt(4) / 2.0;
    strain(0, 1) = strain_voigt(5) / 2.0;
    strain(1, 0) = strain_voigt(5) / 2.0;
    
    return strain;
}

// Compute hydrostatic strain
double compute_hydrostatic_strain(const Eigen::Matrix3d& strain) {
    return strain(0, 0) + strain(1, 1) + strain(2, 2);
}

// Compute biaxial strain
double compute_biaxial_strain(const Eigen::Matrix3d& strain) {
    return 2.0 * strain(2, 2) - strain(0, 0) - strain(1, 1);
}

// Compute shear strain
double compute_shear_strain(const Eigen::Matrix3d& strain) {
    return std::sqrt(std::pow(strain(0, 1), 2) + std::pow(strain(0, 2), 2) + std::pow(strain(1, 2), 2));
}

// Compute conduction band shift due to strain for cubic materials
double compute_conduction_band_shift_cubic(const Eigen::Matrix3d& strain, double a_c) {
    // Hydrostatic strain
    double epsilon_h = compute_hydrostatic_strain(strain);
    
    // Conduction band shift
    return a_c * epsilon_h;
}

// Compute valence band shift due to strain for cubic materials
void compute_valence_band_shift_cubic(const Eigen::Matrix3d& strain, double a_v, double b, double d,
                                    double& delta_E_hh, double& delta_E_lh, double& delta_E_so) {
    // Hydrostatic strain
    double epsilon_h = compute_hydrostatic_strain(strain);
    
    // Biaxial strain
    double epsilon_b = compute_biaxial_strain(strain);
    
    // Shear strain
    double epsilon_s = compute_shear_strain(strain);
    
    // Hydrostatic component
    double delta_E_v_hydro = a_v * epsilon_h;
    
    // Biaxial component
    double delta_E_v_biaxial = -b * epsilon_b / 2.0;
    
    // Shear component
    double delta_E_v_shear = -d * epsilon_s;
    
    // Heavy hole shift
    delta_E_hh = delta_E_v_hydro - delta_E_v_biaxial;
    
    // Light hole shift
    delta_E_lh = delta_E_v_hydro + delta_E_v_biaxial;
    
    // Split-off band shift
    delta_E_so = delta_E_v_hydro + delta_E_v_shear;
}

// Compute conduction band shift due to strain for wurtzite materials
double compute_conduction_band_shift_wurtzite(const Eigen::Matrix3d& strain, double a_cz, double a_ct) {
    // Conduction band shift
    return a_cz * strain(2, 2) + a_ct * (strain(0, 0) + strain(1, 1));
}

// Compute valence band shift due to strain for wurtzite materials
void compute_valence_band_shift_wurtzite(const Eigen::Matrix3d& strain, double a_vz, double a_vt,
                                       double D1, double D2, double D3, double D4, double D5, double D6,
                                       double& delta_E_hh, double& delta_E_lh, double& delta_E_ch) {
    // Hydrostatic component
    double delta_E_v_hydro = a_vz * strain(2, 2) + a_vt * (strain(0, 0) + strain(1, 1));
    
    // Crystal field splitting
    double delta_cr = 0.0; // Typical value for GaN: 0.01-0.02 eV
    
    // Hamiltonian matrix elements
    double F = delta_cr + D1 * strain(2, 2) + D2 * (strain(0, 0) + strain(1, 1));
    double G = std::sqrt(2) * D5 * (strain(0, 2) + strain(2, 0));
    double lambda = std::sqrt(2) * D6 * (strain(0, 1) + strain(1, 0));
    double theta = D4 * (strain(0, 0) - strain(1, 1));
    
    // Eigenvalues of the Hamiltonian
    // Simplified approach for small off-diagonal elements
    delta_E_hh = delta_E_v_hydro + F / 2.0 - std::sqrt(std::pow(F / 2.0, 2) + std::pow(G, 2) + std::pow(lambda, 2) + std::pow(theta, 2));
    delta_E_lh = delta_E_v_hydro + F / 2.0 + std::sqrt(std::pow(F / 2.0, 2) + std::pow(G, 2) + std::pow(lambda, 2) + std::pow(theta, 2));
    delta_E_ch = delta_E_v_hydro + D3 * (strain(0, 0) + strain(1, 1)) + D4 * strain(2, 2);
}

// Compute effective mass change due to strain for electrons
double compute_electron_effective_mass_change(const Eigen::Matrix3d& strain, double Xi) {
    // Hydrostatic strain
    double epsilon_h = compute_hydrostatic_strain(strain);
    
    // Effective mass change
    return Xi * epsilon_h;
}

// Compute effective mass change due to strain for holes
void compute_hole_effective_mass_change(const Eigen::Matrix3d& strain, double L, double M, double N,
                                      double& delta_m_hh, double& delta_m_lh) {
    // Biaxial strain
    double epsilon_b = compute_biaxial_strain(strain);
    
    // Luttinger parameters change
    double delta_gamma1 = L * epsilon_b;
    double delta_gamma2 = M * epsilon_b;
    double delta_gamma3 = N * epsilon_b;
    
    // Heavy hole effective mass change (in-plane)
    delta_m_hh = -delta_gamma1 + delta_gamma2;
    
    // Light hole effective mass change (in-plane)
    delta_m_lh = -delta_gamma1 - delta_gamma2;
}

// Compute bandgap change due to strain
double compute_bandgap_change(const Eigen::Matrix3d& strain, double a_c, double a_v, double b) {
    // Conduction band shift
    double delta_E_c = compute_conduction_band_shift_cubic(strain, a_c);
    
    // Valence band shifts
    double delta_E_hh, delta_E_lh, delta_E_so;
    compute_valence_band_shift_cubic(strain, a_v, b, 0.0, delta_E_hh, delta_E_lh, delta_E_so);
    
    // Bandgap change (using heavy hole band)
    return delta_E_c - delta_E_hh;
}

// Compute strain-induced piezoelectric polarization
Eigen::Vector3d compute_piezoelectric_polarization(const Eigen::Matrix3d& strain, const Eigen::Matrix3d& piezoelectric_tensor) {
    // Initialize polarization vector
    Eigen::Vector3d polarization = Eigen::Vector3d::Zero();
    
    // Compute polarization components
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                polarization(i) += piezoelectric_tensor(i, j) * strain(j, k);
            }
        }
    }
    
    return polarization;
}

// Compute strain-induced electric field
Eigen::Vector3d compute_strain_induced_field(const Eigen::Vector3d& polarization, double epsilon_r) {
    // Constants
    const double epsilon_0 = PhysicalConstants::VACUUM_PERMITTIVITY;
    
    // Compute electric field
    return polarization / (epsilon_0 * epsilon_r);
}

// Compute strain distribution in a quantum dot
Eigen::Matrix3d compute_quantum_dot_strain(double x, double y, double z, double R, double a_dot, double a_matrix) {
    // Distance from the center of the quantum dot
    double r = std::sqrt(x*x + y*y + z*z);
    
    // Lattice mismatch
    double epsilon_0 = (a_dot - a_matrix) / a_matrix;
    
    // Initialize strain tensor
    Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();
    
    if (r <= R) {
        // Inside the quantum dot (uniform strain)
        strain(0, 0) = strain(1, 1) = strain(2, 2) = epsilon_0;
    } else {
        // Outside the quantum dot (decaying strain)
        double factor = std::pow(R / r, 3) * epsilon_0;
        strain(0, 0) = factor * (1.0 - 3.0 * x*x / (r*r));
        strain(1, 1) = factor * (1.0 - 3.0 * y*y / (r*r));
        strain(2, 2) = factor * (1.0 - 3.0 * z*z / (r*r));
        strain(0, 1) = strain(1, 0) = -3.0 * factor * x * y / (r*r);
        strain(0, 2) = strain(2, 0) = -3.0 * factor * x * z / (r*r);
        strain(1, 2) = strain(2, 1) = -3.0 * factor * y * z / (r*r);
    }
    
    return strain;
}

// Compute strain energy density
double compute_strain_energy_density(const Eigen::Matrix3d& strain, const Eigen::Matrix6d& stiffness) {
    // Convert strain tensor to Voigt notation
    Eigen::VectorXd strain_voigt(6);
    strain_voigt << strain(0, 0), strain(1, 1), strain(2, 2), 
                    2.0 * strain(1, 2), 2.0 * strain(0, 2), 2.0 * strain(0, 1);
    
    // Compute stress in Voigt notation
    Eigen::VectorXd stress_voigt = stiffness * strain_voigt;
    
    // Compute strain energy density
    return 0.5 * strain_voigt.dot(stress_voigt);
}

// Compute critical thickness for pseudomorphic growth
double compute_critical_thickness(double a_substrate, double a_layer, double poisson_ratio, double b) {
    // Lattice mismatch
    double f = std::abs((a_substrate - a_layer) / a_layer);
    
    // Critical thickness (Matthews-Blakeslee model)
    return b * (1.0 - poisson_ratio) / (8.0 * M_PI * f * (1.0 + poisson_ratio));
}

} // namespace StrainEffects
