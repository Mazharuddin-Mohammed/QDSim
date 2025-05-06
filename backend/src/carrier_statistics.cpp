/**
 * @file carrier_statistics.cpp
 * @brief Implementation of carrier statistics functions for semiconductor simulations.
 *
 * This file contains the implementation of carrier statistics functions used in
 * semiconductor simulations, including Fermi-Dirac and Boltzmann statistics,
 * as well as functions for computing carrier concentrations and quasi-Fermi levels.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "carrier_statistics.h"
#include "physical_constants.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace CarrierStatistics {

// Fermi-Dirac integral of order 1/2
double fermi_dirac_half(double eta) {
    // For large positive eta, use asymptotic expansion
    if (eta > 10.0) {
        return (2.0/3.0) * std::pow(eta, 1.5) * (1.0 + (1.0/8.0) * std::pow(M_PI/eta, 2));
    }
    
    // For large negative eta, use exponential approximation
    if (eta < -10.0) {
        return std::exp(eta);
    }
    
    // For intermediate values, use Bednarczyk and Bednarczyk approximation
    // D. Bednarczyk, J. Bednarczyk, "The approximation of the Fermi-Dirac integral F_{1/2}(η)"
    // Physics Letters A, 64(4), 1977, pp. 409-410
    double num = std::pow(eta, 4) + 33.6 * std::pow(eta, 2) + 50.0;
    double den = std::pow(eta, 4) + 50.0;
    return std::exp(eta) / (1.0 + 0.27 * std::exp(-1.0 * eta) * std::pow(num/den, 0.75));
}

// Inverse of Fermi-Dirac integral of order 1/2
double inverse_fermi_dirac_half(double x) {
    // For small x, use logarithmic approximation
    if (x < 0.01) {
        return std::log(x);
    }
    
    // For large x, use asymptotic expansion
    if (x > 10.0) {
        return std::pow(1.5 * x, 2.0/3.0) * (1.0 - (1.0/12.0) * std::pow(M_PI/std::pow(1.5*x, 2.0/3.0), 2));
    }
    
    // For intermediate values, use Joyce-Dixon approximation
    // W.B. Joyce, R.W. Dixon, "Analytic approximations for the Fermi energy of an ideal Fermi gas"
    // Applied Physics Letters, 31(5), 1977, pp. 354-356
    double ln_x = std::log(x);
    return ln_x + ln_x * ln_x * (0.1528 + 0.0482 * ln_x) / (1.0 + 0.1292 * ln_x);
}

// Electron concentration using Fermi-Dirac statistics
double electron_concentration_fd(double E_c, double E_f, double N_c, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Reduced Fermi level
    double eta = (E_f - E_c) / kT_eV;
    
    // Compute electron concentration
    return N_c * fermi_dirac_half(eta);
}

// Hole concentration using Fermi-Dirac statistics
double hole_concentration_fd(double E_v, double E_f, double N_v, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Reduced Fermi level
    double eta = (E_v - E_f) / kT_eV;
    
    // Compute hole concentration
    return N_v * fermi_dirac_half(eta);
}

// Electron concentration using Boltzmann statistics
double electron_concentration_boltzmann(double E_c, double E_f, double N_c, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Compute electron concentration
    return N_c * std::exp((E_f - E_c) / kT_eV);
}

// Hole concentration using Boltzmann statistics
double hole_concentration_boltzmann(double E_v, double E_f, double N_v, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Compute hole concentration
    return N_v * std::exp((E_v - E_f) / kT_eV);
}

// Electron quasi-Fermi level using Fermi-Dirac statistics
double electron_quasi_fermi_level_fd(double n, double E_c, double N_c, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Compute quasi-Fermi level
    return E_c + kT_eV * inverse_fermi_dirac_half(n / N_c);
}

// Hole quasi-Fermi level using Fermi-Dirac statistics
double hole_quasi_fermi_level_fd(double p, double E_v, double N_v, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Compute quasi-Fermi level
    return E_v - kT_eV * inverse_fermi_dirac_half(p / N_v);
}

// Electron quasi-Fermi level using Boltzmann statistics
double electron_quasi_fermi_level_boltzmann(double n, double E_c, double N_c, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Compute quasi-Fermi level
    return E_c + kT_eV * std::log(n / N_c);
}

// Hole quasi-Fermi level using Boltzmann statistics
double hole_quasi_fermi_level_boltzmann(double p, double E_v, double N_v, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Compute quasi-Fermi level
    return E_v - kT_eV * std::log(p / N_v);
}

// Intrinsic carrier concentration
double intrinsic_carrier_concentration(double E_g, double N_c, double N_v, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Compute intrinsic carrier concentration
    return std::sqrt(N_c * N_v) * std::exp(-E_g / (2.0 * kT_eV));
}

// Effective density of states in conduction band
double effective_density_of_states_conduction(double m_e, double temperature) {
    // Constants
    const double h = PhysicalConstants::PLANCK_CONSTANT;
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double m0 = PhysicalConstants::ELECTRON_MASS;
    
    // Compute effective density of states
    return 2.0 * std::pow(m_e * m0 * kT / (2.0 * M_PI * h * h), 1.5);
}

// Effective density of states in valence band
double effective_density_of_states_valence(double m_h, double temperature) {
    // Constants
    const double h = PhysicalConstants::PLANCK_CONSTANT;
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double m0 = PhysicalConstants::ELECTRON_MASS;
    
    // Compute effective density of states
    return 2.0 * std::pow(m_h * m0 * kT / (2.0 * M_PI * h * h), 1.5);
}

// Electron mobility with field dependence (Caughey-Thomas model)
double electron_mobility_field_dependent(double mu_0, double E, double v_sat, double beta) {
    // Compute field-dependent mobility
    return mu_0 / std::pow(1.0 + std::pow(mu_0 * E / v_sat, beta), 1.0 / beta);
}

// Hole mobility with field dependence (Caughey-Thomas model)
double hole_mobility_field_dependent(double mu_0, double E, double v_sat, double beta) {
    // Compute field-dependent mobility
    return mu_0 / std::pow(1.0 + std::pow(mu_0 * E / v_sat, beta), 1.0 / beta);
}

// SRH recombination rate
double srh_recombination(double n, double p, double n_i, double tau_n, double tau_p, double n1, double p1) {
    // Compute SRH recombination rate
    return (n * p - n_i * n_i) / (tau_p * (n + n1) + tau_n * (p + p1));
}

// Auger recombination rate
double auger_recombination(double n, double p, double n_i, double C_n, double C_p) {
    // Compute Auger recombination rate
    return (C_n * n + C_p * p) * (n * p - n_i * n_i);
}

// Radiative recombination rate
double radiative_recombination(double n, double p, double n_i, double B) {
    // Compute radiative recombination rate
    return B * (n * p - n_i * n_i);
}

// Total recombination rate
double total_recombination(double n, double p, double n_i, double tau_n, double tau_p, 
                          double n1, double p1, double C_n, double C_p, double B) {
    // Compute total recombination rate
    double R_SRH = srh_recombination(n, p, n_i, tau_n, tau_p, n1, p1);
    double R_Auger = auger_recombination(n, p, n_i, C_n, C_p);
    double R_rad = radiative_recombination(n, p, n_i, B);
    
    return R_SRH + R_Auger + R_rad;
}

// Electron diffusion coefficient (Einstein relation)
double electron_diffusion_coefficient(double mu_n, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double q = PhysicalConstants::ELECTRON_CHARGE;
    
    // Compute diffusion coefficient
    return mu_n * kT / q;
}

// Hole diffusion coefficient (Einstein relation)
double hole_diffusion_coefficient(double mu_p, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double q = PhysicalConstants::ELECTRON_CHARGE;
    
    // Compute diffusion coefficient
    return mu_p * kT / q;
}

// Bandgap narrowing due to heavy doping
double bandgap_narrowing(double N_doping, double temperature) {
    // Constants
    const double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature;
    const double kT_eV = kT * PhysicalConstants::J_TO_EV;
    
    // Critical doping concentration (cm^-3)
    const double N_ref = 1.0e17;
    
    // Bandgap narrowing parameters
    const double A = 0.0187; // eV
    const double B = 0.0063; // eV
    const double C = 0.0057; // eV
    
    // Compute bandgap narrowing
    if (N_doping <= N_ref) {
        return 0.0;
    } else {
        double x = std::log(N_doping / N_ref);
        return A * x + B * x * x + C * x * x * x;
    }
}

} // namespace CarrierStatistics
