/**
 * @file bandgap_models.cpp
 * @brief Implementation of bandgap models for semiconductor simulations.
 *
 * This file contains the implementation of bandgap models used in
 * semiconductor simulations, including temperature and strain dependence.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "bandgap_models.h"
#include "physical_constants.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace BandgapModels {

// Varshni model for temperature-dependent bandgap
double varshni_model(double E_g0, double alpha, double beta, double temperature) {
    return E_g0 - (alpha * temperature * temperature) / (temperature + beta);
}

// Bose-Einstein model for temperature-dependent bandgap
double bose_einstein_model(double E_g0, double alpha_B, double theta_B, double temperature) {
    return E_g0 - alpha_B * (1.0 + 2.0 / (std::exp(theta_B / temperature) - 1.0));
}

// O'Donnell-Chen model for temperature-dependent bandgap
double odonnell_chen_model(double E_g0, double S, double hbar_omega, double temperature) {
    return E_g0 - S * hbar_omega * (std::coth(hbar_omega / (2.0 * PhysicalConstants::BOLTZMANN_CONSTANT * temperature * PhysicalConstants::J_TO_EV)) - 1.0);
}

// Pässler model for temperature-dependent bandgap
double passler_model(double E_g0, double alpha_P, double theta_P, double p, double temperature) {
    return E_g0 - alpha_P * std::pow(temperature, p) / (std::pow(theta_P, p - 1.0) * (theta_P + temperature));
}

// Bandgap narrowing due to heavy doping (Slotboom model)
double bandgap_narrowing_slotboom(double doping) {
    // Constants for Slotboom model
    const double V_bg0 = 9.0e-3; // eV
    const double N_ref = 1.0e17; // cm^-3
    const double C_bg = 0.5;     // dimensionless
    
    // Compute bandgap narrowing
    return V_bg0 * std::log(1.0 + doping / N_ref) * std::pow(1.0 + C_bg * std::log(1.0 + doping / N_ref), 0.5);
}

// Bandgap narrowing due to heavy doping (del Alamo model)
double bandgap_narrowing_del_alamo(double doping) {
    // Constants for del Alamo model
    const double A = 1.87e-2; // eV
    const double N_ref = 1.0e18; // cm^-3
    const double C = 0.5;     // dimensionless
    
    // Compute bandgap narrowing
    return A * std::pow(doping / N_ref, 1.0/3.0);
}

// Bandgap narrowing due to heavy doping (Jain-Roulston model)
double bandgap_narrowing_jain_roulston(double doping, double A_n, double A_p, bool is_n_type) {
    // Constants for Jain-Roulston model
    const double N_ref = 1.0e18; // cm^-3
    
    // Compute bandgap narrowing
    if (is_n_type) {
        return A_n * std::pow(doping / N_ref, 1.0/3.0);
    } else {
        return A_p * std::pow(doping / N_ref, 1.0/3.0);
    }
}

// Bandgap narrowing due to heavy doping (Schenk model)
double bandgap_narrowing_schenk(double doping, double temperature, bool is_n_type) {
    // Constants for Schenk model
    const double kB = PhysicalConstants::BOLTZMANN_CONSTANT * PhysicalConstants::J_TO_EV; // eV/K
    const double q = PhysicalConstants::ELECTRON_CHARGE;
    const double epsilon_r = PhysicalConstants::Silicon::RELATIVE_PERMITTIVITY;
    const double epsilon_0 = PhysicalConstants::VACUUM_PERMITTIVITY;
    
    // Compute Debye length
    double n_i = PhysicalConstants::Silicon::INTRINSIC_CARRIER_CONCENTRATION;
    double L_D = std::sqrt(epsilon_0 * epsilon_r * kB * temperature / (q * q * (doping + n_i)));
    
    // Compute bandgap narrowing
    double prefactor = q * q / (4.0 * M_PI * epsilon_0 * epsilon_r);
    double r_s = std::pow(3.0 / (4.0 * M_PI * doping), 1.0/3.0);
    
    // Exchange and correlation terms
    double E_x = -prefactor * (3.0 / M_PI) * std::pow(3.0 * M_PI * M_PI * doping, 1.0/3.0);
    double E_c = -prefactor * 0.0565 * std::log(1.0 + 21.0 / r_s);
    
    // Total bandgap narrowing
    return -(E_x + E_c);
}

// Bandgap of Si1-xGex alloy
double bandgap_sige(double x_ge, double temperature) {
    // Bandgap of Si and Ge at 0K
    double E_g_Si_0K = 1.17; // eV
    double E_g_Ge_0K = 0.7437; // eV
    
    // Varshni parameters for Si
    double alpha_Si = 4.73e-4; // eV/K
    double beta_Si = 636.0; // K
    
    // Varshni parameters for Ge
    double alpha_Ge = 4.774e-4; // eV/K
    double beta_Ge = 235.0; // K
    
    // Bowing parameter
    double b = 0.21 - 0.1 * x_ge; // eV
    
    // Temperature-dependent bandgaps
    double E_g_Si_T = varshni_model(E_g_Si_0K, alpha_Si, beta_Si, temperature);
    double E_g_Ge_T = varshni_model(E_g_Ge_0K, alpha_Ge, beta_Ge, temperature);
    
    // Bandgap of SiGe alloy (Vegard's law with bowing parameter)
    return (1.0 - x_ge) * E_g_Si_T + x_ge * E_g_Ge_T - b * x_ge * (1.0 - x_ge);
}

// Bandgap of AlxGa1-xAs alloy
double bandgap_algaas(double x_al, double temperature) {
    // Direct bandgap of GaAs at 0K
    double E_g_GaAs_0K = 1.519; // eV
    
    // Direct bandgap of AlAs at 0K
    double E_g_AlAs_0K = 3.099; // eV
    
    // Varshni parameters for GaAs
    double alpha_GaAs = 5.405e-4; // eV/K
    double beta_GaAs = 204.0; // K
    
    // Varshni parameters for AlAs
    double alpha_AlAs = 5.58e-4; // eV/K
    double beta_AlAs = 530.0; // K
    
    // Bowing parameter
    double b = 0.127 + 1.310 * x_al; // eV
    
    // Temperature-dependent bandgaps
    double E_g_GaAs_T = varshni_model(E_g_GaAs_0K, alpha_GaAs, beta_GaAs, temperature);
    double E_g_AlAs_T = varshni_model(E_g_AlAs_0K, alpha_AlAs, beta_AlAs, temperature);
    
    // Bandgap of AlGaAs alloy (Vegard's law with bowing parameter)
    double E_g = (1.0 - x_al) * E_g_GaAs_T + x_al * E_g_AlAs_T - b * x_al * (1.0 - x_al);
    
    // Transition from direct to indirect bandgap
    if (x_al > 0.45) {
        // Indirect bandgap of GaAs at 0K
        double E_g_GaAs_X_0K = 1.981; // eV
        
        // Indirect bandgap of AlAs at 0K
        double E_g_AlAs_X_0K = 2.24; // eV
        
        // Varshni parameters for indirect bandgap
        double alpha_GaAs_X = 4.6e-4; // eV/K
        double beta_GaAs_X = 204.0; // K
        
        double alpha_AlAs_X = 5.0e-4; // eV/K
        double beta_AlAs_X = 530.0; // K
        
        // Bowing parameter for indirect bandgap
        double b_X = 0.055; // eV
        
        // Temperature-dependent indirect bandgaps
        double E_g_GaAs_X_T = varshni_model(E_g_GaAs_X_0K, alpha_GaAs_X, beta_GaAs_X, temperature);
        double E_g_AlAs_X_T = varshni_model(E_g_AlAs_X_0K, alpha_AlAs_X, beta_AlAs_X, temperature);
        
        // Indirect bandgap of AlGaAs alloy
        double E_g_X = (1.0 - x_al) * E_g_GaAs_X_T + x_al * E_g_AlAs_X_T - b_X * x_al * (1.0 - x_al);
        
        // Return the minimum of direct and indirect bandgaps
        return std::min(E_g, E_g_X);
    }
    
    return E_g;
}

// Bandgap of InxGa1-xAs alloy
double bandgap_ingaas(double x_in, double temperature) {
    // Bandgap of GaAs at 0K
    double E_g_GaAs_0K = 1.519; // eV
    
    // Bandgap of InAs at 0K
    double E_g_InAs_0K = 0.417; // eV
    
    // Varshni parameters for GaAs
    double alpha_GaAs = 5.405e-4; // eV/K
    double beta_GaAs = 204.0; // K
    
    // Varshni parameters for InAs
    double alpha_InAs = 2.76e-4; // eV/K
    double beta_InAs = 93.0; // K
    
    // Bowing parameter
    double b = 0.477; // eV
    
    // Temperature-dependent bandgaps
    double E_g_GaAs_T = varshni_model(E_g_GaAs_0K, alpha_GaAs, beta_GaAs, temperature);
    double E_g_InAs_T = varshni_model(E_g_InAs_0K, alpha_InAs, beta_InAs, temperature);
    
    // Bandgap of InGaAs alloy (Vegard's law with bowing parameter)
    return (1.0 - x_in) * E_g_GaAs_T + x_in * E_g_InAs_T - b * x_in * (1.0 - x_in);
}

// Bandgap of InxGa1-xN alloy
double bandgap_ingan(double x_in, double temperature) {
    // Bandgap of GaN at 0K
    double E_g_GaN_0K = 3.507; // eV
    
    // Bandgap of InN at 0K
    double E_g_InN_0K = 0.78; // eV
    
    // Varshni parameters for GaN
    double alpha_GaN = 8.873e-4; // eV/K
    double beta_GaN = 830.0; // K
    
    // Varshni parameters for InN
    double alpha_InN = 2.45e-4; // eV/K
    double beta_InN = 624.0; // K
    
    // Bowing parameter
    double b = 1.43; // eV
    
    // Temperature-dependent bandgaps
    double E_g_GaN_T = varshni_model(E_g_GaN_0K, alpha_GaN, beta_GaN, temperature);
    double E_g_InN_T = varshni_model(E_g_InN_0K, alpha_InN, beta_InN, temperature);
    
    // Bandgap of InGaN alloy (Vegard's law with bowing parameter)
    return (1.0 - x_in) * E_g_GaN_T + x_in * E_g_InN_T - b * x_in * (1.0 - x_in);
}

// Bandgap of AlxGa1-xN alloy
double bandgap_algan(double x_al, double temperature) {
    // Bandgap of GaN at 0K
    double E_g_GaN_0K = 3.507; // eV
    
    // Bandgap of AlN at 0K
    double E_g_AlN_0K = 6.23; // eV
    
    // Varshni parameters for GaN
    double alpha_GaN = 8.873e-4; // eV/K
    double beta_GaN = 830.0; // K
    
    // Varshni parameters for AlN
    double alpha_AlN = 1.799e-3; // eV/K
    double beta_AlN = 1462.0; // K
    
    // Bowing parameter
    double b = 0.7; // eV
    
    // Temperature-dependent bandgaps
    double E_g_GaN_T = varshni_model(E_g_GaN_0K, alpha_GaN, beta_GaN, temperature);
    double E_g_AlN_T = varshni_model(E_g_AlN_0K, alpha_AlN, beta_AlN, temperature);
    
    // Bandgap of AlGaN alloy (Vegard's law with bowing parameter)
    return (1.0 - x_al) * E_g_GaN_T + x_al * E_g_AlN_T - b * x_al * (1.0 - x_al);
}

// Bandgap shift due to strain
double bandgap_shift_strain(const Eigen::Matrix3d& strain, double a_c, double a_v, double b, double d) {
    // Hydrostatic strain
    double epsilon_h = strain(0, 0) + strain(1, 1) + strain(2, 2);
    
    // Biaxial strain
    double epsilon_b = 2.0 * strain(2, 2) - strain(0, 0) - strain(1, 1);
    
    // Shear strain
    double epsilon_s = std::sqrt(std::pow(strain(0, 1), 2) + std::pow(strain(0, 2), 2) + std::pow(strain(1, 2), 2));
    
    // Conduction band shift
    double delta_E_c = a_c * epsilon_h;
    
    // Valence band shift (heavy hole)
    double delta_E_v = a_v * epsilon_h - b * epsilon_b / 2.0 - d * epsilon_s;
    
    // Bandgap shift
    return delta_E_c - delta_E_v;
}

// Bandgap shift due to quantum confinement
double bandgap_shift_quantum_confinement(double well_width, double m_e, double m_h) {
    // Constants
    const double h_bar = PhysicalConstants::REDUCED_PLANCK;
    const double m0 = PhysicalConstants::ELECTRON_MASS;
    const double q = PhysicalConstants::ELECTRON_CHARGE;
    
    // Convert well width from nm to m
    double L = well_width * 1.0e-9;
    
    // Effective masses in kg
    double m_e_kg = m_e * m0;
    double m_h_kg = m_h * m0;
    
    // Ground state energy for electron and hole (infinite well approximation)
    double E_e = std::pow(M_PI * h_bar, 2) / (2.0 * m_e_kg * std::pow(L, 2));
    double E_h = std::pow(M_PI * h_bar, 2) / (2.0 * m_h_kg * std::pow(L, 2));
    
    // Convert from J to eV
    return (E_e + E_h) / q;
}

// Bandgap shift due to hydrostatic pressure
double bandgap_shift_pressure(double pressure, double dE_g_dP) {
    return dE_g_dP * pressure;
}

// Bandgap shift due to electric field (Franz-Keldysh effect)
double bandgap_shift_electric_field(double electric_field, double m_r, double E_g) {
    // Constants
    const double h_bar = PhysicalConstants::REDUCED_PLANCK;
    const double q = PhysicalConstants::ELECTRON_CHARGE;
    const double m0 = PhysicalConstants::ELECTRON_MASS;
    
    // Reduced effective mass in kg
    double m_r_kg = m_r * m0;
    
    // Electro-optic energy
    double hbar_theta = std::pow(std::pow(h_bar, 2) * std::pow(q * electric_field, 2) / (2.0 * m_r_kg), 1.0/3.0);
    
    // Bandgap shift (approximate formula)
    return -hbar_theta;
}

} // namespace BandgapModels
