/**
 * @file mobility_models.cpp
 * @brief Implementation of mobility models for semiconductor simulations.
 *
 * This file contains the implementation of mobility models used in
 * semiconductor simulations, including temperature, doping, and field dependence.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mobility_models.h"
#include "physical_constants.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace MobilityModels {

// Constant mobility model
double constant_mobility(double mu_0, double temperature, double doping, double E_field) {
    return mu_0;
}

// Temperature-dependent mobility model (power law)
double temperature_dependent_mobility(double mu_0, double temperature, double reference_temperature, double alpha) {
    return mu_0 * std::pow(reference_temperature / temperature, alpha);
}

// Doping-dependent mobility model (Caughey-Thomas)
double doping_dependent_mobility(double mu_min, double mu_max, double doping, double N_ref, double alpha) {
    return mu_min + (mu_max - mu_min) / (1.0 + std::pow(doping / N_ref, alpha));
}

// Field-dependent mobility model (Caughey-Thomas)
double field_dependent_mobility(double mu_0, double E_field, double v_sat, double beta) {
    if (E_field <= 0.0) {
        return mu_0;
    }
    return mu_0 / std::pow(1.0 + std::pow(mu_0 * E_field / v_sat, beta), 1.0 / beta);
}

// Combined mobility model (temperature, doping, and field dependence)
double combined_mobility(double mu_0, double temperature, double reference_temperature, double temp_alpha,
                        double mu_min, double mu_max, double doping, double N_ref, double doping_alpha,
                        double E_field, double v_sat, double beta) {
    // Temperature dependence
    double mu_temp = temperature_dependent_mobility(mu_0, temperature, reference_temperature, temp_alpha);
    
    // Doping dependence
    double mu_doping = doping_dependent_mobility(mu_min, mu_max, doping, N_ref, doping_alpha);
    
    // Combine temperature and doping dependence
    double mu_combined = std::min(mu_temp, mu_doping);
    
    // Field dependence
    return field_dependent_mobility(mu_combined, E_field, v_sat, beta);
}

// Lombardi surface mobility model
double lombardi_surface_mobility(double mu_bulk, double mu_sr, double mu_ac, double distance_from_interface) {
    // Mathiessen's rule for combining mobility components
    double mu_surface = 1.0 / (1.0 / mu_sr + 1.0 / mu_ac);
    
    // Exponential decay of surface effects with distance
    double lambda = 1.0e-7; // Decay length (cm)
    double weight = std::exp(-distance_from_interface / lambda);
    
    // Combine bulk and surface mobility
    return 1.0 / (weight / mu_surface + (1.0 - weight) / mu_bulk);
}

// Canali high-field mobility model
double canali_mobility(double mu_0, double E_field, double v_sat, double beta, double temperature) {
    // Temperature-dependent saturation velocity
    double T_ref = 300.0; // Reference temperature (K)
    double v_sat_T = v_sat * std::pow(T_ref / temperature, 0.5);
    
    // Canali model
    if (E_field <= 0.0) {
        return mu_0;
    }
    return mu_0 / std::pow(1.0 + std::pow(mu_0 * E_field / v_sat_T, beta), 1.0 / beta);
}

// Arora mobility model for silicon
double arora_mobility_si(double temperature, double doping, bool is_electron) {
    // Parameters for Arora model
    double mu_min, mu_max, N_ref, alpha;
    
    if (is_electron) {
        // Electron mobility parameters
        mu_min = 88.0 * std::pow(temperature / 300.0, -0.57);
        mu_max = 1252.0 * std::pow(temperature / 300.0, -2.33);
        N_ref = 1.3e17 * std::pow(temperature / 300.0, 2.4);
        alpha = 0.88 * std::pow(temperature / 300.0, -0.146);
    } else {
        // Hole mobility parameters
        mu_min = 54.3 * std::pow(temperature / 300.0, -0.57);
        mu_max = 407.0 * std::pow(temperature / 300.0, -2.23);
        N_ref = 2.35e17 * std::pow(temperature / 300.0, 2.4);
        alpha = 0.88 * std::pow(temperature / 300.0, -0.146);
    }
    
    // Arora model
    return mu_min + (mu_max - mu_min) / (1.0 + std::pow(doping / N_ref, alpha));
}

// Masetti mobility model for silicon
double masetti_mobility_si(double doping, bool is_electron) {
    // Parameters for Masetti model
    double mu_min1, mu_min2, mu_0, mu_1, N_ref1, N_ref2, alpha, beta;
    
    if (is_electron) {
        // Electron mobility parameters
        mu_min1 = 68.5;
        mu_min2 = 68.5;
        mu_0 = 1414.0;
        mu_1 = 56.1;
        N_ref1 = 9.2e16;
        N_ref2 = 3.41e20;
        alpha = 0.711;
        beta = 1.98;
    } else {
        // Hole mobility parameters
        mu_min1 = 44.9;
        mu_min2 = 0.0;
        mu_0 = 470.5;
        mu_1 = 29.0;
        N_ref1 = 2.23e17;
        N_ref2 = 6.1e20;
        alpha = 0.719;
        beta = 2.0;
    }
    
    // Masetti model
    double term1 = mu_min1 / (1.0 + std::pow(N_ref1 / doping, alpha));
    double term2 = mu_0 - mu_min2 / (1.0 + std::pow(doping / N_ref2, beta));
    
    return term1 + term2;
}

// Klaassen mobility model
double klaassen_mobility(double doping_donor, double doping_acceptor, double temperature, bool is_electron) {
    // Total doping
    double N_total = doping_donor + doping_acceptor;
    
    // Parameters for Klaassen model
    double mu_L, mu_max, N_ref, alpha;
    
    if (is_electron) {
        // Electron mobility parameters
        mu_L = 1417.0 * std::pow(temperature / 300.0, -2.5);
        mu_max = 1417.0 * std::pow(temperature / 300.0, -2.5);
        N_ref = 9.68e16 * std::pow(temperature / 300.0, 3.43);
        alpha = 0.91 * std::pow(temperature / 300.0, -0.146);
    } else {
        // Hole mobility parameters
        mu_L = 470.5 * std::pow(temperature / 300.0, -2.2);
        mu_max = 470.5 * std::pow(temperature / 300.0, -2.2);
        N_ref = 2.82e17 * std::pow(temperature / 300.0, 3.43);
        alpha = 0.76 * std::pow(temperature / 300.0, -0.146);
    }
    
    // Klaassen model
    double mu_D = mu_max * std::pow(N_ref / N_total, alpha);
    
    // Screening factor
    double r_H;
    if (is_electron) {
        r_H = 0.012 + 0.045 * std::exp(-doping_donor / 1.0e18);
    } else {
        r_H = 0.012 + 0.045 * std::exp(-doping_acceptor / 1.0e18);
    }
    
    // Combine lattice and impurity scattering
    return 1.0 / (1.0 / mu_L + 1.0 / mu_D);
}

// Lucovsky mobility model for III-V semiconductors
double lucovsky_mobility(double temperature, double doping, double mu_min, double mu_max, double N_ref, double alpha) {
    // Temperature dependence
    double mu_min_T = mu_min * std::pow(300.0 / temperature, 0.5);
    double mu_max_T = mu_max * std::pow(300.0 / temperature, 2.1);
    double N_ref_T = N_ref * std::pow(temperature / 300.0, 3.0);
    
    // Lucovsky model
    return mu_min_T + (mu_max_T - mu_min_T) / (1.0 + std::pow(doping / N_ref_T, alpha));
}

// Farahmand mobility model for GaN
double farahmand_mobility_gan(double temperature, double doping, bool is_electron) {
    // Parameters for Farahmand model
    double mu_min, mu_max, N_ref, gamma, delta, alpha;
    
    if (is_electron) {
        // Electron mobility parameters
        mu_min = 80.0;
        mu_max = 1400.0 * std::pow(300.0 / temperature, 2.0);
        N_ref = 1.0e17;
        gamma = 1.0;
        delta = 0.66;
        alpha = 0.45;
    } else {
        // Hole mobility parameters
        mu_min = 3.0;
        mu_max = 170.0 * std::pow(300.0 / temperature, 2.0);
        N_ref = 1.0e17;
        gamma = 1.0;
        delta = 0.66;
        alpha = 0.45;
    }
    
    // Farahmand model
    return mu_min + (mu_max - mu_min) / (1.0 + std::pow(doping / N_ref, gamma) * std::pow(temperature / 300.0, delta)) * std::pow(300.0 / temperature, alpha);
}

// Quantum well mobility model
double quantum_well_mobility(double bulk_mobility, double well_width, double temperature) {
    // Constants
    const double h_bar = PhysicalConstants::REDUCED_PLANCK;
    const double m0 = PhysicalConstants::ELECTRON_MASS;
    const double kB = PhysicalConstants::BOLTZMANN_CONSTANT;
    
    // Effective mass (typical for GaAs)
    double m_eff = 0.067 * m0;
    
    // Thermal de Broglie wavelength
    double lambda_th = h_bar / std::sqrt(2.0 * m_eff * kB * temperature);
    
    // Size quantization factor
    double size_factor = 1.0 + (lambda_th / well_width) * (lambda_th / well_width);
    
    // Interface roughness scattering
    double roughness_factor = 1.0 / (1.0 + 0.5 * (lambda_th / well_width));
    
    // Combined effect
    return bulk_mobility * size_factor * roughness_factor;
}

// Mobility model for strained silicon
double strained_si_mobility(double unstrained_mobility, double strain, bool is_electron) {
    // Enhancement factor due to strain
    double enhancement_factor;
    
    if (is_electron) {
        // Electron mobility enhancement
        enhancement_factor = 1.0 + 1.8 * strain; // Linear approximation
    } else {
        // Hole mobility enhancement
        enhancement_factor = 1.0 + 2.3 * strain; // Linear approximation
    }
    
    return unstrained_mobility * enhancement_factor;
}

// Mobility model for SiGe alloys
double sige_mobility(double x_ge, double temperature, double doping, bool is_electron) {
    // Mobility for pure Si
    double mu_si = arora_mobility_si(temperature, doping, is_electron);
    
    // Mobility for pure Ge (approximate)
    double mu_ge;
    if (is_electron) {
        mu_ge = 3900.0 * std::pow(temperature / 300.0, -1.66);
    } else {
        mu_ge = 1900.0 * std::pow(temperature / 300.0, -2.33);
    }
    
    // Bowing parameter for mobility
    double bowing = 0.0;
    
    // Vegard's law with bowing parameter
    return mu_si * (1.0 - x_ge) + mu_ge * x_ge - bowing * x_ge * (1.0 - x_ge);
}

// Mobility model for III-V alloys
double iii_v_alloy_mobility(double x, double mu_1, double mu_2, double alloy_scattering) {
    // Matthiessen's rule for alloy scattering
    double mu_alloy = 1.0 / (x * (1.0 - x) * alloy_scattering);
    
    // Vegard's law for binary mobility
    double mu_binary = mu_1 * (1.0 - x) + mu_2 * x;
    
    // Combine binary and alloy scattering
    return 1.0 / (1.0 / mu_binary + 1.0 / mu_alloy);
}

} // namespace MobilityModels
