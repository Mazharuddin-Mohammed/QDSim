#pragma once
/**
 * @file mobility_models.h
 * @brief Defines mobility models for semiconductor simulations.
 *
 * This file contains the declaration of mobility models used in
 * semiconductor simulations, including temperature, doping, and field dependence.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <cmath>

/**
 * @namespace MobilityModels
 * @brief Namespace for mobility models used in semiconductor simulations.
 *
 * This namespace contains functions for computing carrier mobilities
 * with various dependencies, such as temperature, doping, and electric field.
 */
namespace MobilityModels {

/**
 * @brief Constant mobility model.
 *
 * This function returns a constant mobility regardless of temperature,
 * doping, or electric field.
 *
 * @param mu_0 The constant mobility (cm^2/V·s)
 * @param temperature The temperature (K)
 * @param doping The doping concentration (cm^-3)
 * @param E_field The electric field magnitude (V/cm)
 * @return The constant mobility (cm^2/V·s)
 */
double constant_mobility(double mu_0, double temperature, double doping, double E_field);

/**
 * @brief Temperature-dependent mobility model (power law).
 *
 * This function computes the temperature-dependent mobility using a power law:
 * μ(T) = μ(T_ref) * (T_ref/T)^α
 *
 * @param mu_0 The mobility at the reference temperature (cm^2/V·s)
 * @param temperature The temperature (K)
 * @param reference_temperature The reference temperature (K)
 * @param alpha The temperature exponent
 * @return The temperature-dependent mobility (cm^2/V·s)
 */
double temperature_dependent_mobility(double mu_0, double temperature, double reference_temperature, double alpha);

/**
 * @brief Doping-dependent mobility model (Caughey-Thomas).
 *
 * This function computes the doping-dependent mobility using the Caughey-Thomas model:
 * μ = μ_min + (μ_max - μ_min) / (1 + (N/N_ref)^α)
 *
 * @param mu_min The minimum mobility (cm^2/V·s)
 * @param mu_max The maximum mobility (cm^2/V·s)
 * @param doping The doping concentration (cm^-3)
 * @param N_ref The reference doping concentration (cm^-3)
 * @param alpha The doping exponent
 * @return The doping-dependent mobility (cm^2/V·s)
 */
double doping_dependent_mobility(double mu_min, double mu_max, double doping, double N_ref, double alpha);

/**
 * @brief Field-dependent mobility model (Caughey-Thomas).
 *
 * This function computes the field-dependent mobility using the Caughey-Thomas model:
 * μ(E) = μ_0 / (1 + (μ_0 * E / v_sat)^β)^(1/β)
 *
 * @param mu_0 The low-field mobility (cm^2/V·s)
 * @param E_field The electric field magnitude (V/cm)
 * @param v_sat The saturation velocity (cm/s)
 * @param beta The field-dependence parameter
 * @return The field-dependent mobility (cm^2/V·s)
 */
double field_dependent_mobility(double mu_0, double E_field, double v_sat, double beta);

/**
 * @brief Combined mobility model (temperature, doping, and field dependence).
 *
 * This function computes the mobility with temperature, doping, and field dependence.
 *
 * @param mu_0 The reference mobility (cm^2/V·s)
 * @param temperature The temperature (K)
 * @param reference_temperature The reference temperature (K)
 * @param temp_alpha The temperature exponent
 * @param mu_min The minimum mobility (cm^2/V·s)
 * @param mu_max The maximum mobility (cm^2/V·s)
 * @param doping The doping concentration (cm^-3)
 * @param N_ref The reference doping concentration (cm^-3)
 * @param doping_alpha The doping exponent
 * @param E_field The electric field magnitude (V/cm)
 * @param v_sat The saturation velocity (cm/s)
 * @param beta The field-dependence parameter
 * @return The combined mobility (cm^2/V·s)
 */
double combined_mobility(double mu_0, double temperature, double reference_temperature, double temp_alpha,
                        double mu_min, double mu_max, double doping, double N_ref, double doping_alpha,
                        double E_field, double v_sat, double beta);

/**
 * @brief Lombardi surface mobility model.
 *
 * This function computes the mobility near a semiconductor-insulator interface
 * using the Lombardi surface mobility model.
 *
 * @param mu_bulk The bulk mobility (cm^2/V·s)
 * @param mu_sr The surface roughness limited mobility (cm^2/V·s)
 * @param mu_ac The acoustic phonon limited mobility (cm^2/V·s)
 * @param distance_from_interface The distance from the interface (cm)
 * @return The surface mobility (cm^2/V·s)
 */
double lombardi_surface_mobility(double mu_bulk, double mu_sr, double mu_ac, double distance_from_interface);

/**
 * @brief Canali high-field mobility model.
 *
 * This function computes the high-field mobility using the Canali model,
 * which includes temperature dependence of the saturation velocity.
 *
 * @param mu_0 The low-field mobility (cm^2/V·s)
 * @param E_field The electric field magnitude (V/cm)
 * @param v_sat The saturation velocity at 300K (cm/s)
 * @param beta The field-dependence parameter
 * @param temperature The temperature (K)
 * @return The high-field mobility (cm^2/V·s)
 */
double canali_mobility(double mu_0, double E_field, double v_sat, double beta, double temperature);

/**
 * @brief Arora mobility model for silicon.
 *
 * This function computes the mobility in silicon using the Arora model,
 * which includes temperature and doping dependence.
 *
 * @param temperature The temperature (K)
 * @param doping The doping concentration (cm^-3)
 * @param is_electron Whether the carrier is an electron (true) or a hole (false)
 * @return The mobility (cm^2/V·s)
 */
double arora_mobility_si(double temperature, double doping, bool is_electron);

/**
 * @brief Masetti mobility model for silicon.
 *
 * This function computes the mobility in silicon using the Masetti model,
 * which is accurate for a wide range of doping concentrations.
 *
 * @param doping The doping concentration (cm^-3)
 * @param is_electron Whether the carrier is an electron (true) or a hole (false)
 * @return The mobility (cm^2/V·s)
 */
double masetti_mobility_si(double doping, bool is_electron);

/**
 * @brief Klaassen mobility model.
 *
 * This function computes the mobility using the Klaassen model,
 * which accounts for majority and minority carrier mobilities.
 *
 * @param doping_donor The donor doping concentration (cm^-3)
 * @param doping_acceptor The acceptor doping concentration (cm^-3)
 * @param temperature The temperature (K)
 * @param is_electron Whether the carrier is an electron (true) or a hole (false)
 * @return The mobility (cm^2/V·s)
 */
double klaassen_mobility(double doping_donor, double doping_acceptor, double temperature, bool is_electron);

/**
 * @brief Lucovsky mobility model for III-V semiconductors.
 *
 * This function computes the mobility in III-V semiconductors using the Lucovsky model,
 * which includes temperature and doping dependence.
 *
 * @param temperature The temperature (K)
 * @param doping The doping concentration (cm^-3)
 * @param mu_min The minimum mobility (cm^2/V·s)
 * @param mu_max The maximum mobility (cm^2/V·s)
 * @param N_ref The reference doping concentration (cm^-3)
 * @param alpha The doping exponent
 * @return The mobility (cm^2/V·s)
 */
double lucovsky_mobility(double temperature, double doping, double mu_min, double mu_max, double N_ref, double alpha);

/**
 * @brief Farahmand mobility model for GaN.
 *
 * This function computes the mobility in GaN using the Farahmand model,
 * which includes temperature and doping dependence.
 *
 * @param temperature The temperature (K)
 * @param doping The doping concentration (cm^-3)
 * @param is_electron Whether the carrier is an electron (true) or a hole (false)
 * @return The mobility (cm^2/V·s)
 */
double farahmand_mobility_gan(double temperature, double doping, bool is_electron);

/**
 * @brief Quantum well mobility model.
 *
 * This function computes the mobility in a quantum well,
 * accounting for size quantization and interface roughness scattering.
 *
 * @param bulk_mobility The bulk mobility (cm^2/V·s)
 * @param well_width The quantum well width (cm)
 * @param temperature The temperature (K)
 * @return The quantum well mobility (cm^2/V·s)
 */
double quantum_well_mobility(double bulk_mobility, double well_width, double temperature);

/**
 * @brief Mobility model for strained silicon.
 *
 * This function computes the mobility in strained silicon,
 * accounting for the mobility enhancement due to strain.
 *
 * @param unstrained_mobility The mobility in unstrained silicon (cm^2/V·s)
 * @param strain The strain (dimensionless)
 * @param is_electron Whether the carrier is an electron (true) or a hole (false)
 * @return The mobility in strained silicon (cm^2/V·s)
 */
double strained_si_mobility(double unstrained_mobility, double strain, bool is_electron);

/**
 * @brief Mobility model for SiGe alloys.
 *
 * This function computes the mobility in SiGe alloys,
 * accounting for the composition dependence.
 *
 * @param x_ge The germanium fraction (0 to 1)
 * @param temperature The temperature (K)
 * @param doping The doping concentration (cm^-3)
 * @param is_electron Whether the carrier is an electron (true) or a hole (false)
 * @return The mobility in SiGe (cm^2/V·s)
 */
double sige_mobility(double x_ge, double temperature, double doping, bool is_electron);

/**
 * @brief Mobility model for III-V alloys.
 *
 * This function computes the mobility in III-V alloys,
 * accounting for alloy scattering.
 *
 * @param x The alloy composition (0 to 1)
 * @param mu_1 The mobility of the first binary compound (cm^2/V·s)
 * @param mu_2 The mobility of the second binary compound (cm^2/V·s)
 * @param alloy_scattering The alloy scattering parameter (cm^2/V·s)^-1
 * @return The mobility in the III-V alloy (cm^2/V·s)
 */
double iii_v_alloy_mobility(double x, double mu_1, double mu_2, double alloy_scattering);

} // namespace MobilityModels
