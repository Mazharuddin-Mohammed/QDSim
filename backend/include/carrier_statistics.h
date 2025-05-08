#pragma once
/**
 * @file carrier_statistics.h
 * @brief Defines carrier statistics functions for semiconductor simulations.
 *
 * This file contains the declaration of carrier statistics functions used in
 * semiconductor simulations, including Fermi-Dirac and Boltzmann statistics,
 * as well as functions for computing carrier concentrations and quasi-Fermi levels.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <cmath>

/**
 * @namespace CarrierStatistics
 * @brief Namespace for carrier statistics functions used in semiconductor simulations.
 *
 * This namespace contains functions for computing carrier concentrations,
 * quasi-Fermi levels, and other carrier statistics used in semiconductor simulations.
 */
namespace CarrierStatistics {

/**
 * @brief Computes the Fermi-Dirac integral of order 1/2.
 *
 * This function computes the Fermi-Dirac integral of order 1/2, which is used
 * in the calculation of carrier concentrations with Fermi-Dirac statistics.
 *
 * @param eta The reduced Fermi level (E_F - E_C) / kT
 * @return The value of the Fermi-Dirac integral of order 1/2
 */
double fermi_dirac_half(double eta);

/**
 * @brief Computes the inverse of the Fermi-Dirac integral of order 1/2.
 *
 * This function computes the inverse of the Fermi-Dirac integral of order 1/2,
 * which is used in the calculation of quasi-Fermi levels with Fermi-Dirac statistics.
 *
 * @param x The value of the Fermi-Dirac integral of order 1/2
 * @return The reduced Fermi level (E_F - E_C) / kT
 */
double inverse_fermi_dirac_half(double x);

/**
 * @brief Computes the electron concentration using Fermi-Dirac statistics.
 *
 * This function computes the electron concentration using Fermi-Dirac statistics,
 * which is more accurate than Boltzmann statistics for degenerate semiconductors.
 *
 * @param E_c The conduction band edge energy (eV)
 * @param E_f The Fermi level energy (eV)
 * @param N_c The effective density of states in the conduction band (cm^-3)
 * @param temperature The temperature (K)
 * @return The electron concentration (cm^-3)
 */
double electron_concentration_fd(double E_c, double E_f, double N_c, double temperature);

/**
 * @brief Computes the hole concentration using Fermi-Dirac statistics.
 *
 * This function computes the hole concentration using Fermi-Dirac statistics,
 * which is more accurate than Boltzmann statistics for degenerate semiconductors.
 *
 * @param E_v The valence band edge energy (eV)
 * @param E_f The Fermi level energy (eV)
 * @param N_v The effective density of states in the valence band (cm^-3)
 * @param temperature The temperature (K)
 * @return The hole concentration (cm^-3)
 */
double hole_concentration_fd(double E_v, double E_f, double N_v, double temperature);

/**
 * @brief Computes the electron concentration using Boltzmann statistics.
 *
 * This function computes the electron concentration using Boltzmann statistics,
 * which is a good approximation for non-degenerate semiconductors.
 *
 * @param E_c The conduction band edge energy (eV)
 * @param E_f The Fermi level energy (eV)
 * @param N_c The effective density of states in the conduction band (cm^-3)
 * @param temperature The temperature (K)
 * @return The electron concentration (cm^-3)
 */
double electron_concentration_boltzmann(double E_c, double E_f, double N_c, double temperature);

/**
 * @brief Computes the hole concentration using Boltzmann statistics.
 *
 * This function computes the hole concentration using Boltzmann statistics,
 * which is a good approximation for non-degenerate semiconductors.
 *
 * @param E_v The valence band edge energy (eV)
 * @param E_f The Fermi level energy (eV)
 * @param N_v The effective density of states in the valence band (cm^-3)
 * @param temperature The temperature (K)
 * @return The hole concentration (cm^-3)
 */
double hole_concentration_boltzmann(double E_v, double E_f, double N_v, double temperature);

/**
 * @brief Computes the electron quasi-Fermi level using Fermi-Dirac statistics.
 *
 * This function computes the electron quasi-Fermi level using Fermi-Dirac statistics,
 * which is more accurate than Boltzmann statistics for degenerate semiconductors.
 *
 * @param n The electron concentration (cm^-3)
 * @param E_c The conduction band edge energy (eV)
 * @param N_c The effective density of states in the conduction band (cm^-3)
 * @param temperature The temperature (K)
 * @return The electron quasi-Fermi level (eV)
 */
double electron_quasi_fermi_level_fd(double n, double E_c, double N_c, double temperature);

/**
 * @brief Computes the hole quasi-Fermi level using Fermi-Dirac statistics.
 *
 * This function computes the hole quasi-Fermi level using Fermi-Dirac statistics,
 * which is more accurate than Boltzmann statistics for degenerate semiconductors.
 *
 * @param p The hole concentration (cm^-3)
 * @param E_v The valence band edge energy (eV)
 * @param N_v The effective density of states in the valence band (cm^-3)
 * @param temperature The temperature (K)
 * @return The hole quasi-Fermi level (eV)
 */
double hole_quasi_fermi_level_fd(double p, double E_v, double N_v, double temperature);

/**
 * @brief Computes the electron quasi-Fermi level using Boltzmann statistics.
 *
 * This function computes the electron quasi-Fermi level using Boltzmann statistics,
 * which is a good approximation for non-degenerate semiconductors.
 *
 * @param n The electron concentration (cm^-3)
 * @param E_c The conduction band edge energy (eV)
 * @param N_c The effective density of states in the conduction band (cm^-3)
 * @param temperature The temperature (K)
 * @return The electron quasi-Fermi level (eV)
 */
double electron_quasi_fermi_level_boltzmann(double n, double E_c, double N_c, double temperature);

/**
 * @brief Computes the hole quasi-Fermi level using Boltzmann statistics.
 *
 * This function computes the hole quasi-Fermi level using Boltzmann statistics,
 * which is a good approximation for non-degenerate semiconductors.
 *
 * @param p The hole concentration (cm^-3)
 * @param E_v The valence band edge energy (eV)
 * @param N_v The effective density of states in the valence band (cm^-3)
 * @param temperature The temperature (K)
 * @return The hole quasi-Fermi level (eV)
 */
double hole_quasi_fermi_level_boltzmann(double p, double E_v, double N_v, double temperature);

/**
 * @brief Computes the intrinsic carrier concentration.
 *
 * This function computes the intrinsic carrier concentration based on
 * the bandgap and effective densities of states.
 *
 * @param E_g The bandgap energy (eV)
 * @param N_c The effective density of states in the conduction band (cm^-3)
 * @param N_v The effective density of states in the valence band (cm^-3)
 * @param temperature The temperature (K)
 * @return The intrinsic carrier concentration (cm^-3)
 */
double intrinsic_carrier_concentration(double E_g, double N_c, double N_v, double temperature);

/**
 * @brief Computes the effective density of states in the conduction band.
 *
 * This function computes the effective density of states in the conduction band
 * based on the effective mass and temperature.
 *
 * @param m_e The electron effective mass (relative to free electron mass)
 * @param temperature The temperature (K)
 * @return The effective density of states in the conduction band (cm^-3)
 */
double effective_density_of_states_conduction(double m_e, double temperature);

/**
 * @brief Computes the effective density of states in the valence band.
 *
 * This function computes the effective density of states in the valence band
 * based on the effective mass and temperature.
 *
 * @param m_h The hole effective mass (relative to free electron mass)
 * @param temperature The temperature (K)
 * @return The effective density of states in the valence band (cm^-3)
 */
double effective_density_of_states_valence(double m_h, double temperature);

/**
 * @brief Computes the electron mobility with field dependence.
 *
 * This function computes the electron mobility with field dependence
 * using the Caughey-Thomas model.
 *
 * @param mu_0 The low-field electron mobility (cm^2/V·s)
 * @param E The electric field magnitude (V/cm)
 * @param v_sat The saturation velocity (cm/s)
 * @param beta The field-dependence parameter
 * @return The field-dependent electron mobility (cm^2/V·s)
 */
double electron_mobility_field_dependent(double mu_0, double E, double v_sat, double beta);

/**
 * @brief Computes the hole mobility with field dependence.
 *
 * This function computes the hole mobility with field dependence
 * using the Caughey-Thomas model.
 *
 * @param mu_0 The low-field hole mobility (cm^2/V·s)
 * @param E The electric field magnitude (V/cm)
 * @param v_sat The saturation velocity (cm/s)
 * @param beta The field-dependence parameter
 * @return The field-dependent hole mobility (cm^2/V·s)
 */
double hole_mobility_field_dependent(double mu_0, double E, double v_sat, double beta);

/**
 * @brief Computes the SRH recombination rate.
 *
 * This function computes the Shockley-Read-Hall (SRH) recombination rate,
 * which is the dominant recombination mechanism in indirect bandgap semiconductors.
 *
 * @param n The electron concentration (cm^-3)
 * @param p The hole concentration (cm^-3)
 * @param n_i The intrinsic carrier concentration (cm^-3)
 * @param tau_n The electron lifetime (s)
 * @param tau_p The hole lifetime (s)
 * @param n1 The electron concentration when Fermi level is at trap level (cm^-3)
 * @param p1 The hole concentration when Fermi level is at trap level (cm^-3)
 * @return The SRH recombination rate (cm^-3/s)
 */
double srh_recombination(double n, double p, double n_i, double tau_n, double tau_p, double n1, double p1);

/**
 * @brief Computes the Auger recombination rate.
 *
 * This function computes the Auger recombination rate,
 * which is significant in heavily doped semiconductors.
 *
 * @param n The electron concentration (cm^-3)
 * @param p The hole concentration (cm^-3)
 * @param n_i The intrinsic carrier concentration (cm^-3)
 * @param C_n The electron Auger coefficient (cm^6/s)
 * @param C_p The hole Auger coefficient (cm^6/s)
 * @return The Auger recombination rate (cm^-3/s)
 */
double auger_recombination(double n, double p, double n_i, double C_n, double C_p);

/**
 * @brief Computes the radiative recombination rate.
 *
 * This function computes the radiative recombination rate,
 * which is the dominant recombination mechanism in direct bandgap semiconductors.
 *
 * @param n The electron concentration (cm^-3)
 * @param p The hole concentration (cm^-3)
 * @param n_i The intrinsic carrier concentration (cm^-3)
 * @param B The radiative recombination coefficient (cm^3/s)
 * @return The radiative recombination rate (cm^-3/s)
 */
double radiative_recombination(double n, double p, double n_i, double B);

/**
 * @brief Computes the total recombination rate.
 *
 * This function computes the total recombination rate,
 * which is the sum of SRH, Auger, and radiative recombination rates.
 *
 * @param n The electron concentration (cm^-3)
 * @param p The hole concentration (cm^-3)
 * @param n_i The intrinsic carrier concentration (cm^-3)
 * @param tau_n The electron lifetime (s)
 * @param tau_p The hole lifetime (s)
 * @param n1 The electron concentration when Fermi level is at trap level (cm^-3)
 * @param p1 The hole concentration when Fermi level is at trap level (cm^-3)
 * @param C_n The electron Auger coefficient (cm^6/s)
 * @param C_p The hole Auger coefficient (cm^6/s)
 * @param B The radiative recombination coefficient (cm^3/s)
 * @return The total recombination rate (cm^-3/s)
 */
double total_recombination(double n, double p, double n_i, double tau_n, double tau_p, 
                          double n1, double p1, double C_n, double C_p, double B);

/**
 * @brief Computes the electron diffusion coefficient.
 *
 * This function computes the electron diffusion coefficient
 * using the Einstein relation.
 *
 * @param mu_n The electron mobility (cm^2/V·s)
 * @param temperature The temperature (K)
 * @return The electron diffusion coefficient (cm^2/s)
 */
double electron_diffusion_coefficient(double mu_n, double temperature);

/**
 * @brief Computes the hole diffusion coefficient.
 *
 * This function computes the hole diffusion coefficient
 * using the Einstein relation.
 *
 * @param mu_p The hole mobility (cm^2/V·s)
 * @param temperature The temperature (K)
 * @return The hole diffusion coefficient (cm^2/s)
 */
double hole_diffusion_coefficient(double mu_p, double temperature);

/**
 * @brief Computes the bandgap narrowing due to heavy doping.
 *
 * This function computes the bandgap narrowing due to heavy doping,
 * which is significant in heavily doped semiconductors.
 *
 * @param N_doping The doping concentration (cm^-3)
 * @param temperature The temperature (K)
 * @return The bandgap narrowing (eV)
 */
double bandgap_narrowing(double N_doping, double temperature);

} // namespace CarrierStatistics
