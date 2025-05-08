#pragma once
/**
 * @file bandgap_models.h
 * @brief Defines bandgap models for semiconductor simulations.
 *
 * This file contains the declaration of bandgap models used in
 * semiconductor simulations, including temperature and strain dependence.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Dense>

/**
 * @namespace BandgapModels
 * @brief Namespace for bandgap models used in semiconductor simulations.
 *
 * This namespace contains functions for computing bandgaps with various dependencies,
 * such as temperature, doping, strain, and alloy composition.
 */
namespace BandgapModels {

/**
 * @brief Varshni model for temperature-dependent bandgap.
 *
 * This function computes the temperature-dependent bandgap using the Varshni model:
 * E_g(T) = E_g(0) - αT²/(T+β)
 *
 * @param E_g0 The bandgap at 0K (eV)
 * @param alpha The Varshni alpha parameter (eV/K)
 * @param beta The Varshni beta parameter (K)
 * @param temperature The temperature (K)
 * @return The temperature-dependent bandgap (eV)
 */
double varshni_model(double E_g0, double alpha, double beta, double temperature);

/**
 * @brief Bose-Einstein model for temperature-dependent bandgap.
 *
 * This function computes the temperature-dependent bandgap using the Bose-Einstein model:
 * E_g(T) = E_g(0) - α_B[1 + 2/(exp(θ_B/T) - 1)]
 *
 * @param E_g0 The bandgap at 0K (eV)
 * @param alpha_B The Bose-Einstein alpha parameter (eV)
 * @param theta_B The Bose-Einstein theta parameter (K)
 * @param temperature The temperature (K)
 * @return The temperature-dependent bandgap (eV)
 */
double bose_einstein_model(double E_g0, double alpha_B, double theta_B, double temperature);

/**
 * @brief O'Donnell-Chen model for temperature-dependent bandgap.
 *
 * This function computes the temperature-dependent bandgap using the O'Donnell-Chen model:
 * E_g(T) = E_g(0) - S⟨ℏω⟩[coth(⟨ℏω⟩/2kT) - 1]
 *
 * @param E_g0 The bandgap at 0K (eV)
 * @param S The dimensionless coupling constant
 * @param hbar_omega The average phonon energy (eV)
 * @param temperature The temperature (K)
 * @return The temperature-dependent bandgap (eV)
 */
double odonnell_chen_model(double E_g0, double S, double hbar_omega, double temperature);

/**
 * @brief Pässler model for temperature-dependent bandgap.
 *
 * This function computes the temperature-dependent bandgap using the Pässler model:
 * E_g(T) = E_g(0) - α_P T^p/(θ_P^(p-1) (θ_P + T))
 *
 * @param E_g0 The bandgap at 0K (eV)
 * @param alpha_P The Pässler alpha parameter (eV)
 * @param theta_P The Pässler theta parameter (K)
 * @param p The Pässler p parameter (dimensionless)
 * @param temperature The temperature (K)
 * @return The temperature-dependent bandgap (eV)
 */
double passler_model(double E_g0, double alpha_P, double theta_P, double p, double temperature);

/**
 * @brief Bandgap narrowing due to heavy doping (Slotboom model).
 *
 * This function computes the bandgap narrowing due to heavy doping
 * using the Slotboom model.
 *
 * @param doping The doping concentration (cm^-3)
 * @return The bandgap narrowing (eV)
 */
double bandgap_narrowing_slotboom(double doping);

/**
 * @brief Bandgap narrowing due to heavy doping (del Alamo model).
 *
 * This function computes the bandgap narrowing due to heavy doping
 * using the del Alamo model.
 *
 * @param doping The doping concentration (cm^-3)
 * @return The bandgap narrowing (eV)
 */
double bandgap_narrowing_del_alamo(double doping);

/**
 * @brief Bandgap narrowing due to heavy doping (Jain-Roulston model).
 *
 * This function computes the bandgap narrowing due to heavy doping
 * using the Jain-Roulston model.
 *
 * @param doping The doping concentration (cm^-3)
 * @param A_n The n-type coefficient (eV)
 * @param A_p The p-type coefficient (eV)
 * @param is_n_type Whether the doping is n-type (true) or p-type (false)
 * @return The bandgap narrowing (eV)
 */
double bandgap_narrowing_jain_roulston(double doping, double A_n, double A_p, bool is_n_type);

/**
 * @brief Bandgap narrowing due to heavy doping (Schenk model).
 *
 * This function computes the bandgap narrowing due to heavy doping
 * using the Schenk model.
 *
 * @param doping The doping concentration (cm^-3)
 * @param temperature The temperature (K)
 * @param is_n_type Whether the doping is n-type (true) or p-type (false)
 * @return The bandgap narrowing (eV)
 */
double bandgap_narrowing_schenk(double doping, double temperature, bool is_n_type);

/**
 * @brief Bandgap of Si1-xGex alloy.
 *
 * This function computes the bandgap of Si1-xGex alloy
 * as a function of composition and temperature.
 *
 * @param x_ge The germanium fraction (0 to 1)
 * @param temperature The temperature (K)
 * @return The bandgap of the alloy (eV)
 */
double bandgap_sige(double x_ge, double temperature);

/**
 * @brief Bandgap of AlxGa1-xAs alloy.
 *
 * This function computes the bandgap of AlxGa1-xAs alloy
 * as a function of composition and temperature.
 *
 * @param x_al The aluminum fraction (0 to 1)
 * @param temperature The temperature (K)
 * @return The bandgap of the alloy (eV)
 */
double bandgap_algaas(double x_al, double temperature);

/**
 * @brief Bandgap of InxGa1-xAs alloy.
 *
 * This function computes the bandgap of InxGa1-xAs alloy
 * as a function of composition and temperature.
 *
 * @param x_in The indium fraction (0 to 1)
 * @param temperature The temperature (K)
 * @return The bandgap of the alloy (eV)
 */
double bandgap_ingaas(double x_in, double temperature);

/**
 * @brief Bandgap of InxGa1-xN alloy.
 *
 * This function computes the bandgap of InxGa1-xN alloy
 * as a function of composition and temperature.
 *
 * @param x_in The indium fraction (0 to 1)
 * @param temperature The temperature (K)
 * @return The bandgap of the alloy (eV)
 */
double bandgap_ingan(double x_in, double temperature);

/**
 * @brief Bandgap of AlxGa1-xN alloy.
 *
 * This function computes the bandgap of AlxGa1-xN alloy
 * as a function of composition and temperature.
 *
 * @param x_al The aluminum fraction (0 to 1)
 * @param temperature The temperature (K)
 * @return The bandgap of the alloy (eV)
 */
double bandgap_algan(double x_al, double temperature);

/**
 * @brief Bandgap shift due to strain.
 *
 * This function computes the bandgap shift due to strain
 * using the deformation potential theory.
 *
 * @param strain The strain tensor
 * @param a_c The conduction band deformation potential (eV)
 * @param a_v The valence band hydrostatic deformation potential (eV)
 * @param b The valence band biaxial deformation potential (eV)
 * @param d The valence band shear deformation potential (eV)
 * @return The bandgap shift (eV)
 */
double bandgap_shift_strain(const Eigen::Matrix3d& strain, double a_c, double a_v, double b, double d);

/**
 * @brief Bandgap shift due to quantum confinement.
 *
 * This function computes the bandgap shift due to quantum confinement
 * in a quantum well.
 *
 * @param well_width The quantum well width (nm)
 * @param m_e The electron effective mass (m_0)
 * @param m_h The hole effective mass (m_0)
 * @return The bandgap shift (eV)
 */
double bandgap_shift_quantum_confinement(double well_width, double m_e, double m_h);

/**
 * @brief Bandgap shift due to hydrostatic pressure.
 *
 * This function computes the bandgap shift due to hydrostatic pressure.
 *
 * @param pressure The hydrostatic pressure (GPa)
 * @param dE_g_dP The pressure coefficient of the bandgap (eV/GPa)
 * @return The bandgap shift (eV)
 */
double bandgap_shift_pressure(double pressure, double dE_g_dP);

/**
 * @brief Bandgap shift due to electric field (Franz-Keldysh effect).
 *
 * This function computes the bandgap shift due to an electric field
 * (Franz-Keldysh effect).
 *
 * @param electric_field The electric field (V/cm)
 * @param m_r The reduced effective mass (m_0)
 * @param E_g The bandgap (eV)
 * @return The bandgap shift (eV)
 */
double bandgap_shift_electric_field(double electric_field, double m_r, double E_g);

} // namespace BandgapModels
