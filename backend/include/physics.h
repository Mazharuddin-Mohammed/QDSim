#pragma once
/**
 * @file physics.h
 * @brief Defines physics functions for quantum simulations.
 *
 * This file contains the declaration of physics functions used in quantum simulations,
 * including functions for computing effective mass, potential, dielectric constant,
 * charge density, and capacitance.
 *
 * Physical units (UPDATED FOR CONSISTENCY):
 * - Coordinates: meters (m) - SI units for consistency with Schrödinger solver
 * - Effective mass: kilograms (kg) - SI units (use m* × m₀ where m₀ = 9.109e-31 kg)
 * - Potential: Joules (J) - SI units (use V_eV × 1.602e-19 for eV to J conversion)
 * - Dielectric constant: relative to vacuum permittivity (epsilon_0) - dimensionless
 * - Charge density: Coulombs per cubic meter (C/m³) - SI units
 * - Capacitance: Farads per square meter (F/m²) - SI units
 *
 * IMPORTANT: All functions should use SI units consistently to match the
 * Schrödinger solver matrix assembly. This ensures energy eigenvalues
 * are returned in Joules and can be converted to eV by dividing by 1.602e-19.
 *
 * Assumptions and limitations:
 * - The functions assume a 2D simulation domain
 * - The quantum dot is assumed to be circular
 * - The potential includes band offsets and electrostatic potential
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "materials.h"
#include "fe_interpolator.h"
#include <Eigen/Dense>
#include <string>

/**
 * @namespace Physics
 * @brief Namespace for physics functions used in quantum simulations.
 *
 * This namespace contains functions for computing physical quantities
 * used in quantum simulations, such as effective mass, potential,
 * dielectric constant, charge density, and capacitance.
 */
namespace Physics {

/**
 * @brief Computes the effective mass at a given position.
 *
 * This function computes the effective mass at a given position based on
 * the material properties of the quantum dot and the surrounding matrix.
 * The effective mass is position-dependent, with different values inside
 * and outside the quantum dot.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param qd_mat The material properties of the quantum dot
 * @param matrix_mat The material properties of the surrounding matrix
 * @param R The radius of the quantum dot in nanometers (nm)
 * @return The effective mass at the given position relative to electron mass (m_0)
 */
double effective_mass(double x, double y, const Materials::Material& qd_mat,
                      const Materials::Material& matrix_mat, double R);

/**
 * @brief Computes the potential at a given position.
 *
 * This function computes the potential at a given position based on
 * the material properties of the quantum dot and the surrounding matrix,
 * as well as the electrostatic potential from the Poisson equation.
 * The potential includes band offsets and electrostatic potential.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param qd_mat The material properties of the quantum dot
 * @param matrix_mat The material properties of the surrounding matrix
 * @param R The radius of the quantum dot in nanometers (nm)
 * @param type The type of potential ("square", "harmonic", "gaussian", etc.)
 * @param phi The electrostatic potential from the Poisson equation in volts (V)
 * @param interpolator Optional interpolator for the electrostatic potential
 * @return The potential at the given position in electron volts (eV)
 */
double potential(double x, double y, const Materials::Material& qd_mat,
                 const Materials::Material& matrix_mat, double R, const std::string& type,
                 const Eigen::VectorXd& phi, const FEInterpolator* interpolator = nullptr);

/**
 * @brief Computes the relative permittivity at a given position.
 *
 * This function computes the relative permittivity at a given position based on
 * the material properties of the p-type and n-type regions.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param p_mat The material properties of the p-type region
 * @param n_mat The material properties of the n-type region
 * @return The relative permittivity at the given position
 */
double epsilon_r(double x, double y, const Materials::Material& p_mat,
                 const Materials::Material& n_mat);

/**
 * @brief Computes the charge density at a given position.
 *
 * This function computes the charge density at a given position based on
 * the doping concentrations and depletion width.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param N_A The acceptor doping concentration in per cubic nanometer (1/nm^3)
 * @param N_D The donor doping concentration in per cubic nanometer (1/nm^3)
 * @param depletion_width The depletion width in nanometers (nm)
 * @return The charge density at the given position in elementary charges per cubic nanometer (e/nm^3)
 */
// double charge_density(double x, double y, double N_A, double N_D, double depletion_width);
double charge_density(double x, double y, const Eigen::VectorXd& n, const Eigen::VectorXd& p,
                     const FEInterpolator* n_interpolator = nullptr,
                     const FEInterpolator* p_interpolator = nullptr);
/**
 * @brief Computes the capacitance at a given position.
 *
 * This function computes the capacitance at a given position based on
 * the gate geometry and dielectric properties.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param eta The gate efficiency factor (dimensionless)
 * @param Lx The width of the domain in nanometers (nm)
 * @param Ly The height of the domain in nanometers (nm)
 * @param d The gate-to-channel distance in nanometers (nm)
 * @return The capacitance at the given position in farads per square nanometer (F/nm^2)
 */
double cap(double x, double y, double eta, double Lx, double Ly, double d);

double electron_concentration(double x, double y, double phi, const Materials::Material& mat);
double hole_concentration(double x, double y, double phi, const Materials::Material& mat);
double mobility_n(double x, double y, const Materials::Material& mat);
double mobility_p(double x, double y, const Materials::Material& mat);

} // namespace Physics