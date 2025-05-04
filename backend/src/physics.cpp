/**
 * @file physics.cpp
 * @brief Implementation of physics functions for quantum simulations.
 *
 * This file contains the implementation of physics functions used in quantum simulations,
 * including functions for computing effective mass, potential, dielectric constant,
 * charge density, and capacitance.
 *
 * The functions implement realistic physical models for quantum dot simulations,
 * with appropriate scaling and limits to ensure physically meaningful results.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "physics.h"
#include <cmath>

namespace Physics {

/**
 * @brief Computes the effective mass at a given position.
 *
 * This function computes the effective mass at a given position based on
 * the material properties of the quantum dot and the surrounding matrix.
 * The effective mass is position-dependent, with different values inside
 * and outside the quantum dot.
 *
 * The quantum dot is modeled as a circle with radius R centered at the origin.
 * Points inside the circle have the effective mass of the quantum dot material,
 * while points outside have the effective mass of the matrix material.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param qd_mat The material properties of the quantum dot
 * @param matrix_mat The material properties of the surrounding matrix
 * @param R The radius of the quantum dot in nanometers (nm)
 * @return The effective mass at the given position in kilograms (kg)
 */
double effective_mass(double x, double y, const Materials::Material& qd_mat,
                      const Materials::Material& matrix_mat, double R) {
    // Check if the point is inside the quantum dot (circular with radius R)
    return (x * x + y * y <= R * R) ? qd_mat.m_e : matrix_mat.m_e;
}

/**
 * @brief Computes the potential at a given position.
 *
 * This function computes the potential at a given position based on
 * the material properties of the quantum dot and the surrounding matrix,
 * as well as the electrostatic potential from the Poisson equation.
 *
 * The potential includes:
 * 1. The quantum dot potential, which can be either a square well or a Gaussian well
 * 2. The electrostatic potential, which is interpolated from the mesh using the FEInterpolator
 *
 * The function ensures that the potential is within a realistic range by limiting
 * the maximum potential to a specified value.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param qd_mat The material properties of the quantum dot
 * @param matrix_mat The material properties of the surrounding matrix
 * @param R The radius of the quantum dot in nanometers (nm)
 * @param type The type of potential ("square" or "gaussian")
 * @param phi The electrostatic potential from the Poisson equation in volts (V)
 * @param interpolator Optional interpolator for the electrostatic potential
 * @return The potential at the given position in joules (J)
 */
double potential(double x, double y, const Materials::Material& qd_mat,
                 const Materials::Material& matrix_mat, double R, const std::string& type,
                 const Eigen::VectorXd& phi, const FEInterpolator* interpolator) {
    // Constants
    const double e_charge = 1.602e-19; // Electron charge in C
    const double max_potential_eV = 100.0; // Maximum allowed potential in eV

    // Calculate quantum dot potential
    double V_qd_eV = 0.0;
    if (type == "square") {
        if (x * x + y * y <= R * R) {
            // Inside the quantum dot - use negative potential (well)
            V_qd_eV = -std::min(qd_mat.Delta_E_c, max_potential_eV);
        } else {
            // Outside the quantum dot - zero potential
            V_qd_eV = 0.0;
        }
    } else if (type == "gaussian") {
        // Gaussian well with smooth boundaries
        // Negative sign for well (attractive potential)
        double max_depth = std::min(qd_mat.Delta_E_c, max_potential_eV);
        V_qd_eV = -max_depth * std::exp(-(x * x + y * y) / (2 * R * R));
    }

    // Convert QD potential from eV to J
    double V_qd = V_qd_eV * e_charge;

    // Interpolate electrostatic potential at (x,y) using finite element interpolation
    double V_elec_eV = 0.0;
    if (phi.size() > 0 && interpolator != nullptr) {
        try {
            // Use the FEInterpolator to get the potential at (x,y)
            double V_elec_interp = interpolator->interpolate(x, y, phi);

            // Convert from V to eV (multiply by elementary charge)
            V_elec_eV = V_elec_interp;

            // Limit the electrostatic potential to a realistic range
            if (std::abs(V_elec_eV) > max_potential_eV) {
                V_elec_eV = max_potential_eV * (V_elec_eV >= 0 ? 1 : -1);
            }
        } catch (const std::exception& e) {
            // If interpolation fails, use a fallback approach
            if (phi.size() > 0) {
                // Use the first value as a fallback (not ideal but better than crashing)
                V_elec_eV = std::min(std::abs(phi[0]), max_potential_eV) * (phi[0] >= 0 ? 1 : -1);
            }
        }
    } else if (phi.size() > 0) {
        // Fallback to simple approach if interpolator is not available
        V_elec_eV = std::min(std::abs(phi[0]), max_potential_eV) * (phi[0] >= 0 ? 1 : -1);
    }
    double V_elec = V_elec_eV * e_charge;

    // Return the combined potential (QD + electrostatic)
    return V_qd + V_elec;
}

/**
 * @brief Computes the relative permittivity at a given position.
 *
 * This function computes the relative permittivity at a given position based on
 * the material properties of the p-type and n-type regions. The function assumes
 * that the p-type region is at x < 0 and the n-type region is at x > 0.
 *
 * The relative permittivity is used in the Poisson equation to compute the
 * electrostatic potential.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param p_mat The material properties of the p-type region
 * @param n_mat The material properties of the n-type region
 * @return The relative permittivity at the given position (dimensionless)
 */
double epsilon_r(double x, double y, const Materials::Material& p_mat,
                 const Materials::Material& n_mat) {
    // Assume p-side at x < 0, n-side at x > 0
    return (x < 0) ? p_mat.epsilon_r : n_mat.epsilon_r;
}

/**
 * @brief Computes the charge density at a given position.
 *
 * This function computes the charge density at a given position based on
 * the doping concentrations and depletion width. The function models a
 * p-n junction with the p-side at x < 0 and the n-side at x > 0.
 *
 * The charge density is used in the Poisson equation to compute the
 * electrostatic potential.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param N_A The acceptor doping concentration in per cubic nanometer (1/nm^3)
 * @param N_D The donor doping concentration in per cubic nanometer (1/nm^3)
 * @param W_d The total depletion width in nanometers (nm)
 * @return The charge density at the given position in coulombs per cubic nanometer (C/nm^3)
 */
/*
double charge_density(double x, double y, double N_A, double N_D, double W_d) {
    const double q = 1.602e-19; // Elementary charge in coulombs (C)

    // Calculate the depletion widths on the p and n sides
    double x_p = W_d * N_D / (N_A + N_D); // p-side depletion width
    double x_n = W_d * N_A / (N_A + N_D); // n-side depletion width

    // Check if the point is within the depletion region
    if (x >= -x_p && x <= x_n) {
        // Return the charge density based on the doping
        return (x < 0) ? -q * N_A : q * N_D;
    }

    // Outside the depletion region, the charge density is zero
    return 0.0;
}
*/
double charge_density(double x, double y, const Eigen::VectorXd& n, const Eigen::VectorXd& p,
                       const FEInterpolator* n_interpolator, const FEInterpolator* p_interpolator) {
    const double q = 1.602e-19; // Elementary charge in coulombs (C)

    double n_val = 0.0;
    double p_val = 0.0;

    // Interpolate electron concentration if interpolator is available
    if (n.size() > 0 && n_interpolator != nullptr) {
        try {
            n_val = n_interpolator->interpolate(x, y, n);
        } catch (const std::exception& e) {
            // If interpolation fails, use the first value as a fallback
            if (n.size() > 0) {
                n_val = n[0];
            }
        }
    } else if (n.size() > 0) {
        // Fallback to first value if interpolator is not available
        n_val = n[0];
    }

    // Interpolate hole concentration if interpolator is available
    if (p.size() > 0 && p_interpolator != nullptr) {
        try {
            p_val = p_interpolator->interpolate(x, y, p);
        } catch (const std::exception& e) {
            // If interpolation fails, use the first value as a fallback
            if (p.size() > 0) {
                p_val = p[0];
            }
        }
    } else if (p.size() > 0) {
        // Fallback to first value if interpolator is not available
        p_val = p[0];
    }

    // Return the charge density (p - n) * q
    return q * (p_val - n_val);
}

/**
 * @brief Computes the capacitance at a given position.
 *
 * This function computes the capacitance at a given position based on
 * the gate geometry and dielectric properties. The capacitance is modeled
 * as a function of the distance from the edge of the domain, with a
 * quadratic increase near the edges.
 *
 * The capacitance is used in the Schrödinger equation to model the
 * coupling between the quantum dot and the gates.
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @param eta The gate efficiency factor (dimensionless)
 * @param Lx The width of the domain in nanometers (nm)
 * @param Ly The height of the domain in nanometers (nm)
 * @param d The gate-to-channel distance in nanometers (nm)
 * @return The capacitance at the given position in farads per square nanometer (F/nm^2)
 */
double cap(double x, double y, double eta, double Lx, double Ly, double d) {
    // Calculate the capacitance contribution from the x-direction
    // The capacitance increases quadratically near the edges of the domain
    double eta_x = (std::abs(x) > Lx / 2 - d) ? eta * std::pow((std::abs(x) - Lx / 2) / d, 2) : 0.0;

    // Calculate the capacitance contribution from the y-direction
    double eta_y = (std::abs(y) > Ly / 2 - d) ? eta * std::pow((std::abs(y) - Ly / 2) / d, 2) : 0.0;

    // Return the maximum of the two contributions
    return std::max(eta_x, eta_y);
}

double electron_concentration(double x, double y, double phi, const Materials::Material& mat) {
    const double kT = 0.0259; // eV at 300K
    const double q = 1.602e-19; // Elementary charge in C
    double E_F = 0.0; // Simplified; assume constant Fermi level
    return mat.N_c * std::exp((-q * phi - E_F) / kT);
}

double hole_concentration(double x, double y, double phi, const Materials::Material& mat) {
    const double kT = 0.0259; // eV at 300K
    const double q = 1.602e-19; // Elementary charge in C
    double E_F = 0.0; // Simplified; assume constant Fermi level
    return mat.N_v * std::exp((q * phi + mat.E_g - E_F) / kT);
}

double mobility_n(double x, double y, const Materials::Material& mat) {
    return mat.mu_n;
}

double mobility_p(double x, double y, const Materials::Material& mat) {
    return mat.mu_p;
}

} // namespace Physics