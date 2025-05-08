/**
 * @file pn_junction.h
 * @brief Header file for the PNJunction class.
 *
 * This file contains the declaration of the PNJunction class, which implements
 * a physically accurate model of a P-N junction with proper calculation of the
 * potential from charge distributions.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#ifndef PN_JUNCTION_H
#define PN_JUNCTION_H

#include "mesh.h"
#include "poisson.h"
#include "simple_mesh.h"
#include "simple_interpolator.h"
#include <Eigen/Dense>
#include <functional>
#include <memory>

/**
 * @brief Class for modeling P-N junctions with physically accurate potentials.
 *
 * This class implements a physically accurate model of a P-N junction with
 * proper calculation of the potential from charge distributions. It uses the
 * PoissonSolver class to solve the Poisson equation for the electrostatic
 * potential.
 */
class PNJunction {
public:
    /**
     * @brief Constructs a new PNJunction object.
     *
     * @param mesh The mesh to use for the simulation
     * @param epsilon_r The relative permittivity of the semiconductor
     * @param N_A The acceptor concentration (m^-3)
     * @param N_D The donor concentration (m^-3)
     * @param T The temperature (K)
     * @param junction_position The position of the junction (m)
     * @param V_r The reverse bias voltage (V)
     */
    PNJunction(Mesh& mesh, double epsilon_r, double N_A, double N_D, double T,
               double junction_position, double V_r);

    /**
     * @brief Calculates the built-in potential of the P-N junction.
     *
     * @return The built-in potential (V)
     */
    double calculate_built_in_potential() const;

    /**
     * @brief Calculates the depletion width of the P-N junction.
     *
     * @return The depletion width (m)
     */
    double calculate_depletion_width() const;

    /**
     * @brief Calculates the intrinsic carrier concentration.
     *
     * @return The intrinsic carrier concentration (m^-3)
     */
    double calculate_intrinsic_carrier_concentration() const;

    /**
     * @brief Solves the Poisson equation for the electrostatic potential.
     *
     * This method solves the Poisson equation for the electrostatic potential
     * using the PoissonSolver class. It calculates the charge density based on
     * the doping concentrations and carrier concentrations.
     */
    void solve();

    /**
     * @brief Gets the electrostatic potential at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The electrostatic potential (V)
     */
    double get_potential(double x, double y) const;

    /**
     * @brief Gets the electric field at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The electric field vector (V/m)
     */
    Eigen::Vector2d get_electric_field(double x, double y) const;

    /**
     * @brief Gets the electron concentration at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The electron concentration (m^-3)
     */
    double get_electron_concentration(double x, double y) const;

    /**
     * @brief Gets the hole concentration at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The hole concentration (m^-3)
     */
    double get_hole_concentration(double x, double y) const;

    /**
     * @brief Gets the conduction band edge at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The conduction band edge (eV)
     */
    double get_conduction_band_edge(double x, double y) const;

    /**
     * @brief Gets the valence band edge at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The valence band edge (eV)
     */
    double get_valence_band_edge(double x, double y) const;

    /**
     * @brief Gets the quasi-Fermi level for electrons at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The quasi-Fermi level for electrons (eV)
     */
    double get_quasi_fermi_level_electrons(double x, double y) const;

    /**
     * @brief Gets the quasi-Fermi level for holes at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The quasi-Fermi level for holes (eV)
     */
    double get_quasi_fermi_level_holes(double x, double y) const;

    /**
     * @brief Updates the reverse bias voltage.
     *
     * @param V_r The reverse bias voltage (V)
     */
    void update_bias(double V_r);

    /**
     * @brief Updates the doping concentrations.
     *
     * @param N_A The acceptor concentration (m^-3)
     * @param N_D The donor concentration (m^-3)
     */
    void update_doping(double N_A, double N_D);

    // Physical constants
    static constexpr double e_charge = 1.602e-19;  // Elementary charge (C)
    static constexpr double k_B = 1.381e-23;       // Boltzmann constant (J/K)
    static constexpr double epsilon_0 = 8.854e-12; // Vacuum permittivity (F/m)
    static constexpr double h = 6.626e-34;         // Planck constant (J·s)
    static constexpr double m_e = 9.109e-31;       // Electron mass (kg)

    // Accessors for junction parameters
    double get_V_bi() const { return V_bi; }
    double get_V_r() const { return V_r; }
    double get_V_total() const { return V_total; }
    double get_W() const { return W; }
    double get_W_p() const { return W_p; }
    double get_W_n() const { return W_n; }
    double get_N_A() const { return N_A; }
    double get_N_D() const { return N_D; }
    double get_n_i() const { return n_i; }
    double get_junction_position() const { return junction_position; }

private:
    Mesh& mesh;                  // The mesh for the simulation
    double epsilon_r;            // Relative permittivity
    double N_A;                  // Acceptor concentration (m^-3)
    double N_D;                  // Donor concentration (m^-3)
    double T;                    // Temperature (K)
    double junction_position;    // Position of the junction (m)
    double V_r;                  // Reverse bias voltage (V)
    double V_bi;                 // Built-in potential (V)
    double V_total;              // Total potential across the junction (V)
    double W;                    // Depletion width (m)
    double W_p;                  // P-side depletion width (m)
    double W_n;                  // N-side depletion width (m)
    double n_i;                  // Intrinsic carrier concentration (m^-3)
    double E_g;                  // Band gap (eV)
    double chi;                  // Electron affinity (eV)
    double E_F_p;                // Quasi-Fermi level in p-region (eV)
    double E_F_n;                // Quasi-Fermi level in n-region (eV)

    std::unique_ptr<PoissonSolver> poisson_solver; // Poisson solver
    Eigen::VectorXd n;           // Electron concentration at each node
    Eigen::VectorXd p;           // Hole concentration at each node

    // Cached objects for efficient interpolation
    mutable std::unique_ptr<SimpleMesh> simple_mesh;
    mutable std::unique_ptr<SimpleInterpolator> interpolator;

    /**
     * @brief Calculates the charge density at each node.
     *
     * This method calculates the charge density at each node based on the
     * doping concentrations and carrier concentrations.
     *
     * @return The charge density at each node (C/m^3)
     */
    Eigen::VectorXd calculate_charge_density() const;

    /**
     * @brief Updates the carrier concentrations based on the current potential.
     *
     * This method updates the electron and hole concentrations at each node
     * based on the current electrostatic potential.
     */
    void update_carrier_concentrations();

    /**
     * @brief Charge density function for the Poisson solver.
     *
     * This function calculates the charge density at a given position based on
     * the doping concentrations and carrier concentrations.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @param n The electron concentration at each node
     * @param p The hole concentration at each node
     * @return The charge density (C/m^3)
     */
    static double charge_density_function(double x, double y, const Eigen::VectorXd& n, const Eigen::VectorXd& p);

    /**
     * @brief Permittivity function for the Poisson solver.
     *
     * This function returns the relative permittivity at a given position.
     *
     * @param x The x-coordinate (m)
     * @param y The y-coordinate (m)
     * @return The relative permittivity
     */
    static double permittivity_function(double x, double y);
};

#endif // PN_JUNCTION_H
