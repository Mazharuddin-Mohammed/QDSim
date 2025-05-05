/**
 * @file materials.cpp
 * @brief Implementation of the MaterialDatabase class.
 *
 * This file contains the implementation of the MaterialDatabase class,
 * which provides a database of semiconductor material properties for quantum simulations.
 * The database includes properties such as effective masses, bandgaps, band offsets,
 * and dielectric constants for various semiconductor materials.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "materials.h"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>

namespace Materials {

/**
 * @brief Constructs a new MaterialDatabase object.
 *
 * This constructor initializes the database with default values for
 * common semiconductor materials by calling the initialize_database method.
 */
MaterialDatabase::MaterialDatabase() {
    initialize_database();
}

/**
 * @brief Initializes the database with default values.
 *
 * This private method initializes the database with default values for
 * common semiconductor materials. The properties included are:
 * - Effective masses (m_e, m_h, m_lh, m_hh, m_so) relative to electron mass (m_0)
 * - Bandgap (E_g) in eV at 300K
 * - Band offsets (Delta_E_c, Delta_E_v) in eV
 * - Dielectric constant (epsilon_r) relative to vacuum
 * - Mobilities (mu_n, mu_p) in m^2/V·s at 300K
 * - Effective densities of states (N_c, N_v) in m^-3 at 300K
 * - Lattice constant in nm
 * - Spin-orbit splitting energy in eV
 * - Deformation potentials in eV
 * - Elastic constants in GPa
 * - Varshni parameters for temperature dependence of bandgap
 * - Luttinger parameters for k·p calculations
 * - Kane parameter in eV·nm
 * - Bowing parameters for alloys
 */
void MaterialDatabase::initialize_database() {
    // GaAs - Gallium Arsenide
    Material gaas;
    gaas.m_e = 0.067;                // Electron effective mass
    gaas.m_h = 0.45;                 // Average hole effective mass
    gaas.m_lh = 0.082;               // Light hole effective mass
    gaas.m_hh = 0.51;                // Heavy hole effective mass
    gaas.m_so = 0.15;                // Split-off hole effective mass
    gaas.E_g = 1.424;                // Bandgap at 300K
    gaas.Delta_E_c = 0.7;            // Conduction band offset
    gaas.Delta_E_v = 0.3;            // Valence band offset
    gaas.epsilon_r = 12.9;           // Dielectric constant
    gaas.mu_n = 0.85;                // Electron mobility
    gaas.mu_p = 0.04;                // Hole mobility
    gaas.N_c = 4.7e23;               // Effective density of states, conduction band
    gaas.N_v = 9.0e24;               // Effective density of states, valence band
    gaas.lattice_constant = 0.56533; // Lattice constant
    gaas.spin_orbit_splitting = 0.34; // Spin-orbit splitting energy
    gaas.deformation_potential_c = -7.17; // Deformation potential for conduction band
    gaas.deformation_potential_v = 1.16;  // Deformation potential for valence band
    gaas.elastic_c11 = 122.1;        // Elastic constant C11
    gaas.elastic_c12 = 56.6;         // Elastic constant C12
    gaas.elastic_c44 = 60.0;         // Elastic constant C44
    gaas.varshni_alpha = 5.405e-4;   // Varshni parameter alpha
    gaas.varshni_beta = 204;         // Varshni parameter beta
    gaas.luttinger_gamma1 = 6.98;    // Luttinger parameter gamma1
    gaas.luttinger_gamma2 = 2.06;    // Luttinger parameter gamma2
    gaas.luttinger_gamma3 = 2.93;    // Luttinger parameter gamma3
    gaas.kane_parameter = 28.8;      // Kane parameter
    gaas.bowing_bandgap = 0.0;       // Bowing parameter for bandgap
    gaas.bowing_effective_mass = 0.0; // Bowing parameter for effective mass
    gaas.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["GaAs"] = gaas;

    // AlAs - Aluminum Arsenide
    Material alas;
    alas.m_e = 0.15;                 // Electron effective mass
    alas.m_h = 0.76;                 // Average hole effective mass
    alas.m_lh = 0.16;                // Light hole effective mass
    alas.m_hh = 0.79;                // Heavy hole effective mass
    alas.m_so = 0.28;                // Split-off hole effective mass
    alas.E_g = 2.16;                 // Bandgap at 300K
    alas.Delta_E_c = 1.0;            // Conduction band offset
    alas.Delta_E_v = 0.53;           // Valence band offset
    alas.epsilon_r = 10.1;           // Dielectric constant
    alas.mu_n = 0.15;                // Electron mobility
    alas.mu_p = 0.03;                // Hole mobility
    alas.N_c = 1.5e24;               // Effective density of states, conduction band
    alas.N_v = 1.8e25;               // Effective density of states, valence band
    alas.lattice_constant = 0.5661;  // Lattice constant
    alas.spin_orbit_splitting = 0.28; // Spin-orbit splitting energy
    alas.deformation_potential_c = -5.64; // Deformation potential for conduction band
    alas.deformation_potential_v = 2.47;  // Deformation potential for valence band
    alas.elastic_c11 = 125.0;        // Elastic constant C11
    alas.elastic_c12 = 53.4;         // Elastic constant C12
    alas.elastic_c44 = 54.2;         // Elastic constant C44
    alas.varshni_alpha = 3.0e-4;     // Varshni parameter alpha
    alas.varshni_beta = 150;         // Varshni parameter beta
    alas.luttinger_gamma1 = 3.76;    // Luttinger parameter gamma1
    alas.luttinger_gamma2 = 0.82;    // Luttinger parameter gamma2
    alas.luttinger_gamma3 = 1.42;    // Luttinger parameter gamma3
    alas.kane_parameter = 21.1;      // Kane parameter
    alas.bowing_bandgap = 0.0;       // Bowing parameter for bandgap
    alas.bowing_effective_mass = 0.0; // Bowing parameter for effective mass
    alas.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["AlAs"] = alas;

    // InAs - Indium Arsenide
    Material inas;
    inas.m_e = 0.023;                // Electron effective mass
    inas.m_h = 0.41;                 // Average hole effective mass
    inas.m_lh = 0.027;               // Light hole effective mass
    inas.m_hh = 0.41;                // Heavy hole effective mass
    inas.m_so = 0.14;                // Split-off hole effective mass
    inas.E_g = 0.354;                // Bandgap at 300K
    inas.Delta_E_c = 0.7;            // Conduction band offset
    inas.Delta_E_v = 0.27;           // Valence band offset
    inas.epsilon_r = 15.15;          // Dielectric constant
    inas.mu_n = 3.3;                 // Electron mobility
    inas.mu_p = 0.045;               // Hole mobility
    inas.N_c = 8.7e22;               // Effective density of states, conduction band
    inas.N_v = 6.7e24;               // Effective density of states, valence band
    inas.lattice_constant = 0.60583; // Lattice constant
    inas.spin_orbit_splitting = 0.39; // Spin-orbit splitting energy
    inas.deformation_potential_c = -5.08; // Deformation potential for conduction band
    inas.deformation_potential_v = 1.00;  // Deformation potential for valence band
    inas.elastic_c11 = 83.3;         // Elastic constant C11
    inas.elastic_c12 = 45.3;         // Elastic constant C12
    inas.elastic_c44 = 39.6;         // Elastic constant C44
    inas.varshni_alpha = 2.76e-4;    // Varshni parameter alpha
    inas.varshni_beta = 93;          // Varshni parameter beta
    inas.luttinger_gamma1 = 20.0;    // Luttinger parameter gamma1
    inas.luttinger_gamma2 = 8.5;     // Luttinger parameter gamma2
    inas.luttinger_gamma3 = 9.2;     // Luttinger parameter gamma3
    inas.kane_parameter = 81.9;      // Kane parameter
    inas.bowing_bandgap = 0.0;       // Bowing parameter for bandgap
    inas.bowing_effective_mass = 0.0; // Bowing parameter for effective mass
    inas.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["InAs"] = inas;

    // AlGaAs - Aluminum Gallium Arsenide (Al0.3Ga0.7As)
    Material algaas;
    algaas.m_e = 0.09;               // Electron effective mass
    algaas.m_h = 0.51;               // Average hole effective mass
    algaas.m_lh = 0.11;              // Light hole effective mass
    algaas.m_hh = 0.57;              // Heavy hole effective mass
    algaas.m_so = 0.19;              // Split-off hole effective mass
    algaas.E_g = 1.8;                // Bandgap at 300K
    algaas.Delta_E_c = 0.3;          // Conduction band offset
    algaas.Delta_E_v = 0.2;          // Valence band offset
    algaas.epsilon_r = 12.0;         // Dielectric constant
    algaas.mu_n = 0.3;               // Electron mobility
    algaas.mu_p = 0.02;              // Hole mobility
    algaas.N_c = 7.0e23;             // Effective density of states, conduction band
    algaas.N_v = 1.0e25;             // Effective density of states, valence band
    algaas.lattice_constant = 0.5657; // Lattice constant
    algaas.spin_orbit_splitting = 0.32; // Spin-orbit splitting energy
    algaas.deformation_potential_c = -6.7; // Deformation potential for conduction band
    algaas.deformation_potential_v = 1.5;  // Deformation potential for valence band
    algaas.elastic_c11 = 123.0;      // Elastic constant C11
    algaas.elastic_c12 = 55.5;       // Elastic constant C12
    algaas.elastic_c44 = 58.5;       // Elastic constant C44
    algaas.varshni_alpha = 4.6e-4;   // Varshni parameter alpha
    algaas.varshni_beta = 186;       // Varshni parameter beta
    algaas.luttinger_gamma1 = 6.0;   // Luttinger parameter gamma1
    algaas.luttinger_gamma2 = 1.7;   // Luttinger parameter gamma2
    algaas.luttinger_gamma3 = 2.5;   // Luttinger parameter gamma3
    algaas.kane_parameter = 26.5;    // Kane parameter
    algaas.bowing_bandgap = 0.37;    // Bowing parameter for bandgap
    algaas.bowing_effective_mass = 0.05; // Bowing parameter for effective mass
    algaas.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["AlGaAs"] = algaas;

    // InGaAs - Indium Gallium Arsenide (In0.53Ga0.47As)
    Material ingaas;
    ingaas.m_e = 0.041;              // Electron effective mass
    ingaas.m_h = 0.45;               // Average hole effective mass
    ingaas.m_lh = 0.052;             // Light hole effective mass
    ingaas.m_hh = 0.46;              // Heavy hole effective mass
    ingaas.m_so = 0.15;              // Split-off hole effective mass
    ingaas.E_g = 0.74;               // Bandgap at 300K
    ingaas.Delta_E_c = 0.5;          // Conduction band offset
    ingaas.Delta_E_v = 0.25;         // Valence band offset
    ingaas.epsilon_r = 13.9;         // Dielectric constant
    ingaas.mu_n = 1.2;               // Electron mobility
    ingaas.mu_p = 0.03;              // Hole mobility
    ingaas.N_c = 2.1e23;             // Effective density of states, conduction band
    ingaas.N_v = 8.0e24;             // Effective density of states, valence band
    ingaas.lattice_constant = 0.5868; // Lattice constant
    ingaas.spin_orbit_splitting = 0.36; // Spin-orbit splitting energy
    ingaas.deformation_potential_c = -6.0; // Deformation potential for conduction band
    ingaas.deformation_potential_v = 1.1;  // Deformation potential for valence band
    ingaas.elastic_c11 = 102.0;      // Elastic constant C11
    ingaas.elastic_c12 = 51.0;       // Elastic constant C12
    ingaas.elastic_c44 = 47.0;       // Elastic constant C44
    ingaas.varshni_alpha = 4.0e-4;   // Varshni parameter alpha
    ingaas.varshni_beta = 150;       // Varshni parameter beta
    ingaas.luttinger_gamma1 = 13.0;  // Luttinger parameter gamma1
    ingaas.luttinger_gamma2 = 5.0;   // Luttinger parameter gamma2
    ingaas.luttinger_gamma3 = 5.5;   // Luttinger parameter gamma3
    ingaas.kane_parameter = 50.0;    // Kane parameter
    ingaas.bowing_bandgap = 0.477;   // Bowing parameter for bandgap
    ingaas.bowing_effective_mass = 0.0091; // Bowing parameter for effective mass
    ingaas.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["InGaAs"] = ingaas;

    // InP - Indium Phosphide
    Material inp;
    inp.m_e = 0.08;                  // Electron effective mass
    inp.m_h = 0.6;                   // Average hole effective mass
    inp.m_lh = 0.089;                // Light hole effective mass
    inp.m_hh = 0.85;                 // Heavy hole effective mass
    inp.m_so = 0.2;                  // Split-off hole effective mass
    inp.E_g = 1.344;                 // Bandgap at 300K
    inp.Delta_E_c = 0.25;            // Conduction band offset
    inp.Delta_E_v = 0.1;             // Valence band offset
    inp.epsilon_r = 12.5;            // Dielectric constant
    inp.mu_n = 0.46;                 // Electron mobility
    inp.mu_p = 0.015;                // Hole mobility
    inp.N_c = 5.7e23;                // Effective density of states, conduction band
    inp.N_v = 1.1e25;                // Effective density of states, valence band
    inp.lattice_constant = 0.58687;  // Lattice constant
    inp.spin_orbit_splitting = 0.11; // Spin-orbit splitting energy
    inp.deformation_potential_c = -6.0; // Deformation potential for conduction band
    inp.deformation_potential_v = 1.27; // Deformation potential for valence band
    inp.elastic_c11 = 101.1;         // Elastic constant C11
    inp.elastic_c12 = 56.1;          // Elastic constant C12
    inp.elastic_c44 = 45.6;          // Elastic constant C44
    inp.varshni_alpha = 4.9e-4;      // Varshni parameter alpha
    inp.varshni_beta = 327;          // Varshni parameter beta
    inp.luttinger_gamma1 = 5.08;     // Luttinger parameter gamma1
    inp.luttinger_gamma2 = 1.6;      // Luttinger parameter gamma2
    inp.luttinger_gamma3 = 2.1;      // Luttinger parameter gamma3
    inp.kane_parameter = 20.0;       // Kane parameter
    inp.bowing_bandgap = 0.0;        // Bowing parameter for bandgap
    inp.bowing_effective_mass = 0.0; // Bowing parameter for effective mass
    inp.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["InP"] = inp;

    // GaN - Gallium Nitride (Wurtzite)
    Material gan;
    gan.m_e = 0.2;                   // Electron effective mass
    gan.m_h = 1.4;                   // Average hole effective mass
    gan.m_lh = 0.3;                  // Light hole effective mass
    gan.m_hh = 1.4;                  // Heavy hole effective mass
    gan.m_so = 0.6;                  // Split-off hole effective mass
    gan.E_g = 3.39;                  // Bandgap at 300K
    gan.Delta_E_c = 2.1;             // Conduction band offset
    gan.Delta_E_v = 0.7;             // Valence band offset
    gan.epsilon_r = 8.9;             // Dielectric constant
    gan.mu_n = 0.1;                  // Electron mobility
    gan.mu_p = 0.002;                // Hole mobility
    gan.N_c = 2.3e24;                // Effective density of states, conduction band
    gan.N_v = 4.6e25;                // Effective density of states, valence band
    gan.lattice_constant = 0.3189;   // Lattice constant a (c = 0.5185 nm)
    gan.spin_orbit_splitting = 0.017; // Spin-orbit splitting energy
    gan.deformation_potential_c = -9.5; // Deformation potential for conduction band
    gan.deformation_potential_v = 1.7;  // Deformation potential for valence band
    gan.elastic_c11 = 390.0;         // Elastic constant C11
    gan.elastic_c12 = 145.0;         // Elastic constant C12
    gan.elastic_c44 = 105.0;         // Elastic constant C44
    gan.varshni_alpha = 7.7e-4;      // Varshni parameter alpha
    gan.varshni_beta = 600;          // Varshni parameter beta
    gan.luttinger_gamma1 = 2.7;      // Luttinger parameter gamma1
    gan.luttinger_gamma2 = 0.7;      // Luttinger parameter gamma2
    gan.luttinger_gamma3 = 1.1;      // Luttinger parameter gamma3
    gan.kane_parameter = 16.0;       // Kane parameter
    gan.bowing_bandgap = 0.0;        // Bowing parameter for bandgap
    gan.bowing_effective_mass = 0.0; // Bowing parameter for effective mass
    gan.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["GaN"] = gan;

    // Si - Silicon
    Material si;
    si.m_e = 0.26;                   // Electron effective mass (conductivity mass)
    si.m_h = 0.49;                   // Average hole effective mass
    si.m_lh = 0.16;                  // Light hole effective mass
    si.m_hh = 0.49;                  // Heavy hole effective mass
    si.m_so = 0.29;                  // Split-off hole effective mass
    si.E_g = 1.12;                   // Bandgap at 300K
    si.Delta_E_c = 0.5;              // Conduction band offset
    si.Delta_E_v = 0.05;             // Valence band offset
    si.epsilon_r = 11.7;             // Dielectric constant
    si.mu_n = 0.14;                  // Electron mobility
    si.mu_p = 0.045;                 // Hole mobility
    si.N_c = 2.8e25;                 // Effective density of states, conduction band
    si.N_v = 1.04e25;                // Effective density of states, valence band
    si.lattice_constant = 0.5431;    // Lattice constant
    si.spin_orbit_splitting = 0.044; // Spin-orbit splitting energy
    si.deformation_potential_c = -10.5; // Deformation potential for conduction band
    si.deformation_potential_v = 2.3;   // Deformation potential for valence band
    si.elastic_c11 = 165.7;          // Elastic constant C11
    si.elastic_c12 = 63.9;           // Elastic constant C12
    si.elastic_c44 = 79.6;           // Elastic constant C44
    si.varshni_alpha = 4.73e-4;      // Varshni parameter alpha
    si.varshni_beta = 636;           // Varshni parameter beta
    si.luttinger_gamma1 = 4.285;     // Luttinger parameter gamma1
    si.luttinger_gamma2 = 0.339;     // Luttinger parameter gamma2
    si.luttinger_gamma3 = 1.446;     // Luttinger parameter gamma3
    si.kane_parameter = 4.0;         // Kane parameter
    si.bowing_bandgap = 0.0;         // Bowing parameter for bandgap
    si.bowing_effective_mass = 0.0;  // Bowing parameter for effective mass
    si.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["Si"] = si;

    // Ge - Germanium
    Material ge;
    ge.m_e = 0.12;                   // Electron effective mass (conductivity mass)
    ge.m_h = 0.33;                   // Average hole effective mass
    ge.m_lh = 0.043;                 // Light hole effective mass
    ge.m_hh = 0.33;                  // Heavy hole effective mass
    ge.m_so = 0.095;                 // Split-off hole effective mass
    ge.E_g = 0.66;                   // Bandgap at 300K
    ge.Delta_E_c = 0.05;             // Conduction band offset
    ge.Delta_E_v = 0.5;              // Valence band offset
    ge.epsilon_r = 16.2;             // Dielectric constant
    ge.mu_n = 0.38;                  // Electron mobility
    ge.mu_p = 0.18;                  // Hole mobility
    ge.N_c = 1.04e25;                // Effective density of states, conduction band
    ge.N_v = 6.0e24;                 // Effective density of states, valence band
    ge.lattice_constant = 0.5658;    // Lattice constant
    ge.spin_orbit_splitting = 0.29;  // Spin-orbit splitting energy
    ge.deformation_potential_c = -8.6; // Deformation potential for conduction band
    ge.deformation_potential_v = 1.24; // Deformation potential for valence band
    ge.elastic_c11 = 128.5;          // Elastic constant C11
    ge.elastic_c12 = 48.3;           // Elastic constant C12
    ge.elastic_c44 = 66.8;           // Elastic constant C44
    ge.varshni_alpha = 4.77e-4;      // Varshni parameter alpha
    ge.varshni_beta = 235;           // Varshni parameter beta
    ge.luttinger_gamma1 = 13.38;     // Luttinger parameter gamma1
    ge.luttinger_gamma2 = 4.24;      // Luttinger parameter gamma2
    ge.luttinger_gamma3 = 5.69;      // Luttinger parameter gamma3
    ge.kane_parameter = 13.0;        // Kane parameter
    ge.bowing_bandgap = 0.0;         // Bowing parameter for bandgap
    ge.bowing_effective_mass = 0.0;  // Bowing parameter for effective mass
    ge.bowing_lattice_constant = 0.0; // Bowing parameter for lattice constant
    materials["Ge"] = ge;
}

/**
 * @brief Gets the properties of a material by name.
 *
 * This method retrieves the properties of a material from the database
 * by its name. If the material is not found in the database, it throws
 * a runtime error.
 *
 * @param name The name of the material (e.g., "GaAs", "AlGaAs", "InAs")
 * @return A reference to the Material struct containing the properties
 *
 * @throws std::runtime_error If the material is not found in the database
 */
const Material& MaterialDatabase::get_material(const std::string& name) const {
    // Look up the material in the database
    auto it = materials.find(name);

    // Check if the material was found
    if (it == materials.end()) {
        throw std::runtime_error("Material " + name + " not found in database");
    }

    // Return a reference to the material properties
    return it->second;
}

/**
 * @brief Gets the properties of a material at a specific temperature.
 *
 * This method returns the properties of a material at a specific temperature,
 * taking into account temperature-dependent effects such as bandgap narrowing.
 *
 * @param name The name of the material
 * @param temperature The temperature in Kelvin
 * @return A Material struct containing the temperature-dependent properties
 *
 * @throws std::runtime_error If the material is not found in the database
 * @throws std::invalid_argument If the temperature is out of range
 */
Material MaterialDatabase::get_material_at_temperature(const std::string& name, double temperature) const {
    // Check if temperature is valid
    if (temperature <= 0.0) {
        throw std::invalid_argument("Temperature must be positive");
    }

    // Get the material at 300K
    const Material& material_300K = get_material(name);

    // Create a copy for the temperature-dependent properties
    Material material_T = material_300K;

    // Calculate temperature-dependent bandgap using Varshni equation
    material_T.E_g = calculate_bandgap_at_temperature(material_300K, temperature);

    // Calculate temperature-dependent effective masses
    // Simplified model: m*(T) = m*(300K) * (1 + alpha_m * (T - 300))
    double alpha_m = 0.0001; // Approximate temperature coefficient
    double mass_factor = 1.0 + alpha_m * (temperature - 300.0);
    material_T.m_e *= mass_factor;
    material_T.m_h *= mass_factor;
    material_T.m_lh *= mass_factor;
    material_T.m_hh *= mass_factor;
    material_T.m_so *= mass_factor;

    // Calculate temperature-dependent mobilities
    // Simplified model: mu(T) = mu(300K) * (T/300)^(-alpha_mu)
    double alpha_mu_n = 1.5; // Approximate temperature exponent for electrons
    double alpha_mu_p = 1.7; // Approximate temperature exponent for holes
    material_T.mu_n *= std::pow(temperature / 300.0, -alpha_mu_n);
    material_T.mu_p *= std::pow(temperature / 300.0, -alpha_mu_p);

    // Calculate temperature-dependent effective densities of states
    // N_c(T) = N_c(300K) * (T/300)^(3/2)
    double dos_factor = std::pow(temperature / 300.0, 1.5);
    material_T.N_c *= dos_factor;
    material_T.N_v *= dos_factor;

    return material_T;
}

/**
 * @brief Creates an alloy material with the specified composition.
 *
 * This method creates an alloy material by interpolating between the properties
 * of two base materials. It takes into account bowing parameters for non-linear
 * interpolation of certain properties.
 *
 * @param material1 The name of the first base material
 * @param material2 The name of the second base material
 * @param x The composition parameter (0 <= x <= 1)
 * @param name Optional name for the alloy (default: auto-generated)
 * @return A Material struct containing the interpolated properties
 *
 * @throws std::runtime_error If either base material is not found in the database
 * @throws std::invalid_argument If the composition parameter is out of range
 */
Material MaterialDatabase::create_alloy(const std::string& material1, const std::string& material2, double x,
                                      const std::string& name) const {
    // Check if composition parameter is valid
    if (x < 0.0 || x > 1.0) {
        throw std::invalid_argument("Composition parameter must be between 0 and 1");
    }

    // Get the base materials
    const Material& mat1 = get_material(material1);
    const Material& mat2 = get_material(material2);

    // Create a new material for the alloy
    Material alloy;

    // Linear interpolation for most properties
    alloy.m_e = (1.0 - x) * mat1.m_e + x * mat2.m_e - x * (1.0 - x) * mat1.bowing_effective_mass;
    alloy.m_h = (1.0 - x) * mat1.m_h + x * mat2.m_h - x * (1.0 - x) * mat1.bowing_effective_mass;
    alloy.m_lh = (1.0 - x) * mat1.m_lh + x * mat2.m_lh - x * (1.0 - x) * mat1.bowing_effective_mass;
    alloy.m_hh = (1.0 - x) * mat1.m_hh + x * mat2.m_hh - x * (1.0 - x) * mat1.bowing_effective_mass;
    alloy.m_so = (1.0 - x) * mat1.m_so + x * mat2.m_so - x * (1.0 - x) * mat1.bowing_effective_mass;

    // Non-linear interpolation for bandgap (with bowing parameter)
    alloy.E_g = (1.0 - x) * mat1.E_g + x * mat2.E_g - x * (1.0 - x) * mat1.bowing_bandgap;

    // Linear interpolation for band offsets
    alloy.Delta_E_c = (1.0 - x) * mat1.Delta_E_c + x * mat2.Delta_E_c;
    alloy.Delta_E_v = (1.0 - x) * mat1.Delta_E_v + x * mat2.Delta_E_v;

    // Linear interpolation for dielectric constant
    alloy.epsilon_r = (1.0 - x) * mat1.epsilon_r + x * mat2.epsilon_r;

    // Linear interpolation for mobilities
    alloy.mu_n = (1.0 - x) * mat1.mu_n + x * mat2.mu_n;
    alloy.mu_p = (1.0 - x) * mat1.mu_p + x * mat2.mu_p;

    // Linear interpolation for effective densities of states
    alloy.N_c = (1.0 - x) * mat1.N_c + x * mat2.N_c;
    alloy.N_v = (1.0 - x) * mat1.N_v + x * mat2.N_v;

    // Non-linear interpolation for lattice constant (with bowing parameter)
    alloy.lattice_constant = (1.0 - x) * mat1.lattice_constant + x * mat2.lattice_constant -
                           x * (1.0 - x) * mat1.bowing_lattice_constant;

    // Linear interpolation for spin-orbit splitting
    alloy.spin_orbit_splitting = (1.0 - x) * mat1.spin_orbit_splitting + x * mat2.spin_orbit_splitting;

    // Linear interpolation for deformation potentials
    alloy.deformation_potential_c = (1.0 - x) * mat1.deformation_potential_c + x * mat2.deformation_potential_c;
    alloy.deformation_potential_v = (1.0 - x) * mat1.deformation_potential_v + x * mat2.deformation_potential_v;

    // Linear interpolation for elastic constants
    alloy.elastic_c11 = (1.0 - x) * mat1.elastic_c11 + x * mat2.elastic_c11;
    alloy.elastic_c12 = (1.0 - x) * mat1.elastic_c12 + x * mat2.elastic_c12;
    alloy.elastic_c44 = (1.0 - x) * mat1.elastic_c44 + x * mat2.elastic_c44;

    // Linear interpolation for Varshni parameters
    alloy.varshni_alpha = (1.0 - x) * mat1.varshni_alpha + x * mat2.varshni_alpha;
    alloy.varshni_beta = (1.0 - x) * mat1.varshni_beta + x * mat2.varshni_beta;

    // Linear interpolation for Luttinger parameters
    alloy.luttinger_gamma1 = (1.0 - x) * mat1.luttinger_gamma1 + x * mat2.luttinger_gamma1;
    alloy.luttinger_gamma2 = (1.0 - x) * mat1.luttinger_gamma2 + x * mat2.luttinger_gamma2;
    alloy.luttinger_gamma3 = (1.0 - x) * mat1.luttinger_gamma3 + x * mat2.luttinger_gamma3;

    // Linear interpolation for Kane parameter
    alloy.kane_parameter = (1.0 - x) * mat1.kane_parameter + x * mat2.kane_parameter;

    // Copy bowing parameters from the first material
    alloy.bowing_bandgap = mat1.bowing_bandgap;
    alloy.bowing_effective_mass = mat1.bowing_effective_mass;
    alloy.bowing_lattice_constant = mat1.bowing_lattice_constant;

    // Note: We can't modify the materials map in a const method
    // The caller should use add_material() to add the alloy to the database

    return alloy;
}

/**
 * @brief Adds a custom material to the database.
 *
 * This method adds a custom material to the database with the specified properties.
 * If a material with the same name already exists, it will be overwritten.
 *
 * @param name The name of the material
 * @param material The Material struct containing the properties
 */
void MaterialDatabase::add_material(const std::string& name, const Material& material) {
    // Validate the material
    if (!validate_material(material, name)) {
        throw std::invalid_argument("Invalid material properties for: " + name);
    }

    // Add or update the material in the database
    materials[name] = material;
}

/**
 * @brief Gets the bandgap of a material at a specific temperature.
 *
 * This method calculates the bandgap of a material at a specific temperature
 * using the Varshni equation: E_g(T) = E_g(0) - α*T²/(T + β)
 *
 * @param material The Material struct
 * @param temperature The temperature in Kelvin
 * @return The bandgap in electron volts (eV)
 *
 * @throws std::invalid_argument If the temperature is out of range
 */
double MaterialDatabase::calculate_bandgap_at_temperature(const Material& material, double temperature) {
    // Check if temperature is valid
    if (temperature <= 0.0) {
        throw std::invalid_argument("Temperature must be positive");
    }

    // Calculate the bandgap at 0K
    // E_g(0) = E_g(300) + α*300²/(300 + β)
    double E_g_0K = material.E_g + material.varshni_alpha * 300.0 * 300.0 / (300.0 + material.varshni_beta);

    // Calculate the bandgap at the specified temperature using the Varshni equation
    double E_g_T = E_g_0K - material.varshni_alpha * temperature * temperature / (temperature + material.varshni_beta);

    return E_g_T;
}

/**
 * @brief Gets the effective mass of a material under strain.
 *
 * This method calculates the effective mass of a material under strain,
 * taking into account deformation potentials and elastic constants.
 *
 * @param material The Material struct
 * @param strain_xx The xx component of the strain tensor
 * @param strain_yy The yy component of the strain tensor
 * @param strain_zz The zz component of the strain tensor
 * @param is_electron Whether to calculate electron or hole effective mass
 * @return The effective mass under strain relative to electron mass (m_0)
 */
double MaterialDatabase::calculate_effective_mass_under_strain(const Material& material,
                                                            double strain_xx, double strain_yy, double strain_zz,
                                                            bool is_electron) {
    // Calculate hydrostatic strain
    double strain_hydrostatic = strain_xx + strain_yy + strain_zz;

    // Calculate biaxial strain
    double strain_biaxial = strain_xx + strain_yy - 2.0 * strain_zz;

    // Calculate strain-induced shift in band edges
    double delta_E;
    if (is_electron) {
        // For electrons, use conduction band deformation potential
        delta_E = material.deformation_potential_c * strain_hydrostatic;
    } else {
        // For holes, use valence band deformation potential
        delta_E = material.deformation_potential_v * strain_hydrostatic;
    }

    // Calculate strain-induced change in effective mass
    // Simplified model: m*(strain) = m*(0) * (1 + delta_E/E_g)
    double mass_factor = 1.0 + delta_E / material.E_g;

    // Get the appropriate effective mass
    double m_0;
    if (is_electron) {
        m_0 = material.m_e;
    } else {
        // For holes, use the average effective mass
        m_0 = material.m_h;
    }

    // Calculate the effective mass under strain
    double m_strain = m_0 * mass_factor;

    return m_strain;
}

/**
 * @brief Gets a list of all available material names.
 *
 * @return A vector of material names
 */
std::vector<std::string> MaterialDatabase::get_available_materials() const {
    std::vector<std::string> material_names;
    material_names.reserve(materials.size());

    for (const auto& pair : materials) {
        material_names.push_back(pair.first);
    }

    return material_names;
}

/**
 * @brief Validates a Material struct for physical consistency.
 *
 * This private method checks if the properties of a Material struct
 * are physically consistent and within realistic ranges.
 *
 * @param material The Material struct to validate
 * @param name The name of the material (for error messages)
 * @return True if the material is valid, false otherwise
 */
bool MaterialDatabase::validate_material(const Material& material, const std::string& name) const {
    // Check if effective masses are positive
    if (material.m_e <= 0.0 || material.m_h <= 0.0 ||
        material.m_lh <= 0.0 || material.m_hh <= 0.0 || material.m_so <= 0.0) {
        std::cerr << "Warning: Negative or zero effective mass in material: " << name << std::endl;
        return false;
    }

    // Check if bandgap is non-negative
    if (material.E_g < 0.0) {
        std::cerr << "Warning: Negative bandgap in material: " << name << std::endl;
        return false;
    }

    // Check if dielectric constant is positive
    if (material.epsilon_r <= 0.0) {
        std::cerr << "Warning: Non-positive dielectric constant in material: " << name << std::endl;
        return false;
    }

    // Check if mobilities are non-negative
    if (material.mu_n < 0.0 || material.mu_p < 0.0) {
        std::cerr << "Warning: Negative mobility in material: " << name << std::endl;
        return false;
    }

    // Check if effective densities of states are positive
    if (material.N_c <= 0.0 || material.N_v <= 0.0) {
        std::cerr << "Warning: Non-positive effective density of states in material: " << name << std::endl;
        return false;
    }

    // Check if lattice constant is positive
    if (material.lattice_constant <= 0.0) {
        std::cerr << "Warning: Non-positive lattice constant in material: " << name << std::endl;
        return false;
    }

    // Check if Varshni parameters are non-negative
    if (material.varshni_alpha < 0.0 || material.varshni_beta < 0.0) {
        std::cerr << "Warning: Negative Varshni parameter in material: " << name << std::endl;
        return false;
    }

    // Check if Luttinger parameters are physically reasonable
    if (material.luttinger_gamma1 <= 0.0 ||
        material.luttinger_gamma2 < -material.luttinger_gamma1/2.0 ||
        material.luttinger_gamma3 < -material.luttinger_gamma1/2.0) {
        std::cerr << "Warning: Physically inconsistent Luttinger parameters in material: " << name << std::endl;
        return false;
    }

    // All checks passed
    return true;
}

} // namespace Materials