#pragma once
/**
 * @file physical_constants.h
 * @brief Defines physical constants for quantum simulations.
 *
 * This file contains the declaration of physical constants used in quantum simulations,
 * including fundamental constants, unit conversions, and typical parameter ranges.
 * All constants are defined in SI units unless otherwise specified.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <cmath>
#include <limits>

/**
 * @namespace PhysicalConstants
 * @brief Namespace for physical constants used in quantum simulations.
 *
 * This namespace contains fundamental physical constants, unit conversions,
 * and typical parameter ranges used in quantum simulations.
 */
namespace PhysicalConstants {

// Fundamental constants (SI units)
constexpr double ELECTRON_CHARGE = 1.602176634e-19;    // Elementary charge in C
constexpr double ELECTRON_MASS = 9.1093837015e-31;     // Electron mass in kg
constexpr double PLANCK_CONSTANT = 6.62607015e-34;     // Planck constant in J·s
constexpr double REDUCED_PLANCK = 1.054571817e-34;     // Reduced Planck constant (ħ) in J·s
constexpr double BOLTZMANN_CONSTANT = 1.380649e-23;    // Boltzmann constant in J/K
constexpr double VACUUM_PERMITTIVITY = 8.8541878128e-12; // Vacuum permittivity in F/m
constexpr double SPEED_OF_LIGHT = 2.99792458e8;        // Speed of light in m/s
constexpr double AVOGADRO_NUMBER = 6.02214076e23;      // Avogadro's number in mol^-1

// Temperature constants
constexpr double ROOM_TEMPERATURE = 300.0;             // Room temperature in K
constexpr double ZERO_CELSIUS = 273.15;                // 0°C in K
constexpr double LIQUID_NITROGEN = 77.0;               // Liquid nitrogen temperature in K
constexpr double LIQUID_HELIUM = 4.2;                  // Liquid helium temperature in K

// Unit conversions
constexpr double EV_TO_J = ELECTRON_CHARGE;            // Convert eV to J
constexpr double J_TO_EV = 1.0 / ELECTRON_CHARGE;      // Convert J to eV
constexpr double NM_TO_M = 1.0e-9;                     // Convert nm to m
constexpr double M_TO_NM = 1.0e9;                      // Convert m to nm
constexpr double CM_TO_M = 1.0e-2;                     // Convert cm to m
constexpr double M_TO_CM = 1.0e2;                      // Convert m to cm
constexpr double ANGSTROM_TO_NM = 0.1;                 // Convert Å to nm
constexpr double NM_TO_ANGSTROM = 10.0;                // Convert nm to Å
constexpr double CM2_TO_M2 = 1.0e-4;                   // Convert cm² to m²
constexpr double M2_TO_CM2 = 1.0e4;                    // Convert m² to cm²
constexpr double CM3_TO_M3 = 1.0e-6;                   // Convert cm³ to m³
constexpr double M3_TO_CM3 = 1.0e6;                    // Convert m³ to cm³

// Derived constants
constexpr double THERMAL_VOLTAGE_300K = BOLTZMANN_CONSTANT * ROOM_TEMPERATURE / ELECTRON_CHARGE; // Thermal voltage at 300K in V
constexpr double THERMAL_DE_BROGLIE_300K = REDUCED_PLANCK / std::sqrt(2.0 * ELECTRON_MASS * BOLTZMANN_CONSTANT * ROOM_TEMPERATURE); // Thermal de Broglie wavelength at 300K in m
constexpr double VACUUM_IMPEDANCE = 376.730313668; // Vacuum impedance in Ω

// Semiconductor parameters (Silicon at 300K)
namespace Silicon {
    constexpr double LATTICE_CONSTANT = 5.431e-10;        // Lattice constant in m
    constexpr double BANDGAP = 1.12;                      // Bandgap in eV
    constexpr double ELECTRON_AFFINITY = 4.05;            // Electron affinity in eV
    constexpr double RELATIVE_PERMITTIVITY = 11.7;        // Relative permittivity
    constexpr double ELECTRON_EFFECTIVE_MASS = 0.26;      // Electron effective mass (m_e)
    constexpr double HOLE_EFFECTIVE_MASS = 0.386;         // Hole effective mass (m_e)
    constexpr double ELECTRON_MOBILITY = 1400.0;          // Electron mobility in cm²/V·s
    constexpr double HOLE_MOBILITY = 450.0;               // Hole mobility in cm²/V·s
    constexpr double INTRINSIC_CARRIER_CONCENTRATION = 1.0e10;  // Intrinsic carrier concentration in cm^-3
    constexpr double THERMAL_CONDUCTIVITY = 148.0;        // Thermal conductivity in W/m·K
    constexpr double SPECIFIC_HEAT = 700.0;               // Specific heat in J/kg·K
    constexpr double DENSITY = 2329.0;                    // Density in kg/m³
    constexpr double ELECTRON_LIFETIME = 1.0e-6;          // Electron lifetime in s
    constexpr double HOLE_LIFETIME = 1.0e-6;              // Hole lifetime in s
    constexpr double AUGER_COEFFICIENT_N = 2.8e-31;       // Electron Auger coefficient in cm^6/s
    constexpr double AUGER_COEFFICIENT_P = 9.9e-32;       // Hole Auger coefficient in cm^6/s
    constexpr double RADIATIVE_COEFFICIENT = 1.1e-14;     // Radiative recombination coefficient in cm^3/s

    // Deformation potentials
    constexpr double CONDUCTION_DEFORMATION_POTENTIAL = -10.5; // Conduction band deformation potential in eV
    constexpr double VALENCE_DEFORMATION_POTENTIAL = 2.3;     // Valence band deformation potential in eV
    constexpr double SHEAR_DEFORMATION_POTENTIAL = -4.5;      // Shear deformation potential in eV
}

// Semiconductor parameters (Germanium at 300K)
namespace Germanium {
    constexpr double LATTICE_CONSTANT = 5.658e-10;        // Lattice constant in m
    constexpr double BANDGAP = 0.66;                      // Bandgap in eV
    constexpr double ELECTRON_AFFINITY = 4.0;             // Electron affinity in eV
    constexpr double RELATIVE_PERMITTIVITY = 16.2;        // Relative permittivity
    constexpr double ELECTRON_EFFECTIVE_MASS = 0.12;      // Electron effective mass (m_e)
    constexpr double HOLE_EFFECTIVE_MASS = 0.21;          // Hole effective mass (m_e)
    constexpr double ELECTRON_MOBILITY = 3900.0;          // Electron mobility in cm²/V·s
    constexpr double HOLE_MOBILITY = 1900.0;              // Hole mobility in cm²/V·s
    constexpr double INTRINSIC_CARRIER_CONCENTRATION = 2.4e13;  // Intrinsic carrier concentration in cm^-3
    constexpr double THERMAL_CONDUCTIVITY = 60.0;         // Thermal conductivity in W/m·K
    constexpr double SPECIFIC_HEAT = 320.0;               // Specific heat in J/kg·K
    constexpr double DENSITY = 5323.0;                    // Density in kg/m³
    constexpr double ELECTRON_LIFETIME = 1.0e-6;          // Electron lifetime in s
    constexpr double HOLE_LIFETIME = 1.0e-6;              // Hole lifetime in s
    constexpr double AUGER_COEFFICIENT_N = 1.1e-30;       // Electron Auger coefficient in cm^6/s
    constexpr double AUGER_COEFFICIENT_P = 3.0e-31;       // Hole Auger coefficient in cm^6/s
    constexpr double RADIATIVE_COEFFICIENT = 5.0e-14;     // Radiative recombination coefficient in cm^3/s

    // Deformation potentials
    constexpr double CONDUCTION_DEFORMATION_POTENTIAL = -9.5; // Conduction band deformation potential in eV
    constexpr double VALENCE_DEFORMATION_POTENTIAL = 2.0;    // Valence band deformation potential in eV
    constexpr double SHEAR_DEFORMATION_POTENTIAL = -4.0;     // Shear deformation potential in eV
}

// Semiconductor parameters (GaAs at 300K)
namespace GaAs {
    constexpr double LATTICE_CONSTANT = 5.653e-10;        // Lattice constant in m
    constexpr double BANDGAP = 1.42;                      // Bandgap in eV
    constexpr double ELECTRON_AFFINITY = 4.07;            // Electron affinity in eV
    constexpr double RELATIVE_PERMITTIVITY = 12.9;        // Relative permittivity
    constexpr double ELECTRON_EFFECTIVE_MASS = 0.067;     // Electron effective mass (m_e)
    constexpr double HOLE_EFFECTIVE_MASS = 0.45;          // Hole effective mass (m_e)
    constexpr double ELECTRON_MOBILITY = 8500.0;          // Electron mobility in cm²/V·s
    constexpr double HOLE_MOBILITY = 400.0;               // Hole mobility in cm²/V·s
    constexpr double INTRINSIC_CARRIER_CONCENTRATION = 2.1e6;  // Intrinsic carrier concentration in cm^-3
    constexpr double THERMAL_CONDUCTIVITY = 55.0;         // Thermal conductivity in W/m·K
    constexpr double SPECIFIC_HEAT = 350.0;               // Specific heat in J/kg·K
    constexpr double DENSITY = 5320.0;                    // Density in kg/m³
    constexpr double ELECTRON_LIFETIME = 1.0e-8;          // Electron lifetime in s
    constexpr double HOLE_LIFETIME = 1.0e-8;              // Hole lifetime in s
    constexpr double AUGER_COEFFICIENT_N = 5.0e-30;       // Electron Auger coefficient in cm^6/s
    constexpr double AUGER_COEFFICIENT_P = 1.0e-30;       // Hole Auger coefficient in cm^6/s
    constexpr double RADIATIVE_COEFFICIENT = 7.2e-10;     // Radiative recombination coefficient in cm^3/s

    // Deformation potentials
    constexpr double CONDUCTION_DEFORMATION_POTENTIAL = -7.17; // Conduction band deformation potential in eV
    constexpr double VALENCE_DEFORMATION_POTENTIAL = 1.16;    // Valence band deformation potential in eV
    constexpr double SHEAR_DEFORMATION_POTENTIAL = -2.0;      // Shear deformation potential in eV
}

// Semiconductor parameters (InP at 300K)
namespace InP {
    constexpr double LATTICE_CONSTANT = 5.869e-10;        // Lattice constant in m
    constexpr double BANDGAP = 1.35;                      // Bandgap in eV
    constexpr double ELECTRON_AFFINITY = 4.38;            // Electron affinity in eV
    constexpr double RELATIVE_PERMITTIVITY = 12.5;        // Relative permittivity
    constexpr double ELECTRON_EFFECTIVE_MASS = 0.08;      // Electron effective mass (m_e)
    constexpr double HOLE_EFFECTIVE_MASS = 0.6;           // Hole effective mass (m_e)
    constexpr double ELECTRON_MOBILITY = 5400.0;          // Electron mobility in cm²/V·s
    constexpr double HOLE_MOBILITY = 200.0;               // Hole mobility in cm²/V·s
    constexpr double INTRINSIC_CARRIER_CONCENTRATION = 1.3e7;  // Intrinsic carrier concentration in cm^-3
    constexpr double THERMAL_CONDUCTIVITY = 68.0;         // Thermal conductivity in W/m·K
    constexpr double SPECIFIC_HEAT = 310.0;               // Specific heat in J/kg·K
    constexpr double DENSITY = 4810.0;                    // Density in kg/m³
    constexpr double ELECTRON_LIFETIME = 1.0e-8;          // Electron lifetime in s
    constexpr double HOLE_LIFETIME = 1.0e-8;              // Hole lifetime in s
    constexpr double AUGER_COEFFICIENT_N = 3.0e-30;       // Electron Auger coefficient in cm^6/s
    constexpr double AUGER_COEFFICIENT_P = 1.0e-30;       // Hole Auger coefficient in cm^6/s
    constexpr double RADIATIVE_COEFFICIENT = 1.0e-10;     // Radiative recombination coefficient in cm^3/s
}

// Semiconductor parameters (GaN at 300K)
namespace GaN {
    constexpr double LATTICE_CONSTANT_A = 3.189e-10;      // a-lattice constant in m
    constexpr double LATTICE_CONSTANT_C = 5.185e-10;      // c-lattice constant in m
    constexpr double BANDGAP = 3.4;                       // Bandgap in eV
    constexpr double ELECTRON_AFFINITY = 4.1;             // Electron affinity in eV
    constexpr double RELATIVE_PERMITTIVITY = 8.9;         // Relative permittivity
    constexpr double ELECTRON_EFFECTIVE_MASS = 0.2;       // Electron effective mass (m_e)
    constexpr double HOLE_EFFECTIVE_MASS = 1.4;           // Hole effective mass (m_e)
    constexpr double ELECTRON_MOBILITY = 1000.0;          // Electron mobility in cm²/V·s
    constexpr double HOLE_MOBILITY = 30.0;                // Hole mobility in cm²/V·s
    constexpr double INTRINSIC_CARRIER_CONCENTRATION = 1.0e-10;  // Intrinsic carrier concentration in cm^-3
    constexpr double THERMAL_CONDUCTIVITY = 130.0;        // Thermal conductivity in W/m·K
    constexpr double SPECIFIC_HEAT = 490.0;               // Specific heat in J/kg·K
    constexpr double DENSITY = 6150.0;                    // Density in kg/m³
    constexpr double ELECTRON_LIFETIME = 1.0e-9;          // Electron lifetime in s
    constexpr double HOLE_LIFETIME = 1.0e-9;              // Hole lifetime in s
    constexpr double AUGER_COEFFICIENT_N = 1.0e-31;       // Electron Auger coefficient in cm^6/s
    constexpr double AUGER_COEFFICIENT_P = 1.0e-31;       // Hole Auger coefficient in cm^6/s
    constexpr double RADIATIVE_COEFFICIENT = 1.0e-11;     // Radiative recombination coefficient in cm^3/s
    constexpr double PIEZOELECTRIC_CONSTANT = 0.5;        // Piezoelectric constant in C/m²

    // Deformation potentials (wurtzite)
    constexpr double CONDUCTION_DEFORMATION_POTENTIAL_Z = -4.9; // Conduction band deformation potential along c-axis in eV
    constexpr double CONDUCTION_DEFORMATION_POTENTIAL_T = -11.3; // Conduction band deformation potential perpendicular to c-axis in eV
    constexpr double VALENCE_DEFORMATION_POTENTIAL_D1 = -0.89;  // Valence band deformation potential D1 in eV
    constexpr double VALENCE_DEFORMATION_POTENTIAL_D2 = 4.27;   // Valence band deformation potential D2 in eV
    constexpr double VALENCE_DEFORMATION_POTENTIAL_D3 = 5.18;   // Valence band deformation potential D3 in eV
    constexpr double VALENCE_DEFORMATION_POTENTIAL_D4 = -2.93;  // Valence band deformation potential D4 in eV
}

// Typical parameter ranges for semiconductor quantum dots
namespace TypicalRanges {
    // Potential ranges
    constexpr double MIN_POTENTIAL = -10.0;            // Minimum potential in eV
    constexpr double MAX_POTENTIAL = 10.0;             // Maximum potential in eV
    constexpr double TYPICAL_BARRIER_HEIGHT = 0.5;     // Typical barrier height in eV

    // Size ranges
    constexpr double MIN_QD_SIZE = 1.0;                // Minimum quantum dot size in nm
    constexpr double MAX_QD_SIZE = 100.0;              // Maximum quantum dot size in nm
    constexpr double TYPICAL_QD_SIZE = 10.0;           // Typical quantum dot size in nm

    // Effective mass ranges
    constexpr double MIN_EFFECTIVE_MASS = 0.01;        // Minimum effective mass relative to m_0
    constexpr double MAX_EFFECTIVE_MASS = 10.0;        // Maximum effective mass relative to m_0

    // Doping concentration ranges
    constexpr double MIN_DOPING = 1.0e14;              // Minimum doping concentration in cm^-3
    constexpr double MAX_DOPING = 1.0e20;              // Maximum doping concentration in cm^-3
    constexpr double TYPICAL_DOPING = 1.0e17;          // Typical doping concentration in cm^-3

    // Electric field ranges
    constexpr double MIN_ELECTRIC_FIELD = 0.0;         // Minimum electric field in V/cm
    constexpr double MAX_ELECTRIC_FIELD = 1.0e6;       // Maximum electric field in V/cm
    constexpr double TYPICAL_ELECTRIC_FIELD = 1.0e4;   // Typical electric field in V/cm

    // Magnetic field ranges
    constexpr double MIN_MAGNETIC_FIELD = 0.0;         // Minimum magnetic field in T
    constexpr double MAX_MAGNETIC_FIELD = 10.0;        // Maximum magnetic field in T
    constexpr double TYPICAL_MAGNETIC_FIELD = 1.0;     // Typical magnetic field in T
}

/**
 * @brief Checks if a potential value is within a realistic range.
 *
 * This function checks if a potential value is within a realistic range
 * for semiconductor quantum dots. The range is defined by the MIN_POTENTIAL
 * and MAX_POTENTIAL constants.
 *
 * @param potential The potential value to check in eV
 * @return True if the potential is within the realistic range, false otherwise
 */
inline bool is_realistic_potential(double potential) {
    return potential >= TypicalRanges::MIN_POTENTIAL && potential <= TypicalRanges::MAX_POTENTIAL;
}

/**
 * @brief Checks if an effective mass value is within a realistic range.
 *
 * This function checks if an effective mass value is within a realistic range
 * for semiconductor materials. The range is defined by the MIN_EFFECTIVE_MASS
 * and MAX_EFFECTIVE_MASS constants.
 *
 * @param mass The effective mass value to check relative to m_0
 * @return True if the effective mass is within the realistic range, false otherwise
 */
inline bool is_realistic_effective_mass(double mass) {
    return mass >= TypicalRanges::MIN_EFFECTIVE_MASS && mass <= TypicalRanges::MAX_EFFECTIVE_MASS;
}

/**
 * @brief Checks if a doping concentration value is within a realistic range.
 *
 * This function checks if a doping concentration value is within a realistic range
 * for semiconductor materials. The range is defined by the MIN_DOPING
 * and MAX_DOPING constants.
 *
 * @param doping The doping concentration value to check in cm^-3
 * @return True if the doping concentration is within the realistic range, false otherwise
 */
inline bool is_realistic_doping(double doping) {
    return doping >= TypicalRanges::MIN_DOPING && doping <= TypicalRanges::MAX_DOPING;
}

/**
 * @brief Checks if an electric field value is within a realistic range.
 *
 * This function checks if an electric field value is within a realistic range
 * for semiconductor devices. The range is defined by the MIN_ELECTRIC_FIELD
 * and MAX_ELECTRIC_FIELD constants.
 *
 * @param field The electric field value to check in V/cm
 * @return True if the electric field is within the realistic range, false otherwise
 */
inline bool is_realistic_electric_field(double field) {
    return field >= TypicalRanges::MIN_ELECTRIC_FIELD && field <= TypicalRanges::MAX_ELECTRIC_FIELD;
}

/**
 * @brief Checks if a magnetic field value is within a realistic range.
 *
 * This function checks if a magnetic field value is within a realistic range
 * for laboratory experiments. The range is defined by the MIN_MAGNETIC_FIELD
 * and MAX_MAGNETIC_FIELD constants.
 *
 * @param field The magnetic field value to check in T
 * @return True if the magnetic field is within the realistic range, false otherwise
 */
inline bool is_realistic_magnetic_field(double field) {
    return field >= TypicalRanges::MIN_MAGNETIC_FIELD && field <= TypicalRanges::MAX_MAGNETIC_FIELD;
}

} // namespace PhysicalConstants
