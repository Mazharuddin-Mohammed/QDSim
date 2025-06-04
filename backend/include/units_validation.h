#pragma once
/**
 * @file units_validation.h
 * @brief Units validation and conversion utilities for QDSim
 *
 * This file provides utilities to ensure consistent SI units throughout
 * the QDSim codebase, particularly for the Schrödinger solver.
 *
 * All physics calculations should use SI units:
 * - Length: meters (m)
 * - Mass: kilograms (kg)
 * - Energy: Joules (J)
 * - Time: seconds (s)
 * - Charge: Coulombs (C)
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <stdexcept>
#include <string>
#include <cmath>

namespace UnitsValidation {

    // Physical constants in SI units
    constexpr double HBAR = 1.054571817e-34;        // J·s
    constexpr double ELECTRON_MASS = 9.1093837015e-31; // kg
    constexpr double ELEMENTARY_CHARGE = 1.602176634e-19; // C
    constexpr double VACUUM_PERMITTIVITY = 8.8541878128e-12; // F/m

    // Unit conversion factors
    constexpr double EV_TO_JOULE = 1.602176634e-19;
    constexpr double JOULE_TO_EV = 1.0 / EV_TO_JOULE;
    constexpr double NM_TO_METER = 1e-9;
    constexpr double METER_TO_NM = 1e9;

    /**
     * @brief Validate that a mass value is in SI units (kg) and reasonable
     * @param mass Mass value to validate
     * @param context Description of where this mass is used
     * @throws std::invalid_argument if mass is invalid
     */
    inline void validate_mass_SI(double mass, const std::string& context = "") {
        if (mass <= 0) {
            throw std::invalid_argument("Mass must be positive in " + context);
        }
        
        // Reasonable range: 0.001 to 10 times electron mass
        double min_mass = 0.001 * ELECTRON_MASS;
        double max_mass = 10.0 * ELECTRON_MASS;
        
        if (mass < min_mass || mass > max_mass) {
            throw std::invalid_argument("Mass " + std::to_string(mass) + 
                " kg is outside reasonable range [" + std::to_string(min_mass) + 
                ", " + std::to_string(max_mass) + "] in " + context);
        }
    }

    /**
     * @brief Validate that a potential value is in SI units (J) and reasonable
     * @param potential Potential value to validate
     * @param context Description of where this potential is used
     * @throws std::invalid_argument if potential is invalid
     */
    inline void validate_potential_SI(double potential, const std::string& context = "") {
        // Reasonable range: -10 eV to +10 eV
        double min_potential = -10.0 * EV_TO_JOULE;
        double max_potential = 10.0 * EV_TO_JOULE;
        
        if (!std::isfinite(potential)) {
            throw std::invalid_argument("Potential must be finite in " + context);
        }
        
        if (potential < min_potential || potential > max_potential) {
            throw std::invalid_argument("Potential " + std::to_string(potential * JOULE_TO_EV) + 
                " eV is outside reasonable range [" + std::to_string(min_potential * JOULE_TO_EV) + 
                ", " + std::to_string(max_potential * JOULE_TO_EV) + "] eV in " + context);
        }
    }

    /**
     * @brief Validate that coordinates are in SI units (m) and reasonable
     * @param x X coordinate
     * @param y Y coordinate
     * @param context Description of where these coordinates are used
     * @throws std::invalid_argument if coordinates are invalid
     */
    inline void validate_coordinates_SI(double x, double y, const std::string& context = "") {
        // Reasonable range: -1 μm to +1 μm
        double max_coord = 1e-6; // 1 μm
        
        if (!std::isfinite(x) || !std::isfinite(y)) {
            throw std::invalid_argument("Coordinates must be finite in " + context);
        }
        
        if (std::abs(x) > max_coord || std::abs(y) > max_coord) {
            throw std::invalid_argument("Coordinates (" + std::to_string(x * METER_TO_NM) + 
                ", " + std::to_string(y * METER_TO_NM) + ") nm are outside reasonable range ±" + 
                std::to_string(max_coord * METER_TO_NM) + " nm in " + context);
        }
    }

    /**
     * @brief Convert effective mass from relative units (m₀) to SI units (kg)
     * @param m_star_relative Effective mass relative to electron mass
     * @return Effective mass in kg
     */
    inline double effective_mass_to_SI(double m_star_relative) {
        if (m_star_relative <= 0) {
            throw std::invalid_argument("Relative effective mass must be positive");
        }
        return m_star_relative * ELECTRON_MASS;
    }

    /**
     * @brief Convert potential from eV to SI units (J)
     * @param potential_eV Potential in eV
     * @return Potential in J
     */
    inline double potential_to_SI(double potential_eV) {
        return potential_eV * EV_TO_JOULE;
    }

    /**
     * @brief Convert coordinates from nm to SI units (m)
     * @param coord_nm Coordinate in nm
     * @return Coordinate in m
     */
    inline double coordinate_to_SI(double coord_nm) {
        return coord_nm * NM_TO_METER;
    }

    /**
     * @brief Convert energy from SI units (J) to eV
     * @param energy_J Energy in J
     * @return Energy in eV
     */
    inline double energy_to_eV(double energy_J) {
        return energy_J * JOULE_TO_EV;
    }

    /**
     * @brief Calculate expected quantum dot energy scale for validation
     * @param m_star_kg Effective mass in kg
     * @param radius_m QD radius in m
     * @return Expected energy scale in J
     */
    inline double calculate_QD_energy_scale(double m_star_kg, double radius_m) {
        validate_mass_SI(m_star_kg, "QD energy scale calculation");
        
        if (radius_m <= 0 || radius_m > 1e-6) {
            throw std::invalid_argument("QD radius must be positive and < 1 μm");
        }
        
        // E ~ ℏ²/(2m*R²)
        return (HBAR * HBAR) / (2.0 * m_star_kg * radius_m * radius_m);
    }

    /**
     * @brief Validate that computed energy is reasonable for a quantum dot
     * @param energy_J Computed energy in J
     * @param expected_scale_J Expected energy scale in J
     * @param tolerance_factor Tolerance factor (default 10x)
     * @param context Description for error messages
     * @throws std::invalid_argument if energy is unreasonable
     */
    inline void validate_QD_energy(double energy_J, double expected_scale_J, 
                                  double tolerance_factor = 10.0, 
                                  const std::string& context = "") {
        if (!std::isfinite(energy_J)) {
            throw std::invalid_argument("Energy must be finite in " + context);
        }
        
        double energy_eV = energy_to_eV(energy_J);
        double expected_eV = energy_to_eV(expected_scale_J);
        
        // Check if energy is within reasonable range
        double min_energy = expected_scale_J / tolerance_factor;
        double max_energy = expected_scale_J * tolerance_factor;
        
        if (std::abs(energy_J) < min_energy || std::abs(energy_J) > max_energy) {
            throw std::invalid_argument("Energy " + std::to_string(energy_eV) + 
                " eV is outside expected range [" + std::to_string(energy_to_eV(min_energy)) + 
                ", " + std::to_string(energy_to_eV(max_energy)) + "] eV (expected ~" + 
                std::to_string(expected_eV) + " eV) in " + context);
        }
    }

    /**
     * @brief Create a units-validated mass function wrapper
     * @param user_mass_func User-provided mass function (may be in any units)
     * @param assumes_SI If true, assumes user function returns SI units
     * @param assumes_relative If true, assumes user function returns relative units
     * @return Validated mass function that returns SI units
     */
    template<typename MassFunc>
    auto create_validated_mass_function(MassFunc user_mass_func, bool assumes_SI = false, bool assumes_relative = true) {
        return [user_mass_func, assumes_SI, assumes_relative](double x, double y) -> double {
            validate_coordinates_SI(x, y, "mass function evaluation");
            
            double mass_value = user_mass_func(x, y);
            
            if (assumes_relative) {
                // Convert from relative to SI
                mass_value = effective_mass_to_SI(mass_value);
            }
            
            validate_mass_SI(mass_value, "mass function result");
            return mass_value;
        };
    }

    /**
     * @brief Create a units-validated potential function wrapper
     * @param user_potential_func User-provided potential function (may be in any units)
     * @param assumes_SI If true, assumes user function returns SI units
     * @param assumes_eV If true, assumes user function returns eV units
     * @return Validated potential function that returns SI units
     */
    template<typename PotentialFunc>
    auto create_validated_potential_function(PotentialFunc user_potential_func, bool assumes_SI = false, bool assumes_eV = true) {
        return [user_potential_func, assumes_SI, assumes_eV](double x, double y) -> double {
            validate_coordinates_SI(x, y, "potential function evaluation");
            
            double potential_value = user_potential_func(x, y);
            
            if (assumes_eV) {
                // Convert from eV to SI
                potential_value = potential_to_SI(potential_value);
            }
            
            validate_potential_SI(potential_value, "potential function result");
            return potential_value;
        };
    }

} // namespace UnitsValidation
