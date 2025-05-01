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
 * - Electron effective mass (m_e) in kg
 * - Hole effective mass (m_h) in kg
 * - Bandgap (E_g) in eV
 * - Conduction band offset (Delta_E_c) in eV
 * - Dielectric constant (epsilon_r) relative to vacuum
 *
 * The effective masses are given relative to the electron rest mass (9.11e-31 kg).
 */
void MaterialDatabase::initialize_database() {
    // Initialize with common semiconductor materials
    materials["GaAs"] = {0.067 * 9.11e-31, 0.45 * 9.11e-31, 1.43, 0.7, 12.9};
    materials["InAs"] = {0.023 * 9.11e-31, 0.41 * 9.11e-31, 0.35, 0.7, 15.15};
    materials["AlGaAs"] = {0.09 * 9.11e-31, 0.51 * 9.11e-31, 1.8, 0.3, 12.0};
    // Add more materials as needed
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

} // namespace Materials