#pragma once
/**
 * @file materials.h
 * @brief Defines the Material struct and MaterialDatabase class.
 *
 * This file contains the declaration of the Material struct and MaterialDatabase class,
 * which provide a database of semiconductor material properties for quantum simulations.
 * The database includes properties such as effective masses, bandgaps, band offsets,
 * and dielectric constants for various semiconductor materials.
 *
 * Physical units:
 * - Effective masses: relative to electron mass (m_0)
 * - Bandgap: electron volts (eV)
 * - Band offsets: electron volts (eV)
 * - Dielectric constant: relative to vacuum permittivity (epsilon_0)
 *
 * Assumptions and limitations:
 * - The database includes common semiconductor materials
 * - The properties are assumed to be temperature-independent
 * - The properties are assumed to be isotropic
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <string>
#include <unordered_map>

/**
 * @namespace Materials
 * @brief Namespace for semiconductor material properties.
 *
 * This namespace contains the Material struct and MaterialDatabase class,
 * which provide a database of semiconductor material properties for quantum simulations.
 */
namespace Materials {

/**
 * @struct Material
 * @brief Structure containing semiconductor material properties.
 *
 * This structure contains the properties of a semiconductor material,
 * including effective masses, bandgap, band offset, and dielectric constant.
 * These properties are used in quantum simulations to model the behavior
 * of electrons and holes in semiconductor devices.
 */
struct Material {
    /** @brief Electron effective mass relative to electron mass (m_0) */
    double m_e;

    /** @brief Hole effective mass relative to electron mass (m_0) */
    double m_h;

    /** @brief Bandgap in electron volts (eV) */
    double E_g;

    /** @brief Conduction band offset in electron volts (eV) */
    double Delta_E_c;

    /** @brief Dielectric constant relative to vacuum permittivity (epsilon_0) */
    double epsilon_r;
};

/**
 * @class MaterialDatabase
 * @brief Database of semiconductor material properties.
 *
 * This class provides a database of semiconductor material properties for
 * quantum simulations. It includes properties for common semiconductor
 * materials such as GaAs, AlGaAs, InAs, etc.
 *
 * The database is initialized with default values for common materials,
 * and can be queried by material name.
 */
class MaterialDatabase {
public:
    /**
     * @brief Constructs a new MaterialDatabase object.
     *
     * This constructor initializes the database with default values for
     * common semiconductor materials.
     */
    MaterialDatabase();

    /**
     * @brief Gets the properties of a material by name.
     *
     * @param name The name of the material (e.g., "GaAs", "AlGaAs", "InAs")
     * @return A reference to the Material struct containing the properties
     *
     * @throws std::out_of_range If the material is not found in the database
     */
    const Material& get_material(const std::string& name) const;

private:
    /** @brief Map of material names to Material structs */
    std::unordered_map<std::string, Material> materials;

    /**
     * @brief Initializes the database with default values.
     *
     * This private method initializes the database with default values for
     * common semiconductor materials.
     */
    void initialize_database();
};

} // namespace Materials