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
#include <vector>

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

    /** @brief Light hole effective mass relative to electron mass (m_0) */
    double m_lh;

    /** @brief Heavy hole effective mass relative to electron mass (m_0) */
    double m_hh;

    /** @brief Split-off hole effective mass relative to electron mass (m_0) */
    double m_so;

    /** @brief Bandgap in electron volts (eV) at 300K */
    double E_g;

    /** @brief Electron affinity in electron volts (eV) */
    double chi;

    /** @brief Conduction band offset in electron volts (eV) */
    double Delta_E_c;

    /** @brief Valence band offset in electron volts (eV) */
    double Delta_E_v;

    /** @brief Dielectric constant relative to vacuum permittivity (epsilon_0) */
    double epsilon_r;

    /** @brief Electron mobility (m^2/V·s) at 300K */
    double mu_n;

    /** @brief Hole mobility (m^2/V·s) at 300K */
    double mu_p;

    /** @brief Effective density of states, conduction band (m^-3) at 300K */
    double N_c;

    /** @brief Effective density of states, valence band (m^-3) at 300K */
    double N_v;

    /** @brief Lattice constant (nm) */
    double lattice_constant;

    /** @brief Spin-orbit splitting energy (eV) */
    double spin_orbit_splitting;

    /** @brief Deformation potential for conduction band (eV) */
    double deformation_potential_c;

    /** @brief Deformation potential for valence band (eV) */
    double deformation_potential_v;

    /** @brief Elastic constants (GPa) */
    double elastic_c11;
    double elastic_c12;
    double elastic_c44;

    /** @brief Varshni parameters for temperature dependence of bandgap */
    double varshni_alpha;
    double varshni_beta;

    /** @brief Luttinger parameters for k·p calculations */
    double luttinger_gamma1;
    double luttinger_gamma2;
    double luttinger_gamma3;

    /** @brief Kane parameter for k·p calculations (eV·nm) */
    double kane_parameter;

    /** @brief Bowing parameters for alloys */
    double bowing_bandgap;
    double bowing_effective_mass;
    double bowing_lattice_constant;
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
 * and can be queried by material name. It also provides methods for
 * calculating temperature-dependent properties and creating alloys.
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
     * @throws std::runtime_error If the material is not found in the database
     */
    const Material& get_material(const std::string& name) const;

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
    Material get_material_at_temperature(const std::string& name, double temperature) const;

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
    Material create_alloy(const std::string& material1, const std::string& material2, double x,
                         const std::string& name = "") const;

    /**
     * @brief Adds a custom material to the database.
     *
     * This method adds a custom material to the database with the specified properties.
     * If a material with the same name already exists, it will be overwritten.
     *
     * @param name The name of the material
     * @param material The Material struct containing the properties
     */
    void add_material(const std::string& name, const Material& material);

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
    static double calculate_bandgap_at_temperature(const Material& material, double temperature);

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
    static double calculate_effective_mass_under_strain(const Material& material,
                                                      double strain_xx, double strain_yy, double strain_zz,
                                                      bool is_electron = true);

    /**
     * @brief Gets a list of all available material names.
     *
     * @return A vector of material names
     */
    std::vector<std::string> get_available_materials() const;

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
    bool validate_material(const Material& material, const std::string& name) const;
};

} // namespace Materials