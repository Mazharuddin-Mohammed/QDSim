#include "materials.h"
#include <stdexcept>

namespace Materials {

MaterialDatabase::MaterialDatabase() {
    initialize_database();
}

void MaterialDatabase::initialize_database() {
    // Initialize with common semiconductor materials
    materials["GaAs"] = {0.067 * 9.11e-31, 0.45 * 9.11e-31, 1.43, 0.7, 12.9};
    materials["InAs"] = {0.023 * 9.11e-31, 0.41 * 9.11e-31, 0.35, 0.7, 15.15};
    materials["AlGaAs"] = {0.09 * 9.11e-31, 0.51 * 9.11e-31, 1.8, 0.3, 12.0};
    // Add more materials as needed
}

const Material& MaterialDatabase::get_material(const std::string& name) const {
    auto it = materials.find(name);
    if (it == materials.end()) {
        throw std::runtime_error("Material " + name + " not found in database");
    }
    return it->second;
}

} // namespace Materials