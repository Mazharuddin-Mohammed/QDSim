#pragma once
#include <string>
#include <unordered_map>

namespace Materials {

struct Material {
    double m_e;       // Electron effective mass (kg)
    double m_h;       // Hole effective mass (kg)
    double E_g;       // Bandgap (eV)
    double Delta_E_c; // Conduction band offset (eV)
    double epsilon_r; // Dielectric constant
};

class MaterialDatabase {
public:
    MaterialDatabase();
    const Material& get_material(const std::string& name) const;

private:
    std::unordered_map<std::string, Material> materials;
    void initialize_database();
};

} // namespace Materials