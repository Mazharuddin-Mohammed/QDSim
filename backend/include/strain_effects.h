#pragma once
/**
 * @file strain_effects.h
 * @brief Defines strain effects for semiconductor simulations.
 *
 * This file contains the declaration of strain effects used in
 * semiconductor simulations, including deformation potentials, band structure
 * modifications, and effective mass changes due to strain.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Dense>

// Define Matrix6d if not available
namespace Eigen {
    using Matrix6d = Matrix<double, 6, 6>;
}

/**
 * @namespace StrainEffects
 * @brief Namespace for strain effects used in semiconductor simulations.
 *
 * This namespace contains functions for computing strain effects in semiconductors,
 * including deformation potentials, band structure modifications, and effective mass changes.
 */
namespace StrainEffects {

/**
 * @brief Computes the strain tensor for pseudomorphic growth.
 *
 * This function computes the strain tensor for pseudomorphic growth
 * of a layer on a substrate with different lattice constants.
 *
 * @param a_substrate The lattice constant of the substrate (nm)
 * @param a_layer The lattice constant of the layer (nm)
 * @param is_cubic Whether the material is cubic (true) or wurtzite (false)
 * @return The strain tensor
 */
Eigen::Matrix3d compute_strain_tensor_pseudomorphic(double a_substrate, double a_layer, bool is_cubic);

/**
 * @brief Computes the strain tensor from stress and compliance.
 *
 * This function computes the strain tensor from the stress tensor
 * and the compliance matrix.
 *
 * @param stress The stress tensor (GPa)
 * @param compliance The compliance matrix (GPa^-1)
 * @return The strain tensor
 */
Eigen::Matrix3d compute_strain_tensor_from_stress(const Eigen::Matrix3d& stress, const Eigen::Matrix6d& compliance);

/**
 * @brief Computes the hydrostatic strain.
 *
 * This function computes the hydrostatic strain (trace of the strain tensor).
 *
 * @param strain The strain tensor
 * @return The hydrostatic strain
 */
double compute_hydrostatic_strain(const Eigen::Matrix3d& strain);

/**
 * @brief Computes the biaxial strain.
 *
 * This function computes the biaxial strain (2*ε_zz - ε_xx - ε_yy).
 *
 * @param strain The strain tensor
 * @return The biaxial strain
 */
double compute_biaxial_strain(const Eigen::Matrix3d& strain);

/**
 * @brief Computes the shear strain.
 *
 * This function computes the shear strain (sqrt(ε_xy^2 + ε_xz^2 + ε_yz^2)).
 *
 * @param strain The strain tensor
 * @return The shear strain
 */
double compute_shear_strain(const Eigen::Matrix3d& strain);

/**
 * @brief Computes the conduction band shift due to strain for cubic materials.
 *
 * This function computes the conduction band shift due to strain
 * for cubic materials using the deformation potential theory.
 *
 * @param strain The strain tensor
 * @param a_c The conduction band deformation potential (eV)
 * @return The conduction band shift (eV)
 */
double compute_conduction_band_shift_cubic(const Eigen::Matrix3d& strain, double a_c);

/**
 * @brief Computes the valence band shift due to strain for cubic materials.
 *
 * This function computes the valence band shift due to strain
 * for cubic materials using the deformation potential theory.
 *
 * @param strain The strain tensor
 * @param a_v The valence band hydrostatic deformation potential (eV)
 * @param b The valence band biaxial deformation potential (eV)
 * @param d The valence band shear deformation potential (eV)
 * @param delta_E_hh Output parameter for the heavy hole band shift (eV)
 * @param delta_E_lh Output parameter for the light hole band shift (eV)
 * @param delta_E_so Output parameter for the split-off band shift (eV)
 */
void compute_valence_band_shift_cubic(const Eigen::Matrix3d& strain, double a_v, double b, double d,
                                    double& delta_E_hh, double& delta_E_lh, double& delta_E_so);

/**
 * @brief Computes the conduction band shift due to strain for wurtzite materials.
 *
 * This function computes the conduction band shift due to strain
 * for wurtzite materials using the deformation potential theory.
 *
 * @param strain The strain tensor
 * @param a_cz The conduction band deformation potential along c-axis (eV)
 * @param a_ct The conduction band deformation potential perpendicular to c-axis (eV)
 * @return The conduction band shift (eV)
 */
double compute_conduction_band_shift_wurtzite(const Eigen::Matrix3d& strain, double a_cz, double a_ct);

/**
 * @brief Computes the valence band shift due to strain for wurtzite materials.
 *
 * This function computes the valence band shift due to strain
 * for wurtzite materials using the deformation potential theory.
 *
 * @param strain The strain tensor
 * @param a_vz The valence band hydrostatic deformation potential along c-axis (eV)
 * @param a_vt The valence band hydrostatic deformation potential perpendicular to c-axis (eV)
 * @param D1 The valence band deformation potential D1 (eV)
 * @param D2 The valence band deformation potential D2 (eV)
 * @param D3 The valence band deformation potential D3 (eV)
 * @param D4 The valence band deformation potential D4 (eV)
 * @param D5 The valence band deformation potential D5 (eV)
 * @param D6 The valence band deformation potential D6 (eV)
 * @param delta_E_hh Output parameter for the heavy hole band shift (eV)
 * @param delta_E_lh Output parameter for the light hole band shift (eV)
 * @param delta_E_ch Output parameter for the crystal-field split-off band shift (eV)
 */
void compute_valence_band_shift_wurtzite(const Eigen::Matrix3d& strain, double a_vz, double a_vt,
                                       double D1, double D2, double D3, double D4, double D5, double D6,
                                       double& delta_E_hh, double& delta_E_lh, double& delta_E_ch);

/**
 * @brief Computes the effective mass change due to strain for electrons.
 *
 * This function computes the effective mass change due to strain for electrons.
 *
 * @param strain The strain tensor
 * @param Xi The electron effective mass deformation potential
 * @return The relative change in electron effective mass (Δm/m)
 */
double compute_electron_effective_mass_change(const Eigen::Matrix3d& strain, double Xi);

/**
 * @brief Computes the effective mass change due to strain for holes.
 *
 * This function computes the effective mass change due to strain for holes.
 *
 * @param strain The strain tensor
 * @param L The Luttinger parameter deformation potential for γ1
 * @param M The Luttinger parameter deformation potential for γ2
 * @param N The Luttinger parameter deformation potential for γ3
 * @param delta_m_hh Output parameter for the relative change in heavy hole effective mass (Δm/m)
 * @param delta_m_lh Output parameter for the relative change in light hole effective mass (Δm/m)
 */
void compute_hole_effective_mass_change(const Eigen::Matrix3d& strain, double L, double M, double N,
                                      double& delta_m_hh, double& delta_m_lh);

/**
 * @brief Computes the bandgap change due to strain.
 *
 * This function computes the bandgap change due to strain.
 *
 * @param strain The strain tensor
 * @param a_c The conduction band deformation potential (eV)
 * @param a_v The valence band hydrostatic deformation potential (eV)
 * @param b The valence band biaxial deformation potential (eV)
 * @return The bandgap change (eV)
 */
double compute_bandgap_change(const Eigen::Matrix3d& strain, double a_c, double a_v, double b);

/**
 * @brief Computes the strain-induced piezoelectric polarization.
 *
 * This function computes the strain-induced piezoelectric polarization.
 *
 * @param strain The strain tensor
 * @param piezoelectric_tensor The piezoelectric tensor (C/m²)
 * @return The piezoelectric polarization vector (C/m²)
 */
Eigen::Vector3d compute_piezoelectric_polarization(const Eigen::Matrix3d& strain, const Eigen::Matrix3d& piezoelectric_tensor);

/**
 * @brief Computes the strain-induced electric field.
 *
 * This function computes the strain-induced electric field from the polarization.
 *
 * @param polarization The polarization vector (C/m²)
 * @param epsilon_r The relative permittivity
 * @return The electric field vector (V/m)
 */
Eigen::Vector3d compute_strain_induced_field(const Eigen::Vector3d& polarization, double epsilon_r);

/**
 * @brief Computes the strain distribution in a quantum dot.
 *
 * This function computes the strain distribution in and around a spherical quantum dot.
 *
 * @param x The x-coordinate (nm)
 * @param y The y-coordinate (nm)
 * @param z The z-coordinate (nm)
 * @param R The radius of the quantum dot (nm)
 * @param a_dot The lattice constant of the quantum dot (nm)
 * @param a_matrix The lattice constant of the matrix (nm)
 * @return The strain tensor at the specified position
 */
Eigen::Matrix3d compute_quantum_dot_strain(double x, double y, double z, double R, double a_dot, double a_matrix);

/**
 * @brief Computes the strain energy density.
 *
 * This function computes the strain energy density.
 *
 * @param strain The strain tensor
 * @param stiffness The stiffness matrix (GPa)
 * @return The strain energy density (J/m³)
 */
double compute_strain_energy_density(const Eigen::Matrix3d& strain, const Eigen::Matrix6d& stiffness);

/**
 * @brief Computes the critical thickness for pseudomorphic growth.
 *
 * This function computes the critical thickness for pseudomorphic growth
 * using the Matthews-Blakeslee model.
 *
 * @param a_substrate The lattice constant of the substrate (nm)
 * @param a_layer The lattice constant of the layer (nm)
 * @param poisson_ratio The Poisson ratio of the layer
 * @param b The Burgers vector magnitude (nm)
 * @return The critical thickness (nm)
 */
double compute_critical_thickness(double a_substrate, double a_layer, double poisson_ratio, double b);

} // namespace StrainEffects
