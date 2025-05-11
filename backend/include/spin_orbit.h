#pragma once
/**
 * @file spin_orbit.h
 * @brief Defines the SpinOrbitCoupling class for including spin-orbit effects in quantum simulations.
 *
 * This file contains the declaration of the SpinOrbitCoupling class, which provides
 * methods for including spin-orbit coupling effects in quantum simulations.
 * It supports both Rashba and Dresselhaus spin-orbit coupling.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include "materials.h"
#include <Eigen/Sparse>
#include <complex>
#include <vector>
#include <string>
#include <functional>

/**
 * @enum SpinOrbitType
 * @brief Enumeration of spin-orbit coupling types.
 */
enum class SpinOrbitType {
    NONE,       ///< No spin-orbit coupling
    RASHBA,     ///< Rashba spin-orbit coupling
    DRESSELHAUS, ///< Dresselhaus spin-orbit coupling
    BOTH        ///< Both Rashba and Dresselhaus spin-orbit coupling
};

/**
 * @class SpinOrbitCoupling
 * @brief Provides methods for including spin-orbit coupling effects in quantum simulations.
 *
 * The SpinOrbitCoupling class provides methods for including spin-orbit coupling
 * effects in quantum simulations. It supports both Rashba and Dresselhaus
 * spin-orbit coupling, which arise from structural and bulk inversion asymmetry,
 * respectively.
 */
class SpinOrbitCoupling {
public:
    /**
     * @brief Constructs a new SpinOrbitCoupling object.
     *
     * @param mesh The mesh to use for the simulation
     * @param material The material to use for the simulation
     * @param type The type of spin-orbit coupling to include
     * @param rashba_parameter The Rashba parameter (in eV·nm)
     * @param dresselhaus_parameter The Dresselhaus parameter (in eV·nm)
     */
    SpinOrbitCoupling(const Mesh& mesh, const Materials::Material& material, SpinOrbitType type = SpinOrbitType::NONE,
                     double rashba_parameter = 0.0, double dresselhaus_parameter = 0.0);

    /**
     * @brief Assembles the spin-orbit coupling Hamiltonian matrix.
     *
     * This method assembles the spin-orbit coupling Hamiltonian matrix using
     * the finite element method. The matrix is added to the existing Hamiltonian
     * matrix to include spin-orbit coupling effects.
     *
     * @param H The Hamiltonian matrix to modify
     * @param electric_field_x Function that returns the x-component of the electric field at a given position
     * @param electric_field_y Function that returns the y-component of the electric field at a given position
     * @param electric_field_z Function that returns the z-component of the electric field at a given position
     *
     * @throws std::runtime_error If the assembly fails
     */
    void assemble_spin_orbit_hamiltonian(Eigen::SparseMatrix<std::complex<double>>& H,
                                        std::function<double(double, double)> electric_field_x = nullptr,
                                        std::function<double(double, double)> electric_field_y = nullptr,
                                        std::function<double(double, double)> electric_field_z = nullptr);

    /**
     * @brief Sets the type of spin-orbit coupling to include.
     *
     * @param type The type of spin-orbit coupling to include
     */
    void set_spin_orbit_type(SpinOrbitType type);

    /**
     * @brief Sets the Rashba parameter.
     *
     * @param parameter The Rashba parameter (in eV·nm)
     */
    void set_rashba_parameter(double parameter);

    /**
     * @brief Sets the Dresselhaus parameter.
     *
     * @param parameter The Dresselhaus parameter (in eV·nm)
     */
    void set_dresselhaus_parameter(double parameter);

    /**
     * @brief Gets the type of spin-orbit coupling.
     *
     * @return The type of spin-orbit coupling
     */
    SpinOrbitType get_spin_orbit_type() const;

    /**
     * @brief Gets the Rashba parameter.
     *
     * @return The Rashba parameter (in eV·nm)
     */
    double get_rashba_parameter() const;

    /**
     * @brief Gets the Dresselhaus parameter.
     *
     * @return The Dresselhaus parameter (in eV·nm)
     */
    double get_dresselhaus_parameter() const;

    /**
     * @brief Calculates the Rashba parameter from the electric field.
     *
     * This method calculates the Rashba parameter from the electric field
     * using the formula α = α_0 * E_z, where α_0 is the Rashba coefficient
     * and E_z is the z-component of the electric field.
     *
     * @param material The material to use for the calculation
     * @param electric_field_z The z-component of the electric field (in V/m)
     * @return The Rashba parameter (in eV·nm)
     */
    static double calculate_rashba_parameter(const Materials::Material& material, double electric_field_z);

    /**
     * @brief Calculates the Dresselhaus parameter from the material properties.
     *
     * This method calculates the Dresselhaus parameter from the material properties
     * using the formula γ = γ_0 * (π/a)^2, where γ_0 is the Dresselhaus coefficient
     * and a is the lattice constant.
     *
     * @param material The material to use for the calculation
     * @return The Dresselhaus parameter (in eV·nm)
     */
    static double calculate_dresselhaus_parameter(const Materials::Material& material);

private:
    const Mesh& mesh_;                  ///< The mesh to use for the simulation
    const Materials::Material& material_;          ///< The material to use for the simulation
    SpinOrbitType type_;                ///< The type of spin-orbit coupling to include
    double rashba_parameter_;           ///< The Rashba parameter (in eV·nm)
    double dresselhaus_parameter_;      ///< The Dresselhaus parameter (in eV·nm)

    /**
     * @brief Assembles the Rashba spin-orbit coupling Hamiltonian matrix.
     *
     * This private method assembles the Rashba spin-orbit coupling Hamiltonian
     * matrix using the finite element method.
     *
     * @param H The Hamiltonian matrix to modify
     * @param electric_field_z Function that returns the z-component of the electric field at a given position
     */
    void assemble_rashba_hamiltonian(Eigen::SparseMatrix<std::complex<double>>& H,
                                   std::function<double(double, double)> electric_field_z);

    /**
     * @brief Assembles the Dresselhaus spin-orbit coupling Hamiltonian matrix.
     *
     * This private method assembles the Dresselhaus spin-orbit coupling Hamiltonian
     * matrix using the finite element method.
     *
     * @param H The Hamiltonian matrix to modify
     */
    void assemble_dresselhaus_hamiltonian(Eigen::SparseMatrix<std::complex<double>>& H);

    /**
     * @brief Computes the Pauli matrices.
     *
     * This private method computes the Pauli matrices σ_x, σ_y, and σ_z,
     * which are used in the spin-orbit coupling Hamiltonian.
     *
     * @param sigma_x The σ_x Pauli matrix (output)
     * @param sigma_y The σ_y Pauli matrix (output)
     * @param sigma_z The σ_z Pauli matrix (output)
     */
    void compute_pauli_matrices(Eigen::Matrix2cd& sigma_x, Eigen::Matrix2cd& sigma_y, Eigen::Matrix2cd& sigma_z) const;
};
