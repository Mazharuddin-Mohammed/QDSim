/**
 * @file schrodinger.h
 * @brief Defines the SchrodingerSolver class for quantum simulations.
 *
 * This file contains the declaration of the SchrodingerSolver class, which implements
 * methods for solving the Schrödinger equation in quantum dot simulations. The solver
 * supports GPU acceleration for higher-order elements.
 *
 * Physical units:
 * - Coordinates: nanometers (nm)
 * - Energy: electron volts (eV)
 * - Mass: effective electron mass (m_e)
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include "mesh.h"
#include "gpu_accelerator.h"
#include <Eigen/Sparse>
#include <complex>
#include <functional>
#include <vector>

/**
 * @class SchrodingerSolver
 * @brief Implements the Schrödinger equation solver for quantum simulations.
 *
 * The SchrodingerSolver class implements methods for solving the Schrödinger equation
 * in quantum dot simulations. It assembles the Hamiltonian and mass matrices, and
 * solves the generalized eigenvalue problem to find the energy levels and wavefunctions.
 *
 * The Schrödinger equation is solved in the form:
 * \f[ -\frac{\hbar^2}{2} \nabla \cdot \left(\frac{1}{m^*(\mathbf{r})} \nabla \psi(\mathbf{r})\right) + V(\mathbf{r}) \psi(\mathbf{r}) = E \psi(\mathbf{r}) \f]
 *
 * where \f$m^*(\mathbf{r})\f$ is the effective mass, \f$V(\mathbf{r})\f$ is the potential,
 * \f$\psi(\mathbf{r})\f$ is the wavefunction, and \f$E\f$ is the energy eigenvalue.
 *
 * The solver supports GPU acceleration for higher-order elements, which can significantly
 * improve performance for large-scale simulations.
 */
class SchrodingerSolver {
public:
    /**
     * @brief Constructs a new SchrodingerSolver object.
     *
     * @param mesh The mesh on which to solve the Schrödinger equation
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param use_gpu Whether to use GPU acceleration (if available)
     */
    SchrodingerSolver(Mesh& mesh,
                    std::function<double(double, double)> m_star,
                    std::function<double(double, double)> V,
                    bool use_gpu = false);

    /**
     * @brief Solves the Schrödinger equation.
     *
     * This method solves the generalized eigenvalue problem arising from the
     * finite element discretization of the Schrödinger equation. It computes
     * the lowest `num_eigenvalues` eigenvalues and corresponding eigenvectors.
     *
     * @param num_eigenvalues The number of eigenvalues to compute
     * @return A pair containing the eigenvalues and eigenvectors
     */
    std::pair<std::vector<double>, std::vector<Eigen::VectorXd>> solve(int num_eigenvalues);

    /**
     * @brief Gets the Hamiltonian matrix.
     *
     * @return The Hamiltonian matrix
     */
    const Eigen::SparseMatrix<std::complex<double>>& get_H() const { return H; }

    /**
     * @brief Gets the mass matrix.
     *
     * @return The mass matrix
     */
    const Eigen::SparseMatrix<std::complex<double>>& get_M() const { return M; }

    /**
     * @brief Gets the mesh.
     *
     * @return The mesh
     */
    const Mesh& get_mesh() const { return mesh; }

    /**
     * @brief Gets the eigenvalues.
     *
     * @return The eigenvalues
     */
    const std::vector<double>& get_eigenvalues() const { return eigenvalues; }

    /**
     * @brief Gets the eigenvectors.
     *
     * @return The eigenvectors
     */
    const std::vector<Eigen::VectorXd>& get_eigenvectors() const { return eigenvectors; }

    /**
     * @brief Checks if GPU acceleration is enabled.
     *
     * @return True if GPU acceleration is enabled, false otherwise
     */
    bool is_gpu_enabled() const { return use_gpu && gpu_accelerator.is_gpu_enabled(); }

    /**
     * @brief Gets the GPU accelerator.
     *
     * @return The GPU accelerator
     */
    const GPUAccelerator& get_gpu_accelerator() const { return gpu_accelerator; }

private:
    /**
     * @brief Assembles the Hamiltonian and mass matrices.
     *
     * This method assembles the Hamiltonian and mass matrices for the
     * finite element discretization of the Schrödinger equation.
     */
    void assemble_matrices();

    /**
     * @brief Assembles the Hamiltonian and mass matrices on the CPU.
     *
     * This method assembles the Hamiltonian and mass matrices on the CPU.
     */
    void assemble_matrices_cpu();

    /**
     * @brief Assembles the Hamiltonian and mass matrices on the GPU.
     *
     * This method assembles the Hamiltonian and mass matrices on the GPU.
     */
    void assemble_matrices_gpu();

    /**
     * @brief Solves the generalized eigenvalue problem on the CPU.
     *
     * This method solves the generalized eigenvalue problem on the CPU.
     *
     * @param num_eigenvalues The number of eigenvalues to compute
     */
    void solve_eigen_cpu(int num_eigenvalues);

    /**
     * @brief Solves the generalized eigenvalue problem on the GPU.
     *
     * This method solves the generalized eigenvalue problem on the GPU.
     *
     * @param num_eigenvalues The number of eigenvalues to compute
     */
    void solve_eigen_gpu(int num_eigenvalues);

    Mesh& mesh;                                        ///< The mesh on which to solve the Schrödinger equation
    std::function<double(double, double)> m_star;      ///< Function that returns the effective mass at a given position
    std::function<double(double, double)> V;           ///< Function that returns the potential at a given position
    Eigen::SparseMatrix<std::complex<double>> H;       ///< The Hamiltonian matrix
    Eigen::SparseMatrix<std::complex<double>> M;       ///< The mass matrix
    std::vector<double> eigenvalues;                   ///< The computed eigenvalues
    std::vector<Eigen::VectorXd> eigenvectors;         ///< The computed eigenvectors
    bool use_gpu;                                      ///< Whether to use GPU acceleration
    GPUAccelerator gpu_accelerator;                    ///< The GPU accelerator
};
