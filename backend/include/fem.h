#pragma once
/**
 * @file fem.h
 * @brief Defines the FEMSolver class for quantum simulations.
 *
 * This file contains the declaration of the FEMSolver class, which implements
 * the finite element method for solving the Schrödinger equation in quantum
 * dot simulations. The solver supports linear (P1), quadratic (P2), and cubic (P3)
 * elements, and can use MPI for parallel computations.
 *
 * Physical units:
 * - Coordinates: nanometers (nm)
 * - Energy: electron volts (eV)
 * - Mass: effective electron mass (m_e)
 *
 * Assumptions and limitations:
 * - The solver uses the effective mass approximation
 * - The potential is assumed to be real-valued
 * - The solver supports 2D simulations only
 * - The solver can use MPI for parallel computations
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include "self_consistent.h"
#include "poisson.h"
#include "fe_interpolator.h"
#include "error_estimator.h"
#include "mesh_quality.h"
#include "adaptive_refinement.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <string>
#include <functional>
#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @class FEMSolver
 * @brief Implements the finite element method for quantum simulations.
 *
 * The FEMSolver class implements the finite element method for solving the
 * Schrödinger equation in quantum dot simulations. It assembles the Hamiltonian
 * and mass matrices, and provides methods for adaptive mesh refinement.
 *
 * The Schrödinger equation is solved in the form:
 * \f[ -\frac{\hbar^2}{2} \nabla \cdot \left(\frac{1}{m^*(\mathbf{r})} \nabla \psi(\mathbf{r})\right) + V(\mathbf{r}) \psi(\mathbf{r}) = E \psi(\mathbf{r}) \f]
 *
 * where \f$m^*(\mathbf{r})\f$ is the effective mass, \f$V(\mathbf{r})\f$ is the potential,
 * \f$\psi(\mathbf{r})\f$ is the wavefunction, and \f$E\f$ is the energy eigenvalue.
 */
class FEMSolver {
public:
    /**
     * @brief Constructs a new FEMSolver object.
     *
     * @param mesh The mesh to use for the simulation
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param cap Function that returns the capacitance at a given position
     * @param sc_solver The self-consistent solver to use for electrostatic calculations
     * @param order The order of the finite elements (1 for P1, 2 for P2, 3 for P3)
     * @param use_mpi Whether to use MPI for parallel computations (only effective if compiled with USE_MPI)
     *
     * @throws std::invalid_argument If the input parameters are invalid
     * @note If MPI support is not enabled at compile time (USE_MPI not defined),
     *       the use_mpi parameter will be ignored and serial execution will be used.
     */
    FEMSolver(Mesh& mesh, double (*m_star)(double, double), double (*V)(double, double),
              double (*cap)(double, double), SelfConsistentSolver& sc_solver, int order, bool use_mpi = false);

    /**
     * @brief Sets the potential values at mesh nodes for interpolation.
     *
     * This method sets the potential values at mesh nodes for proper finite element
     * interpolation. The potential values are used in the assemble_element_matrix method.
     *
     * @param potential_values The potential values at mesh nodes
     *
     * @throws std::invalid_argument If the potential_values vector has an invalid size
     */
    void set_potential_values(const Eigen::VectorXd& potential_values);

    /**
     * @brief Enables or disables the use of interpolated potentials.
     *
     * When enabled, the solver will use finite element interpolation for potentials
     * instead of calling the V function directly. This provides more accurate results,
     * especially for higher-order elements.
     *
     * @param enable Whether to enable interpolated potentials
     */
    void use_interpolated_potential(bool enable);
    /**
     * @brief Destroys the FEMSolver object.
     */
    ~FEMSolver();

    /**
     * @brief Assembles the Hamiltonian and mass matrices.
     *
     * This method assembles the Hamiltonian and mass matrices using the finite
     * element method. If MPI is enabled, the assembly is performed in parallel.
     *
     * @throws std::runtime_error If the assembly fails
     */
    void assemble_matrices();

    /**
     * @brief Adapts the mesh based on the given eigenvector.
     *
     * This method adapts the mesh by refining elements where the eigenvector
     * has a large gradient. The refinement is controlled by the threshold parameter.
     *
     * @param eigenvector The eigenvector to use for adaptation
     * @param threshold The threshold for refinement (elements with gradient > threshold are refined)
     * @param cache_dir The directory to use for caching intermediate results
     *
     * @throws std::invalid_argument If the eigenvector has an invalid size
     * @throws std::runtime_error If the adaptation fails
     */
    void adapt_mesh(const Eigen::VectorXd& eigenvector, double threshold, const std::string& cache_dir);

    /**
     * @brief Gets the Hamiltonian matrix.
     *
     * @return A reference to the Hamiltonian matrix
     */
    const Eigen::SparseMatrix<std::complex<double>>& get_H() const { return H; }

    /**
     * @brief Gets the mass matrix.
     *
     * @return A reference to the mass matrix
     */
    const Eigen::SparseMatrix<std::complex<double>>& get_M() const { return M; }

    /**
     * @brief Checks if MPI is being used.
     *
     * @return True if MPI is being used, false otherwise
     * @note This returns true only if both use_mpi is true AND the code was compiled with USE_MPI defined
     */
    bool is_using_mpi() const {
#ifdef USE_MPI
        return use_mpi;
#else
        return false;
#endif
    }

#ifdef USE_MPI
    /**
     * @brief Enables or disables MPI for parallel computations.
     *
     * This method allows enabling or disabling MPI at runtime. When enabled,
     * the solver will use MPI for parallel matrix assembly and mesh refinement.
     * When disabled, the solver will use serial implementations.
     *
     * @param enable Whether to enable MPI
     */
    void enable_mpi(bool enable);
#endif

    /**
     * @brief Gets the finite element interpolator.
     *
     * @return A pointer to the finite element interpolator
     */
    const FEInterpolator* get_interpolator() const { return interpolator; }

private:
    /** @brief Reference to the mesh used for the simulation */
    Mesh& mesh;

    /** @brief Reference to the Poisson solver used for electrostatic calculations */
    // PoissonSolver& poisson;
    SelfConsistentSolver& sc_solver;
    /** @brief Hamiltonian matrix (sparse complex matrix) */
    Eigen::SparseMatrix<std::complex<double>> H;

    /** @brief Mass matrix (sparse complex matrix) */
    Eigen::SparseMatrix<std::complex<double>> M;

    /** @brief Function that returns the effective mass at a given position */
    double (*m_star)(double, double);

    /** @brief Function that returns the potential at a given position */
    double (*V)(double, double);

    /** @brief Function that returns the capacitance at a given position */
    double (*cap)(double, double);

    /** @brief Order of the finite elements (1 for P1, 2 for P2, 3 for P3) */
    int order;

    /** @brief Flag to enable/disable MPI for parallel computations */
    bool use_mpi;

    /** @brief Finite element interpolator for field interpolation */
    FEInterpolator* interpolator;

    /** @brief Potential values at mesh nodes for interpolation */
    Eigen::VectorXd potential_values;

    /** @brief Flag to enable/disable interpolated potentials */
    bool use_interpolation;

    /**
     * @brief Maps a point from the reference element to the physical element.
     *
     * This method maps a point from the reference element (unit triangle) to the
     * physical element using linear interpolation.
     *
     * @param ref_point The point in the reference element
     * @param nodes The nodes of the physical element
     * @return Eigen::Vector2d The corresponding point in the physical element
     */
    Eigen::Vector2d map_reference_to_physical(const Eigen::Vector2d& ref_point,
                                            const std::vector<Eigen::Vector2d>& nodes) const;

    /**
     * @brief Evaluates the shape functions and their gradients at a point.
     *
     * This method evaluates the shape functions and their gradients at a point
     * in the reference element.
     *
     * @param ref_point The point in the reference element
     * @param shape_values Output parameter for the shape function values
     * @param shape_gradients Output parameter for the shape function gradients
     * @param nodes The nodes of the physical element
     * @param order The order of the finite elements
     */
    void evaluate_shape_functions(const Eigen::Vector2d& ref_point,
                                std::vector<double>& shape_values,
                                std::vector<Eigen::Vector2d>& shape_gradients,
                                const std::vector<Eigen::Vector2d>& nodes,
                                int order) const;

    /**
     * @brief Maps gradients from the reference element to the physical element.
     *
     * This method maps gradients from the reference element to the physical element
     * using the Jacobian of the transformation.
     *
     * @param ref_gradients The gradients in the reference element
     * @param phys_gradients Output parameter for the gradients in the physical element
     * @param nodes The nodes of the physical element
     */
    void map_gradients_to_physical(const std::vector<Eigen::Vector2d>& ref_gradients,
                                 std::vector<Eigen::Vector2d>& phys_gradients,
                                 const std::vector<Eigen::Vector2d>& nodes) const;

    /**
     * @brief Assembles the element matrices for a single element.
     *
     * This private method assembles the Hamiltonian and mass matrices for a single element.
     *
     * @param e The index of the element
     * @param H_e The element Hamiltonian matrix (output)
     * @param M_e The element mass matrix (output)
     */
    void assemble_element_matrix(size_t e, Eigen::MatrixXcd& H_e, Eigen::MatrixXcd& M_e);

    /**
     * @brief Assembles the matrices in serial mode.
     *
     * This private method assembles the Hamiltonian and mass matrices in serial mode.
     */
    void assemble_matrices_serial();

#ifdef USE_MPI
    /**
     * @brief Assembles the matrices in parallel mode using MPI.
     *
     * This private method assembles the Hamiltonian and mass matrices in parallel mode using MPI.
     */
    void assemble_matrices_parallel();
#endif
};

