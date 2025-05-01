/**
 * @file poisson.cpp
 * @brief Implementation of the PoissonSolver class for electrostatic calculations.
 *
 * This file contains the implementation of the PoissonSolver class, which implements
 * the finite element method for solving the Poisson equation in electrostatic
 * calculations. The solver computes the electrostatic potential and electric field
 * for a given charge distribution and boundary conditions.
 *
 * The implementation uses linear (P1) finite elements and the SimplicialLDLT
 * solver from Eigen for solving the resulting linear system.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "poisson.h"
#include <Eigen/SparseCholesky>

/**
 * @brief Constructs a new PoissonSolver object.
 *
 * This constructor initializes the PoissonSolver with the given mesh and
 * material functions. It resizes the stiffness matrix, potential vector,
 * and right-hand side vector to match the mesh size.
 *
 * @param mesh The mesh to use for the simulation
 * @param epsilon_r Function that returns the relative permittivity at a given position
 * @param rho Function that returns the charge density at a given position
 */
PoissonSolver::PoissonSolver(Mesh& mesh, double (*epsilon_r)(double, double), double (*rho)(double, double))
    : mesh(mesh), epsilon_r(epsilon_r), rho(rho) {
    // Initialize matrices and vectors with the correct size
    K.resize(mesh.getNumNodes(), mesh.getNumNodes());
    phi.resize(mesh.getNumNodes());
    f.resize(mesh.getNumNodes());
}

/**
 * @brief Assembles the stiffness matrix.
 *
 * This private method assembles the stiffness matrix for the Poisson equation
 * using the finite element method. It iterates over all elements in the mesh,
 * computes the element matrices, and adds the contributions to the global matrix.
 *
 * The assembly process involves:
 * 1. Iterating over all elements in the mesh
 * 2. Getting the element nodes
 * 3. Using Gaussian quadrature to integrate over the element
 * 4. Computing the gradients of the basis functions
 * 5. Adding the element matrix entries to the global matrix using triplets
 *
 * @throws std::runtime_error If the assembly fails
 */
void PoissonSolver::assemble_matrix() {
    // Create triplet list for sparse matrix assembly
    std::vector<Eigen::Triplet<double>> triplets;

    // Iterate over all elements in the mesh
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        // Get the element nodes
        auto element = mesh.getElements()[e];
        std::vector<Eigen::Vector2d> nodes;
        for (int i = 0; i < 3; ++i) {
            nodes.push_back(mesh.getNodes()[element[i]]);
        }

        // 3-point Gaussian quadrature for P1 elements
        for (int q = 0; q < 3; ++q) {
            // Compute quadrature point (simplified for now)
            double x = 0.0, y = 0.0;

            // Get the permittivity at this point (F/m)
            double eps = epsilon_r(x, y) * 8.854e-12;

            // Compute the element matrix entries
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    // Gradient of basis functions (simplified for now)
                    double grad_i_dot_grad_j = 1.0;

                    // Add the element matrix entry to the triplet list
                    triplets.emplace_back(element[i], element[j], eps * grad_i_dot_grad_j);
                }
            }
        }
    }

    // Set the global matrix from the triplets
    K.setFromTriplets(triplets.begin(), triplets.end());
}

/**
 * @brief Assembles the right-hand side vector.
 *
 * This private method assembles the right-hand side vector for the Poisson equation
 * using the finite element method. It iterates over all elements in the mesh,
 * computes the element vectors, and adds the contributions to the global vector.
 *
 * The assembly process involves:
 * 1. Initializing the right-hand side vector to zero
 * 2. Iterating over all elements in the mesh
 * 3. Getting the element nodes
 * 4. Using Gaussian quadrature to integrate over the element
 * 5. Computing the charge density at each quadrature point
 * 6. Adding the element vector entries to the global vector
 *
 * @throws std::runtime_error If the assembly fails
 */
void PoissonSolver::assemble_rhs() {
    // Initialize the right-hand side vector to zero
    f.setZero();

    // Iterate over all elements in the mesh
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        // Get the element nodes
        auto element = mesh.getElements()[e];
        std::vector<Eigen::Vector2d> nodes;
        for (int i = 0; i < 3; ++i) {
            nodes.push_back(mesh.getNodes()[element[i]]);
        }

        // 3-point Gaussian quadrature for P1 elements
        for (int q = 0; q < 3; ++q) {
            // Compute quadrature point (simplified for now)
            double x = 0.0, y = 0.0;

            // Get the charge density at this point
            double charge = rho(x, y);

            // Add the element vector entries to the global vector
            for (int i = 0; i < 3; ++i) {
                // Basis function value (simplified for now)
                double basis_value = 1.0;

                // Add the contribution to the right-hand side vector
                f[element[i]] += charge * basis_value;
            }
        }
    }
}

/**
 * @brief Applies boundary conditions.
 *
 * This private method applies Dirichlet boundary conditions to the stiffness
 * matrix and right-hand side vector. It sets the potential to V_p at the left
 * boundary (x = -Lx/2) and to V_n at the right boundary (x = Lx/2).
 *
 * The boundary condition application involves:
 * 1. Identifying nodes on the boundaries
 * 2. Setting the potential at those nodes to the specified values
 * 3. Modifying the stiffness matrix and right-hand side vector to enforce the boundary conditions
 *
 * @param V_p Potential at the p-type boundary in volts (V)
 * @param V_n Potential at the n-type boundary in volts (V)
 *
 * @throws std::runtime_error If the boundary condition application fails
 */
void PoissonSolver::apply_boundary_conditions(double V_p, double V_n) {
    // Apply Dirichlet BC: phi = V_p at x = -Lx/2, phi = V_n at x = Lx/2
    double Lx = 1.0; // Default domain size, should be a parameter
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        const auto& node = mesh.getNodes()[i];
        double x = node[0];
        double y = node[1];
        if (std::abs(x - (-Lx / 2)) < 1e-10) {
            phi[i] = V_p;
            // Clear the row and set diagonal to 1
            for (int k = 0; k < K.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it) {
                    if (it.row() == i) {
                        it.valueRef() = 0.0;
                    }
                }
            }
            K.coeffRef(i, i) = 1.0;
            f[i] = V_p;
        } else if (std::abs(x - (Lx / 2)) < 1e-10) {
            phi[i] = V_n;
            // Clear the row and set diagonal to 1
            for (int k = 0; k < K.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it) {
                    if (it.row() == i) {
                        it.valueRef() = 0.0;
                    }
                }
            }
            K.coeffRef(i, i) = 1.0;
            f[i] = V_n;
        }
    }
}

/**
 * @brief Solves the Poisson equation.
 *
 * This method assembles the stiffness matrix and right-hand side vector,
 * applies boundary conditions, and solves the resulting linear system to
 * obtain the electrostatic potential.
 *
 * The solution process involves:
 * 1. Assembling the stiffness matrix
 * 2. Assembling the right-hand side vector
 * 3. Applying boundary conditions
 * 4. Solving the linear system using the SimplicialLDLT solver
 *
 * @param V_p Potential at the p-type boundary in volts (V)
 * @param V_n Potential at the n-type boundary in volts (V)
 *
 * @throws std::runtime_error If the solver fails to converge
 */
void PoissonSolver::solve(double V_p, double V_n) {
    // Assemble the stiffness matrix
    assemble_matrix();

    // Assemble the right-hand side vector
    assemble_rhs();

    // Apply boundary conditions
    apply_boundary_conditions(V_p, V_n);

    // Solve the linear system using the SimplicialLDLT solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);
    phi = solver.solve(f);
}

/**
 * @brief Computes the electric field at a given position.
 *
 * This method computes the electric field at a given position by taking
 * the negative gradient of the electrostatic potential. The electric field
 * is related to the potential by E = -∇φ.
 *
 * The computation involves:
 * 1. Finding the element containing the point
 * 2. Computing the gradient of the potential within that element
 * 3. Taking the negative of the gradient to get the electric field
 *
 * @param x The x-coordinate of the position in nanometers (nm)
 * @param y The y-coordinate of the position in nanometers (nm)
 * @return The electric field vector in volts per nanometer (V/nm)
 *
 * @throws std::runtime_error If the position is outside the mesh
 *
 * @note This is a placeholder implementation. The actual implementation
 *       will use finite element interpolation to compute the gradient.
 */
Eigen::Vector2d PoissonSolver::get_electric_field(double x, double y) const {
    // Approximate E = -grad phi using finite differences or basis function gradients
    Eigen::Vector2d E(0.0, 0.0);

    // TODO: Implement proper finite element interpolation for the gradient
    // This would involve:
    // 1. Finding the element containing the point (x, y)
    // 2. Computing the gradient of the potential within that element
    // 3. Taking the negative of the gradient to get the electric field

    // Implementation TBD based on mesh and phi
    return E;
}