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
#include "simple_mesh.h"
#include "simple_interpolator.h"
#include <Eigen/SparseCholesky>
#include <iostream>
#include <stdexcept>

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
PoissonSolver::PoissonSolver(Mesh& mesh, double (*epsilon_r)(double, double),
                             double (*rho)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&))
    : mesh(mesh), epsilon_r(epsilon_r), rho(rho) {
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

        // Calculate element area
        double x1 = nodes[0][0], y1 = nodes[0][1];
        double x2 = nodes[1][0], y2 = nodes[1][1];
        double x3 = nodes[2][0], y3 = nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Calculate shape function gradients
        // For linear elements, the gradients are constant over the element
        std::vector<Eigen::Vector2d> grad_N(3);

        // Calculate the Jacobian matrix for mapping from reference to physical element
        Eigen::Matrix2d J;
        J << x2 - x1, x3 - x1,
             y2 - y1, y3 - y1;

        // Calculate the inverse of the Jacobian
        Eigen::Matrix2d J_inv = J.inverse();

        // Reference element gradients
        std::vector<Eigen::Vector2d> ref_grad = {
            {-1.0, -1.0},  // dN1/d(xi), dN1/d(eta)
            {1.0, 0.0},    // dN2/d(xi), dN2/d(eta)
            {0.0, 1.0}     // dN3/d(xi), dN3/d(eta)
        };

        // Transform gradients from reference to physical element
        for (int i = 0; i < 3; ++i) {
            grad_N[i] = J_inv.transpose() * ref_grad[i];
        }

        // Calculate element centroid
        double xc = (x1 + x2 + x3) / 3.0;
        double yc = (y1 + y2 + y3) / 3.0;

        // Get the permittivity at the centroid
        double eps = epsilon_r(xc, yc) * 8.854e-12;

        // Compute the element matrix entries
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Gradient of basis functions dot product
                double grad_i_dot_grad_j = grad_N[i].dot(grad_N[j]);

                // Add the element matrix entry to the triplet list
                triplets.emplace_back(element[i], element[j], eps * grad_i_dot_grad_j * area);
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
void PoissonSolver::assemble_rhs(const Eigen::VectorXd& n, const Eigen::VectorXd& p) {
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

        // Calculate element area
        double x1 = nodes[0][0], y1 = nodes[0][1];
        double x2 = nodes[1][0], y2 = nodes[1][1];
        double x3 = nodes[2][0], y3 = nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Gauss quadrature points and weights for triangular elements
        std::vector<std::array<double, 3>> quad_points = {
            {1.0/6.0, 1.0/6.0, 2.0/3.0},
            {1.0/6.0, 2.0/3.0, 1.0/6.0},
            {2.0/3.0, 1.0/6.0, 1.0/6.0}
        };
        std::vector<double> quad_weights = {1.0/6.0, 1.0/6.0, 1.0/6.0};

        // Integrate over quadrature points
        for (size_t q = 0; q < quad_points.size(); ++q) {
            // Barycentric coordinates
            double lambda1 = quad_points[q][0];
            double lambda2 = quad_points[q][1];
            double lambda3 = quad_points[q][2];

            // Calculate position at quadrature point
            double x_q = lambda1 * x1 + lambda2 * x2 + lambda3 * x3;
            double y_q = lambda1 * y1 + lambda2 * y2 + lambda3 * y3;

            // Get the charge density at this point
            double charge = rho(x_q, y_q, n, p);

            // Add the element vector entries to the global vector
            for (int i = 0; i < 3; ++i) {
                // Basis function value at quadrature point
                double basis_value;
                if (i == 0) basis_value = lambda1;
                else if (i == 1) basis_value = lambda2;
                else basis_value = lambda3;

                // Add the contribution to the right-hand side vector
                f[element[i]] += charge * basis_value * quad_weights[q] * area;
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
    double Lx = mesh.get_lx(); // Get the actual domain size from the mesh
    double Ly = mesh.get_ly();
    double tolerance = 1e-10; // Tolerance for boundary detection

    // Create a vector to mark boundary nodes
    std::vector<bool> is_boundary(mesh.getNumNodes(), false);
    std::vector<double> boundary_values(mesh.getNumNodes(), 0.0);

    // Identify boundary nodes and set their values
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        const auto& node = mesh.getNodes()[i];
        double x = node[0];
        double y = node[1];

        // Left boundary (p-type)
        if (std::abs(x - (-Lx / 2)) < tolerance) {
            is_boundary[i] = true;
            boundary_values[i] = V_p;
            phi[i] = V_p;
        }
        // Right boundary (n-type)
        else if (std::abs(x - (Lx / 2)) < tolerance) {
            is_boundary[i] = true;
            boundary_values[i] = V_n;
            phi[i] = V_n;
        }
        // Top and bottom boundaries (Neumann boundary conditions)
        // We don't need to do anything special for Neumann boundaries in the FEM formulation
    }

    // Apply Dirichlet boundary conditions to the stiffness matrix and right-hand side vector
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        if (is_boundary[i]) {
            // Modify the right-hand side vector
            for (int k = 0; k < K.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it) {
                    if (it.row() == i) {
                        // Clear the row
                        it.valueRef() = 0.0;
                    }
                    else if (it.col() == i) {
                        // Modify the right-hand side vector
                        f[it.row()] -= it.value() * boundary_values[i];
                    }
                }
            }

            // Set the diagonal entry to 1
            K.coeffRef(i, i) = 1.0;

            // Set the right-hand side value
            f[i] = boundary_values[i];
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
    // Create empty electron and hole concentration vectors
    Eigen::VectorXd n = Eigen::VectorXd::Zero(mesh.getNumNodes());
    Eigen::VectorXd p = Eigen::VectorXd::Zero(mesh.getNumNodes());

    // Call the overloaded solve method
    solve(V_p, V_n, n, p);
}

/**
 * @brief Solves the Poisson equation with specified carrier concentrations.
 *
 * This method assembles the stiffness matrix and right-hand side vector,
 * applies boundary conditions, and solves the resulting linear system to
 * obtain the electrostatic potential. It uses the provided electron and
 * hole concentrations to calculate the charge density.
 *
 * The solution process involves:
 * 1. Assembling the stiffness matrix
 * 2. Assembling the right-hand side vector
 * 3. Applying boundary conditions
 * 4. Solving the linear system using the SimplicialLDLT solver
 *
 * @param V_p Potential at the p-type boundary in volts (V)
 * @param V_n Potential at the n-type boundary in volts (V)
 * @param n The electron concentration at each node
 * @param p The hole concentration at each node
 *
 * @throws std::runtime_error If the solver fails to converge
 */
void PoissonSolver::solve(double V_p, double V_n, const Eigen::VectorXd& n, const Eigen::VectorXd& p) {
    // Assemble the stiffness matrix
    assemble_matrix();

    // Assemble the right-hand side vector
    assemble_rhs(n, p);

    // Apply boundary conditions
    apply_boundary_conditions(V_p, V_n);

    // Solve the linear system using the SimplicialLDLT solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);
    phi = solver.solve(f);
}

/**
 * @brief Sets the potential values directly.
 *
 * This method allows setting the potential values directly, which is useful
 * for implementing custom solvers or for testing.
 *
 * @param potential The potential values to set
 */
void PoissonSolver::set_potential(const Eigen::VectorXd& potential) {
    // Check that the potential vector has the correct size
    if (potential.size() != mesh.getNumNodes()) {
        throw std::invalid_argument("Potential vector size does not match number of mesh nodes");
    }

    // Set the potential values
    phi = potential;
}

/**
 * @brief Updates the potential values and solves the Poisson equation.
 *
 * This method updates the potential values and then solves the Poisson equation
 * with the updated values. It's useful for implementing self-consistent solvers
 * that need to update the potential iteratively.
 *
 * @param potential The potential values to set
 * @param V_p Potential at the p-type boundary in volts (V)
 * @param V_n Potential at the n-type boundary in volts (V)
 * @param n The electron concentration at each node
 * @param p The hole concentration at each node
 */
void PoissonSolver::update_and_solve(const Eigen::VectorXd& potential, double V_p, double V_n,
                                   const Eigen::VectorXd& n, const Eigen::VectorXd& p) {
    // Set the potential values
    set_potential(potential);

    // Assemble the stiffness matrix
    assemble_matrix();

    // Assemble the right-hand side vector
    assemble_rhs(n, p);

    // Apply boundary conditions
    apply_boundary_conditions(V_p, V_n);

    // Solve the linear system using the SimplicialLDLT solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);

    // Check if the factorization was successful
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize the stiffness matrix");
    }

    // Solve the linear system
    phi = solver.solve(f);

    // Check if the solve was successful
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve the linear system");
    }
}

/**
 * @brief Initializes the PoissonSolver with a new mesh and functions.
 *
 * This method reinitializes the PoissonSolver with a new mesh and functions.
 * It's useful when the mesh has been refined or changed.
 *
 * @param mesh The mesh to use
 * @param epsilon_r Function that returns the relative permittivity at a given position
 * @param rho Function that returns the charge density at a given position
 */
void PoissonSolver::initialize(Mesh& mesh, double (*epsilon_r)(double, double),
                             double (*rho)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)) {
    // Store the function pointers
    this->epsilon_r = epsilon_r;
    this->rho = rho;

    // Resize the potential vector
    phi.resize(mesh.getNumNodes());
    phi.setZero();

    // Resize the stiffness matrix and right-hand side vector
    K.resize(mesh.getNumNodes(), mesh.getNumNodes());
    f.resize(mesh.getNumNodes());

    // Initialize the stiffness matrix and right-hand side vector
    K.setZero();
    f.setZero();
}

/**
 * @brief Sets the charge density values directly.
 *
 * This method allows setting the charge density values directly, which is useful
 * for implementing custom solvers or for testing.
 *
 * @param charge_density The charge density values to set
 */
void PoissonSolver::set_charge_density(const Eigen::VectorXd& charge_density) {
    // Check that the charge density vector has the correct size
    if (charge_density.size() != mesh.getNumNodes()) {
        throw std::invalid_argument("Charge density vector size does not match number of mesh nodes");
    }

    // Set the right-hand side vector directly
    f = charge_density;

    // We need to assemble the stiffness matrix if it hasn't been done yet
    if (K.nonZeros() == 0) {
        assemble_matrix();
    }
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
 */
Eigen::Vector2d PoissonSolver::get_electric_field(double x, double y) const {
    // Create a simple mesh from the mesh nodes and elements
    std::vector<Eigen::Vector2d> nodes;
    std::vector<std::array<int, 3>> elements;

    // Convert mesh nodes to Eigen::Vector2d
    for (const auto& node : mesh.getNodes()) {
        nodes.push_back(Eigen::Vector2d(node[0], node[1]));
    }

    // Convert mesh elements to std::array<int, 3>
    for (const auto& element : mesh.getElements()) {
        elements.push_back({element[0], element[1], element[2]});
    }

    // Create a simple mesh and interpolator
    SimpleMesh simple_mesh(nodes, elements);
    SimpleInterpolator interpolator(simple_mesh);

    // Find the element containing the point (x, y)
    int elem_idx = interpolator.findElement(x, y);

    if (elem_idx >= 0) {
        // Get the element
        const auto& element = elements[elem_idx];

        // Get the node coordinates and potential values
        double x1 = nodes[element[0]][0];
        double y1 = nodes[element[0]][1];
        double x2 = nodes[element[1]][0];
        double y2 = nodes[element[1]][1];
        double x3 = nodes[element[2]][0];
        double y3 = nodes[element[2]][1];

        double phi1 = phi[element[0]];
        double phi2 = phi[element[1]];
        double phi3 = phi[element[2]];

        // Compute the gradient using shape function derivatives
        // For linear elements, the gradient is constant within each element
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Compute the derivatives of the shape functions
        double dN1_dx = (y2 - y3) / (2.0 * area);
        double dN1_dy = (x3 - x2) / (2.0 * area);
        double dN2_dx = (y3 - y1) / (2.0 * area);
        double dN2_dy = (x1 - x3) / (2.0 * area);
        double dN3_dx = (y1 - y2) / (2.0 * area);
        double dN3_dy = (x2 - x1) / (2.0 * area);

        // Compute the gradient of the potential
        double dphi_dx = phi1 * dN1_dx + phi2 * dN2_dx + phi3 * dN3_dx;
        double dphi_dy = phi1 * dN1_dy + phi2 * dN2_dy + phi3 * dN3_dy;

        // The electric field is the negative gradient of the potential
        return Eigen::Vector2d(-dphi_dx, -dphi_dy);
    } else {
        // If the point is outside the mesh, return zero electric field
        return Eigen::Vector2d::Zero();
    }
}