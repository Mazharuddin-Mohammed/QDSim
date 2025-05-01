/**
 * @file bindings.cpp
 * @brief Python bindings for the C++ library.
 *
 * This file contains the Python bindings for the C++ library using pybind11.
 * It exposes the C++ classes and functions to Python, allowing them to be
 * used from Python code.
 *
 * The bindings include:
 * - Mesh class for finite element discretization
 * - FEInterpolator class for field interpolation
 * - MaterialDatabase class for material properties
 * - PoissonSolver class for electrostatic calculations
 * - FEMSolver class for quantum simulations
 * - EigenSolver class for eigenvalue problems
 * - Physics functions for material properties and potentials
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include "fem.h"
#include "solver.h"
#include "poisson.h"
#include "materials.h"
#include "physics.h"
#include "fe_interpolator.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

/**
 * @brief Python module definition for the C++ library.
 *
 * This macro defines the Python module for the C++ library.
 * The module name is qdsim_cpp, and it exposes the C++ classes
 * and functions to Python.
 */
PYBIND11_MODULE(qdsim_cpp, m) {
    // Set module docstring
    m.doc() = "QDSim C++ backend for quantum dot simulations";

    // Mesh class for finite element discretization
    pybind11::class_<Mesh>(m, "Mesh")
        .def(pybind11::init<double, double, int, int, int>(),
             pybind11::arg("Lx"), pybind11::arg("Ly"), pybind11::arg("nx"), pybind11::arg("ny"), pybind11::arg("element_order"),
             "Construct a new Mesh object with the specified dimensions, number of elements, and element order")
        .def("get_nodes", &Mesh::getNodes, "Get the mesh nodes")
        .def("get_elements", &Mesh::getElements, "Get the mesh elements")
        .def("get_num_nodes", &Mesh::getNumNodes, "Get the number of nodes in the mesh")
        .def("get_num_elements", &Mesh::getNumElements, "Get the number of elements in the mesh")
        .def("get_element_order", &Mesh::getElementOrder, "Get the element order (1 for P1, 2 for P2, 3 for P3)")
        .def("get_lx", &Mesh::get_lx, "Get the width of the domain")
        .def("get_ly", &Mesh::get_ly, "Get the height of the domain")
        .def("get_nx", &Mesh::get_nx, "Get the number of elements in the x-direction")
        .def("get_ny", &Mesh::get_ny, "Get the number of elements in the y-direction")
        .def("getNumNodes", &Mesh::getNumNodes, "Get the number of nodes in the mesh (alias for get_num_nodes)")
        .def("getNumElements", &Mesh::getNumElements, "Get the number of elements in the mesh (alias for get_num_elements)")
        .def("getElementOrder", &Mesh::getElementOrder, "Get the element order (alias for get_element_order)");

    // FEInterpolator class for field interpolation
    pybind11::class_<FEInterpolator>(m, "FEInterpolator")
        .def(pybind11::init<const Mesh&>(),
             pybind11::arg("mesh"),
             "Construct a new FEInterpolator object for the specified mesh")
        .def("interpolate", &FEInterpolator::interpolate,
             pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("field"),
             "Interpolate a scalar field at a point (x, y)")
        .def("interpolate_with_gradient", [](FEInterpolator& self, double x, double y, const Eigen::VectorXd& field) {
                double grad_x = 0.0, grad_y = 0.0;
                double value = self.interpolateWithGradient(x, y, field, grad_x, grad_y);
                return std::make_tuple(value, grad_x, grad_y);
             },
             pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("field"),
             "Interpolate a scalar field and its gradient at a point (x, y)")
        .def("find_element", &FEInterpolator::findElement,
             pybind11::arg("x"), pybind11::arg("y"),
             "Find the element containing a point (x, y)")
        .def("compute_barycentric_coordinates", [](FEInterpolator& self, double x, double y, const std::vector<Eigen::Vector2d>& vertices) {
                std::vector<double> lambda(3);
                bool inside = self.computeBarycentricCoordinates(x, y, vertices, lambda);
                return std::make_tuple(inside, lambda);
             },
             pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("vertices"),
             "Compute the barycentric coordinates of a point in a triangle")
        .def("evaluate_shape_functions", &FEInterpolator::evaluateShapeFunctions,
             pybind11::arg("lambda"), pybind11::arg("shape_values"),
             "Evaluate the shape functions at a point given by barycentric coordinates")
        .def("evaluate_shape_function_gradients", &FEInterpolator::evaluateShapeFunctionGradients,
             pybind11::arg("lambda"), pybind11::arg("vertices"), pybind11::arg("shape_gradients"),
             "Evaluate the shape function gradients at a point given by barycentric coordinates");

    // MaterialDatabase class for material properties
    pybind11::class_<Materials::MaterialDatabase>(m, "MaterialDatabase")
        .def(pybind11::init<>(),
             "Construct a new MaterialDatabase object with default materials")
        .def("get_material", [](const Materials::MaterialDatabase& db, const std::string& name) {
                auto mat = db.get_material(name);
                return std::make_tuple(mat.m_e, mat.m_h, mat.E_g, mat.Delta_E_c, mat.epsilon_r);
             },
             pybind11::arg("name"),
             "Get the properties of a material by name, returns a tuple of (m_e, m_h, E_g, Delta_E_c, epsilon_r)");

    // PoissonSolver class for electrostatic calculations
    pybind11::class_<PoissonSolver>(m, "PoissonSolver")
        .def(pybind11::init<Mesh&, double(*)(double, double), double(*)(double, double)>(),
             pybind11::arg("mesh"), pybind11::arg("rho"), pybind11::arg("epsilon_r"),
             "Construct a new PoissonSolver object with the specified mesh, charge density function, and permittivity function")
        .def("solve", &PoissonSolver::solve,
             pybind11::arg("V_p"), pybind11::arg("V_n"),
             "Solve the Poisson equation with the specified boundary potentials")
        .def("get_potential", [](const PoissonSolver& solver) { return solver.phi; },
             "Get the computed electrostatic potential")
        .def("get_electric_field", [](const PoissonSolver& solver, double x, double y) {
                // Call the actual implementation
                return solver.get_electric_field(x, y);
             },
             pybind11::arg("x"), pybind11::arg("y"),
             "Get the electric field at a point (x, y)");

    // FEMSolver class for quantum simulations
    pybind11::class_<FEMSolver>(m, "FEMSolver")
        .def(pybind11::init<Mesh&, double(*)(double, double), double(*)(double, double), double(*)(double, double), PoissonSolver&, int, bool>(),
             pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"), pybind11::arg("cap"),
             pybind11::arg("poisson"), pybind11::arg("order"), pybind11::arg("use_mpi") = true,
             "Construct a new FEMSolver object with the specified mesh, effective mass function, potential function, capacitance function, Poisson solver, element order, and MPI flag")
        .def("assemble_matrices", &FEMSolver::assemble_matrices,
             "Assemble the Hamiltonian and mass matrices")
        .def("adapt_mesh", &FEMSolver::adapt_mesh,
             pybind11::arg("psi"), pybind11::arg("threshold"),
             "Adapt the mesh based on the solution gradient")
        .def("is_using_mpi", &FEMSolver::is_using_mpi,
             "Check if the solver is using MPI")
        .def("get_H", [](const FEMSolver& solver) { return solver.get_H(); },
             "Get the Hamiltonian matrix")
        .def("get_M", [](const FEMSolver& solver) { return solver.get_M(); },
             "Get the mass matrix")
        .def("get_interpolator", [](const FEMSolver& solver) { return solver.get_interpolator(); },
             "Get the field interpolator");

    // EigenSolver class for eigenvalue problems
    pybind11::class_<EigenSolver>(m, "EigenSolver")
        .def(pybind11::init<FEMSolver&>(),
             pybind11::arg("fem_solver"),
             "Construct a new EigenSolver object for the specified FEM solver")
        .def("solve", &EigenSolver::solve,
             pybind11::arg("num_eigenpairs") = 10,
             "Solve the eigenvalue problem and compute the specified number of eigenpairs")
        .def("get_eigenvalues", &EigenSolver::get_eigenvalues,
             "Get the computed eigenvalues")
        .def("get_eigenvectors", &EigenSolver::get_eigenvectors,
             "Get the computed eigenvectors");

    // Physics functions
    m.def("effective_mass", &Physics::effective_mass,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("qd_mat"), pybind11::arg("matrix_mat"), pybind11::arg("R"),
          "Compute the effective mass at a point (x, y) based on the quantum dot and matrix materials");

    m.def("potential", static_cast<double(*)(double, double, const Materials::Material&, const Materials::Material&, double, const std::string&, const Eigen::VectorXd&, const FEInterpolator*)>(&Physics::potential),
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("qd_mat"), pybind11::arg("matrix_mat"),
          pybind11::arg("R"), pybind11::arg("type"), pybind11::arg("phi"), pybind11::arg("interpolator") = nullptr,
          "Compute the potential at a point (x, y) based on the quantum dot and matrix materials, and the electrostatic potential");

    m.def("epsilon_r", &Physics::epsilon_r,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("p_mat"), pybind11::arg("n_mat"),
          "Compute the relative permittivity at a point (x, y) based on the p-type and n-type materials");

    m.def("charge_density", &Physics::charge_density,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("N_A"), pybind11::arg("N_D"), pybind11::arg("W_d"),
          "Compute the charge density at a point (x, y) based on the doping concentrations and depletion width");

    m.def("cap", &Physics::cap,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("eta"), pybind11::arg("Lx"), pybind11::arg("Ly"), pybind11::arg("d"),
          "Compute the capacitance at a point (x, y) based on the gate geometry and dielectric properties");
}