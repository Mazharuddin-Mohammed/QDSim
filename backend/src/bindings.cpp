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
#include "self_consistent.h"
#include "simple_self_consistent.h"
#include "basic_solver.h"
#include "improved_self_consistent.h"
#include "materials.h"
#include "physics.h"
#include "fe_interpolator.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <iostream>
#include <memory>

// Global Python callback functions for SelfConsistentSolver
// We use shared_ptr to ensure proper reference counting
std::shared_ptr<pybind11::function> g_epsilon_r_py;
std::shared_ptr<pybind11::function> g_rho_py;
std::shared_ptr<pybind11::function> g_n_conc_py;
std::shared_ptr<pybind11::function> g_p_conc_py;
std::shared_ptr<pybind11::function> g_mu_n_py;
std::shared_ptr<pybind11::function> g_mu_p_py;

// C++ wrapper functions that call the Python callbacks
double epsilon_r_wrapper(double x, double y) {
    pybind11::gil_scoped_acquire gil;
    try {
        if (g_epsilon_r_py) {
            return (*g_epsilon_r_py)(x, y).cast<double>();
        } else {
            std::cerr << "Error in epsilon_r callback: Python callback is null" << std::endl;
            return 1.0; // Default value
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in epsilon_r callback: " << e.what() << std::endl;
        return 1.0; // Default value
    }
}

double rho_wrapper(double x, double y, const Eigen::VectorXd& n, const Eigen::VectorXd& p) {
    pybind11::gil_scoped_acquire gil;
    try {
        if (g_rho_py) {
            return (*g_rho_py)(x, y, n, p).cast<double>();
        } else {
            std::cerr << "Error in rho callback: Python callback is null" << std::endl;
            return 0.0; // Default value
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in rho callback: " << e.what() << std::endl;
        return 0.0; // Default value
    }
}

double n_conc_wrapper(double x, double y, double phi, const Materials::Material& mat) {
    pybind11::gil_scoped_acquire gil;
    try {
        if (g_n_conc_py) {
            return (*g_n_conc_py)(x, y, phi, mat).cast<double>();
        } else {
            std::cerr << "Error in n_conc callback: Python callback is null" << std::endl;
            return 1e10; // Default value
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in n_conc callback: " << e.what() << std::endl;
        return 1e10; // Default value
    }
}

double p_conc_wrapper(double x, double y, double phi, const Materials::Material& mat) {
    pybind11::gil_scoped_acquire gil;
    try {
        if (g_p_conc_py) {
            return (*g_p_conc_py)(x, y, phi, mat).cast<double>();
        } else {
            std::cerr << "Error in p_conc callback: Python callback is null" << std::endl;
            return 1e10; // Default value
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in p_conc callback: " << e.what() << std::endl;
        return 1e10; // Default value
    }
}

double mu_n_wrapper(double x, double y, const Materials::Material& mat) {
    pybind11::gil_scoped_acquire gil;
    try {
        if (g_mu_n_py) {
            return (*g_mu_n_py)(x, y, mat).cast<double>();
        } else {
            std::cerr << "Error in mu_n callback: Python callback is null" << std::endl;
            return 0.1; // Default value
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in mu_n callback: " << e.what() << std::endl;
        return 0.1; // Default value
    }
}

double mu_p_wrapper(double x, double y, const Materials::Material& mat) {
    pybind11::gil_scoped_acquire gil;
    try {
        if (g_mu_p_py) {
            return (*g_mu_p_py)(x, y, mat).cast<double>();
        } else {
            std::cerr << "Error in mu_p callback: Python callback is null" << std::endl;
            return 0.01; // Default value
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in mu_p callback: " << e.what() << std::endl;
        return 0.01; // Default value
    }
}

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
             "Find the element containing a point (x, y)");

    // MaterialDatabase class for material properties
    pybind11::class_<Materials::MaterialDatabase>(m, "MaterialDatabase")
        .def(pybind11::init<>(),
             "Construct a new MaterialDatabase object with default materials")
        .def("get_material", [](const Materials::MaterialDatabase& db, const std::string& name) {
                auto mat = db.get_material(name);
                return std::make_tuple(mat.m_e, mat.m_h, mat.E_g, mat.Delta_E_c, mat.epsilon_r,
                                         mat.mu_n, mat.mu_p, mat.N_c, mat.N_v);
             },
             pybind11::arg("name"),
             "Get the properties of a material by name, returns a tuple of (m_e, m_h, E_g, Delta_E_c, epsilon_r, mat.mu_n, mat.mu_p, mat.N_c, mat.N_v)");

    // PoissonSolver class for electrostatic calculations
    pybind11::class_<PoissonSolver>(m, "PoissonSolver")
        .def(pybind11::init<Mesh&, double(*)(double, double),
                           double(*)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)>(),
             pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
             "Construct a new PoissonSolver object with the specified mesh, permittivity function, and charge density function")
        .def("solve", &PoissonSolver::solve,
             pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("n"), pybind11::arg("p"),
             "Solve the Poisson equation with the specified boundary potentials and carrier concentrations")
        .def("get_potential", [](const PoissonSolver& solver) { return solver.phi; },
             "Get the computed electrostatic potential")
        .def("get_electric_field", [](const PoissonSolver& solver, double x, double y) {
                // Call the actual implementation
                return solver.get_electric_field(x, y);
             },
             pybind11::arg("x"), pybind11::arg("y"),
             "Get the electric field at a point (x, y)");

     // SelfConsistentSolver class for self-consistent Poisson-drift-diffusion simulations
     pybind11::class_<SelfConsistentSolver>(m, "SelfConsistentSolver")
             .def(pybind11::init<Mesh&, double(*)(double, double),
                                double(*)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&),
                                double(*)(double, double, double, const Materials::Material&),
                                double(*)(double, double, double, const Materials::Material&),
                                double(*)(double, double, const Materials::Material&),
                                double(*)(double, double, const Materials::Material&)>(),
                  pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
                  pybind11::arg("n_conc"), pybind11::arg("p_conc"), pybind11::arg("mu_n"), pybind11::arg("mu_p"),
                  "Construct a new SelfConsistentSolver object with the specified mesh and callback functions")
             .def("solve", &SelfConsistentSolver::solve,
                  pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("N_A"), pybind11::arg("N_D"),
                  pybind11::arg("tolerance") = 1e-6, pybind11::arg("max_iter") = 100,
                  "Solve the self-consistent Poisson-drift-diffusion equations")
             .def("get_potential", &SelfConsistentSolver::get_potential,
                  "Get the computed electrostatic potential")
             .def("get_n", &SelfConsistentSolver::get_n,
                  "Get the computed electron concentration")
             .def("get_p", &SelfConsistentSolver::get_p,
                  "Get the computed hole concentration")
             .def("get_electric_field", &SelfConsistentSolver::get_electric_field,
                  pybind11::arg("x"), pybind11::arg("y"),
                  "Get the electric field at a point (x, y)");

    // FEMSolver class for quantum simulations
    pybind11::class_<FEMSolver>(m, "FEMSolver")
        .def(pybind11::init<Mesh&, double(*)(double, double), double(*)(double, double), double(*)(double, double), SelfConsistentSolver&, int, bool>(),
             pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"), pybind11::arg("cap"),
             pybind11::arg("sc_solver"), pybind11::arg("order"), pybind11::arg("use_mpi") = false,
             "Construct a new FEMSolver object with the specified mesh, effective mass function, potential function, capacitance function, Self-Consistent solver, element order, and MPI flag")
        .def("assemble_matrices", &FEMSolver::assemble_matrices,
             "Assemble the Hamiltonian and mass matrices")
        .def("adapt_mesh", &FEMSolver::adapt_mesh,
             pybind11::arg("psi"), pybind11::arg("threshold"), pybind11::arg("output_file") = "",
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

    /*
    m.def("charge_density", &Physics::charge_density,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("N_A"), pybind11::arg("N_D"), pybind11::arg("W_d"),
          "Compute the charge density at a point (x, y) based on the doping concentrations and depletion width");
    */

    m.def("charge_density", &Physics::charge_density,
               pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("n"), pybind11::arg("p"),
               pybind11::arg("n_interpolator") = nullptr, pybind11::arg("p_interpolator") = nullptr,
               "Compute the charge density at a point (x, y) based on the carrier concentrations");

     m.def("cap", &Physics::cap,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("eta"), pybind11::arg("Lx"), pybind11::arg("Ly"), pybind11::arg("d"),
          "Compute the capacitance at a point (x, y) based on the gate geometry and dielectric properties");

     m.def("electron_concentration", &Physics::electron_concentration,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("phi"), pybind11::arg("mat"),
          "Compute the electron concentration at a point (x, y) based on the electrostatic potential and material properties");

     m.def("hole_concentration", &Physics::hole_concentration,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("phi"), pybind11::arg("mat"),
          "Compute the hole concentration at a point (x, y) based on the electrostatic potential and material properties");

     m.def("mobility_n", &Physics::mobility_n,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("mat"),
          "Compute the electron mobility at a point (x, y) based on the material properties");

     m.def("mobility_p", &Physics::mobility_p,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("mat"),
          "Compute the hole mobility at a point (x, y) based on the material properties");

     // SimpleSelfConsistentSolver class for simplified self-consistent Poisson-drift-diffusion simulations
     pybind11::class_<SimpleSelfConsistentSolver>(m, "SimpleSelfConsistentSolver")
             .def(pybind11::init<Mesh&, std::function<double(double, double)>,
                                std::function<double(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)>>(),
                  pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
                  "Construct a new SimpleSelfConsistentSolver object with the specified mesh and callback functions")
             .def("solve", &SimpleSelfConsistentSolver::solve,
                  pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("N_A"), pybind11::arg("N_D"),
                  pybind11::arg("tolerance") = 1e-6, pybind11::arg("max_iter") = 100,
                  "Solve the self-consistent Poisson-drift-diffusion equations")
             .def("get_potential", &SimpleSelfConsistentSolver::get_potential,
                  "Get the computed electrostatic potential")
             .def("get_n", &SimpleSelfConsistentSolver::get_n,
                  "Get the computed electron concentration")
             .def("get_p", &SimpleSelfConsistentSolver::get_p,
                  "Get the computed hole concentration");

     // BasicSolver class for demonstration purposes
     pybind11::class_<BasicSolver>(m, "BasicSolver")
             .def(pybind11::init<Mesh&>(),
                  pybind11::arg("mesh"),
                  "Construct a new BasicSolver object with the specified mesh")
             .def("solve", &BasicSolver::solve,
                  pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("N_A"), pybind11::arg("N_D"),
                  "Set up simple fields without doing any actual solving")
             .def("get_potential", &BasicSolver::get_potential,
                  "Get the electrostatic potential")
             .def("get_n", &BasicSolver::get_n,
                  "Get the electron concentration")
             .def("get_p", &BasicSolver::get_p,
                  "Get the hole concentration");

     // ImprovedSelfConsistentSolver class for self-consistent Poisson-drift-diffusion simulations
     pybind11::class_<ImprovedSelfConsistentSolver>(m, "ImprovedSelfConsistentSolver")
             .def(pybind11::init<Mesh&, std::function<double(double, double)>,
                                std::function<double(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)>>(),
                  pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
                  "Construct a new ImprovedSelfConsistentSolver object with the specified mesh and callback functions")
             .def("solve", &ImprovedSelfConsistentSolver::solve,
                  pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("N_A"), pybind11::arg("N_D"),
                  pybind11::arg("tolerance") = 1e-6, pybind11::arg("max_iter") = 100,
                  "Solve the self-consistent Poisson-drift-diffusion equations")
             .def("get_potential", &ImprovedSelfConsistentSolver::get_potential,
                  "Get the computed electrostatic potential")
             .def("get_n", &ImprovedSelfConsistentSolver::get_n,
                  "Get the computed electron concentration")
             .def("get_p", &ImprovedSelfConsistentSolver::get_p,
                  "Get the computed hole concentration");

     // Helper functions to create callbacks for SelfConsistentSolver
     m.def("create_self_consistent_solver", [](Mesh& mesh, pybind11::function epsilon_r_py, pybind11::function rho_py,
                                             pybind11::function n_conc_py, pybind11::function p_conc_py,
                                             pybind11::function mu_n_py, pybind11::function mu_p_py) {
         // Store the Python callbacks in global variables using shared_ptr
         g_epsilon_r_py = std::make_shared<pybind11::function>(epsilon_r_py);
         g_rho_py = std::make_shared<pybind11::function>(rho_py);
         g_n_conc_py = std::make_shared<pybind11::function>(n_conc_py);
         g_p_conc_py = std::make_shared<pybind11::function>(p_conc_py);
         g_mu_n_py = std::make_shared<pybind11::function>(mu_n_py);
         g_mu_p_py = std::make_shared<pybind11::function>(mu_p_py);

         // Create and return the SelfConsistentSolver with the wrapper functions
         return new SelfConsistentSolver(mesh, epsilon_r_wrapper, rho_wrapper, n_conc_wrapper, p_conc_wrapper, mu_n_wrapper, mu_p_wrapper);
     }, pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
        pybind11::arg("n_conc"), pybind11::arg("p_conc"), pybind11::arg("mu_n"), pybind11::arg("mu_p"),
        "Create a new SelfConsistentSolver object with the specified mesh and Python callback functions");

     // Helper function to create a SimpleSelfConsistentSolver
     m.def("create_simple_self_consistent_solver", [](Mesh& mesh, pybind11::function epsilon_r_py, pybind11::function rho_py) {
         // Store the Python callbacks in global variables using shared_ptr
         g_epsilon_r_py = std::make_shared<pybind11::function>(epsilon_r_py);
         g_rho_py = std::make_shared<pybind11::function>(rho_py);

         // Create and return the SimpleSelfConsistentSolver with the wrapper functions
         return new SimpleSelfConsistentSolver(mesh, epsilon_r_wrapper, rho_wrapper);
     }, pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
        "Create a new SimpleSelfConsistentSolver object with the specified mesh and Python callback functions");

     // Helper function to create an ImprovedSelfConsistentSolver
     m.def("create_improved_self_consistent_solver", [](Mesh& mesh, pybind11::function epsilon_r_py, pybind11::function rho_py) {
         // Store the Python callbacks in global variables using shared_ptr
         g_epsilon_r_py = std::make_shared<pybind11::function>(epsilon_r_py);
         g_rho_py = std::make_shared<pybind11::function>(rho_py);

         // Create and return the ImprovedSelfConsistentSolver with the wrapper functions
         return new ImprovedSelfConsistentSolver(mesh, epsilon_r_wrapper, rho_wrapper);
     }, pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
        "Create a new ImprovedSelfConsistentSolver object with the specified mesh and Python callback functions");

     // Helper function to clear the global variables
     m.def("clear_callbacks", []() {
         g_epsilon_r_py.reset();
         g_rho_py.reset();
         g_n_conc_py.reset();
         g_p_conc_py.reset();
         g_mu_n_py.reset();
         g_mu_p_py.reset();
     }, "Clear the global Python callbacks to avoid memory leaks");
}