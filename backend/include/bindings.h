#pragma once
#include "mesh.h"
#include "fem.h"
#include "solver.h"
#include "poisson.h"
#include "materials.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

PYBIND11_MODULE(qdsim_cpp, m) {
    pybind11::class_<Mesh>(m, "Mesh")
        .def(pybind11::init<double, double, int, int, int>())
        .def("get_nodes", &Mesh::getNodes)
        .def("get_elements", &Mesh::getElements)
        .def("get_num_nodes", &Mesh::getNumNodes)
        .def("get_num_elements", &Mesh::getNumElements)
        .def("get_element_order", &Mesh::getElementOrder)
        .def("getNumNodes", &Mesh::getNumNodes)
        .def("getNumElements", &Mesh::getNumElements)
        .def("getElementOrder", &Mesh::getElementOrder);

    pybind11::class_<Materials::MaterialDatabase>(m, "MaterialDatabase")
        .def(pybind11::init<>())
        .def("get_material", [](const Materials::MaterialDatabase& db, const std::string& name) {
            auto mat = db.get_material(name);
            return std::make_tuple(mat.m_e, mat.m_h, mat.E_g, mat.Delta_E_c, mat.epsilon_r);
        });

    pybind11::class_<PoissonSolver>(m, "PoissonSolver")
        .def(pybind11::init<Mesh&, double(*)(double, double), double(*)(double, double)>())
        .def("solve", &PoissonSolver::solve)
        .def("get_potential", [](const PoissonSolver& solver) { return solver.phi; })
        .def("get_electric_field", [](const PoissonSolver& solver, double x, double y) {
            // Placeholder implementation
            return Eigen::Vector2d(0.0, 0.0);
        });

    pybind11::class_<FEMSolver>(m, "FEMSolver")
        .def(pybind11::init<Mesh&, double(*)(double, double), double(*)(double, double), double(*)(double, double), PoissonSolver&, int, bool>(),
             pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"), pybind11::arg("cap"),
             pybind11::arg("poisson"), pybind11::arg("order"), pybind11::arg("use_mpi") = true)
        .def("assemble_matrices", &FEMSolver::assemble_matrices)
        .def("adapt_mesh", &FEMSolver::adapt_mesh)
        .def("is_using_mpi", &FEMSolver::is_using_mpi)
        .def("get_H", [](const FEMSolver& solver) { return solver.get_H(); })
        .def("get_M", [](const FEMSolver& solver) { return solver.get_M(); });

    pybind11::class_<EigenSolver>(m, "EigenSolver")
        .def(pybind11::init<FEMSolver&>())
        .def("solve", &EigenSolver::solve)
        .def("get_eigenvalues", &EigenSolver::get_eigenvalues)
        .def("get_eigenvectors", &EigenSolver::get_eigenvectors);

    // Physics functions are now provided directly by the user in Python
    m.def("effective_mass", [](double x, double y) { return 0.067; });
    m.def("potential", [](double x, double y) { return 0.0; });
    m.def("epsilon_r", [](double x, double y) { return 12.9; });
    m.def("charge_density", [](double x, double y) { return 0.0; });
    m.def("cap", [](double x, double y) { return 0.0; });
}