#pragma once
#include "mesh.h"
#include "fem.h"
#include "solver.h"
#include "poisson.h"
#include "materials.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

PYBIND11_MODULE(qdsim, m) {
    pybind11::class_<Mesh>(m, "Mesh")
        .def(pybind11::init<double, double, int, int, int>())
        .def("get_nodes", &Mesh::get_nodes)
        .def("get_elements", &Mesh::get_elements)
        .def("get_num_nodes", &Mesh::get_num_nodes)
        .def("get_num_elements", &Mesh::get_num_elements)
        .def("get_Lx", &Mesh::get_Lx)
        .def("get_Ly", &Mesh::get_Ly);

    pybind11::class_<Materials::MaterialDatabase>(m, "MaterialDatabase")
        .def(pybind11::init<>())
        .def("get_material", [](const Materials::MaterialDatabase& db, const std::string& name) {
            auto mat = db.get_material(name);
            return std::make_tuple(mat.m_e, mat.m_h, mat.E_g, mat.Delta_E_c, mat.epsilon_r);
        });

    pybind11::class_<PoissonSolver>(m, "PoissonSolver")
        .def(pybind11::init<Mesh&, pybind11::function, pybind11::function>())
        .def("solve", &PoissonSolver::solve)
        .def("get_potential", &PoissonSolver::get_potential)
        .def("get_electric_field", &PoissonSolver::get_electric_field);

    pybind11::class_<FEMSolver>(m, "FEMSolver")
        .def(pybind11::init<Mesh&, pybind11::function, pybind11::function, pybind11::function, PoissonSolver&, int, bool>(),
             pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"), pybind11::arg("cap"),
             pybind11::arg("poisson"), pybind11::arg("order"), pybind11::arg("use_mpi") = true)
        .def("assemble_matrices", &FEMSolver::assemble_matrices)
        .def("adapt_mesh", &FEMSolver::adapt_mesh)
        .def("is_using_mpi", &FEMSolver::is_using_mpi);

    pybind11::class_<EigenSolver>(m, "EigenSolver")
        .def(pybind11::init<FEMSolver&>())
        .def("solve", &EigenSolver::solve)
        .def("get_eigenvalues", &EigenSolver::get_eigenvalues)
        .def("get_eigenvectors", &EigenSolver::get_eigenvectors);

    m.def("effective_mass", &Physics::effective_mass);
    m.def("potential", &Physics::potential);
    m.def("epsilon_r", &Physics::epsilon_r);
    m.def("charge_density", &Physics::charge_density);
    m.def("cap", &Physics::cap);
}