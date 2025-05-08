/**
 * @file fe_interpolator_module.cpp
 * @brief Python bindings for the FEInterpolator class.
 *
 * This file contains the Python bindings for the FEInterpolator class,
 * which provides methods for interpolating fields on finite element meshes.
 * It uses pybind11 to expose the C++ classes and methods to Python.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include "fe_interpolator.h"
#include "mesh.h"

namespace py = pybind11;

PYBIND11_MODULE(fe_interpolator_module, m) {
    m.doc() = "Python bindings for the FEInterpolator class";

    // First expose the Mesh class since FEInterpolator depends on it
    py::class_<Mesh>(m, "Mesh")
        .def(py::init<double, double, int, int, int>())
        .def("get_nodes", &Mesh::getNodes)
        .def("get_elements", &Mesh::getElements)
        .def("get_num_nodes", &Mesh::getNumNodes)
        .def("get_num_elements", &Mesh::getNumElements)
        .def("get_element_order", &Mesh::getElementOrder)
        .def("get_lx", &Mesh::get_lx)
        .def("get_ly", &Mesh::get_ly)
        .def("get_nx", &Mesh::get_nx)
        .def("get_ny", &Mesh::get_ny);

    // Now expose the FEInterpolator class
    py::class_<FEInterpolator>(m, "FEInterpolator")
        .def(py::init<const Mesh&>())
        .def("interpolate", &FEInterpolator::interpolate)
        .def("interpolate_with_gradient", [](FEInterpolator& self, double x, double y, const Eigen::VectorXd& field) {
            double grad_x = 0.0, grad_y = 0.0;
            double value = self.interpolateWithGradient(x, y, field, grad_x, grad_y);
            return py::make_tuple(value, grad_x, grad_y);
        })
        .def("find_element", &FEInterpolator::findElement)
        .def("compute_barycentric_coordinates", [](FEInterpolator& self, double x, double y, const std::vector<Eigen::Vector2d>& vertices) {
            std::vector<double> lambda(3);
            bool inside = self.computeBarycentricCoordinates(x, y, vertices, lambda);
            return py::make_tuple(inside, lambda);
        })
        .def("evaluate_shape_functions", &FEInterpolator::evaluateShapeFunctions)
        .def("evaluate_shape_function_gradients", &FEInterpolator::evaluateShapeFunctionGradients);
}
