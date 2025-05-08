/**
 * @file simple_interpolator.cpp
 * @brief Implementation of the SimpleInterpolator class.
 *
 * This file contains the implementation of the SimpleInterpolator class,
 * which provides a simplified interpolator for finite element interpolation.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "simple_interpolator.h"
#include <cmath>
#include <limits>

SimpleInterpolator::SimpleInterpolator(const SimpleMesh& mesh)
    : mesh_(mesh) {
    // Nothing to do here
}

double SimpleInterpolator::interpolate(double x, double y, const Eigen::VectorXd& values) const {
    // Find the element containing the point
    int elem_idx = findElement(x, y);
    if (elem_idx < 0) {
        // Point is outside the mesh
        return 0.0;
    }

    // Compute the barycentric coordinates
    std::array<double, 3> lambda;
    if (!computeBarycentricCoordinates(x, y, elem_idx, lambda)) {
        // Point is outside the element
        return 0.0;
    }

    // Get the element nodes
    const auto& element = mesh_.getElements()[elem_idx];

    // Interpolate the field value
    double value = 0.0;
    for (int i = 0; i < 3; ++i) {
        int node_idx = element[i];
        if (node_idx < values.size()) {
            value += lambda[i] * values[node_idx];
        }
    }

    return value;
}

int SimpleInterpolator::findElement(double x, double y) const {
    // Iterate over all elements
    for (size_t e = 0; e < mesh_.getElements().size(); ++e) {
        // Compute barycentric coordinates
        std::array<double, 3> lambda;
        bool inside = computeBarycentricCoordinates(x, y, e, lambda);
        if (inside) {
            return e;
        }
    }

    // Point is outside the mesh
    return -1;
}

bool SimpleInterpolator::computeBarycentricCoordinates(double x, double y, int elem_idx,
                                                     std::array<double, 3>& lambda) const {
    // Get the element nodes
    const auto& element = mesh_.getElements()[elem_idx];
    const auto& nodes = mesh_.getNodes();

    // Get the node coordinates
    double x1 = nodes[element[0]][0];
    double y1 = nodes[element[0]][1];
    double x2 = nodes[element[1]][0];
    double y2 = nodes[element[1]][1];
    double x3 = nodes[element[2]][0];
    double y3 = nodes[element[2]][1];

    // Compute the area of the triangle
    double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
    if (area < std::numeric_limits<double>::epsilon()) {
        // Degenerate triangle
        return false;
    }

    // Compute the barycentric coordinates
    lambda[0] = 0.5 * std::abs((x2 - x) * (y3 - y) - (x3 - x) * (y2 - y)) / area;
    lambda[1] = 0.5 * std::abs((x3 - x) * (y1 - y) - (x1 - x) * (y3 - y)) / area;
    lambda[2] = 0.5 * std::abs((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / area;

    // Check if the point is inside the triangle
    const double eps = 1e-10;
    return (lambda[0] >= -eps && lambda[1] >= -eps && lambda[2] >= -eps &&
            std::abs(lambda[0] + lambda[1] + lambda[2] - 1.0) < eps);
}
