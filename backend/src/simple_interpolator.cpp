/**
 * @file simple_interpolator.cpp
 * @brief Implementation of the SimpleInterpolator class.
 *
 * This file contains the implementation of the SimpleInterpolator class,
 * which provides a simplified interpolator for finite element interpolation.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "bindings.h"
#include <cmath>

double SimpleInterpolator::interpolate(double x, double y, const std::vector<double>& values) const {
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

    // Get the element
    const auto& elem = mesh.getElements()[elem_idx];

    // Interpolate the value
    double value = 0.0;
    for (int i = 0; i < 3; ++i) {
        value += lambda[i] * values[elem[i]];
    }

    return value;
}

int SimpleInterpolator::findElement(double x, double y) const {
    // Get the elements
    const auto& elements = mesh.getElements();

    // Check each element
    for (size_t e = 0; e < elements.size(); ++e) {
        // Compute the barycentric coordinates
        std::array<double, 3> lambda;
        if (computeBarycentricCoordinates(x, y, e, lambda)) {
            // Point is inside this element
            return e;
        }
    }

    // Point is outside the mesh
    return -1;
}

bool SimpleInterpolator::computeBarycentricCoordinates(double x, double y, int elem_idx, std::array<double, 3>& lambda) const {
    // Get the nodes of the element
    const auto& elem = mesh.getElements()[elem_idx];
    const auto& nodes = mesh.getNodes();

    // Get the coordinates of the vertices
    double x1 = nodes[elem[0]][0];
    double y1 = nodes[elem[0]][1];
    double x2 = nodes[elem[1]][0];
    double y2 = nodes[elem[1]][1];
    double x3 = nodes[elem[2]][0];
    double y3 = nodes[elem[2]][1];

    // Compute the area of the triangle
    double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
    if (area < 1e-10) {
        // Degenerate triangle
        return false;
    }

    // Compute the barycentric coordinates
    lambda[0] = 0.5 * std::abs((x2 - x) * (y3 - y) - (x3 - x) * (y2 - y)) / area;
    lambda[1] = 0.5 * std::abs((x3 - x) * (y1 - y) - (x1 - x) * (y3 - y)) / area;
    lambda[2] = 1.0 - lambda[0] - lambda[1];

    // Check if the point is inside the triangle
    return lambda[0] >= -1e-10 && lambda[1] >= -1e-10 && lambda[2] >= -1e-10;
}
