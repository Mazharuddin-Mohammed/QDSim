/**
 * @file fe_interpolator.cpp
 * @brief Implementation of the FEInterpolator class for finite element interpolation.
 *
 * This file contains the implementation of the FEInterpolator class, which provides
 * methods for interpolating scalar fields on a finite element mesh. The interpolation
 * is based on the finite element shape functions and supports linear (P1), quadratic (P2),
 * and cubic (P3) elements.
 *
 * The implementation includes methods for:
 * - Finding the element containing a point
 * - Computing barycentric coordinates
 * - Evaluating shape functions and their gradients
 * - Interpolating field values and gradients
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "fe_interpolator.h"
#include <cmath>
#include <limits>

/**
 * @brief Constructs a new FEInterpolator object.
 *
 * This constructor initializes the FEInterpolator with a reference to the mesh
 * and sets the element order based on the mesh's element order.
 *
 * @param mesh The mesh to use for interpolation, must be a valid mesh with nodes and elements
 */
FEInterpolator::FEInterpolator(const Mesh& mesh) : mesh(mesh), element_order(mesh.getElementOrder()) {
    // No additional initialization needed
}

/**
 * @brief Interpolate a scalar field at a point.
 *
 * This method interpolates a scalar field at a given point using the finite
 * element shape functions. If the point is outside the mesh, a default value
 * of 0.0 is returned.
 *
 * The interpolation process involves:
 * 1. Finding the element containing the point
 * 2. Getting the element vertices and node indices
 * 3. Computing the barycentric coordinates of the point
 * 4. Evaluating the shape functions at the point
 * 5. Computing the interpolated value as a weighted sum of field values
 *
 * @param x The x-coordinate of the point in nanometers (nm)
 * @param y The y-coordinate of the point in nanometers (nm)
 * @param field The scalar field values at mesh nodes, must have the same length as the number of nodes
 * @return The interpolated value at the point
 */
double FEInterpolator::interpolate(double x, double y, const Eigen::VectorXd& field) const {
    // Find the element containing the point
    int elem_idx = findElement(x, y);
    if (elem_idx < 0) {
        // Point is outside the mesh, return a default value
        return 0.0;
    }

    // Get the element vertices
    std::vector<Eigen::Vector2d> vertices;
    std::vector<int> node_indices;

    if (element_order == 1) {
        // P1 element (linear)
        const auto& elem = mesh.getElements()[elem_idx];
        for (int i = 0; i < 3; ++i) {
            vertices.push_back(mesh.getNodes()[elem[i]]);
            node_indices.push_back(elem[i]);
        }
    } else if (element_order == 2) {
        // P2 element (quadratic)
        const auto& elem = mesh.getQuadraticElements()[elem_idx];
        for (int i = 0; i < 6; ++i) {
            vertices.push_back(mesh.getNodes()[elem[i]]);
            node_indices.push_back(elem[i]);
        }
    } else if (element_order == 3) {
        // P3 element (cubic)
        const auto& elem = mesh.getCubicElements()[elem_idx];
        for (int i = 0; i < 10; ++i) {
            vertices.push_back(mesh.getNodes()[elem[i]]);
            node_indices.push_back(elem[i]);
        }
    }

    // Compute barycentric coordinates
    std::vector<double> lambda(3);
    if (!computeBarycentricCoordinates(x, y, {vertices[0], vertices[1], vertices[2]}, lambda)) {
        // This shouldn't happen if findElement worked correctly
        return 0.0;
    }

    // Evaluate shape functions
    std::vector<double> shape_values;
    if (element_order == 1) {
        shape_values.resize(3);
        // For P1 elements, shape functions are just the barycentric coordinates
        shape_values = lambda;
    } else {
        // For higher-order elements, evaluate the shape functions
        if (element_order == 2) {
            shape_values.resize(6);
        } else { // element_order == 3
            shape_values.resize(10);
        }
        evaluateShapeFunctions(lambda, shape_values);
    }

    // Interpolate the field
    double value = 0.0;
    for (size_t i = 0; i < node_indices.size(); ++i) {
        value += field[node_indices[i]] * shape_values[i];
    }

    return value;
}

/**
 * @brief Interpolate a scalar field at a point with gradient.
 *
 * This method interpolates a scalar field at a given point using the finite
 * element shape functions, and also computes the gradient of the field at
 * that point. If the point is outside the mesh, a default value of 0.0 is
 * returned, and the gradient is set to (0.0, 0.0).
 *
 * The interpolation process involves:
 * 1. Finding the element containing the point
 * 2. Getting the element vertices and node indices
 * 3. Computing the barycentric coordinates of the point
 * 4. Evaluating the shape functions and their gradients at the point
 * 5. Computing the interpolated value and gradient as weighted sums
 *
 * @param x The x-coordinate of the point in nanometers (nm)
 * @param y The y-coordinate of the point in nanometers (nm)
 * @param field The scalar field values at mesh nodes, must have the same length as the number of nodes
 * @param grad_x Output parameter for the x-component of the gradient in field units per nanometer (field units/nm)
 * @param grad_y Output parameter for the y-component of the gradient in field units per nanometer (field units/nm)
 * @return The interpolated value at the point
 */
double FEInterpolator::interpolateWithGradient(double x, double y, const Eigen::VectorXd& field,
                                             double& grad_x, double& grad_y) const {
    // Find the element containing the point
    int elem_idx = findElement(x, y);
    if (elem_idx < 0) {
        // Point is outside the mesh, return a default value
        grad_x = grad_y = 0.0;
        return 0.0;
    }

    // Get the element vertices
    std::vector<Eigen::Vector2d> vertices;
    std::vector<int> node_indices;

    if (element_order == 1) {
        // P1 element (linear)
        const auto& elem = mesh.getElements()[elem_idx];
        for (int i = 0; i < 3; ++i) {
            vertices.push_back(mesh.getNodes()[elem[i]]);
            node_indices.push_back(elem[i]);
        }
    } else if (element_order == 2) {
        // P2 element (quadratic)
        const auto& elem = mesh.getQuadraticElements()[elem_idx];
        for (int i = 0; i < 6; ++i) {
            vertices.push_back(mesh.getNodes()[elem[i]]);
            node_indices.push_back(elem[i]);
        }
    } else if (element_order == 3) {
        // P3 element (cubic)
        const auto& elem = mesh.getCubicElements()[elem_idx];
        for (int i = 0; i < 10; ++i) {
            vertices.push_back(mesh.getNodes()[elem[i]]);
            node_indices.push_back(elem[i]);
        }
    }

    // Compute barycentric coordinates
    std::vector<double> lambda(3);
    if (!computeBarycentricCoordinates(x, y, {vertices[0], vertices[1], vertices[2]}, lambda)) {
        // This shouldn't happen if findElement worked correctly
        grad_x = grad_y = 0.0;
        return 0.0;
    }

    // Evaluate shape functions and gradients
    std::vector<double> shape_values;
    std::vector<Eigen::Vector2d> shape_gradients;

    if (element_order == 1) {
        // For P1 elements, shape functions are just the barycentric coordinates
        shape_values = lambda;
        shape_gradients.resize(3);
        evaluateShapeFunctionGradients(lambda, {vertices[0], vertices[1], vertices[2]}, shape_gradients);
    } else {
        // For higher-order elements, evaluate the shape functions and gradients
        if (element_order == 2) {
            shape_values.resize(6);
            shape_gradients.resize(6);
        } else { // element_order == 3
            shape_values.resize(10);
            shape_gradients.resize(10);
        }
        evaluateShapeFunctions(lambda, shape_values);
        evaluateShapeFunctionGradients(lambda, vertices, shape_gradients);
    }

    // Interpolate the field and gradient
    double value = 0.0;
    grad_x = grad_y = 0.0;
    for (size_t i = 0; i < node_indices.size(); ++i) {
        value += field[node_indices[i]] * shape_values[i];
        grad_x += field[node_indices[i]] * shape_gradients[i](0);
        grad_y += field[node_indices[i]] * shape_gradients[i](1);
    }

    return value;
}

/**
 * @brief Find the element containing a point.
 *
 * This method finds the element containing a given point by performing a
 * brute-force search through all elements in the mesh. For each element,
 * it computes the barycentric coordinates of the point and checks if they
 * are all non-negative (indicating the point is inside the element).
 *
 * In a production implementation, a more efficient search algorithm such as
 * a quadtree or R-tree would be used to reduce the search complexity from
 * O(n) to O(log n).
 *
 * @param x The x-coordinate of the point in nanometers (nm)
 * @param y The y-coordinate of the point in nanometers (nm)
 * @return The index of the element containing the point, or -1 if the point is outside the mesh
 */
int FEInterpolator::findElement(double x, double y) const {
    // Simple brute-force search for the element containing the point
    // In a real implementation, we would use a more efficient search algorithm (e.g., quadtree)
    const auto& nodes = mesh.getNodes();

    if (element_order == 1) {
        const auto& elements = mesh.getElements();
        for (size_t e = 0; e < elements.size(); ++e) {
            const auto& elem = elements[e];
            std::vector<Eigen::Vector2d> vertices = {
                nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
            };
            std::vector<double> lambda(3);
            if (computeBarycentricCoordinates(x, y, vertices, lambda)) {
                return e;
            }
        }
    } else if (element_order == 2) {
        const auto& elements = mesh.getQuadraticElements();
        for (size_t e = 0; e < elements.size(); ++e) {
            const auto& elem = elements[e];
            std::vector<Eigen::Vector2d> vertices = {
                nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
            };
            std::vector<double> lambda(3);
            if (computeBarycentricCoordinates(x, y, vertices, lambda)) {
                return e;
            }
        }
    } else if (element_order == 3) {
        const auto& elements = mesh.getCubicElements();
        for (size_t e = 0; e < elements.size(); ++e) {
            const auto& elem = elements[e];
            std::vector<Eigen::Vector2d> vertices = {
                nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
            };
            std::vector<double> lambda(3);
            if (computeBarycentricCoordinates(x, y, vertices, lambda)) {
                return e;
            }
        }
    }

    // Point is outside the mesh
    return -1;
}

/**
 * @brief Compute the barycentric coordinates of a point in a triangle.
 *
 * This method computes the barycentric coordinates of a point in a triangle.
 * The barycentric coordinates are the weights of the vertices in the linear
 * combination that gives the point. They are used for interpolation within
 * the triangle.
 *
 * The computation involves:
 * 1. Computing the area of the triangle
 * 2. Computing the barycentric coordinates using the formula:
 *    lambda_i = A_i / A, where A_i is the area of the subtriangle formed by
 *    the point and the two vertices opposite to vertex i, and A is the area
 *    of the original triangle
 * 3. Checking if the point is inside the triangle by verifying that all
 *    barycentric coordinates are non-negative
 *
 * @param x The x-coordinate of the point in nanometers (nm)
 * @param y The y-coordinate of the point in nanometers (nm)
 * @param vertices The vertices of the triangle in nanometers (nm)
 * @param lambda Output parameter for the barycentric coordinates (3 values)
 * @return True if the point is inside the triangle, false otherwise
 */
bool FEInterpolator::computeBarycentricCoordinates(double x, double y,
                                                 const std::vector<Eigen::Vector2d>& vertices,
                                                 std::vector<double>& lambda) const {
    // Compute barycentric coordinates for a point (x,y) in a triangle
    // lambda[0], lambda[1], lambda[2] are the barycentric coordinates

    // Vertices of the triangle
    const Eigen::Vector2d& v0 = vertices[0];
    const Eigen::Vector2d& v1 = vertices[1];
    const Eigen::Vector2d& v2 = vertices[2];

    // Compute the area of the triangle
    double area = 0.5 * ((v1(0) - v0(0)) * (v2(1) - v0(1)) - (v2(0) - v0(0)) * (v1(1) - v0(1)));

    // Compute the barycentric coordinates
    lambda[0] = 0.5 * ((v1(0) * v2(1) - v2(0) * v1(1)) + (v1(1) - v2(1)) * x + (v2(0) - v1(0)) * y) / area;
    lambda[1] = 0.5 * ((v2(0) * v0(1) - v0(0) * v2(1)) + (v2(1) - v0(1)) * x + (v0(0) - v2(0)) * y) / area;
    lambda[2] = 0.5 * ((v0(0) * v1(1) - v1(0) * v0(1)) + (v0(1) - v1(1)) * x + (v1(0) - v0(0)) * y) / area;

    // Check if the point is inside the triangle
    const double eps = 1e-10;
    return lambda[0] >= -eps && lambda[1] >= -eps && lambda[2] >= -eps;
}

/**
 * @brief Evaluate the shape functions at a point given by barycentric coordinates.
 *
 * This method evaluates the finite element shape functions at a point given
 * by its barycentric coordinates. The shape functions depend on the element
 * order (P1, P2, or P3).
 *
 * For P1 elements (linear), the shape functions are simply the barycentric coordinates.
 * For P2 elements (quadratic), the shape functions include additional terms for the edge nodes.
 * For P3 elements (cubic), the shape functions include additional terms for the edge and interior nodes.
 *
 * The shape functions have the property that they are 1 at one node and 0 at all other nodes.
 *
 * @param lambda The barycentric coordinates (3 values)
 * @param shape_values Output parameter for the shape function values
 *                    (3 values for P1, 6 values for P2, 10 values for P3)
 */
void FEInterpolator::evaluateShapeFunctions(const std::vector<double>& lambda,
                                          std::vector<double>& shape_values) const {
    // Evaluate the shape functions at a point given by barycentric coordinates

    if (element_order == 1) {
        // P1 elements (linear)
        // Shape functions are just the barycentric coordinates
        shape_values[0] = lambda[0];
        shape_values[1] = lambda[1];
        shape_values[2] = lambda[2];
    } else if (element_order == 2) {
        // P2 elements (quadratic)
        // Vertex nodes
        shape_values[0] = lambda[0] * (2.0 * lambda[0] - 1.0);
        shape_values[1] = lambda[1] * (2.0 * lambda[1] - 1.0);
        shape_values[2] = lambda[2] * (2.0 * lambda[2] - 1.0);
        // Edge nodes
        shape_values[3] = 4.0 * lambda[0] * lambda[1];
        shape_values[4] = 4.0 * lambda[1] * lambda[2];
        shape_values[5] = 4.0 * lambda[2] * lambda[0];
    } else if (element_order == 3) {
        // P3 elements (cubic)
        double l0 = lambda[0];
        double l1 = lambda[1];
        double l2 = lambda[2];

        // Vertex nodes
        shape_values[0] = 0.5 * l0 * (3.0 * l0 - 1.0) * (3.0 * l0 - 2.0);
        shape_values[1] = 0.5 * l1 * (3.0 * l1 - 1.0) * (3.0 * l1 - 2.0);
        shape_values[2] = 0.5 * l2 * (3.0 * l2 - 1.0) * (3.0 * l2 - 2.0);

        // Edge nodes (2 per edge)
        shape_values[3] = 4.5 * l0 * l1 * (3.0 * l0 - 1.0);
        shape_values[4] = 4.5 * l0 * l1 * (3.0 * l1 - 1.0);

        shape_values[5] = 4.5 * l1 * l2 * (3.0 * l1 - 1.0);
        shape_values[6] = 4.5 * l1 * l2 * (3.0 * l2 - 1.0);

        shape_values[7] = 4.5 * l2 * l0 * (3.0 * l2 - 1.0);
        shape_values[8] = 4.5 * l2 * l0 * (3.0 * l0 - 1.0);

        // Interior node
        shape_values[9] = 27.0 * l0 * l1 * l2;
    }
}

/**
 * @brief Evaluate the shape function gradients at a point given by barycentric coordinates.
 *
 * This method evaluates the gradients of the finite element shape functions
 * at a point given by its barycentric coordinates. The shape function gradients
 * depend on the element order (P1, P2, or P3) and the geometry of the element.
 *
 * The computation involves:
 * 1. Computing the gradients of the barycentric coordinates
 * 2. Computing the shape function gradients based on the element order
 *
 * For P1 elements (linear), the shape function gradients are simply the gradients
 * of the barycentric coordinates, which are constant within each element.
 *
 * For P2 elements (quadratic) and P3 elements (cubic), the shape function gradients
 * are more complex and depend on the barycentric coordinates.
 *
 * @param lambda The barycentric coordinates (3 values)
 * @param vertices The vertices of the triangle in nanometers (nm)
 * @param shape_gradients Output parameter for the shape function gradients
 *                       (3 values for P1, 6 values for P2, 10 values for P3)
 */
void FEInterpolator::evaluateShapeFunctionGradients(const std::vector<double>& lambda,
                                                  const std::vector<Eigen::Vector2d>& vertices,
                                                  std::vector<Eigen::Vector2d>& shape_gradients) const {
    // Evaluate the shape function gradients at a point given by barycentric coordinates

    // Compute the gradients of the barycentric coordinates
    std::vector<Eigen::Vector2d> lambda_gradients(3);

    // Vertices of the triangle
    const Eigen::Vector2d& v0 = vertices[0];
    const Eigen::Vector2d& v1 = vertices[1];
    const Eigen::Vector2d& v2 = vertices[2];

    // Compute the area of the triangle
    double area = 0.5 * ((v1(0) - v0(0)) * (v2(1) - v0(1)) - (v2(0) - v0(0)) * (v1(1) - v0(1)));

    // Compute the gradients of the barycentric coordinates
    lambda_gradients[0](0) = 0.5 * (v1(1) - v2(1)) / area;
    lambda_gradients[0](1) = 0.5 * (v2(0) - v1(0)) / area;

    lambda_gradients[1](0) = 0.5 * (v2(1) - v0(1)) / area;
    lambda_gradients[1](1) = 0.5 * (v0(0) - v2(0)) / area;

    lambda_gradients[2](0) = 0.5 * (v0(1) - v1(1)) / area;
    lambda_gradients[2](1) = 0.5 * (v1(0) - v0(0)) / area;

    if (element_order == 1) {
        // P1 elements (linear)
        // Shape function gradients are just the gradients of the barycentric coordinates
        shape_gradients[0] = lambda_gradients[0];
        shape_gradients[1] = lambda_gradients[1];
        shape_gradients[2] = lambda_gradients[2];
    } else if (element_order == 2) {
        // P2 elements (quadratic)
        double l0 = lambda[0];
        double l1 = lambda[1];
        double l2 = lambda[2];

        // Vertex nodes
        shape_gradients[0] = lambda_gradients[0] * (4.0 * l0 - 1.0);
        shape_gradients[1] = lambda_gradients[1] * (4.0 * l1 - 1.0);
        shape_gradients[2] = lambda_gradients[2] * (4.0 * l2 - 1.0);

        // Edge nodes
        shape_gradients[3] = 4.0 * (lambda_gradients[0] * l1 + lambda_gradients[1] * l0);
        shape_gradients[4] = 4.0 * (lambda_gradients[1] * l2 + lambda_gradients[2] * l1);
        shape_gradients[5] = 4.0 * (lambda_gradients[2] * l0 + lambda_gradients[0] * l2);
    } else if (element_order == 3) {
        // P3 elements (cubic)
        double l0 = lambda[0];
        double l1 = lambda[1];
        double l2 = lambda[2];

        // Vertex nodes
        shape_gradients[0] = lambda_gradients[0] * (0.5 * (27.0 * l0 * l0 - 18.0 * l0 + 2.0));
        shape_gradients[1] = lambda_gradients[1] * (0.5 * (27.0 * l1 * l1 - 18.0 * l1 + 2.0));
        shape_gradients[2] = lambda_gradients[2] * (0.5 * (27.0 * l2 * l2 - 18.0 * l2 + 2.0));

        // Edge nodes (2 per edge)
        shape_gradients[3] = 4.5 * (lambda_gradients[0] * l1 * (3.0 * l0 - 1.0) + l0 * l1 * 3.0 * lambda_gradients[0] + lambda_gradients[1] * l0 * (3.0 * l0 - 1.0));
        shape_gradients[4] = 4.5 * (lambda_gradients[0] * l1 * (3.0 * l1 - 1.0) + lambda_gradients[1] * l0 * (3.0 * l1 - 1.0) + l0 * l1 * 3.0 * lambda_gradients[1]);

        shape_gradients[5] = 4.5 * (lambda_gradients[1] * l2 * (3.0 * l1 - 1.0) + l1 * l2 * 3.0 * lambda_gradients[1] + lambda_gradients[2] * l1 * (3.0 * l1 - 1.0));
        shape_gradients[6] = 4.5 * (lambda_gradients[1] * l2 * (3.0 * l2 - 1.0) + lambda_gradients[2] * l1 * (3.0 * l2 - 1.0) + l1 * l2 * 3.0 * lambda_gradients[2]);

        shape_gradients[7] = 4.5 * (lambda_gradients[2] * l0 * (3.0 * l2 - 1.0) + l2 * l0 * 3.0 * lambda_gradients[2] + lambda_gradients[0] * l2 * (3.0 * l2 - 1.0));
        shape_gradients[8] = 4.5 * (lambda_gradients[2] * l0 * (3.0 * l0 - 1.0) + lambda_gradients[0] * l2 * (3.0 * l0 - 1.0) + l2 * l0 * 3.0 * lambda_gradients[0]);

        // Interior node
        shape_gradients[9] = 27.0 * (lambda_gradients[0] * l1 * l2 + l0 * lambda_gradients[1] * l2 + l0 * l1 * lambda_gradients[2]);
    }
}
