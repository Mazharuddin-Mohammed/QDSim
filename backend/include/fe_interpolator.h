#pragma once
/**
 * @file fe_interpolator.h
 * @brief Defines the FEInterpolator class for finite element interpolation.
 *
 * This file contains the declaration of the FEInterpolator class, which provides
 * methods for interpolating scalar fields on a finite element mesh. The interpolation
 * is based on the finite element shape functions and supports linear, quadratic,
 * and cubic elements.
 *
 * Physical units:
 * - Coordinates: nanometers (nm)
 * - Field values: arbitrary (depends on the field being interpolated)
 * - Gradient: field units per nanometer (field units/nm)
 *
 * Assumptions and limitations:
 * - The interpolation is based on the finite element shape functions
 * - The field is assumed to be defined at the mesh nodes
 * - Points outside the mesh are handled by returning the value at the nearest node
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <functional>

/**
 * @class FEInterpolator
 * @brief Class for finite element interpolation of scalar fields.
 *
 * This class provides methods to interpolate scalar fields (like potentials)
 * at arbitrary points in the mesh using finite element shape functions.
 * It supports linear (P1), quadratic (P2), and cubic (P3) elements.
 *
 * The interpolation is based on the finite element shape functions, which
 * are defined on each element of the mesh. For points outside the mesh,
 * the value at the nearest node is returned.
 *
 * The class also provides methods for computing the gradient of the field,
 * which is useful for computing electric fields from potentials, for example.
 */
class FEInterpolator {
public:
    /**
     * @brief Construct a new FEInterpolator object.
     *
     * @param mesh The mesh to use for interpolation, must be a valid mesh with nodes and elements
     *
     * @throws std::invalid_argument If the mesh is invalid or has an unsupported element order
     */
    FEInterpolator(const Mesh& mesh);

    /**
     * @brief Interpolate a scalar field at a point.
     *
     * This method interpolates a scalar field at a given point using the finite
     * element shape functions. If the point is outside the mesh, the value at
     * the nearest node is returned.
     *
     * @param x The x-coordinate of the point in nanometers (nm)
     * @param y The y-coordinate of the point in nanometers (nm)
     * @param field The scalar field values at the mesh nodes, must have the same length as the number of nodes
     * @return double The interpolated value
     *
     * @throws std::invalid_argument If the field has an invalid length
     */
    double interpolate(double x, double y, const Eigen::VectorXd& field) const;

    /**
     * @brief Interpolate a scalar field at a point with gradient.
     *
     * This method interpolates a scalar field at a given point using the finite
     * element shape functions, and also computes the gradient of the field at
     * that point. If the point is outside the mesh, the value at the nearest
     * node is returned, and the gradient is set to zero.
     *
     * @param x The x-coordinate of the point in nanometers (nm)
     * @param y The y-coordinate of the point in nanometers (nm)
     * @param field The scalar field values at the mesh nodes, must have the same length as the number of nodes
     * @param grad_x Output parameter for the x-component of the gradient in field units per nanometer (field units/nm)
     * @param grad_y Output parameter for the y-component of the gradient in field units per nanometer (field units/nm)
     * @return double The interpolated value
     *
     * @throws std::invalid_argument If the field has an invalid length
     */
    double interpolateWithGradient(double x, double y, const Eigen::VectorXd& field,
                                  double& grad_x, double& grad_y) const;

    /**
     * @brief Find the element containing a point.
     *
     * This method finds the element containing a given point. If the point is
     * outside the mesh, -1 is returned.
     *
     * @param x The x-coordinate of the point in nanometers (nm)
     * @param y The y-coordinate of the point in nanometers (nm)
     * @return int The index of the element containing the point, or -1 if the point is outside the mesh
     *
     * @note This method is useful for determining if a point is inside the mesh,
     *       and for finding the element containing a point for further processing.
     */
    int findElement(double x, double y) const;

private:
    /** @brief Reference to the mesh used for interpolation */
    const Mesh& mesh;

    /** @brief Order of the finite elements (1 for P1, 2 for P2, 3 for P3) */
    int element_order;

    /**
     * @brief Compute the barycentric coordinates of a point in a triangle.
     *
     * This method computes the barycentric coordinates of a point in a triangle.
     * The barycentric coordinates are the weights of the vertices in the linear
     * combination that gives the point. They are used for interpolation within
     * the triangle.
     *
     * @param x The x-coordinate of the point in nanometers (nm)
     * @param y The y-coordinate of the point in nanometers (nm)
     * @param vertices The vertices of the triangle in nanometers (nm)
     * @param lambda Output parameter for the barycentric coordinates (3 values)
     * @return bool True if the point is inside the triangle, false otherwise
     *
     * @note A point is inside the triangle if all barycentric coordinates are non-negative.
     *       The barycentric coordinates sum to 1 for any point in the plane.
     */
    bool computeBarycentricCoordinates(double x, double y,
                                      const std::vector<Eigen::Vector2d>& vertices,
                                      std::vector<double>& lambda) const;

    /**
     * @brief Evaluate the shape functions at a point given by barycentric coordinates.
     *
     * This method evaluates the finite element shape functions at a point given
     * by its barycentric coordinates. The shape functions depend on the element
     * order (P1, P2, or P3).
     *
     * @param lambda The barycentric coordinates (3 values)
     * @param shape_values Output parameter for the shape function values
     *                    (3 values for P1, 6 values for P2, 10 values for P3)
     *
     * @note The shape functions are the basis functions used for interpolation
     *       within each element. They have the property that they are 1 at one
     *       node and 0 at all other nodes.
     */
    void evaluateShapeFunctions(const std::vector<double>& lambda,
                               std::vector<double>& shape_values) const;

    /**
     * @brief Evaluate the shape function gradients at a point given by barycentric coordinates.
     *
     * This method evaluates the gradients of the finite element shape functions
     * at a point given by its barycentric coordinates. The shape function gradients
     * depend on the element order (P1, P2, or P3) and the geometry of the element.
     *
     * @param lambda The barycentric coordinates (3 values)
     * @param vertices The vertices of the triangle in nanometers (nm)
     * @param shape_gradients Output parameter for the shape function gradients
     *                       (3 values for P1, 6 values for P2, 10 values for P3)
     *
     * @note The shape function gradients are used for computing the gradient of
     *       interpolated fields. For linear elements (P1), the gradients are constant
     *       within each element.
     */
    void evaluateShapeFunctionGradients(const std::vector<double>& lambda,
                                       const std::vector<Eigen::Vector2d>& vertices,
                                       std::vector<Eigen::Vector2d>& shape_gradients) const;
};
