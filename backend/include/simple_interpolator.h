/**
 * @file simple_interpolator.h
 * @brief Defines a simplified interpolator class.
 *
 * This file contains the declaration of the SimpleInterpolator class, which provides
 * a simplified interface for field interpolation.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#ifndef SIMPLE_INTERPOLATOR_H
#define SIMPLE_INTERPOLATOR_H

#include "simple_mesh.h"
#include <Eigen/Dense>
#include <vector>
#include <array>

/**
 * @class SimpleInterpolator
 * @brief A simplified interpolator class.
 *
 * This class provides a simplified interface for field interpolation. It uses
 * linear interpolation within triangular elements.
 */
class SimpleInterpolator {
public:
    /**
     * @brief Constructs a new SimpleInterpolator object.
     *
     * @param mesh The mesh to use for interpolation
     */
    SimpleInterpolator(const SimpleMesh& mesh);

    /**
     * @brief Interpolates a field at a given position.
     *
     * @param x The x-coordinate of the position
     * @param y The y-coordinate of the position
     * @param values The field values at the mesh nodes
     * @return The interpolated field value
     */
    double interpolate(double x, double y, const Eigen::VectorXd& values) const;

    /**
     * @brief Finds the element containing a given position.
     *
     * @param x The x-coordinate of the position
     * @param y The y-coordinate of the position
     * @return The index of the element containing the position, or -1 if not found
     */
    int findElement(double x, double y) const;

    /**
     * @brief Computes the barycentric coordinates of a point in an element.
     *
     * @param x The x-coordinate of the position
     * @param y The y-coordinate of the position
     * @param elem_idx The index of the element
     * @param lambda The barycentric coordinates (output)
     * @return True if the point is inside the element, false otherwise
     */
    bool computeBarycentricCoordinates(double x, double y, int elem_idx,
                                      std::array<double, 3>& lambda) const;

private:
    const SimpleMesh& mesh_; ///< The mesh to use for interpolation
};

#endif // SIMPLE_INTERPOLATOR_H
