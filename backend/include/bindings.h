#pragma once
/**
 * @file bindings.h
 * @brief Python bindings for QDSim.
 *
 * This file contains the declarations for the Python bindings implementation.
 * The implementation is in bindings.cpp.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <vector>
#include <array>
#include <Eigen/Dense>

/**
 * @class SimpleMesh
 * @brief A simple mesh class for interpolation.
 *
 * This class provides a simplified mesh representation for interpolation.
 * It contains only the nodes and elements, without the additional functionality
 * of the full Mesh class.
 */
class SimpleMesh {
public:
    /**
     * @brief Constructs a new SimpleMesh object.
     *
     * @param nodes The nodes of the mesh
     * @param elements The elements of the mesh
     */
    SimpleMesh(const std::vector<Eigen::Vector2d>& nodes, const std::vector<std::array<int, 3>>& elements)
        : nodes(nodes), elements(elements) {}

    /**
     * @brief Get the nodes of the mesh.
     * @return A reference to the vector of node coordinates
     */
    const std::vector<Eigen::Vector2d>& getNodes() const { return nodes; }

    /**
     * @brief Get the elements of the mesh.
     * @return A reference to the vector of elements
     */
    const std::vector<std::array<int, 3>>& getElements() const { return elements; }

private:
    /** @brief Vector of node coordinates */
    std::vector<Eigen::Vector2d> nodes;

    /** @brief Vector of elements, each with 3 nodes */
    std::vector<std::array<int, 3>> elements;
};

/**
 * @class SimpleInterpolator
 * @brief A simple interpolator class for finite element interpolation.
 *
 * This class provides a simplified interpolator for finite element interpolation.
 * It can interpolate values at arbitrary points within the mesh.
 */
class SimpleInterpolator {
public:
    /**
     * @brief Constructs a new SimpleInterpolator object.
     *
     * @param mesh The mesh to interpolate on
     */
    SimpleInterpolator(const SimpleMesh& mesh) : mesh(mesh) {}

    /**
     * @brief Interpolate a value at a point.
     *
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @param values The values at the mesh nodes
     * @return The interpolated value at the point
     */
    double interpolate(double x, double y, const std::vector<double>& values) const;

    /**
     * @brief Find the element containing a point.
     *
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return The index of the element containing the point, or -1 if not found
     */
    int findElement(double x, double y) const;

    /**
     * @brief Compute the barycentric coordinates of a point in an element.
     *
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @param elem_idx The index of the element
     * @param lambda The barycentric coordinates (output)
     * @return True if the point is inside the element, false otherwise
     */
    bool computeBarycentricCoordinates(double x, double y, int elem_idx, std::array<double, 3>& lambda) const;

private:
    /** @brief The mesh to interpolate on */
    const SimpleMesh& mesh;
};