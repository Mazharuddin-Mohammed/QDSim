/**
 * @file simple_mesh.h
 * @brief Defines a simplified mesh class for interpolation.
 *
 * This file contains the declaration of the SimpleMesh class, which provides
 * a simplified interface for mesh operations used in interpolation.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#ifndef SIMPLE_MESH_H
#define SIMPLE_MESH_H

#include <Eigen/Dense>
#include <vector>
#include <array>

/**
 * @class SimpleMesh
 * @brief A simplified mesh class for interpolation.
 *
 * This class provides a simplified interface for mesh operations used in
 * interpolation. It stores the nodes and elements of the mesh.
 */
class SimpleMesh {
public:
    /**
     * @brief Constructs a new SimpleMesh object.
     *
     * @param nodes The nodes of the mesh
     * @param elements The elements of the mesh
     */
    SimpleMesh(const std::vector<Eigen::Vector2d>& nodes,
               const std::vector<std::array<int, 3>>& elements);

    /**
     * @brief Gets the nodes of the mesh.
     *
     * @return The nodes of the mesh
     */
    const std::vector<Eigen::Vector2d>& getNodes() const { return nodes_; }

    /**
     * @brief Gets the elements of the mesh.
     *
     * @return The elements of the mesh
     */
    const std::vector<std::array<int, 3>>& getElements() const { return elements_; }

private:
    std::vector<Eigen::Vector2d> nodes_;       ///< The nodes of the mesh
    std::vector<std::array<int, 3>> elements_; ///< The elements of the mesh
};

#endif // SIMPLE_MESH_H
