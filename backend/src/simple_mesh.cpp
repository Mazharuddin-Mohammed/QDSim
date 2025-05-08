/**
 * @file simple_mesh.cpp
 * @brief Implementation of the SimpleMesh class.
 *
 * This file contains the implementation of the SimpleMesh class, which provides
 * a simplified interface for mesh operations used in interpolation.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "simple_mesh.h"

SimpleMesh::SimpleMesh(const std::vector<Eigen::Vector2d>& nodes,
                       const std::vector<std::array<int, 3>>& elements)
    : nodes_(nodes), elements_(elements) {
    // Nothing to do here
}
