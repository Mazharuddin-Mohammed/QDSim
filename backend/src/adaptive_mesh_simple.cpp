/**
 * @file adaptive_mesh_simple.cpp
 * @brief Implementation of simplified adaptive mesh refinement.
 *
 * This file contains a simplified implementation of adaptive mesh refinement
 * that doesn't use MPI. It provides functions for refining elements, computing
 * refinement flags based on solution gradients, smoothing the mesh, and checking
 * mesh quality and conformity.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "adaptive_mesh.h"
#include <map>
#include <set>
#include <cmath>

void AdaptiveMesh::refineMesh(Mesh& mesh, const std::vector<bool>& refine_flags, bool allow_hanging_nodes) {
    // This is a simplified version of the refineMesh function that doesn't use MPI

    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    auto& elements = const_cast<std::vector<std::array<int, 3>>&>(mesh.getElements());
    auto& quadratic_elements = const_cast<std::vector<std::array<int, 6>>&>(mesh.getQuadraticElements());
    auto& cubic_elements = const_cast<std::vector<std::array<int, 10>>&>(mesh.getCubicElements());
    int order = mesh.getElementOrder();

    // Create new elements and nodes
    std::vector<std::array<int, 3>> new_elements;
    std::vector<std::array<int, 6>> new_quadratic_elements;
    std::vector<std::array<int, 10>> new_cubic_elements;
    std::vector<Eigen::Vector2d> new_nodes;

    // Maps to keep track of edge midpoints and triangle centroids
    std::map<std::pair<int, int>, int> edge_midpoints;
    std::map<std::array<int, 3>, int> triangle_centroids;

    // Helper function to get or create a midpoint node
    auto getMidpoint = [&](int n1, int n2) -> int {
        auto edge = std::minmax(n1, n2);
        if (edge_midpoints.find(edge) == edge_midpoints.end()) {
            Eigen::Vector2d mid = 0.5 * (nodes[n1] + nodes[n2]);
            new_nodes.push_back(mid);
            edge_midpoints[edge] = nodes.size() + new_nodes.size() - 1;
        }
        return edge_midpoints[edge];
    };

    // Helper function to get or create a centroid node
    auto getCentroid = [&](int n0, int n1, int n2) -> int {
        std::array<int, 3> tri = {n0, n1, n2};
        std::sort(tri.begin(), tri.end());
        if (triangle_centroids.find(tri) == triangle_centroids.end()) {
            Eigen::Vector2d centroid = (nodes[n0] + nodes[n1] + nodes[n2]) / 3.0;
            new_nodes.push_back(centroid);
            triangle_centroids[tri] = nodes.size() + new_nodes.size() - 1;
        }
        return triangle_centroids[tri];
    };

    // Refine elements
    for (size_t e = 0; e < elements.size(); ++e) {
        if (e < refine_flags.size() && refine_flags[e]) {
            // Refine this element
            const auto& elem = elements[e];
            int n0 = elem[0], n1 = elem[1], n2 = elem[2];

            // Get or create midpoint nodes
            int m01 = getMidpoint(n0, n1);
            int m12 = getMidpoint(n1, n2);
            int m20 = getMidpoint(n2, n0);

            // Create four new elements
            new_elements.push_back({n0, m01, m20});
            new_elements.push_back({m01, n1, m12});
            new_elements.push_back({m20, m12, n2});
            new_elements.push_back({m01, m12, m20});

            if (order >= 2) {
                // Create quadratic elements
                // This is a simplified version that doesn't handle all the details
                // of creating proper quadratic elements
                for (const auto& new_elem : new_elements) {
                    std::array<int, 6> qelem;
                    qelem[0] = new_elem[0];
                    qelem[1] = new_elem[1];
                    qelem[2] = new_elem[2];
                    qelem[3] = getMidpoint(new_elem[0], new_elem[1]);
                    qelem[4] = getMidpoint(new_elem[1], new_elem[2]);
                    qelem[5] = getMidpoint(new_elem[2], new_elem[0]);
                    new_quadratic_elements.push_back(qelem);
                }
            }

            if (order == 3) {
                // Create cubic elements
                // This is a simplified version that doesn't handle all the details
                // of creating proper cubic elements
                for (const auto& new_elem : new_elements) {
                    std::array<int, 10> celem;
                    celem[0] = new_elem[0];
                    celem[1] = new_elem[1];
                    celem[2] = new_elem[2];
                    celem[3] = getMidpoint(new_elem[0], new_elem[1]);
                    celem[4] = celem[3]; // Simplified
                    celem[5] = getMidpoint(new_elem[1], new_elem[2]);
                    celem[6] = celem[5]; // Simplified
                    celem[7] = getMidpoint(new_elem[2], new_elem[0]);
                    celem[8] = celem[7]; // Simplified
                    celem[9] = getCentroid(new_elem[0], new_elem[1], new_elem[2]);
                    new_cubic_elements.push_back(celem);
                }
            }
        } else {
            // Keep this element as is
            new_elements.push_back(elements[e]);
            if (order >= 2 && e < quadratic_elements.size()) {
                new_quadratic_elements.push_back(quadratic_elements[e]);
            }
            if (order == 3 && e < cubic_elements.size()) {
                new_cubic_elements.push_back(cubic_elements[e]);
            }
        }
    }

    // Add new nodes to the mesh
    for (const auto& node : new_nodes) {
        nodes.push_back(node);
    }

    // Replace elements with new elements
    elements = new_elements;
    if (order >= 2) {
        quadratic_elements = new_quadratic_elements;
    }
    if (order == 3) {
        cubic_elements = new_cubic_elements;
    }
}

std::vector<bool> AdaptiveMesh::computeRefinementFlags(const Mesh& mesh, const Eigen::VectorXd& psi, double threshold) {
    // Compute refinement flags based on the gradient of psi
    std::vector<bool> refine_flags(mesh.getNumElements(), false);

    // Compute the gradient of psi at each element
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        const auto& elem = mesh.getElements()[e];

        // Get the nodes of the element
        std::vector<Eigen::Vector2d> vertices;
        for (int i = 0; i < 3; ++i) {
            vertices.push_back(mesh.getNodes()[elem[i]]);
        }

        // Compute the area of the element
        Eigen::Vector2d v1 = vertices[1] - vertices[0];
        Eigen::Vector2d v2 = vertices[2] - vertices[0];
        double area = 0.5 * std::abs(v1(0) * v2(1) - v1(1) * v2(0));

        // Compute the gradient of psi
        double grad_mag = 0.0;
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;
            double dpsi = std::abs(psi[elem[j]] - psi[elem[k]]);
            grad_mag += dpsi;
        }
        grad_mag /= 3.0;

        // Refine if the gradient is above the threshold
        if (grad_mag > threshold) {
            refine_flags[e] = true;
        }
    }

    return refine_flags;
}

void AdaptiveMesh::smoothMesh(Mesh& mesh, int num_iterations, double quality_threshold) {
    // Smooth the mesh by moving nodes to the average of their neighbors
    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());

    // Find neighbors of each node
    std::vector<std::vector<int>> neighbors(nodes.size());
    for (const auto& elem : mesh.getElements()) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i != j) {
                    neighbors[elem[i]].push_back(elem[j]);
                }
            }
        }
    }

    // Remove duplicates
    for (auto& n : neighbors) {
        std::sort(n.begin(), n.end());
        n.erase(std::unique(n.begin(), n.end()), n.end());
    }

    // Smooth nodes
    std::vector<Eigen::Vector2d> new_nodes = nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!neighbors[i].empty()) {
            Eigen::Vector2d avg = Eigen::Vector2d::Zero();
            for (int j : neighbors[i]) {
                avg += nodes[j];
            }
            avg /= neighbors[i].size();
            new_nodes[i] = 0.5 * (nodes[i] + avg);
        }
    }

    // Update nodes
    nodes = new_nodes;
}

double AdaptiveMesh::computeTriangleQuality(const Mesh& mesh, int elem_idx) {
    // Compute the quality of a triangle element
    const auto& elem = mesh.getElements()[elem_idx];

    // Get the nodes of the element
    std::vector<Eigen::Vector2d> vertices;
    for (int i = 0; i < 3; ++i) {
        vertices.push_back(mesh.getNodes()[elem[i]]);
    }

    // Compute the side lengths
    double a = (vertices[1] - vertices[0]).norm();
    double b = (vertices[2] - vertices[1]).norm();
    double c = (vertices[0] - vertices[2]).norm();

    // Compute the area
    double s = 0.5 * (a + b + c);
    double area = std::sqrt(s * (s - a) * (s - b) * (s - c));

    // Compute the quality (ratio of area to perimeter)
    double quality = 4.0 * std::sqrt(3.0) * area / (a * b * c);

    return quality;
}

bool AdaptiveMesh::isMeshConforming(const Mesh& mesh) {
    // Check if the mesh is conforming (no hanging nodes)

    // Find all edges
    std::set<std::pair<int, int>> edges;
    for (const auto& elem : mesh.getElements()) {
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            edges.insert(std::minmax(elem[i], elem[j]));
        }
    }

    // Count how many elements share each edge
    std::map<std::pair<int, int>, int> edge_count;
    for (const auto& elem : mesh.getElements()) {
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            edge_count[std::minmax(elem[i], elem[j])]++;
        }
    }

    // Check if all edges are shared by exactly 1 or 2 elements
    for (const auto& [edge, count] : edge_count) {
        if (count > 2) {
            return false;
        }
    }

    return true;
}
