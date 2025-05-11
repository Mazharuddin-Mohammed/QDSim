/**
 * @file mesh_quality.cpp
 * @brief Implementation of the MeshQuality class for mesh quality control.
 *
 * This file contains the implementation of the MeshQuality class, which provides
 * methods for assessing and improving mesh quality. The implementation includes
 * algorithms for computing quality metrics, detecting poor-quality elements,
 * and improving mesh quality through smoothing and edge swapping.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh_quality.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <map>
#include <iostream>

/**
 * @brief Constructs a new MeshQuality object.
 *
 * @param mesh The mesh to assess and improve
 * @param metric The quality metric to use
 */
MeshQuality::MeshQuality(Mesh& mesh, QualityMetric metric)
    : mesh(mesh), metric(metric) {
    // Initialize quality metrics
    quality.resize(mesh.getNumElements(), 0.0);
}

/**
 * @brief Computes quality metrics for all elements.
 *
 * This method computes the specified quality metric for all elements in the mesh.
 * Higher values indicate better quality for all metrics except ASPECT_RATIO,
 * where lower values indicate better quality.
 *
 * @return A vector of quality metrics for each element
 */
std::vector<double> MeshQuality::computeQualityMetrics() {
    // Compute quality metrics for all elements
    for (size_t i = 0; i < mesh.getNumElements(); ++i) {
        switch (metric) {
            case QualityMetric::ASPECT_RATIO:
                quality[i] = computeAspectRatio(i);
                break;
            case QualityMetric::MINIMUM_ANGLE:
                quality[i] = computeMinimumAngle(i);
                break;
            case QualityMetric::SHAPE_REGULARITY:
                quality[i] = computeShapeRegularity(i);
                break;
            case QualityMetric::CONDITION_NUMBER:
                quality[i] = computeConditionNumber(i);
                break;
        }
    }
    return quality;
}

/**
 * @brief Detects poor-quality elements.
 *
 * This method detects elements with quality below the specified threshold.
 * For ASPECT_RATIO, elements with quality above the threshold are considered poor.
 * For all other metrics, elements with quality below the threshold are considered poor.
 *
 * @param threshold The quality threshold
 * @return A vector of boolean flags indicating which elements have poor quality
 */
std::vector<bool> MeshQuality::detectPoorQualityElements(double threshold) {
    // Ensure quality metrics are computed
    if (quality.empty()) {
        computeQualityMetrics();
    }

    // Initialize flags
    std::vector<bool> poor_quality(mesh.getNumElements(), false);

    // Detect poor-quality elements
    for (size_t i = 0; i < mesh.getNumElements(); ++i) {
        if (metric == QualityMetric::ASPECT_RATIO) {
            // For aspect ratio, lower is better
            poor_quality[i] = quality[i] > threshold;
        } else {
            // For all other metrics, higher is better
            poor_quality[i] = quality[i] < threshold;
        }
    }

    return poor_quality;
}

/**
 * @brief Gets the minimum quality metric value.
 *
 * @return The minimum quality metric value
 */
double MeshQuality::getMinQuality() const {
    if (quality.empty()) {
        return 0.0;
    }

    if (metric == QualityMetric::ASPECT_RATIO) {
        // For aspect ratio, higher values indicate worse quality
        return *std::max_element(quality.begin(), quality.end());
    } else {
        // For all other metrics, lower values indicate worse quality
        return *std::min_element(quality.begin(), quality.end());
    }
}

/**
 * @brief Gets the average quality metric value.
 *
 * @return The average quality metric value
 */
double MeshQuality::getAvgQuality() const {
    if (quality.empty()) {
        return 0.0;
    }

    double sum = std::accumulate(quality.begin(), quality.end(), 0.0);
    return sum / quality.size();
}

/**
 * @brief Computes the aspect ratio of a triangular element.
 *
 * The aspect ratio is the ratio of the longest edge to the shortest edge.
 * Lower values indicate better quality, with 1.0 being the best (equilateral triangle).
 *
 * @param elem_idx The index of the element
 * @return The aspect ratio of the element
 */
double MeshQuality::computeAspectRatio(int elem_idx) {
    // Get element vertices
    const auto& elem = mesh.getElements()[elem_idx];
    const auto& nodes = mesh.getNodes();

    // Compute edge lengths
    double l1 = (nodes[elem[1]] - nodes[elem[0]]).norm();
    double l2 = (nodes[elem[2]] - nodes[elem[1]]).norm();
    double l3 = (nodes[elem[0]] - nodes[elem[2]]).norm();

    // Compute aspect ratio
    double min_length = std::min({l1, l2, l3});
    double max_length = std::max({l1, l2, l3});

    // Avoid division by zero
    if (min_length < 1e-10) {
        return 1e10;
    }

    return max_length / min_length;
}

/**
 * @brief Computes the minimum angle of a triangular element.
 *
 * The minimum angle is the smallest of the three angles in the triangle.
 * Higher values indicate better quality, with 60 degrees being the best (equilateral triangle).
 *
 * @param elem_idx The index of the element
 * @return The minimum angle of the element in degrees
 */
double MeshQuality::computeMinimumAngle(int elem_idx) {
    // Get element vertices
    const auto& elem = mesh.getElements()[elem_idx];
    const auto& nodes = mesh.getNodes();

    // Get vertex coordinates
    Eigen::Vector2d a = nodes[elem[0]];
    Eigen::Vector2d b = nodes[elem[1]];
    Eigen::Vector2d c = nodes[elem[2]];

    // Compute edge vectors
    Eigen::Vector2d ab = b - a;
    Eigen::Vector2d bc = c - b;
    Eigen::Vector2d ca = a - c;

    // Compute edge lengths
    double lab = ab.norm();
    double lbc = bc.norm();
    double lca = ca.norm();

    // Avoid division by zero
    if (lab < 1e-10 || lbc < 1e-10 || lca < 1e-10) {
        return 0.0;
    }

    // Compute angles using the law of cosines
    double cos_a = (lab * lab + lca * lca - lbc * lbc) / (2.0 * lab * lca);
    double cos_b = (lab * lab + lbc * lbc - lca * lca) / (2.0 * lab * lbc);
    double cos_c = (lbc * lbc + lca * lca - lab * lab) / (2.0 * lbc * lca);

    // Clamp cosines to [-1, 1] to avoid numerical issues
    cos_a = std::max(-1.0, std::min(1.0, cos_a));
    cos_b = std::max(-1.0, std::min(1.0, cos_b));
    cos_c = std::max(-1.0, std::min(1.0, cos_c));

    // Compute angles in degrees
    double angle_a = std::acos(cos_a) * 180.0 / M_PI;
    double angle_b = std::acos(cos_b) * 180.0 / M_PI;
    double angle_c = std::acos(cos_c) * 180.0 / M_PI;

    // Return the minimum angle
    return std::min({angle_a, angle_b, angle_c});
}

/**
 * @brief Computes the shape regularity of a triangular element.
 *
 * The shape regularity is the ratio of the inscribed circle radius to the circumscribed
 * circle radius, normalized to give a value of 1 for an equilateral triangle.
 * Higher values indicate better quality, with 1.0 being the best (equilateral triangle).
 *
 * @param elem_idx The index of the element
 * @return The shape regularity of the element
 */
double MeshQuality::computeShapeRegularity(int elem_idx) {
    // Get element vertices
    const auto& elem = mesh.getElements()[elem_idx];
    const auto& nodes = mesh.getNodes();

    // Get vertex coordinates
    Eigen::Vector2d a = nodes[elem[0]];
    Eigen::Vector2d b = nodes[elem[1]];
    Eigen::Vector2d c = nodes[elem[2]];

    // Compute edge lengths
    double ab = (b - a).norm();
    double bc = (c - b).norm();
    double ca = (a - c).norm();

    // Compute semi-perimeter
    double s = (ab + bc + ca) / 2.0;

    // Compute area using Heron's formula
    double area = std::sqrt(s * (s - ab) * (s - bc) * (s - ca));

    // Avoid division by zero
    if (area < 1e-10) {
        return 0.0;
    }

    // Compute inscribed circle radius
    double r_in = area / s;

    // Compute circumscribed circle radius
    double r_out = (ab * bc * ca) / (4.0 * area);

    // Compute shape regularity (normalized to 1 for equilateral triangle)
    return 2.0 * r_in / r_out;
}

/**
 * @brief Computes the condition number of a triangular element.
 *
 * The condition number is the ratio of the largest to smallest singular value
 * of the element Jacobian. Lower values indicate better quality, with 1.0 being
 * the best (equilateral triangle).
 *
 * @param elem_idx The index of the element
 * @return The condition number of the element
 */
double MeshQuality::computeConditionNumber(int elem_idx) {
    // Get element vertices
    const auto& elem = mesh.getElements()[elem_idx];
    const auto& nodes = mesh.getNodes();

    // Get vertex coordinates
    Eigen::Vector2d a = nodes[elem[0]];
    Eigen::Vector2d b = nodes[elem[1]];
    Eigen::Vector2d c = nodes[elem[2]];

    // Compute edge vectors
    Eigen::Vector2d ab = b - a;
    Eigen::Vector2d ac = c - a;

    // Construct Jacobian matrix
    Eigen::Matrix2d J;
    J.col(0) = ab;
    J.col(1) = ac;

    // Compute singular values
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(J);
    double sigma_max = svd.singularValues()(0);
    double sigma_min = svd.singularValues()(1);

    // Avoid division by zero
    if (sigma_min < 1e-10) {
        return 1e10;
    }

    // Compute condition number
    return sigma_max / sigma_min;
}

/**
 * @brief Improves mesh quality using Laplacian smoothing.
 *
 * This method improves mesh quality by moving interior nodes towards the
 * average position of their neighbors. It preserves boundary nodes to
 * maintain the domain shape.
 *
 * @param num_iterations The number of smoothing iterations
 * @param relaxation The relaxation factor (0 to 1)
 * @return True if the mesh quality improved, false otherwise
 */
bool MeshQuality::improveMeshQualityLaplacian(int num_iterations, double relaxation) {
    // Get mesh data
    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    const auto& elements = mesh.getElements();

    // Compute initial quality metrics
    std::vector<double> initial_quality = computeQualityMetrics();
    double initial_min_quality = getMinQuality();
    double initial_avg_quality = getAvgQuality();

    // Build node-to-node connectivity
    std::vector<std::set<int>> neighbors(nodes.size());
    for (const auto& elem : elements) {
        for (int i = 0; i < 3; ++i) {
            neighbors[elem[i]].insert(elem[(i + 1) % 3]);
            neighbors[elem[i]].insert(elem[(i + 2) % 3]);
        }
    }

    // Identify boundary nodes
    std::vector<bool> is_boundary(nodes.size(), false);
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        for (int j = 0; j < 3; ++j) {
            int n1 = elem[j];
            int n2 = elem[(j + 1) % 3];

            // Count how many elements share this edge
            int count = 0;
            for (size_t k = 0; k < elements.size(); ++k) {
                const auto& other_elem = elements[k];
                for (int l = 0; l < 3; ++l) {
                    int m1 = other_elem[l];
                    int m2 = other_elem[(l + 1) % 3];
                    if ((n1 == m1 && n2 == m2) || (n1 == m2 && n2 == m1)) {
                        count++;
                    }
                }
            }

            // If the edge is on the boundary, mark both nodes as boundary nodes
            if (count == 1) {
                is_boundary[n1] = true;
                is_boundary[n2] = true;
            }
        }
    }

    // Perform Laplacian smoothing
    std::vector<Eigen::Vector2d> original_nodes = nodes;
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<Eigen::Vector2d> new_nodes = nodes;

        // Smooth interior nodes
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (is_boundary[i] || neighbors[i].empty()) {
                continue;
            }

            // Compute average position of neighbors
            Eigen::Vector2d avg(0, 0);
            for (int j : neighbors[i]) {
                avg += nodes[j];
            }
            avg /= neighbors[i].size();

            // Apply relaxation
            new_nodes[i] = (1.0 - relaxation) * nodes[i] + relaxation * avg;
        }

        // Update node positions
        nodes = new_nodes;

        // Check if any elements have become inverted
        bool has_inverted = false;
        for (size_t i = 0; i < elements.size(); ++i) {
            const auto& elem = elements[i];
            Eigen::Vector2d a = nodes[elem[0]];
            Eigen::Vector2d b = nodes[elem[1]];
            Eigen::Vector2d c = nodes[elem[2]];

            // Compute signed area
            double area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));

            if (area <= 0) {
                has_inverted = true;
                break;
            }
        }

        // If any elements have become inverted, revert to original positions
        if (has_inverted) {
            nodes = original_nodes;
            break;
        }
    }

    // Compute final quality metrics
    std::vector<double> final_quality = computeQualityMetrics();
    double final_min_quality = getMinQuality();
    double final_avg_quality = getAvgQuality();

    // Check if quality has improved
    bool improved = false;
    if (metric == QualityMetric::ASPECT_RATIO) {
        // For aspect ratio, lower is better
        improved = final_min_quality < initial_min_quality || final_avg_quality < initial_avg_quality;
    } else {
        // For all other metrics, higher is better
        improved = final_min_quality > initial_min_quality || final_avg_quality > initial_avg_quality;
    }

    return improved;
}

/**
 * @brief Improves mesh quality using optimization-based smoothing.
 *
 * This method improves mesh quality by optimizing node positions to maximize
 * the minimum quality metric. It uses a gradient-based optimization algorithm.
 *
 * @param num_iterations The number of optimization iterations
 * @return True if the mesh quality improved, false otherwise
 */
bool MeshQuality::improveMeshQualityOptimization(int num_iterations) {
    // Get mesh data
    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    const auto& elements = mesh.getElements();

    // Compute initial quality metrics
    std::vector<double> initial_quality = computeQualityMetrics();
    double initial_min_quality = getMinQuality();
    double initial_avg_quality = getAvgQuality();

    // Identify boundary nodes
    std::vector<bool> is_boundary(nodes.size(), false);
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        for (int j = 0; j < 3; ++j) {
            int n1 = elem[j];
            int n2 = elem[(j + 1) % 3];

            // Count how many elements share this edge
            int count = 0;
            for (size_t k = 0; k < elements.size(); ++k) {
                const auto& other_elem = elements[k];
                for (int l = 0; l < 3; ++l) {
                    int m1 = other_elem[l];
                    int m2 = other_elem[(l + 1) % 3];
                    if ((n1 == m1 && n2 == m2) || (n1 == m2 && n2 == m1)) {
                        count++;
                    }
                }
            }

            // If the edge is on the boundary, mark both nodes as boundary nodes
            if (count == 1) {
                is_boundary[n1] = true;
                is_boundary[n2] = true;
            }
        }
    }

    // Build node-to-element connectivity
    std::vector<std::vector<int>> node_elements(nodes.size());
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        for (int j = 0; j < 3; ++j) {
            node_elements[elem[j]].push_back(i);
        }
    }

    // Perform optimization-based smoothing
    std::vector<Eigen::Vector2d> original_nodes = nodes;
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Iterate over all interior nodes
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (is_boundary[i] || node_elements[i].empty()) {
                continue;
            }

            // Compute current minimum quality for elements connected to this node
            double current_min_quality = std::numeric_limits<double>::max();
            if (metric == QualityMetric::ASPECT_RATIO) {
                current_min_quality = 0.0;
            }

            for (int elem_idx : node_elements[i]) {
                double q;
                switch (metric) {
                    case QualityMetric::ASPECT_RATIO:
                        q = computeAspectRatio(elem_idx);
                        current_min_quality = std::max(current_min_quality, q);
                        break;
                    case QualityMetric::MINIMUM_ANGLE:
                        q = computeMinimumAngle(elem_idx);
                        current_min_quality = std::min(current_min_quality, q);
                        break;
                    case QualityMetric::SHAPE_REGULARITY:
                        q = computeShapeRegularity(elem_idx);
                        current_min_quality = std::min(current_min_quality, q);
                        break;
                    case QualityMetric::CONDITION_NUMBER:
                        q = computeConditionNumber(elem_idx);
                        current_min_quality = std::max(current_min_quality, q);
                        break;
                }
            }

            // Try to improve the minimum quality by moving the node
            const double step_size = 0.1;
            const double delta = 0.01;

            // Compute approximate gradient using finite differences
            Eigen::Vector2d gradient(0, 0);
            Eigen::Vector2d original_pos = nodes[i];

            // Compute gradient in x direction
            nodes[i][0] += delta;
            double q_plus_x = std::numeric_limits<double>::max();
            if (metric == QualityMetric::ASPECT_RATIO || metric == QualityMetric::CONDITION_NUMBER) {
                q_plus_x = 0.0;
            }

            for (int elem_idx : node_elements[i]) {
                double q;
                switch (metric) {
                    case QualityMetric::ASPECT_RATIO:
                        q = computeAspectRatio(elem_idx);
                        q_plus_x = std::max(q_plus_x, q);
                        break;
                    case QualityMetric::MINIMUM_ANGLE:
                        q = computeMinimumAngle(elem_idx);
                        q_plus_x = std::min(q_plus_x, q);
                        break;
                    case QualityMetric::SHAPE_REGULARITY:
                        q = computeShapeRegularity(elem_idx);
                        q_plus_x = std::min(q_plus_x, q);
                        break;
                    case QualityMetric::CONDITION_NUMBER:
                        q = computeConditionNumber(elem_idx);
                        q_plus_x = std::max(q_plus_x, q);
                        break;
                }
            }

            nodes[i][0] = original_pos[0] - delta;
            double q_minus_x = std::numeric_limits<double>::max();
            if (metric == QualityMetric::ASPECT_RATIO || metric == QualityMetric::CONDITION_NUMBER) {
                q_minus_x = 0.0;
            }

            for (int elem_idx : node_elements[i]) {
                double q;
                switch (metric) {
                    case QualityMetric::ASPECT_RATIO:
                        q = computeAspectRatio(elem_idx);
                        q_minus_x = std::max(q_minus_x, q);
                        break;
                    case QualityMetric::MINIMUM_ANGLE:
                        q = computeMinimumAngle(elem_idx);
                        q_minus_x = std::min(q_minus_x, q);
                        break;
                    case QualityMetric::SHAPE_REGULARITY:
                        q = computeShapeRegularity(elem_idx);
                        q_minus_x = std::min(q_minus_x, q);
                        break;
                    case QualityMetric::CONDITION_NUMBER:
                        q = computeConditionNumber(elem_idx);
                        q_minus_x = std::max(q_minus_x, q);
                        break;
                }
            }

            // Compute gradient in y direction
            nodes[i] = original_pos;
            nodes[i][1] += delta;
            double q_plus_y = std::numeric_limits<double>::max();
            if (metric == QualityMetric::ASPECT_RATIO || metric == QualityMetric::CONDITION_NUMBER) {
                q_plus_y = 0.0;
            }

            for (int elem_idx : node_elements[i]) {
                double q;
                switch (metric) {
                    case QualityMetric::ASPECT_RATIO:
                        q = computeAspectRatio(elem_idx);
                        q_plus_y = std::max(q_plus_y, q);
                        break;
                    case QualityMetric::MINIMUM_ANGLE:
                        q = computeMinimumAngle(elem_idx);
                        q_plus_y = std::min(q_plus_y, q);
                        break;
                    case QualityMetric::SHAPE_REGULARITY:
                        q = computeShapeRegularity(elem_idx);
                        q_plus_y = std::min(q_plus_y, q);
                        break;
                    case QualityMetric::CONDITION_NUMBER:
                        q = computeConditionNumber(elem_idx);
                        q_plus_y = std::max(q_plus_y, q);
                        break;
                }
            }

            nodes[i][1] = original_pos[1] - delta;
            double q_minus_y = std::numeric_limits<double>::max();
            if (metric == QualityMetric::ASPECT_RATIO || metric == QualityMetric::CONDITION_NUMBER) {
                q_minus_y = 0.0;
            }

            for (int elem_idx : node_elements[i]) {
                double q;
                switch (metric) {
                    case QualityMetric::ASPECT_RATIO:
                        q = computeAspectRatio(elem_idx);
                        q_minus_y = std::max(q_minus_y, q);
                        break;
                    case QualityMetric::MINIMUM_ANGLE:
                        q = computeMinimumAngle(elem_idx);
                        q_minus_y = std::min(q_minus_y, q);
                        break;
                    case QualityMetric::SHAPE_REGULARITY:
                        q = computeShapeRegularity(elem_idx);
                        q_minus_y = std::min(q_minus_y, q);
                        break;
                    case QualityMetric::CONDITION_NUMBER:
                        q = computeConditionNumber(elem_idx);
                        q_minus_y = std::max(q_minus_y, q);
                        break;
                }
            }

            // Compute gradient
            if (metric == QualityMetric::ASPECT_RATIO || metric == QualityMetric::CONDITION_NUMBER) {
                // For these metrics, lower is better
                gradient[0] = (q_minus_x - q_plus_x) / (2.0 * delta);
                gradient[1] = (q_minus_y - q_plus_y) / (2.0 * delta);
            } else {
                // For these metrics, higher is better
                gradient[0] = (q_plus_x - q_minus_x) / (2.0 * delta);
                gradient[1] = (q_plus_y - q_minus_y) / (2.0 * delta);
            }

            // Normalize gradient
            double grad_norm = gradient.norm();
            if (grad_norm > 1e-10) {
                gradient /= grad_norm;
            }

            // Move node in direction of gradient
            nodes[i] = original_pos + step_size * gradient;

            // Check if any elements have become inverted
            bool has_inverted = false;
            for (int elem_idx : node_elements[i]) {
                const auto& elem = elements[elem_idx];
                Eigen::Vector2d a = nodes[elem[0]];
                Eigen::Vector2d b = nodes[elem[1]];
                Eigen::Vector2d c = nodes[elem[2]];

                // Compute signed area
                double area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));

                if (area <= 0) {
                    has_inverted = true;
                    break;
                }
            }

            // If any elements have become inverted, revert to original position
            if (has_inverted) {
                nodes[i] = original_pos;
            }
        }
    }

    // Compute final quality metrics
    std::vector<double> final_quality = computeQualityMetrics();
    double final_min_quality = getMinQuality();
    double final_avg_quality = getAvgQuality();

    // Check if quality has improved
    bool improved = false;
    if (metric == QualityMetric::ASPECT_RATIO || metric == QualityMetric::CONDITION_NUMBER) {
        // For these metrics, lower is better
        improved = final_min_quality < initial_min_quality || final_avg_quality < initial_avg_quality;
    } else {
        // For all other metrics, higher is better
        improved = final_min_quality > initial_min_quality || final_avg_quality > initial_avg_quality;
    }

    return improved;
}

/**
 * @brief Improves mesh quality using edge swapping.
 *
 * This method improves mesh quality by swapping edges between adjacent elements
 * when the swap improves the minimum quality of the affected elements.
 *
 * @return True if the mesh quality improved, false otherwise
 */
bool MeshQuality::improveMeshQualityEdgeSwap() {
    // Get mesh data
    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    auto& elements = const_cast<std::vector<std::array<int, 3>>&>(mesh.getElements());

    // Compute initial quality metrics
    std::vector<double> initial_quality = computeQualityMetrics();
    double initial_min_quality = getMinQuality();
    double initial_avg_quality = getAvgQuality();

    // Build edge-to-element connectivity
    std::map<std::pair<int, int>, std::vector<int>> edge_elements;
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        for (int j = 0; j < 3; ++j) {
            int n1 = elem[j];
            int n2 = elem[(j + 1) % 3];
            auto edge = std::minmax(n1, n2);
            edge_elements[edge].push_back(i);
        }
    }

    // Perform edge swapping
    bool any_swapped = false;
    for (const auto& [edge, elem_indices] : edge_elements) {
        // Only consider interior edges (shared by exactly two elements)
        if (elem_indices.size() != 2) {
            continue;
        }

        int elem1_idx = elem_indices[0];
        int elem2_idx = elem_indices[1];

        // Check if edge swap is valid
        if (!isEdgeSwapValid(elem1_idx, elem2_idx)) {
            continue;
        }

        // Compute quality before swap
        double q1_before = quality[elem1_idx];
        double q2_before = quality[elem2_idx];
        double min_quality_before = std::min(q1_before, q2_before);

        // Perform edge swap
        std::array<int, 3> elem1_before = elements[elem1_idx];
        std::array<int, 3> elem2_before = elements[elem2_idx];

        // Find the nodes that are not part of the shared edge
        int n1 = -1, n2 = -1;
        for (int i = 0; i < 3; ++i) {
            if (elem1_before[i] != edge.first && elem1_before[i] != edge.second) {
                n1 = elem1_before[i];
            }
            if (elem2_before[i] != edge.first && elem2_before[i] != edge.second) {
                n2 = elem2_before[i];
            }
        }

        // Create new elements with swapped edge
        std::array<int, 3> elem1_after = {n1, n2, edge.first};
        std::array<int, 3> elem2_after = {n1, edge.second, n2};

        // Temporarily swap the edge
        elements[elem1_idx] = elem1_after;
        elements[elem2_idx] = elem2_after;

        // Compute quality after swap
        double q1_after = computeShapeRegularity(elem1_idx);
        double q2_after = computeShapeRegularity(elem2_idx);
        double min_quality_after = std::min(q1_after, q2_after);

        // Decide whether to keep the swap
        bool keep_swap = false;
        if (metric == QualityMetric::ASPECT_RATIO || metric == QualityMetric::CONDITION_NUMBER) {
            // For these metrics, lower is better
            keep_swap = min_quality_after < min_quality_before;
        } else {
            // For all other metrics, higher is better
            keep_swap = min_quality_after > min_quality_before;
        }

        if (keep_swap) {
            // Update quality metrics
            quality[elem1_idx] = q1_after;
            quality[elem2_idx] = q2_after;
            any_swapped = true;
        } else {
            // Revert the swap
            elements[elem1_idx] = elem1_before;
            elements[elem2_idx] = elem2_before;
        }
    }

    // Compute final quality metrics
    std::vector<double> final_quality = computeQualityMetrics();
    double final_min_quality = getMinQuality();
    double final_avg_quality = getAvgQuality();

    // Check if quality has improved
    bool improved = false;
    if (metric == QualityMetric::ASPECT_RATIO || metric == QualityMetric::CONDITION_NUMBER) {
        // For these metrics, lower is better
        improved = final_min_quality < initial_min_quality || final_avg_quality < initial_avg_quality;
    } else {
        // For all other metrics, higher is better
        improved = final_min_quality > initial_min_quality || final_avg_quality > initial_avg_quality;
    }

    return improved && any_swapped;
}

/**
 * @brief Checks if an edge swap is valid.
 *
 * This private method checks if an edge swap is valid, i.e., if it does not
 * create inverted elements or non-conforming mesh.
 *
 * @param elem1_idx The index of the first element
 * @param elem2_idx The index of the second element
 * @return True if the edge swap is valid, false otherwise
 */
bool MeshQuality::isEdgeSwapValid(int elem1_idx, int elem2_idx) {
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();

    // Get element vertices
    const auto& elem1 = elements[elem1_idx];
    const auto& elem2 = elements[elem2_idx];

    // Find the shared edge
    std::pair<int, int> shared_edge(-1, -1);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (elem1[i] == elem2[j]) {
                if (shared_edge.first == -1) {
                    shared_edge.first = elem1[i];
                } else {
                    shared_edge.second = elem1[i];
                    break;
                }
            }
        }
        if (shared_edge.second != -1) {
            break;
        }
    }

    // Ensure shared_edge.first < shared_edge.second
    if (shared_edge.first > shared_edge.second) {
        std::swap(shared_edge.first, shared_edge.second);
    }

    // Find the nodes that are not part of the shared edge
    int n1 = -1, n2 = -1;
    for (int i = 0; i < 3; ++i) {
        if (elem1[i] != shared_edge.first && elem1[i] != shared_edge.second) {
            n1 = elem1[i];
        }
        if (elem2[i] != shared_edge.first && elem2[i] != shared_edge.second) {
            n2 = elem2[i];
        }
    }

    // Check if the quadrilateral formed by the two triangles is convex
    Eigen::Vector2d a = nodes[shared_edge.first];
    Eigen::Vector2d b = nodes[shared_edge.second];
    Eigen::Vector2d c = nodes[n1];
    Eigen::Vector2d d = nodes[n2];

    // Compute signed areas of the triangles after swap
    double area1 = 0.5 * ((c[0] - a[0]) * (d[1] - a[1]) - (d[0] - a[0]) * (c[1] - a[1]));
    double area2 = 0.5 * ((d[0] - b[0]) * (c[1] - b[1]) - (c[0] - b[0]) * (d[1] - b[1]));

    // The swap is valid if both triangles have positive area
    return area1 > 0 && area2 > 0;
}

/**
 * @brief Performs an edge swap between two adjacent elements.
 *
 * This private method performs an edge swap between two adjacent elements.
 * It updates the element connectivity and node positions as needed.
 *
 * @param elem1_idx The index of the first element
 * @param elem2_idx The index of the second element
 * @return True if the edge swap was successful, false otherwise
 */
bool MeshQuality::performEdgeSwap(int elem1_idx, int elem2_idx) {
    // Check if the edge swap is valid
    if (!isEdgeSwapValid(elem1_idx, elem2_idx)) {
        return false;
    }

    // Get mesh data
    auto& elements = const_cast<std::vector<std::array<int, 3>>&>(mesh.getElements());

    // Get element vertices
    const auto& elem1 = elements[elem1_idx];
    const auto& elem2 = elements[elem2_idx];

    // Find the shared edge
    std::pair<int, int> shared_edge(-1, -1);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (elem1[i] == elem2[j]) {
                if (shared_edge.first == -1) {
                    shared_edge.first = elem1[i];
                } else {
                    shared_edge.second = elem1[i];
                    break;
                }
            }
        }
        if (shared_edge.second != -1) {
            break;
        }
    }

    // Find the nodes that are not part of the shared edge
    int n1 = -1, n2 = -1;
    for (int i = 0; i < 3; ++i) {
        if (elem1[i] != shared_edge.first && elem1[i] != shared_edge.second) {
            n1 = elem1[i];
        }
        if (elem2[i] != shared_edge.first && elem2[i] != shared_edge.second) {
            n2 = elem2[i];
        }
    }

    // Create new elements with swapped edge
    std::array<int, 3> elem1_new = {n1, n2, shared_edge.first};
    std::array<int, 3> elem2_new = {n1, shared_edge.second, n2};

    // Update elements
    elements[elem1_idx] = elem1_new;
    elements[elem2_idx] = elem2_new;

    return true;
}
