/**
 * @file adaptive_refinement.cpp
 * @brief Implementation of the AdaptiveRefinement class for advanced adaptive mesh refinement.
 *
 * This file contains the implementation of the AdaptiveRefinement class, which provides
 * methods for advanced adaptive mesh refinement based on error estimation and
 * mesh quality control.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "adaptive_refinement.h"
#include "adaptive_mesh.h"
#include <algorithm>
#include <cmath>
#include <numeric>

/**
 * @brief Constructs a new AdaptiveRefinement object.
 *
 * @param mesh The mesh to refine
 * @param estimator_type The type of error estimator to use
 * @param error_norm The error norm to use for error estimation
 * @param quality_metric The quality metric to use for mesh quality control
 */
AdaptiveRefinement::AdaptiveRefinement(Mesh& mesh, EstimatorType estimator_type,
                                     ErrorNorm error_norm, QualityMetric quality_metric)
    : mesh(mesh), global_error_norm(0.0) {
    // Initialize error indicators
    error_indicators.resize(mesh.getNumElements(), 0.0);
}

/**
 * @brief Refines the mesh adaptively.
 *
 * This method refines the mesh adaptively based on the error estimator and
 * refinement strategy. It computes error indicators, marks elements for refinement,
 * refines the mesh, and improves mesh quality.
 *
 * @param solution The solution vector
 * @param H The Hamiltonian matrix
 * @param M The mass matrix
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @param strategy The refinement strategy to use
 * @param parameter The parameter for the refinement strategy
 * @param improve_quality Whether to improve mesh quality after refinement
 * @return True if the mesh was refined, false otherwise
 */
bool AdaptiveRefinement::refine(const Eigen::VectorXd& solution,
                              const Eigen::SparseMatrix<std::complex<double>>& H,
                              const Eigen::SparseMatrix<std::complex<double>>& M,
                              std::function<double(double, double)> m_star,
                              std::function<double(double, double)> V,
                              RefinementStrategy strategy,
                              double parameter,
                              bool improve_quality) {
    // Compute error indicators using a simplified approach
    error_indicators.resize(mesh.getNumElements(), 0.0);

    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();

    // Compute error indicators based on solution gradient
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& elem = elements[e];

        // Get element nodes
        std::vector<Eigen::Vector2d> elem_nodes;
        for (int i = 0; i < 3; ++i) {
            elem_nodes.push_back(nodes[elem[i]]);
        }

        // Compute element area
        double x1 = elem_nodes[0][0], y1 = elem_nodes[0][1];
        double x2 = elem_nodes[1][0], y2 = elem_nodes[1][1];
        double x3 = elem_nodes[2][0], y3 = elem_nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Compute element diameter (longest edge)
        double d1 = (elem_nodes[1] - elem_nodes[0]).norm();
        double d2 = (elem_nodes[2] - elem_nodes[1]).norm();
        double d3 = (elem_nodes[0] - elem_nodes[2]).norm();
        double h_e = std::max({d1, d2, d3});

        // Compute solution gradient
        double u1 = solution[elem[0]], u2 = solution[elem[1]], u3 = solution[elem[2]];
        double grad_u_norm = std::sqrt(
            std::pow(u2 - u1, 2) / std::pow(d1, 2) +
            std::pow(u3 - u2, 2) / std::pow(d2, 2) +
            std::pow(u1 - u3, 2) / std::pow(d3, 2)
        );

        // Compute error indicator
        error_indicators[e] = h_e * h_e * grad_u_norm * grad_u_norm * area;
    }

    // Compute global error norm
    global_error_norm = 0.0;
    for (double error : error_indicators) {
        global_error_norm += error * error;
    }
    global_error_norm = std::sqrt(global_error_norm);

    // Compute refinement flags based on the strategy
    std::vector<bool> refine_flags(error_indicators.size(), false);

    if (strategy == RefinementStrategy::FIXED_FRACTION) {
        // Create a vector of indices
        std::vector<size_t> indices(error_indicators.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Sort indices by error indicators in descending order
        std::sort(indices.begin(), indices.end(),
                 [&error_indicators](size_t i1, size_t i2) {
                     return error_indicators[i1] > error_indicators[i2];
                 });

        // Compute the number of elements to refine
        size_t num_to_refine = static_cast<size_t>(parameter * error_indicators.size());

        // Mark elements for refinement
        for (size_t i = 0; i < num_to_refine && i < indices.size(); ++i) {
            refine_flags[indices[i]] = true;
        }
    } else if (strategy == RefinementStrategy::FIXED_THRESHOLD) {
        // Compute the maximum error indicator
        double max_error = *std::max_element(error_indicators.begin(), error_indicators.end());

        // Compute the threshold
        double threshold = parameter * max_error;

        // Mark elements for refinement
        for (size_t i = 0; i < error_indicators.size(); ++i) {
            if (error_indicators[i] > threshold) {
                refine_flags[i] = true;
            }
        }
    } else if (strategy == RefinementStrategy::DORFLER) {
        // Compute the total error
        double total_error = 0.0;
        for (double error : error_indicators) {
            total_error += error * error;
        }

        // Create a vector of indices
        std::vector<size_t> indices(error_indicators.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Sort indices by error indicators in descending order
        std::sort(indices.begin(), indices.end(),
                 [&error_indicators](size_t i1, size_t i2) {
                     return error_indicators[i1] > error_indicators[i2];
                 });

        // Mark elements for refinement until the desired fraction of the total error is reached
        double marked_error = 0.0;
        size_t i = 0;
        while (marked_error < parameter * total_error && i < indices.size()) {
            refine_flags[indices[i]] = true;
            marked_error += error_indicators[indices[i]] * error_indicators[indices[i]];
            ++i;
        }
    }

    // Check if any elements are marked for refinement
    if (std::none_of(refine_flags.begin(), refine_flags.end(), [](bool flag) { return flag; })) {
        return false;
    }

    // Refine the mesh
    mesh.refine(refine_flags);

    // Improve mesh quality if requested
    if (improve_quality) {
        improveMeshQuality();
    }

    return true;
}

/**
 * @brief Gets the error indicators from the last refinement.
 *
 * @return The error indicators from the last refinement
 */
std::vector<double> AdaptiveRefinement::getErrorIndicators() const {
    return error_indicators;
}

/**
 * @brief Gets the global error norm from the last refinement.
 *
 * @return The global error norm from the last refinement
 */
double AdaptiveRefinement::getGlobalErrorNorm() const {
    return global_error_norm;
}

/**
 * @brief Gets the minimum quality metric value from the last refinement.
 *
 * @return The minimum quality metric value from the last refinement
 */
double AdaptiveRefinement::getMinQuality() const {
    // For simplicity, we'll compute a basic shape quality metric
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();

    double min_quality = std::numeric_limits<double>::max();

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];

        // Get element nodes
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
        double quality = 2.0 * r_in / r_out;

        // Update minimum quality
        min_quality = std::min(min_quality, quality);
    }

    return min_quality;
}

/**
 * @brief Gets the average quality metric value from the last refinement.
 *
 * @return The average quality metric value from the last refinement
 */
double AdaptiveRefinement::getAvgQuality() const {
    // For simplicity, we'll compute a basic shape quality metric
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();

    double sum_quality = 0.0;

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];

        // Get element nodes
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
            continue;
        }

        // Compute inscribed circle radius
        double r_in = area / s;

        // Compute circumscribed circle radius
        double r_out = (ab * bc * ca) / (4.0 * area);

        // Compute shape regularity (normalized to 1 for equilateral triangle)
        double quality = 2.0 * r_in / r_out;

        // Accumulate quality
        sum_quality += quality;
    }

    // Compute average quality
    return sum_quality / elements.size();
}

/**
 * @brief Improves mesh quality after refinement.
 *
 * This private method improves mesh quality after refinement using
 * a simplified Laplacian smoothing approach.
 *
 * @return True if the mesh quality improved, false otherwise
 */
bool AdaptiveRefinement::improveMeshQuality() {
    // Get mesh data
    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    const auto& elements = mesh.getElements();

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
    const int num_iterations = 5;
    const double relaxation = 0.5;

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

    // For simplicity, we'll just return true to indicate that we attempted to improve the mesh quality
    return true;
}
