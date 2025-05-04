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
    : mesh(mesh), error_estimator(mesh, estimator_type, error_norm),
      mesh_quality(mesh, quality_metric), global_error_norm(0.0) {
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
    // Compute error indicators
    error_indicators = error_estimator.estimateError(solution, H, M, m_star, V);
    
    // Compute global error norm
    global_error_norm = error_estimator.computeGlobalErrorNorm(error_indicators);
    
    // Compute refinement flags based on the strategy
    std::vector<bool> refine_flags;
    switch (strategy) {
        case RefinementStrategy::FIXED_FRACTION:
            refine_flags = error_estimator.computeRefinementFlags(error_indicators, 1, parameter);
            break;
        case RefinementStrategy::FIXED_THRESHOLD:
            refine_flags = error_estimator.computeRefinementFlags(error_indicators, 2, parameter);
            break;
        case RefinementStrategy::DORFLER:
            refine_flags = error_estimator.computeRefinementFlags(error_indicators, 3, parameter);
            break;
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
    return mesh_quality.getMinQuality();
}

/**
 * @brief Gets the average quality metric value from the last refinement.
 *
 * @return The average quality metric value from the last refinement
 */
double AdaptiveRefinement::getAvgQuality() const {
    return mesh_quality.getAvgQuality();
}

/**
 * @brief Improves mesh quality after refinement.
 *
 * This private method improves mesh quality after refinement using
 * Laplacian smoothing, optimization-based smoothing, and edge swapping.
 *
 * @return True if the mesh quality improved, false otherwise
 */
bool AdaptiveRefinement::improveMeshQuality() {
    // Compute initial quality metrics
    mesh_quality.computeQualityMetrics();
    double initial_min_quality = mesh_quality.getMinQuality();
    double initial_avg_quality = mesh_quality.getAvgQuality();
    
    // Apply Laplacian smoothing
    bool improved_laplacian = mesh_quality.improveMeshQualityLaplacian(5, 0.5);
    
    // Apply edge swapping
    bool improved_edge_swap = mesh_quality.improveMeshQualityEdgeSwap();
    
    // Apply optimization-based smoothing
    bool improved_optimization = mesh_quality.improveMeshQualityOptimization(10);
    
    // Compute final quality metrics
    mesh_quality.computeQualityMetrics();
    double final_min_quality = mesh_quality.getMinQuality();
    double final_avg_quality = mesh_quality.getAvgQuality();
    
    // Check if quality has improved
    bool improved = improved_laplacian || improved_edge_swap || improved_optimization;
    
    return improved;
}
