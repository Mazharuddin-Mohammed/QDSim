#pragma once
/**
 * @file adaptive_refinement.h
 * @brief Defines the AdaptiveRefinement class for advanced adaptive mesh refinement.
 *
 * This file contains the declaration of the AdaptiveRefinement class, which provides
 * methods for advanced adaptive mesh refinement based on error estimation and
 * mesh quality control. The class integrates the ErrorEstimator and MeshQuality
 * classes to provide a comprehensive adaptive refinement framework.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include "error_estimator.h"
#include "mesh_quality.h"
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <set>
#include <limits>

/**
 * @enum RefinementStrategy
 * @brief Enumeration of refinement strategies.
 */
enum class RefinementStrategy {
    FIXED_FRACTION,  ///< Refine a fixed fraction of elements with the largest error
    FIXED_THRESHOLD, ///< Refine elements with error exceeding a fixed threshold
    DORFLER          ///< Refine elements that contribute a fixed fraction of the total error
};

/**
 * @class AdaptiveRefinement
 * @brief Implements advanced adaptive mesh refinement techniques.
 *
 * The AdaptiveRefinement class provides methods for advanced adaptive mesh refinement
 * based on error estimation and mesh quality control. It integrates the ErrorEstimator
 * and MeshQuality classes to provide a comprehensive adaptive refinement framework.
 */
class AdaptiveRefinement {
public:
    /**
     * @brief Constructs a new AdaptiveRefinement object.
     *
     * @param mesh The mesh to refine
     * @param estimator_type The type of error estimator to use
     * @param error_norm The error norm to use for error estimation
     * @param quality_metric The quality metric to use for mesh quality control
     */
    AdaptiveRefinement(Mesh& mesh, EstimatorType estimator_type = EstimatorType::RESIDUAL,
                      ErrorNorm error_norm = ErrorNorm::ENERGY,
                      QualityMetric quality_metric = QualityMetric::SHAPE_REGULARITY);

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
    bool refine(const Eigen::VectorXd& solution,
               const Eigen::SparseMatrix<std::complex<double>>& H,
               const Eigen::SparseMatrix<std::complex<double>>& M,
               std::function<double(double, double)> m_star,
               std::function<double(double, double)> V,
               RefinementStrategy strategy = RefinementStrategy::FIXED_FRACTION,
               double parameter = 0.3,
               bool improve_quality = true);

    /**
     * @brief Gets the error indicators from the last refinement.
     *
     * @return The error indicators from the last refinement
     */
    std::vector<double> getErrorIndicators() const;

    /**
     * @brief Gets the global error norm from the last refinement.
     *
     * @return The global error norm from the last refinement
     */
    double getGlobalErrorNorm() const;

    /**
     * @brief Gets the minimum quality metric value from the last refinement.
     *
     * @return The minimum quality metric value from the last refinement
     */
    double getMinQuality() const;

    /**
     * @brief Gets the average quality metric value from the last refinement.
     *
     * @return The average quality metric value from the last refinement
     */
    double getAvgQuality() const;

private:
    Mesh& mesh;                          ///< Reference to the mesh
    std::vector<double> error_indicators; ///< Error indicators from the last refinement
    double global_error_norm;            ///< Global error norm from the last refinement

    /**
     * @brief Improves mesh quality after refinement.
     *
     * This private method improves mesh quality after refinement using
     * Laplacian smoothing, optimization-based smoothing, and edge swapping.
     *
     * @return True if the mesh quality improved, false otherwise
     */
    bool improveMeshQuality();
};
