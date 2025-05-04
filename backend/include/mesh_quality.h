#pragma once
/**
 * @file mesh_quality.h
 * @brief Defines the MeshQuality class for mesh quality control.
 *
 * This file contains the declaration of the MeshQuality class, which provides
 * methods for assessing and improving mesh quality. The class implements
 * algorithms for computing quality metrics, detecting poor-quality elements,
 * and improving mesh quality through smoothing and edge swapping.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include <Eigen/Dense>
#include <vector>
#include <functional>

/**
 * @enum QualityMetric
 * @brief Enumeration of quality metrics for mesh elements.
 */
enum class QualityMetric {
    ASPECT_RATIO,      ///< Ratio of longest to shortest edge
    MINIMUM_ANGLE,     ///< Minimum angle in the element
    SHAPE_REGULARITY,  ///< Ratio of inscribed to circumscribed circle radii
    CONDITION_NUMBER   ///< Condition number of the element Jacobian
};

/**
 * @class MeshQuality
 * @brief Implements mesh quality assessment and improvement techniques.
 *
 * The MeshQuality class provides methods for assessing and improving mesh quality.
 * It supports various quality metrics and improvement techniques, including
 * Laplacian smoothing, optimization-based smoothing, and edge swapping.
 */
class MeshQuality {
public:
    /**
     * @brief Constructs a new MeshQuality object.
     *
     * @param mesh The mesh to assess and improve
     * @param metric The quality metric to use
     */
    MeshQuality(Mesh& mesh, QualityMetric metric = QualityMetric::SHAPE_REGULARITY);

    /**
     * @brief Computes quality metrics for all elements.
     *
     * This method computes the specified quality metric for all elements in the mesh.
     * Higher values indicate better quality for all metrics except ASPECT_RATIO,
     * where lower values indicate better quality.
     *
     * @return A vector of quality metrics for each element
     */
    std::vector<double> computeQualityMetrics();

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
    std::vector<bool> detectPoorQualityElements(double threshold);

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
    bool improveMeshQualityLaplacian(int num_iterations = 5, double relaxation = 0.5);

    /**
     * @brief Improves mesh quality using optimization-based smoothing.
     *
     * This method improves mesh quality by optimizing node positions to maximize
     * the minimum quality metric. It uses a gradient-based optimization algorithm.
     *
     * @param num_iterations The number of optimization iterations
     * @return True if the mesh quality improved, false otherwise
     */
    bool improveMeshQualityOptimization(int num_iterations = 10);

    /**
     * @brief Improves mesh quality using edge swapping.
     *
     * This method improves mesh quality by swapping edges between adjacent elements
     * when the swap improves the minimum quality of the affected elements.
     *
     * @return True if the mesh quality improved, false otherwise
     */
    bool improveMeshQualityEdgeSwap();

    /**
     * @brief Gets the minimum quality metric value.
     *
     * @return The minimum quality metric value
     */
    double getMinQuality() const;

    /**
     * @brief Gets the average quality metric value.
     *
     * @return The average quality metric value
     */
    double getAvgQuality() const;

private:
    Mesh& mesh;                  ///< Reference to the mesh
    QualityMetric metric;        ///< Quality metric to use
    std::vector<double> quality; ///< Quality metrics for each element

    /**
     * @brief Computes the aspect ratio of a triangular element.
     *
     * The aspect ratio is the ratio of the longest edge to the shortest edge.
     * Lower values indicate better quality, with 1.0 being the best (equilateral triangle).
     *
     * @param elem_idx The index of the element
     * @return The aspect ratio of the element
     */
    double computeAspectRatio(int elem_idx);

    /**
     * @brief Computes the minimum angle of a triangular element.
     *
     * The minimum angle is the smallest of the three angles in the triangle.
     * Higher values indicate better quality, with 60 degrees being the best (equilateral triangle).
     *
     * @param elem_idx The index of the element
     * @return The minimum angle of the element in degrees
     */
    double computeMinimumAngle(int elem_idx);

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
    double computeShapeRegularity(int elem_idx);

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
    double computeConditionNumber(int elem_idx);

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
    bool isEdgeSwapValid(int elem1_idx, int elem2_idx);

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
    bool performEdgeSwap(int elem1_idx, int elem2_idx);
};
