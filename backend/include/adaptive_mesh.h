#pragma once
/**
 * @file adaptive_mesh.h
 * @brief Defines the AdaptiveMesh class for adaptive mesh refinement.
 *
 * This file contains the declaration of the AdaptiveMesh class, which provides
 * methods for adaptive mesh refinement based on solution features. The class
 * implements algorithms for computing refinement flags, refining the mesh,
 * ensuring mesh conformity, and improving mesh quality.
 *
 * The adaptive refinement process involves:
 * 1. Computing refinement flags based on solution gradients
 * 2. Refining the mesh by subdividing marked elements
 * 3. Ensuring mesh conformity by adding transition elements
 * 4. Smoothing the mesh to improve element quality
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include <Eigen/Dense>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @class AdaptiveMesh
 * @brief Implements adaptive mesh refinement algorithms.
 *
 * The AdaptiveMesh class provides static methods for adaptive mesh refinement
 * based on solution features. It includes methods for computing refinement flags,
 * refining the mesh, ensuring mesh conformity, and improving mesh quality.
 *
 * The class is designed to work with the Mesh class and supports both serial
 * and parallel (MPI) execution.
 */
class AdaptiveMesh {
public:
#ifdef USE_MPI
    /**
     * @brief Refines the mesh in parallel using MPI.
     *
     * This method refines the mesh by subdividing elements marked for refinement.
     * It ensures mesh conformity by adding transition elements as needed.
     * The refinement is performed in parallel using MPI.
     *
     * @param mesh The mesh to refine
     * @param refine_flags A vector of boolean flags indicating which elements to refine
     * @param comm The MPI communicator
     *
     * @throws std::invalid_argument If the size of refine_flags does not match the number of elements
     */
    static void refineMesh(Mesh& mesh, const std::vector<bool>& refine_flags, MPI_Comm comm = MPI_COMM_WORLD);
#else
    /**
     * @brief Refines the mesh.
     *
     * This method refines the mesh by subdividing elements marked for refinement.
     * It ensures mesh conformity by adding transition elements as needed.
     *
     * @param mesh The mesh to refine
     * @param refine_flags A vector of boolean flags indicating which elements to refine
     *
     * @throws std::invalid_argument If the size of refine_flags does not match the number of elements
     */
    static void refineMesh(Mesh& mesh, const std::vector<bool>& refine_flags, bool allow_hanging_nodes = false);
#endif

    /**
     * @brief Computes refinement flags based on solution gradients.
     *
     * This method computes refinement flags based on the gradient of the solution.
     * Elements with large gradients are marked for refinement.
     *
     * @param mesh The mesh
     * @param psi The solution vector
     * @param threshold The threshold for refinement (elements with gradient > threshold are refined)
     * @return A vector of boolean flags indicating which elements to refine
     */
    static std::vector<bool> computeRefinementFlags(const Mesh& mesh, const Eigen::VectorXd& psi, double threshold);

    /**
     * @brief Computes anisotropic refinement directions.
     *
     * This method computes anisotropic refinement directions based on the solution
     * gradient and hessian. It returns a vector of refinement directions for each
     * element, where each direction is a unit vector indicating the direction of
     * maximum error.
     *
     * @param mesh The mesh
     * @param psi The solution vector
     * @return A vector of refinement directions for each element
     */
    static std::vector<Eigen::Vector2d> computeAnisotropicDirections(const Mesh& mesh, const Eigen::VectorXd& psi);

    /**
     * @brief Refines the mesh anisotropically.
     *
     * This method refines the mesh anisotropically by subdividing elements in
     * specific directions based on the solution features. It is particularly
     * useful for problems with directional features like boundary layers or shocks.
     *
     * @param mesh The mesh to refine
     * @param refine_flags A vector of boolean flags indicating which elements to refine
     * @param directions A vector of refinement directions for each element
     */
    static void refineAnisotropic(Mesh& mesh, const std::vector<bool>& refine_flags, const std::vector<Eigen::Vector2d>& directions);

    /**
     * @brief Smooths the mesh to improve element quality.
     *
     * This method smooths the mesh by adjusting node positions to improve
     * element quality. It uses a Laplacian smoothing algorithm.
     *
     * @param mesh The mesh to smooth
     * @param num_iterations The number of smoothing iterations to perform (default: 3)
     * @param quality_threshold The minimum acceptable element quality (default: 0.3)
     */
    static void smoothMesh(Mesh& mesh, int num_iterations = 3, double quality_threshold = 0.3);

    /**
     * @brief Improves mesh quality using optimization-based techniques.
     *
     * This method improves mesh quality using optimization-based techniques
     * such as centroidal Voronoi tessellation (CVT) or optimization of element
     * quality measures. It is more effective than simple Laplacian smoothing
     * but also more computationally expensive.
     *
     * @param mesh The mesh to improve
     * @param quality_threshold The minimum acceptable element quality (default: 0.3)
     * @param max_iterations The maximum number of optimization iterations (default: 10)
     */
    static void improveQuality(Mesh& mesh, double quality_threshold = 0.3, int max_iterations = 10);

    /**
     * @brief Computes the quality of a triangular element.
     *
     * This method computes the quality of a triangular element using the
     * ratio of the inscribed circle radius to the circumscribed circle radius.
     * A value of 1 indicates an equilateral triangle, and a value of 0 indicates
     * a degenerate triangle.
     *
     * @param mesh The mesh
     * @param elem_idx The index of the element
     * @return The quality of the element (0 to 1)
     */
    static double computeTriangleQuality(const Mesh& mesh, int elem_idx);

    /**
     * @brief Checks if the mesh is conforming.
     *
     * This method checks if the mesh is conforming, i.e., if there are no
     * hanging nodes. A conforming mesh is required for finite element analysis.
     *
     * @param mesh The mesh to check
     * @return True if the mesh is conforming, false otherwise
     */
    static bool isMeshConforming(const Mesh& mesh);

    /**
     * @brief Coarsens the mesh in regions with low error.
     *
     * This method coarsens the mesh by merging elements in regions with low error.
     * It is useful for reducing the computational cost in regions where high
     * resolution is not needed.
     *
     * @param mesh The mesh to coarsen
     * @param error_indicators The error indicators for each element
     * @param threshold The threshold for coarsening (elements with error < threshold are coarsened)
     * @return True if the mesh was coarsened, false otherwise
     */
    static bool coarsenMesh(Mesh& mesh, const std::vector<double>& error_indicators, double threshold);

    /**
     * @brief Computes coarsening flags based on error indicators.
     *
     * This method computes coarsening flags based on the error indicators.
     * Elements with error indicators below the threshold are marked for coarsening.
     *
     * @param mesh The mesh
     * @param error_indicators The error indicators for each element
     * @param threshold The threshold for coarsening (elements with error < threshold are coarsened)
     * @return A vector of boolean flags indicating which elements to coarsen
     */
    static std::vector<bool> computeCoarseningFlags(const Mesh& mesh, const std::vector<double>& error_indicators, double threshold);

    /**
     * @brief Computes physics-based refinement flags.
     *
     * This method computes refinement flags based on the physics of the problem.
     * It takes into account the potential function, effective mass, and other
     * physical parameters to determine where refinement is needed.
     *
     * @param mesh The mesh
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param threshold The threshold for refinement
     * @return A vector of boolean flags indicating which elements to refine
     */
    static std::vector<bool> computePhysicsBasedRefinementFlags(
        const Mesh& mesh,
        std::function<double(double, double)> m_star,
        std::function<double(double, double)> V,
        double threshold);

    /**
     * @brief Computes the optimal position for a node to improve element quality.
     *
     * This is a helper method for improveQuality that computes the optimal position
     * for a node to improve the quality of its connected elements.
     *
     * @param mesh The mesh
     * @param node_idx The index of the node to optimize
     * @param connected_elements The indices of elements connected to the node
     * @return The optimal position for the node
     */
    static Eigen::Vector2d computeOptimalPosition(const Mesh& mesh, int node_idx, const std::vector<int>& connected_elements);

    /**
     * @brief Performs multi-level refinement.
     *
     * This method performs multi-level refinement by repeatedly refining
     * the mesh based on error indicators. It allows for different refinement
     * levels in different regions of the mesh.
     *
     * @param mesh The mesh to refine
     * @param error_estimator The error estimator to use
     * @param solution The solution vector
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param m_star Function that returns the effective mass at a given position
     * @param V Function that returns the potential at a given position
     * @param max_levels The maximum number of refinement levels
     * @param threshold The threshold for refinement
     * @param allow_hanging_nodes Whether to allow hanging nodes in the mesh (default: false)
     */
    static void refineMultiLevel(
        Mesh& mesh,
        ErrorEstimator& error_estimator,
        const Eigen::VectorXd& solution,
        const Eigen::SparseMatrix<std::complex<double>>& H,
        const Eigen::SparseMatrix<std::complex<double>>& M,
        std::function<double(double, double)> m_star,
        std::function<double(double, double)> V,
        int max_levels,
        double threshold,
        bool allow_hanging_nodes = false);
};