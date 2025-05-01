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
    static void refineMesh(Mesh& mesh, const std::vector<bool>& refine_flags);
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
     * @brief Smooths the mesh to improve element quality.
     *
     * This method smooths the mesh by adjusting node positions to improve
     * element quality. It uses a Laplacian smoothing algorithm.
     *
     * @param mesh The mesh to smooth
     */
    static void smoothMesh(Mesh& mesh);

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
};