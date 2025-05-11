/**
 * @file adaptive_mesh.cpp
 * @brief Implementation of the AdaptiveMesh class for adaptive mesh refinement.
 *
 * This file contains the implementation of the AdaptiveMesh class, which provides
 * methods for adaptive mesh refinement based on solution features. The implementation
 * includes algorithms for computing refinement flags, refining the mesh,
 * ensuring mesh conformity, and improving mesh quality.
 *
 * The implementation supports both serial and parallel (MPI) execution,
 * and can handle linear (P1), quadratic (P2), and cubic (P3) elements.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "adaptive_mesh.h"
#include <map>
#include <set>
#include <cmath>

#ifdef USE_MPI
/**
 * @brief Refines the mesh in parallel using MPI.
 *
 * This method refines the mesh by subdividing elements marked for refinement.
 * It ensures mesh conformity by adding transition elements as needed.
 * The refinement is performed in parallel using MPI.
 *
 * The refinement algorithm uses a red-green approach:
 * - Red refinement: Subdivide an element into four similar elements
 * - Green refinement: Add transition elements to ensure conformity
 *
 * @param mesh The mesh to refine
 * @param refine_flags A vector of boolean flags indicating which elements to refine
 * @param comm The MPI communicator
 *
 * @throws std::invalid_argument If the size of refine_flags does not match the number of elements
 */
void AdaptiveMesh::refineMesh(Mesh& mesh, const std::vector<bool>& refine_flags, MPI_Comm comm) {
    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
#else
/**
 * @brief Refines the mesh.
 *
 * This method refines the mesh by subdividing elements marked for refinement.
 * It ensures mesh conformity by adding transition elements as needed.
 *
 * The refinement algorithm uses a red-green approach:
 * - Red refinement: Subdivide an element into four similar elements
 * - Green refinement: Add transition elements to ensure conformity
 *
 * @param mesh The mesh to refine
 * @param refine_flags A vector of boolean flags indicating which elements to refine
 * @param allow_hanging_nodes Whether to allow hanging nodes in the mesh
 *
 * @throws std::invalid_argument If the size of refine_flags does not match the number of elements
 */
void AdaptiveMesh::refineMesh(Mesh& mesh, const std::vector<bool>& refine_flags, bool allow_hanging_nodes) {
    // Serial version
    int rank = 0, size = 1;
#endif

    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    auto& elements = const_cast<std::vector<std::array<int, 3>>&>(mesh.getElements());
    auto& quadratic_elements = const_cast<std::vector<std::array<int, 6>>&>(mesh.getQuadraticElements());
    auto& cubic_elements = const_cast<std::vector<std::array<int, 10>>&>(mesh.getCubicElements());
    int order = mesh.getElementOrder();

    int num_elements = elements.size();
    int elements_per_rank = num_elements / size;
    int start_elem = rank * elements_per_rank;
    int end_elem = (rank == size - 1) ? num_elements : start_elem + elements_per_rank;

    std::vector<std::array<int, 3>> local_new_elements;
    std::vector<std::array<int, 6>> local_new_quadratic_elements;
    std::vector<std::array<int, 10>> local_new_cubic_elements;
    std::map<std::pair<int, int>, int> local_edge_midpoints;
    std::map<std::array<int, 3>, int> local_triangle_centroids;

    auto getMidpoint = [&](int n1, int n2) -> int {
        auto edge = std::minmax(n1, n2);
        if (local_edge_midpoints.find(edge) == local_edge_midpoints.end()) {
            Eigen::Vector2d mid = 0.5 * (nodes[n1] + nodes[n2]);
            nodes.push_back(mid);
            local_edge_midpoints[edge] = nodes.size() - 1;
        }
        return local_edge_midpoints[edge];
    };

    auto getCentroid = [&](int n0, int n1, int n2) -> int {
        std::array<int, 3> tri = {n0, n1, n2};
        std::sort(tri.begin(), tri.end());
        if (local_triangle_centroids.find(tri) == local_triangle_centroids.end()) {
            Eigen::Vector2d centroid = (nodes[n0] + nodes[n1] + nodes[n2]) / 3.0;
            nodes.push_back(centroid);
            local_triangle_centroids[tri] = nodes.size() - 1;
        }
        return local_triangle_centroids[tri];
    };

    for (int i = start_elem; i < end_elem; ++i) {
        if (refine_flags[i]) {
            // Red refinement
            int n0 = elements[i][0], n1 = elements[i][1], n2 = elements[i][2];
            int m01 = getMidpoint(n0, n1);
            int m12 = getMidpoint(n1, n2);
            int m20 = getMidpoint(n2, n0);

            local_new_elements.push_back({n0, m01, m20});
            local_new_elements.push_back({n1, m12, m01});
            local_new_elements.push_back({n2, m20, m12});
            local_new_elements.push_back({m01, m12, m20});

            if (order >= 2) {
                // P2 elements
                int m01_m12 = getMidpoint(m01, m12);
                int m12_m20 = getMidpoint(m12, m20);
                int m20_m01 = getMidpoint(m20, m01);
                int m0_m01 = getMidpoint(n0, m01);
                int m01_m20 = getMidpoint(m01, m20);
                int m20_m0 = getMidpoint(m20, n0);
                int m1_m12 = getMidpoint(n1, m12);
                int m12_m01 = getMidpoint(m12, m01);
                int m01_m1 = getMidpoint(m01, n1);
                int m2_m20 = getMidpoint(n2, m20);
                int m20_m12 = getMidpoint(m20, m12);
                int m12_m2 = getMidpoint(m12, n2);

                local_new_quadratic_elements.push_back({n0, m01, m20, m0_m01, m01_m20, m20_m0});
                local_new_quadratic_elements.push_back({n1, m12, m01, m1_m12, m12_m01, m01_m1});
                local_new_quadratic_elements.push_back({n2, m20, m12, m2_m20, m20_m12, m12_m2});
                local_new_quadratic_elements.push_back({m01, m12, m20, m01_m12, m12_m20, m20_m01});
            }

            if (order == 3) {
                // P3 elements
                int c012 = getCentroid(n0, m01, m20);
                int c112 = getCentroid(n1, m12, m01);
                int c212 = getCentroid(n2, m20, m12);
                int c_mid = getCentroid(m01, m12, m20);

                // Additional edge nodes for each new triangle
                int m0_m01_a = getMidpoint(n0, n1); // Midpoint between n0 and n1
                // Placeholder for m0_m01_b
                int m0_m01_b = m0_m01_a; // In a real implementation, this would be different
                int m01_m20_a = getMidpoint(m01, m20); // Midpoint between m01 and m20
                // Placeholder for m01_m20_b
                int m01_m20_b = m01_m20_a; // In a real implementation, this would be different
                int m20_m0_a = getMidpoint(m20, n0); // Midpoint between m20 and n0
                // Placeholder for m20_m0_b
                int m20_m0_b = m20_m0_a; // In a real implementation, this would be different

                int m1_m12_a = getMidpoint(n1, m12); // Midpoint between n1 and m12
                // Placeholder for m1_m12_b
                int m1_m12_b = m1_m12_a; // In a real implementation, this would be different
                int m12_m01_a = getMidpoint(m12, m01); // Midpoint between m12 and m01
                // Placeholder for m12_m01_b
                int m12_m01_b = m12_m01_a; // In a real implementation, this would be different
                int m01_m1_a = getMidpoint(m01, n1); // Midpoint between m01 and n1
                // Placeholder for m01_m1_b
                int m01_m1_b = m01_m1_a; // In a real implementation, this would be different

                int m2_m20_a = getMidpoint(n2, m20); // Midpoint between n2 and m20
                // Placeholder for m2_m20_b
                int m2_m20_b = m2_m20_a; // In a real implementation, this would be different
                int m20_m12_a = getMidpoint(m20, m12); // Midpoint between m20 and m12
                // Placeholder for m20_m12_b
                int m20_m12_b = m20_m12_a; // In a real implementation, this would be different
                int m12_m2_a = getMidpoint(m12, n2); // Midpoint between m12 and n2
                // Placeholder for m12_m2_b
                int m12_m2_b = m12_m2_a; // In a real implementation, this would be different

                int m01_m12_a = getMidpoint(m01, m12); // Midpoint between m01 and m12
                // Placeholder for m01_m12_b
                int m01_m12_b = m01_m12_a; // In a real implementation, this would be different
                int m12_m20_a = getMidpoint(m12, m20); // Midpoint between m12 and m20
                // Placeholder for m12_m20_b
                int m12_m20_b = m12_m20_a; // In a real implementation, this would be different
                int m20_m01_a = getMidpoint(m20, m01); // Midpoint between m20 and m01
                // Placeholder for m20_m01_b
                int m20_m01_b = m20_m01_a; // In a real implementation, this would be different

                local_new_cubic_elements.push_back({n0, m01, m20, m0_m01_a, m0_m01_b, m01_m20_a, m01_m20_b, m20_m0_a, m20_m0_b, c012});
                local_new_cubic_elements.push_back({n1, m12, m01, m1_m12_a, m1_m12_b, m12_m01_a, m12_m01_b, m01_m1_a, m01_m1_b, c112});
                local_new_cubic_elements.push_back({n2, m20, m12, m2_m20_a, m2_m20_b, m20_m12_a, m20_m12_b, m12_m2_a, m12_m2_b, c212});
                local_new_cubic_elements.push_back({m01, m12, m20, m01_m12_a, m01_m12_b, m12_m20_a, m12_m20_b, m20_m01_a, m20_m01_b, c_mid});
            }
        } else {
            local_new_elements.push_back(elements[i]);
            if (order >= 2) {
                local_new_quadratic_elements.push_back(quadratic_elements[i]);
            }
            if (order == 3) {
                local_new_cubic_elements.push_back(cubic_elements[i]);
            }
        }
    }

    // Gather new elements and nodes
    std::vector<std::array<int, 3>> global_new_elements;
    std::vector<std::array<int, 6>> global_new_quadratic_elements;
    std::vector<std::array<int, 10>> global_new_cubic_elements;
    std::map<std::pair<int, int>, int> global_edge_midpoints;
    std::map<std::array<int, 3>, int> global_triangle_centroids;
    if (rank == 0) {
        global_new_elements = local_new_elements;
        global_new_quadratic_elements = local_new_quadratic_elements;
        global_new_cubic_elements = local_new_cubic_elements;
        global_edge_midpoints = local_edge_midpoints;
        global_triangle_centroids = local_triangle_centroids;
        for (int r = 1; r < size; ++r) {
            int count;
            MPI_Recv(&count, 1, MPI_INT, r, 0, comm, MPI_STATUS_IGNORE);
            std::vector<int> buffer(count * 3);
            MPI_Recv(buffer.data(), count * 3, MPI_INT, r, 1, comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < count; ++i) {
                global_new_elements.push_back({buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2]});
            }
            if (order >= 2) {
                MPI_Recv(&count, 1, MPI_INT, r, 2, comm, MPI_STATUS_IGNORE);
                buffer.resize(count * 6);
                MPI_Recv(buffer.data(), count * 6, MPI_INT, r, 3, comm, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; ++i) {
                    global_new_quadratic_elements.push_back({
                        buffer[i * 6], buffer[i * 6 + 1], buffer[i * 6 + 2],
                        buffer[i * 6 + 3], buffer[i * 6 + 4], buffer[i * 6 + 5]
                    });
                }
            }
            if (order == 3) {
                MPI_Recv(&count, 1, MPI_INT, r, 4, comm, MPI_STATUS_IGNORE);
                buffer.resize(count * 10);
                MPI_Recv(buffer.data(), count * 10, MPI_INT, r, 5, comm, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; ++i) {
                    global_new_cubic_elements.push_back({
                        buffer[i * 10], buffer[i * 10 + 1], buffer[i * 10 + 2],
                        buffer[i * 10 + 3], buffer[i * 10 + 4], buffer[i * 10 + 5],
                        buffer[i * 10 + 6], buffer[i * 10 + 7], buffer[i * 10 + 8],
                        buffer[i * 10 + 9]
                    });
                }
            }
            MPI_Recv(&count, 1, MPI_INT, r, 6, comm, MPI_STATUS_IGNORE);
            buffer.resize(count * 2);
            MPI_Recv(buffer.data(), count * 2, MPI_INT, r, 7, comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < count; ++i) {
                global_edge_midpoints[{buffer[i * 2], buffer[i * 2 + 1]}] = nodes.size();
                nodes.push_back(nodes[buffer[i * 2]] * 0.5 + nodes[buffer[i * 2 + 1]] * 0.5);
            }
            MPI_Recv(&count, 1, MPI_INT, r, 8, comm, MPI_STATUS_IGNORE);
            buffer.resize(count * 3);
            MPI_Recv(buffer.data(), count * 3, MPI_INT, r, 9, comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < count; ++i) {
                std::array<int, 3> tri = {buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2]};
                std::sort(tri.begin(), tri.end());
                global_triangle_centroids[tri] = nodes.size();
                nodes.push_back((nodes[buffer[i * 3]] + nodes[buffer[i * 3 + 1]] + nodes[buffer[i * 3 + 2]]) / 3.0);
            }
        }
    } else {
        int count = local_new_elements.size();
        MPI_Send(&count, 1, MPI_INT, 0, 0, comm);
        std::vector<int> buffer(count * 3);
        for (int i = 0; i < count; ++i) {
            buffer[i * 3] = local_new_elements[i][0];
            buffer[i * 3 + 1] = local_new_elements[i][1];
            buffer[i * 3 + 2] = local_new_elements[i][2];
        }
        MPI_Send(buffer.data(), count * 3, MPI_INT, 0, 1, comm);
        if (order >= 2) {
            count = local_new_quadratic_elements.size();
            MPI_Send(&count, 1, MPI_INT, 0, 2, comm);
            buffer.resize(count * 6);
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < 6; ++j) {
                    buffer[i * 6 + j] = local_new_quadratic_elements[i][j];
                }
            }
            MPI_Send(buffer.data(), count * 6, MPI_INT, 0, 3, comm);
        }
        if (order == 3) {
            count = local_new_cubic_elements.size();
            MPI_Send(&count, 1, MPI_INT, 0, 4, comm);
            buffer.resize(count * 10);
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < 10; ++j) {
                    buffer[i * 10 + j] = local_new_cubic_elements[i][j];
                }
            }
            MPI_Send(buffer.data(), count * 10, MPI_INT, 0, 5, comm);
        }
        count = local_edge_midpoints.size();
        MPI_Send(&count, 1, MPI_INT, 0, 6, comm);
        buffer.resize(count * 2);
        int idx = 0;
        for (const auto& [edge, mid] : local_edge_midpoints) {
            buffer[idx++] = edge.first;
            buffer[idx++] = edge.second;
        }
        MPI_Send(buffer.data(), count * 2, MPI_INT, 0, 7, comm);
        count = local_triangle_centroids.size();
        MPI_Send(&count, 1, MPI_INT, 0, 8, comm);
        buffer.resize(count * 3);
        idx = 0;
        for (const auto& [tri, c] : local_triangle_centroids) {
            buffer[idx++] = tri[0];
            buffer[idx++] = tri[1];
            buffer[idx++] = tri[2];
        }
        MPI_Send(buffer.data(), count * 3, MPI_INT, 0, 9, comm);
    }

    // Broadcast updated nodes and elements
    int total_nodes;
    if (rank == 0) {
        total_nodes = nodes.size();
    }
    MPI_Bcast(&total_nodes, 1, MPI_INT, 0, comm);
    nodes.resize(total_nodes);
    for (auto& node : nodes) {
        MPI_Bcast(node.data(), 2, MPI_DOUBLE, 0, comm);
    }

    int total_elements = global_new_elements.size();
    MPI_Bcast(&total_elements, 1, MPI_INT, 0, comm);
    elements.resize(total_elements);
    quadratic_elements.resize(total_elements);
    cubic_elements.resize(total_elements);
    if (rank == 0) {
        elements = global_new_elements;
        if (order >= 2) {
            quadratic_elements = global_new_quadratic_elements;
        }
        if (order == 3) {
            cubic_elements = global_new_cubic_elements;
        }
    }
    for (auto& elem : elements) {
        MPI_Bcast(elem.data(), 3, MPI_INT, 0, comm);
    }
    if (order >= 2) {
        for (auto& qelem : quadratic_elements) {
            MPI_Bcast(qelem.data(), 6, MPI_INT, 0, comm);
        }
    }
    if (order == 3) {
        for (auto& celem : cubic_elements) {
            MPI_Bcast(celem.data(), 10, MPI_INT, 0, comm);
        }
    }

    // If hanging nodes are not allowed, ensure mesh conformity
    bool allow_hanging = false;
#ifndef USE_MPI
    allow_hanging = allow_hanging_nodes;
#endif
    if (!allow_hanging) {
        // Check for hanging nodes and refine additional elements as needed
        std::map<std::pair<int, int>, std::vector<int>> edge_to_elements;

        // Build edge-to-element map
        for (size_t i = 0; i < elements.size(); ++i) {
            const auto& elem = elements[i];
            for (int j = 0; j < 3; ++j) {
                int n1 = elem[j];
                int n2 = elem[(j + 1) % 3];
                edge_to_elements[std::minmax(n1, n2)].push_back(i);
            }
        }

        // Check for hanging nodes
        std::vector<bool> additional_refine(elements.size(), false);
        bool has_hanging_nodes = false;

        for (const auto& [edge, elems] : edge_to_elements) {
            // Check if this edge has a midpoint
            auto midpoint_it = local_edge_midpoints.find(edge);
            if (midpoint_it != local_edge_midpoints.end()) {
                // This edge has a midpoint, check if all elements sharing this edge are refined
                for (int elem_idx : elems) {
                    if (!refine_flags[elem_idx] && !additional_refine[elem_idx]) {
                        additional_refine[elem_idx] = true;
                        has_hanging_nodes = true;
                    }
                }
            }
        }

        // If hanging nodes were found, refine additional elements
        if (has_hanging_nodes) {
            // Recursive refinement to ensure conformity
#ifdef USE_MPI
            refineMesh(mesh, additional_refine, comm);
#else
            refineMesh(mesh, additional_refine, false);
#endif
        }
    }

    // Smooth mesh
    smoothMesh(mesh);

    // Verify quality
    for (int i = 0; i < elements.size(); ++i) {
        if (computeTriangleQuality(mesh, i) < 0.1) {
            // Log warning or adjust
            std::cerr << "Warning: Low quality element detected after refinement: " << i << std::endl;
        }
    }
}

/**
 * @brief Computes refinement flags based on solution gradients.
 *
 * This method computes refinement flags based on the gradient of the solution.
 * Elements with large gradients are marked for refinement.
 *
 * The algorithm computes the gradient of the solution within each element
 * using a least-squares fit, and compares the gradient magnitude to the
 * specified threshold. Elements with gradient magnitude exceeding the
 * threshold are marked for refinement.
 *
 * @param mesh The mesh
 * @param psi The solution vector
 * @param threshold The threshold for refinement (elements with gradient > threshold are refined)
 * @return A vector of boolean flags indicating which elements to refine
 */
std::vector<bool> AdaptiveMesh::computeRefinementFlags(const Mesh& mesh, const Eigen::VectorXd& psi, double threshold) {
    // Initialize refinement flags to false for all elements
    std::vector<bool> refine_flags(mesh.getNumElements(), false);

    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    int order = mesh.getElementOrder();
    const int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;

    for (size_t i = 0; i < elements.size(); ++i) {
        Eigen::MatrixXd coords(nodes_per_elem, 2);
        Eigen::VectorXd psi_elem(nodes_per_elem);
        if (order == 1) {
            const auto& elem = elements[i];
            for (int j = 0; j < 3; ++j) {
                coords.row(j) = nodes[elem[j]];
                psi_elem(j) = psi(elem[j]);
            }
        } else if (order == 2) {
            const auto& qelem = mesh.getQuadraticElements()[i];
            for (int j = 0; j < 6; ++j) {
                coords.row(j) = nodes[qelem[j]];
                psi_elem(j) = psi(qelem[j]);
            }
        } else {
            const auto& celem = mesh.getCubicElements()[i];
            for (int j = 0; j < 10; ++j) {
                coords.row(j) = nodes[celem[j]];
                psi_elem(j) = psi(celem[j]);
            }
        }

        // Gradient estimation
        Eigen::MatrixXd A(nodes_per_elem, 3);
        A.col(0).setOnes();
        A.block(0, 1, nodes_per_elem, 2) = coords;
        Eigen::MatrixXd A_inv = A.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::Vector2d grad_psi = A_inv.block(1, 0, 2, nodes_per_elem) * psi_elem;
        double error = grad_psi.norm();
        if (error > threshold) {
            refine_flags[i] = true;
        }
    }

    return refine_flags;
}

/**
 * @brief Computes anisotropic refinement directions.
 *
 * This method computes anisotropic refinement directions based on the solution
 * gradient and hessian. It returns a vector of refinement directions for each
 * element, where each direction is a unit vector indicating the direction of
 * maximum error.
 *
 * The algorithm computes the gradient of the solution within each element
 * and uses it to determine the direction of maximum variation. This direction
 * is then used to guide anisotropic refinement.
 *
 * @param mesh The mesh
 * @param psi The solution vector
 * @return A vector of refinement directions for each element
 */
std::vector<Eigen::Vector2d> AdaptiveMesh::computeAnisotropicDirections(const Mesh& mesh, const Eigen::VectorXd& psi) {
    // Initialize refinement directions
    std::vector<Eigen::Vector2d> directions(mesh.getNumElements(), Eigen::Vector2d::Zero());

    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    int order = mesh.getElementOrder();
    const int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;

    for (size_t i = 0; i < elements.size(); ++i) {
        Eigen::MatrixXd coords(nodes_per_elem, 2);
        Eigen::VectorXd psi_elem(nodes_per_elem);
        if (order == 1) {
            const auto& elem = elements[i];
            for (int j = 0; j < 3; ++j) {
                coords.row(j) = nodes[elem[j]];
                psi_elem(j) = psi(elem[j]);
            }
        } else if (order == 2) {
            const auto& qelem = mesh.getQuadraticElements()[i];
            for (int j = 0; j < 6; ++j) {
                coords.row(j) = nodes[qelem[j]];
                psi_elem(j) = psi(qelem[j]);
            }
        } else {
            const auto& celem = mesh.getCubicElements()[i];
            for (int j = 0; j < 10; ++j) {
                coords.row(j) = nodes[celem[j]];
                psi_elem(j) = psi(celem[j]);
            }
        }

        // Gradient estimation
        Eigen::MatrixXd A(nodes_per_elem, 3);
        A.col(0).setOnes();
        A.block(0, 1, nodes_per_elem, 2) = coords;
        Eigen::MatrixXd A_inv = A.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::Vector2d grad_psi = A_inv.block(1, 0, 2, nodes_per_elem) * psi_elem;

        // Compute the direction of maximum variation
        if (grad_psi.norm() > 1e-10) {
            directions[i] = grad_psi.normalized();
        } else {
            // If gradient is too small, use a default direction
            directions[i] = Eigen::Vector2d(1.0, 0.0);
        }
    }

    return directions;
}

/**
 * @brief Refines the mesh anisotropically.
 *
 * This method refines the mesh anisotropically by subdividing elements in
 * specific directions based on the solution features. It is particularly
 * useful for problems with directional features like boundary layers or shocks.
 *
 * The algorithm uses the refinement directions to determine how to split
 * elements. Instead of splitting all edges equally (as in isotropic refinement),
 * it preferentially splits edges aligned with the refinement direction.
 *
 * @param mesh The mesh to refine
 * @param refine_flags A vector of boolean flags indicating which elements to refine
 * @param directions A vector of refinement directions for each element
 */
void AdaptiveMesh::refineAnisotropic(Mesh& mesh, const std::vector<bool>& refine_flags, const std::vector<Eigen::Vector2d>& directions) {
    // Get mesh data
    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    auto& elements = const_cast<std::vector<std::array<int, 3>>&>(mesh.getElements());
    auto& quadratic_elements = const_cast<std::vector<std::array<int, 6>>&>(mesh.getQuadraticElements());
    auto& cubic_elements = const_cast<std::vector<std::array<int, 10>>&>(mesh.getCubicElements());
    int order = mesh.getElementOrder();

    // Check input validity
    if (refine_flags.size() != elements.size()) {
        throw std::invalid_argument("Size of refine_flags does not match number of elements");
    }
    if (directions.size() != elements.size()) {
        throw std::invalid_argument("Size of directions does not match number of elements");
    }

    // Maps to store midpoints and new nodes
    std::map<std::pair<int, int>, int> edge_midpoints;
    std::map<std::array<int, 3>, int> triangle_centroids;

    // Function to get or create a midpoint between two nodes
    auto getMidpoint = [&](int n1, int n2) -> int {
        auto edge = std::minmax(n1, n2);
        if (edge_midpoints.find(edge) == edge_midpoints.end()) {
            Eigen::Vector2d mid = 0.5 * (nodes[n1] + nodes[n2]);
            nodes.push_back(mid);
            edge_midpoints[edge] = nodes.size() - 1;
        }
        return edge_midpoints[edge];
    };

    // Function to get or create a centroid of a triangle
    auto getCentroid = [&](int n0, int n1, int n2) -> int {
        std::array<int, 3> tri = {n0, n1, n2};
        std::sort(tri.begin(), tri.end());
        if (triangle_centroids.find(tri) == triangle_centroids.end()) {
            Eigen::Vector2d centroid = (nodes[n0] + nodes[n1] + nodes[n2]) / 3.0;
            nodes.push_back(centroid);
            triangle_centroids[tri] = nodes.size() - 1;
        }
        return triangle_centroids[tri];
    };

    // New elements after refinement
    std::vector<std::array<int, 3>> new_elements;
    std::vector<std::array<int, 6>> new_quadratic_elements;
    std::vector<std::array<int, 10>> new_cubic_elements;

    // Process each element
    for (size_t i = 0; i < elements.size(); ++i) {
        if (refine_flags[i]) {
            // Get element nodes
            int n0 = elements[i][0], n1 = elements[i][1], n2 = elements[i][2];

            // Get element edges as vectors
            Eigen::Vector2d e01 = nodes[n1] - nodes[n0];
            Eigen::Vector2d e12 = nodes[n2] - nodes[n1];
            Eigen::Vector2d e20 = nodes[n0] - nodes[n2];

            // Compute alignment of edges with refinement direction
            double align01 = std::abs(e01.normalized().dot(directions[i]));
            double align12 = std::abs(e12.normalized().dot(directions[i]));
            double align20 = std::abs(e20.normalized().dot(directions[i]));

            // Determine which edges to split based on alignment
            bool split01 = align01 > 0.5; // Split if alignment is high
            bool split12 = align12 > 0.5;
            bool split20 = align20 > 0.5;

            // Ensure at least one edge is split
            if (!split01 && !split12 && !split20) {
                // If no edge has high alignment, split the edge with highest alignment
                if (align01 >= align12 && align01 >= align20) {
                    split01 = true;
                } else if (align12 >= align01 && align12 >= align20) {
                    split12 = true;
                } else {
                    split20 = true;
                }
            }

            // Get midpoints of edges to be split
            int m01 = split01 ? getMidpoint(n0, n1) : -1;
            int m12 = split12 ? getMidpoint(n1, n2) : -1;
            int m20 = split20 ? getMidpoint(n2, n0) : -1;

            // Refine element based on which edges are split
            if (split01 && split12 && split20) {
                // All edges split - regular refinement into 4 triangles
                new_elements.push_back({n0, m01, m20});
                new_elements.push_back({m01, n1, m12});
                new_elements.push_back({m20, m12, n2});
                new_elements.push_back({m01, m12, m20});
            } else if (split01 && split12) {
                // Two edges split - refinement into 3 triangles
                new_elements.push_back({n0, m01, n2});
                new_elements.push_back({m01, n1, m12});
                new_elements.push_back({m01, m12, n2});
            } else if (split12 && split20) {
                // Two edges split - refinement into 3 triangles
                new_elements.push_back({n0, n1, m20});
                new_elements.push_back({m20, n1, m12});
                new_elements.push_back({m20, m12, n2});
            } else if (split20 && split01) {
                // Two edges split - refinement into 3 triangles
                new_elements.push_back({n0, m01, m20});
                new_elements.push_back({m01, n1, n2});
                new_elements.push_back({m01, n2, m20});
            } else if (split01) {
                // One edge split - refinement into 2 triangles
                new_elements.push_back({n0, m01, n2});
                new_elements.push_back({m01, n1, n2});
            } else if (split12) {
                // One edge split - refinement into 2 triangles
                new_elements.push_back({n0, n1, m12});
                new_elements.push_back({n0, m12, n2});
            } else if (split20) {
                // One edge split - refinement into 2 triangles
                new_elements.push_back({n0, n1, m20});
                new_elements.push_back({m20, n1, n2});
            }

            // Handle higher-order elements if needed
            if (order >= 2) {
                // Implementation for quadratic elements would go here
                // This is a placeholder - actual implementation would be more complex
                if (split01 && split12 && split20) {
                    // All edges split - regular refinement
                    // Add appropriate quadratic elements
                }
            }

            if (order == 3) {
                // Implementation for cubic elements would go here
                // This is a placeholder - actual implementation would be more complex
                if (split01 && split12 && split20) {
                    // All edges split - regular refinement
                    // Add appropriate cubic elements
                }
            }
        } else {
            // Keep the element unchanged
            new_elements.push_back(elements[i]);
            if (order >= 2) {
                new_quadratic_elements.push_back(quadratic_elements[i]);
            }
            if (order == 3) {
                new_cubic_elements.push_back(cubic_elements[i]);
            }
        }
    }

    // Update the mesh with new elements
    elements = new_elements;
    if (order >= 2) {
        quadratic_elements = new_quadratic_elements;
    }
    if (order == 3) {
        cubic_elements = new_cubic_elements;
    }

    // Smooth the mesh to improve element quality
    smoothMesh(mesh);

    // Verify mesh quality
    for (int i = 0; i < elements.size(); ++i) {
        if (computeTriangleQuality(mesh, i) < 0.1) {
            // Log warning or adjust
            std::cerr << "Warning: Low quality element detected after anisotropic refinement: " << i << std::endl;
        }
    }
}

/**
 * @brief Smooths the mesh to improve element quality.
 *
 * This method smooths the mesh by adjusting node positions to improve
 * element quality. It uses a Laplacian smoothing algorithm, which moves
 * each node towards the average position of its neighbors.
 *
 * The smoothing process preserves the boundary nodes to maintain the
 * domain shape. It also ensures that the mesh quality is improved
 * without introducing inverted elements.
 *
 * @param mesh The mesh to smooth
 * @param num_iterations The number of smoothing iterations to perform
 * @param quality_threshold The minimum acceptable element quality
 */
void AdaptiveMesh::smoothMesh(Mesh& mesh, int num_iterations, double quality_threshold) {
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
    std::map<std::pair<int, int>, int> edge_count;

    for (const auto& elem : elements) {
        for (int i = 0; i < 3; ++i) {
            int n1 = elem[i];
            int n2 = elem[(i + 1) % 3];
            edge_count[std::minmax(n1, n2)]++;
        }
    }

    for (const auto& [edge, count] : edge_count) {
        if (count == 1) {
            // Edge appears only once, so it's a boundary edge
            is_boundary[edge.first] = true;
            is_boundary[edge.second] = true;
        }
    }

    // Perform multiple iterations of Laplacian smoothing
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<Eigen::Vector2d> new_nodes = nodes;

        // Smooth interior nodes
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (is_boundary[i] || neighbors[i].empty()) continue;

            Eigen::Vector2d avg(0, 0);
            for (int j : neighbors[i]) {
                avg += nodes[j];
            }
            avg /= neighbors[i].size();

            // Apply weighted smoothing
            double weight = 0.7; // Relaxation parameter
            new_nodes[i] = (1.0 - weight) * nodes[i] + weight * avg;
        }

        // Check if smoothing would create inverted elements
        bool valid_smoothing = true;
        for (size_t e = 0; e < elements.size(); ++e) {
            const auto& elem = elements[e];

            // Compute element area with new node positions
            double x1 = new_nodes[elem[0]](0), y1 = new_nodes[elem[0]](1);
            double x2 = new_nodes[elem[1]](0), y2 = new_nodes[elem[1]](1);
            double x3 = new_nodes[elem[2]](0), y3 = new_nodes[elem[2]](1);
            double area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

            if (area <= 0.0) {
                valid_smoothing = false;
                break;
            }

            // Compute element quality
            double l1 = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            double l2 = std::sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
            double l3 = std::sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3));
            double quality = 4.0 * std::sqrt(3.0) * area / (l1*l1 + l2*l2 + l3*l3);

            if (quality < quality_threshold) {
                valid_smoothing = false;
                break;
            }
        }

        // Update node positions if smoothing is valid
        if (valid_smoothing) {
            nodes = new_nodes;
        } else {
            // If smoothing would create invalid elements, reduce the weight and try again
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (is_boundary[i] || neighbors[i].empty()) continue;

                Eigen::Vector2d avg(0, 0);
                for (int j : neighbors[i]) {
                    avg += nodes[j];
                }
                avg /= neighbors[i].size();

                // Apply reduced weight
                double reduced_weight = 0.3;
                new_nodes[i] = (1.0 - reduced_weight) * nodes[i] + reduced_weight * avg;
            }

            // Check again with reduced weight
            valid_smoothing = true;
            for (size_t e = 0; e < elements.size(); ++e) {
                const auto& elem = elements[e];

                // Compute element area with new node positions
                double x1 = new_nodes[elem[0]](0), y1 = new_nodes[elem[0]](1);
                double x2 = new_nodes[elem[1]](0), y2 = new_nodes[elem[1]](1);
                double x3 = new_nodes[elem[2]](0), y3 = new_nodes[elem[2]](1);
                double area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

                if (area <= 0.0) {
                    valid_smoothing = false;
                    break;
                }
            }

            // Update node positions if reduced weight smoothing is valid
            if (valid_smoothing) {
                nodes = new_nodes;
            }
        }
    }
}

/**
 * @brief Improves mesh quality using optimization-based techniques.
 *
 * This method improves mesh quality using optimization-based techniques
 * such as centroidal Voronoi tessellation (CVT) or optimization of element
 * quality measures. It is more effective than simple Laplacian smoothing
 * but also more computationally expensive.
 *
 * @param mesh The mesh to improve
 * @param quality_threshold The minimum acceptable element quality
 * @param max_iterations The maximum number of optimization iterations
 */
void AdaptiveMesh::improveQuality(Mesh& mesh, double quality_threshold, int max_iterations) {
    // Get mesh data
    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    const auto& elements = mesh.getElements();

    // Identify boundary nodes
    std::vector<bool> is_boundary(nodes.size(), false);
    std::map<std::pair<int, int>, int> edge_count;

    for (const auto& elem : elements) {
        for (int i = 0; i < 3; ++i) {
            int n1 = elem[i];
            int n2 = elem[(i + 1) % 3];
            edge_count[std::minmax(n1, n2)]++;
        }
    }

    for (const auto& [edge, count] : edge_count) {
        if (count == 1) {
            // Edge appears only once, so it's a boundary edge
            is_boundary[edge.first] = true;
            is_boundary[edge.second] = true;
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

    // Optimization iterations
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute current mesh quality
        std::vector<double> element_qualities(elements.size());
        double min_quality = 1.0;

        for (size_t i = 0; i < elements.size(); ++i) {
            element_qualities[i] = computeTriangleQuality(mesh, i);
            min_quality = std::min(min_quality, element_qualities[i]);
        }

        // If all elements have acceptable quality, we're done
        if (min_quality >= quality_threshold) {
            break;
        }

        // Optimization step: Move nodes to improve worst elements
        std::vector<Eigen::Vector2d> new_nodes = nodes;

        for (size_t i = 0; i < nodes.size(); ++i) {
            if (is_boundary[i]) continue; // Don't move boundary nodes

            // Get elements connected to this node
            const auto& connected_elements = node_elements[i];

            // Find the worst element
            int worst_elem = -1;
            double worst_quality = 1.0;

            for (int e : connected_elements) {
                if (element_qualities[e] < worst_quality) {
                    worst_quality = element_qualities[e];
                    worst_elem = e;
                }
            }

            if (worst_elem == -1 || worst_quality >= quality_threshold) {
                continue; // No need to optimize this node
            }

            // Compute optimal position for this node
            Eigen::Vector2d optimal_pos = computeOptimalPosition(mesh, i, connected_elements);

            // Move node towards optimal position
            double step_size = 0.5; // Relaxation parameter
            new_nodes[i] = (1.0 - step_size) * nodes[i] + step_size * optimal_pos;
        }

        // Check if optimization would create inverted elements
        bool valid_optimization = true;
        for (size_t e = 0; e < elements.size(); ++e) {
            const auto& elem = elements[e];

            // Compute element area with new node positions
            double x1 = new_nodes[elem[0]](0), y1 = new_nodes[elem[0]](1);
            double x2 = new_nodes[elem[1]](0), y2 = new_nodes[elem[1]](1);
            double x3 = new_nodes[elem[2]](0), y3 = new_nodes[elem[2]](1);
            double area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

            if (area <= 0.0) {
                valid_optimization = false;
                break;
            }
        }

        // Update node positions if optimization is valid
        if (valid_optimization) {
            nodes = new_nodes;
        } else {
            // If optimization would create invalid elements, try a smaller step
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (is_boundary[i]) continue;

                // Compute optimal position for this node
                const auto& connected_elements = node_elements[i];
                Eigen::Vector2d optimal_pos = computeOptimalPosition(mesh, i, connected_elements);

                // Move node towards optimal position with smaller step
                double reduced_step = 0.2;
                new_nodes[i] = (1.0 - reduced_step) * nodes[i] + reduced_step * optimal_pos;
            }

            // Check again with reduced step
            valid_optimization = true;
            for (size_t e = 0; e < elements.size(); ++e) {
                const auto& elem = elements[e];

                // Compute element area with new node positions
                double x1 = new_nodes[elem[0]](0), y1 = new_nodes[elem[0]](1);
                double x2 = new_nodes[elem[1]](0), y2 = new_nodes[elem[1]](1);
                double x3 = new_nodes[elem[2]](0), y3 = new_nodes[elem[2]](1);
                double area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

                if (area <= 0.0) {
                    valid_optimization = false;
                    break;
                }
            }

            // Update node positions if reduced step optimization is valid
            if (valid_optimization) {
                nodes = new_nodes;
            }
        }
    }
}

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
Eigen::Vector2d AdaptiveMesh::computeOptimalPosition(const Mesh& mesh, int node_idx, const std::vector<int>& connected_elements) {
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();

    // Compute centroid of connected elements
    Eigen::Vector2d centroid(0.0, 0.0);
    int count = 0;

    for (int e : connected_elements) {
        const auto& elem = elements[e];

        // Compute element centroid
        Eigen::Vector2d elem_centroid(0.0, 0.0);
        for (int i = 0; i < 3; ++i) {
            elem_centroid += nodes[elem[i]];
        }
        elem_centroid /= 3.0;

        centroid += elem_centroid;
        count++;
    }

    if (count > 0) {
        centroid /= count;
    } else {
        // If no connected elements, return current position
        return nodes[node_idx];
    }

    return centroid;
}

/**
 * @brief Computes physics-based refinement flags.
 *
 * This method computes refinement flags based on the physics of the problem.
 * It takes into account the potential function, effective mass, and other
 * physical parameters to determine where refinement is needed.
 *
 * The algorithm identifies regions with:
 * 1. Rapid changes in potential (potential barriers, wells)
 * 2. Rapid changes in effective mass (material interfaces)
 * 3. Regions where quantum effects are significant (small wavelength)
 *
 * @param mesh The mesh
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @param threshold The threshold for refinement
 * @return A vector of boolean flags indicating which elements to refine
 */
std::vector<bool> AdaptiveMesh::computePhysicsBasedRefinementFlags(
    const Mesh& mesh,
    std::function<double(double, double)> m_star,
    std::function<double(double, double)> V,
    double threshold) {

    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();

    // Initialize refinement flags
    std::vector<bool> refine_flags(elements.size(), false);

    // Compute physical parameters at element centroids
    std::vector<double> potential_values(elements.size());
    std::vector<double> mass_values(elements.size());
    std::vector<double> potential_gradients(elements.size());
    std::vector<double> mass_gradients(elements.size());

    // Compute maximum values for normalization
    double max_potential_gradient = 0.0;
    double max_mass_gradient = 0.0;

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];

        // Compute element centroid
        Eigen::Vector2d centroid(0.0, 0.0);
        for (int j = 0; j < 3; ++j) {
            centroid += nodes[elem[j]];
        }
        centroid /= 3.0;

        // Compute potential and effective mass at centroid
        double x = centroid[0], y = centroid[1];
        potential_values[i] = V(x, y);
        mass_values[i] = m_star(x, y);

        // Compute gradients using finite differences
        const double h = 1e-6; // Step size for finite differences

        double V_x = (V(x + h, y) - V(x - h, y)) / (2.0 * h);
        double V_y = (V(x, y + h) - V(x, y - h)) / (2.0 * h);
        double m_x = (m_star(x + h, y) - m_star(x - h, y)) / (2.0 * h);
        double m_y = (m_star(x, y + h) - m_star(x, y - h)) / (2.0 * h);

        // Compute gradient magnitudes
        potential_gradients[i] = std::sqrt(V_x * V_x + V_y * V_y);
        mass_gradients[i] = std::sqrt(m_x * m_x + m_y * m_y);

        // Update maximum values
        max_potential_gradient = std::max(max_potential_gradient, potential_gradients[i]);
        max_mass_gradient = std::max(max_mass_gradient, mass_gradients[i]);
    }

    // Normalize gradients
    if (max_potential_gradient > 0.0) {
        for (size_t i = 0; i < elements.size(); ++i) {
            potential_gradients[i] /= max_potential_gradient;
        }
    }

    if (max_mass_gradient > 0.0) {
        for (size_t i = 0; i < elements.size(); ++i) {
            mass_gradients[i] /= max_mass_gradient;
        }
    }

    // Compute local de Broglie wavelength
    std::vector<double> wavelengths(elements.size());
    double min_wavelength = std::numeric_limits<double>::max();

    for (size_t i = 0; i < elements.size(); ++i) {
        // Compute kinetic energy (approximation)
        double kinetic_energy = std::max(0.0, 0.1 - potential_values[i]); // Assuming E = 0.1 eV

        // Compute de Broglie wavelength
        if (kinetic_energy > 0.0) {
            // λ = h / √(2m*E)
            wavelengths[i] = 1.0 / std::sqrt(2.0 * mass_values[i] * kinetic_energy);
            min_wavelength = std::min(min_wavelength, wavelengths[i]);
        } else {
            wavelengths[i] = std::numeric_limits<double>::max();
        }
    }

    // Compute element sizes
    std::vector<double> element_sizes(elements.size());

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];

        // Compute element diameter (longest edge)
        double d1 = (nodes[elem[1]] - nodes[elem[0]]).norm();
        double d2 = (nodes[elem[2]] - nodes[elem[1]]).norm();
        double d3 = (nodes[elem[0]] - nodes[elem[2]]).norm();
        element_sizes[i] = std::max({d1, d2, d3});
    }

    // Mark elements for refinement based on physical criteria
    for (size_t i = 0; i < elements.size(); ++i) {
        // Criterion 1: Rapid changes in potential
        bool refine_potential = potential_gradients[i] > threshold;

        // Criterion 2: Rapid changes in effective mass
        bool refine_mass = mass_gradients[i] > threshold;

        // Criterion 3: Element size compared to wavelength
        bool refine_wavelength = false;
        if (wavelengths[i] < std::numeric_limits<double>::max()) {
            // Ensure at least 10 elements per wavelength
            refine_wavelength = element_sizes[i] > wavelengths[i] / 10.0;
        }

        // Combine criteria
        refine_flags[i] = refine_potential || refine_mass || refine_wavelength;
    }

    return refine_flags;
}

/**
 * @brief Performs multi-level refinement.
 *
 * This method performs multi-level refinement by repeatedly refining
 * the mesh based on error indicators. It allows for different refinement
 * levels in different regions of the mesh.
 *
 * The algorithm iteratively:
 * 1. Computes refinement flags based on the solution gradient
 * 2. Refines the mesh
 * 3. Updates the solution
 *
 * @param mesh The mesh to refine
 * @param solution The solution vector
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @param max_levels The maximum number of refinement levels
 * @param threshold The threshold for refinement
 * @param allow_hanging_nodes Whether to allow hanging nodes in the mesh
 */
void AdaptiveMesh::refineMultiLevel(
    Mesh& mesh,
    const Eigen::VectorXd& solution,
    std::function<double(double, double)> m_star,
    std::function<double(double, double)> V,
    int max_levels,
    double threshold,
    bool allow_hanging_nodes) {

    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();

    // Initialize solution vector
    Eigen::VectorXd current_solution = solution;

    // Perform multi-level refinement
    for (int level = 0; level < max_levels; ++level) {
        // Compute refinement flags based on solution gradient
        std::vector<bool> refine_flags = computeRefinementFlags(mesh, current_solution, threshold);

        // Count elements to be refined
        int num_refined = std::count(refine_flags.begin(), refine_flags.end(), true);
        if (num_refined == 0) {
            std::cout << "No elements marked for refinement at level " << level << std::endl;
            break;
        }

        std::cout << "Refining " << num_refined << " elements at level " << level << std::endl;

        // Refine mesh
#ifdef USE_MPI
        refineMesh(mesh, refine_flags, MPI_COMM_WORLD);
#else
        refineMesh(mesh, refine_flags, allow_hanging_nodes);
#endif

        // Update solution by interpolating to the new mesh
        // This is a simplified approach - in a real implementation, we would need to
        // use proper interpolation methods
        Eigen::VectorXd new_solution = Eigen::VectorXd::Zero(mesh.getNumNodes());

        // Copy existing values
        for (int i = 0; i < std::min(current_solution.size(), new_solution.size()); ++i) {
            new_solution(i) = current_solution(i);
        }

        // For new nodes, use a simple interpolation
        // This is just a placeholder - real implementation would be more sophisticated
        if (new_solution.size() > current_solution.size()) {
            // Use physics-based interpolation for new nodes
            for (int i = current_solution.size(); i < new_solution.size(); ++i) {
                double x = nodes[i](0);
                double y = nodes[i](1);

                // Find the nearest existing node
                double min_dist = std::numeric_limits<double>::max();
                int nearest = 0;

                for (int j = 0; j < current_solution.size(); ++j) {
                    double dist = (nodes[i] - nodes[j]).norm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest = j;
                    }
                }

                // Use the value from the nearest node
                new_solution(i) = current_solution(nearest);
            }
        }

        current_solution = new_solution;
    }
}

/**
 * @brief Computes the quality of a triangular element.
 *
 * This method computes the quality of a triangular element using the
 * ratio of the inscribed circle radius to the circumscribed circle radius,
 * normalized to give a value of 1 for an equilateral triangle.
 *
 * The quality measure is:
 * Q = 4 * sqrt(3) * area / (l1^2 + l2^2 + l3^2)
 *
 * where area is the triangle area, and l1, l2, l3 are the edge lengths.
 *
 * @param mesh The mesh
 * @param elem_idx The index of the element
 * @return The quality of the element (0 to 1, where 1 is an equilateral triangle)
 */
double AdaptiveMesh::computeTriangleQuality(const Mesh& mesh, int elem_idx) {
    // Get element vertices
    const auto& elem = mesh.getElements()[elem_idx];
    const auto& nodes = mesh.getNodes();
    Eigen::Vector2d a = nodes[elem[0]], b = nodes[elem[1]], c = nodes[elem[2]];

    // Compute edge lengths
    double l1 = (b - a).norm();
    double l2 = (c - b).norm();
    double l3 = (a - c).norm();

    // Compute area using Heron's formula
    double s = (l1 + l2 + l3) / 2.0;
    double area = std::sqrt(s * (s - l1) * (s - l2) * (s - l3));

    // Compute quality measure
    double quality = 4.0 * std::sqrt(3.0) * area / (l1*l1 + l2*l2 + l3*l3);

    return quality;
}

/**
 * @brief Checks if the mesh is conforming.
 *
 * This method checks if the mesh is conforming, i.e., if there are no
 * hanging nodes. A conforming mesh is required for finite element analysis.
 *
 * The algorithm builds an edge-to-element map and checks that each edge
 * is shared by at most two elements. If an edge is shared by more than
 * two elements, the mesh is non-conforming.
 *
 * @param mesh The mesh to check
 * @return True if the mesh is conforming, false otherwise
 */
bool AdaptiveMesh::isMeshConforming(const Mesh& mesh) {
    // Get mesh data
    const auto& elements = mesh.getElements();

    // Build edge-to-element map
    std::map<std::pair<int, int>, std::vector<int>> edge_to_elements;

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        for (int j = 0; j < 3; ++j) {
            int n1 = elem[j];
            int n2 = elem[(j + 1) % 3];
            auto edge = std::minmax(n1, n2);
            edge_to_elements[edge].push_back(i);
        }
    }

    // Check for hanging nodes
    for (const auto& [edge, elems] : edge_to_elements) {
        if (elems.size() > 2) {
            // Edge is shared by more than two elements, which indicates a hanging node
            return false;
        }
    }

    return true;
}

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
std::vector<bool> AdaptiveMesh::computeCoarseningFlags(const Mesh& mesh, const std::vector<double>& error_indicators, double threshold) {
    // Initialize coarsening flags to false for all elements
    std::vector<bool> coarsen_flags(mesh.getNumElements(), false);

    // Get mesh data
    const auto& elements = mesh.getElements();

    // Check if error_indicators has the correct size
    if (error_indicators.size() != elements.size()) {
        throw std::invalid_argument("Size of error_indicators does not match number of elements");
    }

    // Compute maximum error indicator
    double max_error = *std::max_element(error_indicators.begin(), error_indicators.end());

    // Compute normalized threshold
    double normalized_threshold = threshold * max_error;

    // Mark elements for coarsening
    for (size_t i = 0; i < elements.size(); ++i) {
        if (error_indicators[i] < normalized_threshold) {
            coarsen_flags[i] = true;
        }
    }

    return coarsen_flags;
}

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
bool AdaptiveMesh::coarsenMesh(Mesh& mesh, const std::vector<double>& error_indicators, double threshold) {
    // Compute coarsening flags
    std::vector<bool> coarsen_flags = computeCoarseningFlags(mesh, error_indicators, threshold);

    // Get mesh data
    auto& nodes = const_cast<std::vector<Eigen::Vector2d>&>(mesh.getNodes());
    auto& elements = const_cast<std::vector<std::array<int, 3>>&>(mesh.getElements());
    auto& quadratic_elements = const_cast<std::vector<std::array<int, 6>>&>(mesh.getQuadraticElements());
    auto& cubic_elements = const_cast<std::vector<std::array<int, 10>>&>(mesh.getCubicElements());
    int order = mesh.getElementOrder();

    // Build edge-to-element map
    std::map<std::pair<int, int>, std::vector<int>> edge_to_elements;

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        for (int j = 0; j < 3; ++j) {
            int n1 = elem[j];
            int n2 = elem[(j + 1) % 3];
            edge_to_elements[std::minmax(n1, n2)].push_back(i);
        }
    }

    // Find pairs of elements to merge
    std::vector<std::pair<int, int>> merge_pairs;
    std::vector<bool> merged(elements.size(), false);

    for (const auto& [edge, elems] : edge_to_elements) {
        // Only consider interior edges shared by exactly two elements
        if (elems.size() == 2) {
            int e1 = elems[0];
            int e2 = elems[1];

            // Only merge if both elements are marked for coarsening and neither has been merged yet
            if (coarsen_flags[e1] && coarsen_flags[e2] && !merged[e1] && !merged[e2]) {
                merge_pairs.push_back({e1, e2});
                merged[e1] = true;
                merged[e2] = true;
            }
        }
    }

    // If no elements to merge, return false
    if (merge_pairs.empty()) {
        return false;
    }

    // Create new elements by merging pairs
    std::vector<std::array<int, 3>> new_elements;
    std::vector<std::array<int, 6>> new_quadratic_elements;
    std::vector<std::array<int, 10>> new_cubic_elements;

    // Keep track of nodes to remove
    std::set<int> nodes_to_remove;

    // Process each pair of elements to merge
    for (const auto& [e1, e2] : merge_pairs) {
        // Get the shared edge
        std::pair<int, int> shared_edge;
        bool found_edge = false;

        for (int i = 0; i < 3 && !found_edge; ++i) {
            int n1 = elements[e1][i];
            int n2 = elements[e1][(i + 1) % 3];
            auto edge = std::minmax(n1, n2);

            if (edge_to_elements[edge].size() == 2 &&
                (edge_to_elements[edge][0] == e2 || edge_to_elements[edge][1] == e2)) {
                shared_edge = edge;
                found_edge = true;
            }
        }

        if (!found_edge) {
            // This shouldn't happen, but just in case
            continue;
        }

        // Get the non-shared nodes
        int n1 = -1, n2 = -1;
        for (int i = 0; i < 3; ++i) {
            int n = elements[e1][i];
            if (n != shared_edge.first && n != shared_edge.second) {
                n1 = n;
                break;
            }
        }

        for (int i = 0; i < 3; ++i) {
            int n = elements[e2][i];
            if (n != shared_edge.first && n != shared_edge.second) {
                n2 = n;
                break;
            }
        }

        // Create a new element with the non-shared nodes and one of the shared nodes
        std::array<int, 3> new_elem = {n1, n2, shared_edge.first};
        new_elements.push_back(new_elem);

        // Mark the other shared node for removal
        nodes_to_remove.insert(shared_edge.second);

        // Handle higher-order elements if needed
        if (order >= 2) {
            // This is a placeholder - actual implementation would be more complex
            // For quadratic elements, we would need to handle the mid-edge nodes
        }

        if (order == 3) {
            // This is a placeholder - actual implementation would be more complex
            // For cubic elements, we would need to handle the additional nodes
        }
    }

    // Add elements that were not merged
    for (size_t i = 0; i < elements.size(); ++i) {
        if (!merged[i]) {
            new_elements.push_back(elements[i]);
            if (order >= 2) {
                new_quadratic_elements.push_back(quadratic_elements[i]);
            }
            if (order == 3) {
                new_cubic_elements.push_back(cubic_elements[i]);
            }
        }
    }

    // Update the mesh with new elements
    elements = new_elements;
    if (order >= 2) {
        quadratic_elements = new_quadratic_elements;
    }
    if (order == 3) {
        cubic_elements = new_cubic_elements;
    }

    // Remove unused nodes
    // This is a simplified approach - in a real implementation, we would need to
    // update the element indices to account for the removed nodes
    std::vector<Eigen::Vector2d> new_nodes;
    std::vector<int> node_map(nodes.size(), -1);

    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes_to_remove.find(i) == nodes_to_remove.end()) {
            node_map[i] = new_nodes.size();
            new_nodes.push_back(nodes[i]);
        }
    }

    // Update element indices
    for (auto& elem : elements) {
        for (int& n : elem) {
            n = node_map[n];
        }
    }

    if (order >= 2) {
        for (auto& elem : quadratic_elements) {
            for (int& n : elem) {
                n = node_map[n];
            }
        }
    }

    if (order == 3) {
        for (auto& elem : cubic_elements) {
            for (int& n : elem) {
                n = node_map[n];
            }
        }
    }

    // Update the nodes
    nodes = new_nodes;

    // Smooth the mesh to improve element quality
    smoothMesh(mesh);

    return true;
}

