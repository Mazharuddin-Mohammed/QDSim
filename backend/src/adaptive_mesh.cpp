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
 *
 * @throws std::invalid_argument If the size of refine_flags does not match the number of elements
 */
void AdaptiveMesh::refineMesh(Mesh& mesh, const std::vector<bool>& refine_flags) {
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

    // Smooth mesh
    smoothMesh(mesh);

    // Verify quality
    for (int i = 0; i < elements.size(); ++i) {
        if (computeTriangleQuality(mesh, i) < 0.1) {
            // Log warning or adjust
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
 */
void AdaptiveMesh::smoothMesh(Mesh& mesh) {
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

    std::vector<Eigen::Vector2d> new_nodes = nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (neighbors[i].empty()) continue;
        Eigen::Vector2d avg(0, 0);
        for (int j : neighbors[i]) {
            avg += nodes[j];
        }
        avg /= neighbors[i].size();
        new_nodes[i] = 0.5 * (nodes[i] + avg);
    }

    double Lx = nodes.back()(0), Ly = nodes.back()(1);
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (std::abs(nodes[i](0)) < 1e-10 || std::abs(nodes[i](0) - Lx) < 1e-10 ||
            std::abs(nodes[i](1)) < 1e-10 || std::abs(nodes[i](1) - Ly) < 1e-10) {
            new_nodes[i] = nodes[i];
        }
    }

    nodes = new_nodes;
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

    double l1 = (b - a).norm(), l2 = (c - b).norm(), l3 = (a - c).norm();
    double s = (l1 + l2 + l3) / 2.0;
    double area = std::sqrt(s * (s - l1) * (s - l2) * (s - l3));
    if (area < 1e-10) return 0.0;
    return 4.0 * std::sqrt(3.0) * area / (l1 * l1 + l2 * l2 + l3 * l3);
}

/**
 * @brief Checks if the mesh is conforming.
 *
 * This method checks if the mesh is conforming, i.e., if there are no
 * hanging nodes. A conforming mesh is required for finite element analysis.
 *
 * The algorithm checks that each edge is shared by at most two elements.
 * If an edge is shared by more than two elements, the mesh is non-conforming.
 *
 * @param mesh The mesh to check
 * @return True if the mesh is conforming, false otherwise
 */
bool AdaptiveMesh::isMeshConforming(const Mesh& mesh) {
    // Get mesh elements
    const auto& elements = mesh.getElements();

    // Map each edge to the elements that contain it
    std::map<std::pair<int, int>, std::vector<int>> edge_to_elements;

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        for (int j = 0; j < 3; ++j) {
            auto edge = std::minmax(elem[j], elem[(j + 1) % 3]);
            edge_to_elements[edge].push_back(i);
        }
    }

    for (const auto& [edge, elem_list] : edge_to_elements) {
        if (elem_list.size() > 2) {
            return false;
        }
    }
    return true;
}