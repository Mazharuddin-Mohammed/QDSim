#include "mesh.h"
#include "adaptive_mesh.h"
#include <fstream>

Mesh::Mesh(double Lx, double Ly, int nx, int ny, int element_order) : element_order(element_order) {
    if (element_order < 1 || element_order > 3) {
        throw std::invalid_argument("Element order must be 1 (linear), 2 (quadratic), or 3 (cubic)");
    }
    generateTriangularMesh(Lx, Ly, nx, ny);
}

void Mesh::generateTriangularMesh(double Lx, double Ly, int nx, int ny) {
    double dx = Lx / nx, dy = Ly / ny;
    nodes.clear();
    elements.clear();
    quadratic_elements.clear();
    cubic_elements.clear();

    // Generate vertex nodes
    for (int i = 0; i <= ny; ++i) {
        for (int j = 0; j <= nx; ++j) {
            nodes.emplace_back(j * dx, i * dy);
        }
    }

    // Generate elements and higher-order nodes
    std::map<std::pair<int, int>, int> edge_midpoints;
    std::map<std::array<int, 3>, int> triangle_centroids; // For P3
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            int n0 = i * (nx + 1) + j;
            int n1 = n0 + 1;
            int n2 = n0 + nx + 1;
            int n3 = n2 + 1;
            elements.push_back({n0, n1, n2});
            elements.push_back({n1, n3, n2});

            auto getMidpoint = [&](int n1, int n2) -> int {
                auto edge = std::minmax(n1, n2);
                if (edge_midpoints.find(edge) == edge_midpoints.end()) {
                    Eigen::Vector2d mid = 0.5 * (nodes[n1] + nodes[n2]);
                    nodes.push_back(mid);
                    edge_midpoints[edge] = nodes.size() - 1;
                }
                return edge_midpoints[edge];
            };

            if (element_order >= 2) {
                // P2 elements
                int m01 = getMidpoint(n0, n1);
                int m12 = getMidpoint(n1, n2);
                int m20 = getMidpoint(n2, n0);
                int m23 = getMidpoint(n2, n3);
                int m13 = getMidpoint(n1, n3);
                int m03 = getMidpoint(n0, n3);
                quadratic_elements.push_back({n0, n1, n2, m01, m12, m20});
                quadratic_elements.push_back({n1, n3, n2, m13, m23, m12});
            }

            if (element_order == 3) {
                // P3 elements: additional edge nodes and centroid
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

                int m01_a = getMidpoint(n0, n1); // Split edge n0-n1
                // Placeholder for m01_b
                int m01_b = m01_a; // In a real implementation, this would be different
                int m12_a = getMidpoint(n1, n2); // Split edge n1-n2
                // Placeholder for m12_b
                int m12_b = m12_a; // In a real implementation, this would be different
                int m20_a = getMidpoint(n2, n0); // Split edge n2-n0
                // Placeholder for m20_b
                int m20_b = m20_a; // In a real implementation, this would be different
                int c012 = getCentroid(n0, n1, n2);

                int m13_a = getMidpoint(n1, n3); // Split edge n1-n3
                // Placeholder for m13_b
                int m13_b = m13_a; // In a real implementation, this would be different
                int m23_a = getMidpoint(n2, n3); // Split edge n2-n3
                // Placeholder for m23_b
                int m23_b = m23_a; // In a real implementation, this would be different
                int m03_a = getMidpoint(n0, n3); // Split edge n0-n3
                // Placeholder for m03_b
                int m03_b = m03_a; // In a real implementation, this would be different
                int c132 = getCentroid(n1, n3, n2);

                cubic_elements.push_back({n0, n1, n2, m01_a, m01_b, m12_a, m12_b, m20_a, m20_b, c012});
                cubic_elements.push_back({n1, n3, n2, m13_a, m13_b, m23_a, m23_b, m12_a, m12_b, c132});
            }
        }
    }
}

void Mesh::refine(const std::vector<bool>& refine_flags) {
    AdaptiveMesh::refineMesh(*this, refine_flags);
}

#ifdef USE_MPI
void Mesh::refine(const std::vector<bool>& refine_flags, MPI_Comm comm) {
    // MPI version of mesh refinement
    // For now, just call the serial version
    refine(refine_flags);

    // In a real implementation, we would use MPI to parallelize the refinement
    // and synchronize the mesh across processes
}
#endif

void Mesh::save(const std::string& filename) const {
    std::ofstream out(filename);
    out << nodes.size() << " " << elements.size() << " " << element_order << "\n";
    for (const auto& node : nodes) {
        out << node(0) << " " << node(1) << "\n";
    }
    for (const auto& elem : elements) {
        out << elem[0] << " " << elem[1] << " " << elem[2] << "\n";
    }
    if (element_order >= 2) {
        for (const auto& qelem : quadratic_elements) {
            out << qelem[0] << " " << qelem[1] << " " << qelem[2] << " "
                << qelem[3] << " " << qelem[4] << " " << qelem[5] << "\n";
        }
    }
    if (element_order == 3) {
        for (const auto& celem : cubic_elements) {
            out << celem[0] << " " << celem[1] << " " << celem[2] << " "
                << celem[3] << " " << celem[4] << " " << celem[5] << " "
                << celem[6] << " " << celem[7] << " " << celem[8] << " "
                << celem[9] << "\n";
        }
    }
}

Mesh Mesh::load(const std::string& filename) {
    std::ifstream in(filename);
    size_t num_nodes, num_elements;
    int element_order;
    in >> num_nodes >> num_elements >> element_order;
    Mesh mesh(0, 0, 0, 0, element_order);
    mesh.nodes.resize(num_nodes);
    mesh.elements.resize(num_elements);
    for (auto& node : mesh.nodes) {
        in >> node(0) >> node(1);
    }
    for (auto& elem : mesh.elements) {
        in >> elem[0] >> elem[1] >> elem[2];
    }
    if (element_order >= 2) {
        mesh.quadratic_elements.resize(num_elements);
        for (auto& qelem : mesh.quadratic_elements) {
            in >> qelem[0] >> qelem[1] >> qelem[2] >> qelem[3] >> qelem[4] >> qelem[5];
        }
    }
    if (element_order == 3) {
        mesh.cubic_elements.resize(num_elements);
        for (auto& celem : mesh.cubic_elements) {
            in >> celem[0] >> celem[1] >> celem[2] >> celem[3] >> celem[4] >> celem[5]
               >> celem[6] >> celem[7] >> celem[8] >> celem[9];
        }
    }
    return mesh;
}