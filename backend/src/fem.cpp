#include "mesh.h"
#include "fem.h"
#include "adaptive_mesh.h"
#include <Eigen/Dense>
#include <vector>
#include <filesystem>

FEMSolver::FEMSolver(Mesh& mesh, double (*m_star)(double, double), double (*V)(double, double),
                     double (*cap)(double, double), PoissonSolver& poisson, int order, bool use_mpi)
    : mesh(mesh), poisson(poisson), m_star(m_star), V(V), cap(cap), order(order), use_mpi(use_mpi) {
    H.resize(mesh.getNumNodes(), mesh.getNumNodes());
    M.resize(mesh.getNumNodes(), mesh.getNumNodes());
}

void FEMSolver::assemble_matrices() {
    // Resize matrices to match the current mesh
    H.resize(mesh.getNumNodes(), mesh.getNumNodes());
    M.resize(mesh.getNumNodes(), mesh.getNumNodes());

    if (use_mpi) {
#ifdef USE_MPI
        assemble_matrices_parallel();
#else
        // Fallback to serial if MPI is disabled at compile time
        assemble_matrices_serial();
#endif
    } else {
        assemble_matrices_serial();
    }
}

void FEMSolver::assemble_matrices_serial() {
    std::vector<Eigen::Triplet<std::complex<double>>> H_triplets, M_triplets;
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        Eigen::MatrixXcd H_e, M_e;
        assemble_element_matrix(e, H_e, M_e);

        // Get the element nodes based on element order
        std::vector<int> element;
        if (order == 1) {
            element.assign(mesh.getElements()[e].begin(), mesh.getElements()[e].end());
        } else if (order == 2) {
            element.assign(mesh.getQuadraticElements()[e].begin(), mesh.getQuadraticElements()[e].end());
        } else { // order == 3
            element.assign(mesh.getCubicElements()[e].begin(), mesh.getCubicElements()[e].end());
        }

        for (int i = 0; i < H_e.rows(); ++i) {
            for (int j = 0; j < H_e.cols(); ++j) {
                H_triplets.emplace_back(element[i], element[j], H_e(i, j));
                M_triplets.emplace_back(element[i], element[j], M_e(i, j));
            }
        }
    }
    H.setFromTriplets(H_triplets.begin(), H_triplets.end());
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());
}

#ifdef USE_MPI
void FEMSolver::assemble_matrices_parallel() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_elements = mesh.getNumElements();
    int elements_per_rank = num_elements / size;
    int start_elem = rank * elements_per_rank;
    int end_elem = (rank == size - 1) ? num_elements : start_elem + elements_per_rank;

    std::vector<Eigen::Triplet<std::complex<double>>> H_triplets, M_triplets;
    for (int e = start_elem; e < end_elem; ++e) {
        Eigen::MatrixXcd H_e, M_e;
        assemble_element_matrix(e, H_e, M_e);

        // Get the element nodes based on element order
        std::vector<int> element;
        if (order == 1) {
            element.assign(mesh.getElements()[e].begin(), mesh.getElements()[e].end());
        } else if (order == 2) {
            element.assign(mesh.getQuadraticElements()[e].begin(), mesh.getQuadraticElements()[e].end());
        } else { // order == 3
            element.assign(mesh.getCubicElements()[e].begin(), mesh.getCubicElements()[e].end());
        }

        for (int i = 0; i < H_e.rows(); ++i) {
            for (int j = 0; j < H_e.cols(); ++j) {
                H_triplets.emplace_back(element[i], element[j], H_e(i, j));
                M_triplets.emplace_back(element[i], element[j], M_e(i, j));
            }
        }
    }

    // Gather all triplets to rank 0
    std::vector<Eigen::Triplet<std::complex<double>>> global_H_triplets, global_M_triplets;
    if (rank == 0) {
        global_H_triplets = H_triplets;
        global_M_triplets = M_triplets;

        for (int r = 1; r < size; ++r) {
            // Receive H triplets
            int h_count;
            MPI_Recv(&h_count, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < h_count; ++i) {
                int row, col;
                double real, imag;
                MPI_Recv(&row, 1, MPI_INT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&col, 1, MPI_INT, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&real, 1, MPI_DOUBLE, r, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&imag, 1, MPI_DOUBLE, r, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                global_H_triplets.emplace_back(row, col, std::complex<double>(real, imag));
            }

            // Receive M triplets
            int m_count;
            MPI_Recv(&m_count, 1, MPI_INT, r, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < m_count; ++i) {
                int row, col;
                double real, imag;
                MPI_Recv(&row, 1, MPI_INT, r, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&col, 1, MPI_INT, r, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&real, 1, MPI_DOUBLE, r, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&imag, 1, MPI_DOUBLE, r, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                global_M_triplets.emplace_back(row, col, std::complex<double>(real, imag));
            }
        }

        H.setFromTriplets(global_H_triplets.begin(), global_H_triplets.end());
        M.setFromTriplets(global_M_triplets.begin(), global_M_triplets.end());
    } else {
        // Send H triplets
        int h_count = H_triplets.size();
        MPI_Send(&h_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        for (const auto& triplet : H_triplets) {
            int row = triplet.row();
            int col = triplet.col();
            double real = triplet.value().real();
            double imag = triplet.value().imag();
            MPI_Send(&row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&col, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&real, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
            MPI_Send(&imag, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        }

        // Send M triplets
        int m_count = M_triplets.size();
        MPI_Send(&m_count, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);

        for (const auto& triplet : M_triplets) {
            int row = triplet.row();
            int col = triplet.col();
            double real = triplet.value().real();
            double imag = triplet.value().imag();
            MPI_Send(&row, 1, MPI_INT, 0, 6, MPI_COMM_WORLD);
            MPI_Send(&col, 1, MPI_INT, 0, 7, MPI_COMM_WORLD);
            MPI_Send(&real, 1, MPI_DOUBLE, 0, 8, MPI_COMM_WORLD);
            MPI_Send(&imag, 1, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD);
        }
    }

    // Broadcast matrices to all ranks
    // Note: In a real implementation, we would need to broadcast the actual matrix data
    // For now, we'll just synchronize the number of non-zeros
    if (rank == 0) {
        int nnz_H = H.nonZeros();
        int nnz_M = M.nonZeros();
        MPI_Bcast(&nnz_H, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nnz_M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int nnz_H = 0;
        int nnz_M = 0;
        MPI_Bcast(&nnz_H, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nnz_M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
}
#endif





void FEMSolver::adapt_mesh(const Eigen::VectorXd& eigenvector, double threshold, const std::string& cache_dir) {
    std::string cache_file = cache_dir.empty() ? "" : cache_dir + "/mesh_" + std::to_string(eigenvector.norm()) + ".txt";
    if (!cache_file.empty() && std::filesystem::exists(cache_file)) {
        mesh = Mesh::load(cache_file);
        assemble_matrices();
        return;
    }

    std::vector<bool> refine_flags = AdaptiveMesh::computeRefinementFlags(mesh, eigenvector, threshold);

    // Use MPI for mesh refinement if enabled
    if (use_mpi) {
#ifdef USE_MPI
        mesh.refine(refine_flags, MPI_COMM_WORLD);
#else
        mesh.refine(refine_flags);
#endif
    } else {
        mesh.refine(refine_flags);
    }

    if (!cache_file.empty()) {
        std::filesystem::create_directories(cache_dir);
        mesh.save(cache_file);
    }

    assemble_matrices();
}



void FEMSolver::assemble_element_matrix(size_t e, Eigen::MatrixXcd& H_e, Eigen::MatrixXcd& M_e) {
    // Simplified; use existing P1/P2/P3 quadrature
    H_e.setZero(order == 1 ? 3 : (order == 2 ? 6 : 10), order == 1 ? 3 : (order == 2 ? 6 : 10));
    M_e.setZero(H_e.rows(), H_e.cols());

    // Get element nodes
    std::vector<Eigen::Vector2d> element_nodes;
    if (order == 1) {
        const auto& elem = mesh.getElements()[e];
        for (int i = 0; i < 3; ++i) {
            element_nodes.push_back(mesh.getNodes()[elem[i]]);
        }
    } else if (order == 2) {
        const auto& elem = mesh.getQuadraticElements()[e];
        for (int i = 0; i < 6; ++i) {
            element_nodes.push_back(mesh.getNodes()[elem[i]]);
        }
    } else { // order == 3
        const auto& elem = mesh.getCubicElements()[e];
        for (int i = 0; i < 10; ++i) {
            element_nodes.push_back(mesh.getNodes()[elem[i]]);
        }
    }

    // Number of quadrature points based on element order
    int num_quad_points = (order == 1) ? 3 : (order == 2) ? 7 : 12;

    for (int q = 0; q < num_quad_points; ++q) {
        // For simplicity, we'll just use the centroid of the element as the quadrature point
        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const auto& node : element_nodes) {
            centroid += node;
        }
        centroid /= element_nodes.size();

        double x = centroid.x();
        double y = centroid.y();

        double m = m_star(x, y);
        double V_val = V(x, y);
        double eta = cap(x, y);

        // Compute basis functions and gradients (simplified)
        for (int i = 0; i < H_e.rows(); ++i) {
            for (int j = 0; j < H_e.cols(); ++j) {
                // Simplified matrix assembly - in a real implementation, we would compute the actual basis functions
                H_e(i, j) += (1.054e-34 * 1.054e-34 / (2 * m)) * 1.0 + (V_val + std::complex<double>(0, eta)) * 1.0;
                M_e(i, j) += 1.0; // Basis function product
            }
        }
    }
}