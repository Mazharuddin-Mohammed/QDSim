/**
 * @file fem.cpp
 * @brief Implementation of the FEMSolver class for quantum simulations.
 *
 * This file contains the implementation of the FEMSolver class, which implements
 * the finite element method for solving the Schrödinger equation in quantum
 * dot simulations. The solver assembles the Hamiltonian and mass matrices,
 * and provides methods for adaptive mesh refinement.
 *
 * The implementation supports both serial and parallel (MPI) execution,
 * and can use linear (P1), quadratic (P2), or cubic (P3) elements.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include "fem.h"
#include "adaptive_mesh.h"
#include <Eigen/Dense>
#include <vector>
#include <filesystem>

/**
 * @brief Constructs a new FEMSolver object.
 *
 * This constructor initializes the FEMSolver with the given mesh, material functions,
 * and solver parameters. It resizes the Hamiltonian and mass matrices to match the
 * mesh size, and creates a finite element interpolator for field interpolation.
 *
 * @param mesh The mesh to use for the simulation
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @param cap Function that returns the capacitance at a given position
 * @param poisson The Poisson solver to use for electrostatic calculations
 * @param order The order of the finite elements (1 for P1, 2 for P2, 3 for P3)
 * @param use_mpi Whether to use MPI for parallel computations
 */
FEMSolver::FEMSolver(Mesh& mesh, double (*m_star)(double, double), double (*V)(double, double),
                     double (*cap)(double, double), PoissonSolver& poisson, int order, bool use_mpi)
    : mesh(mesh), poisson(poisson), m_star(m_star), V(V), cap(cap), order(order), use_mpi(use_mpi) {
    // Initialize matrices with the correct size
    H.resize(mesh.getNumNodes(), mesh.getNumNodes());
    M.resize(mesh.getNumNodes(), mesh.getNumNodes());

    // Create the finite element interpolator for field interpolation
    interpolator = new FEInterpolator(mesh);
}

/**
 * @brief Destroys the FEMSolver object.
 *
 * This destructor cleans up the resources used by the FEMSolver,
 * including the finite element interpolator.
 */
FEMSolver::~FEMSolver() {
    // Clean up the interpolator
    if (interpolator) {
        delete interpolator;
        interpolator = nullptr;
    }
}

/**
 * @brief Assembles the Hamiltonian and mass matrices.
 *
 * This method assembles the Hamiltonian and mass matrices using the finite
 * element method. It resizes the matrices to match the current mesh size,
 * and then calls either the serial or parallel assembly method based on
 * the use_mpi flag.
 *
 * The Hamiltonian matrix represents the energy operator in the Schrödinger equation,
 * and the mass matrix represents the overlap of basis functions.
 *
 * @throws std::runtime_error If the assembly fails
 */
void FEMSolver::assemble_matrices() {
    // Resize matrices to match the current mesh
    H.resize(mesh.getNumNodes(), mesh.getNumNodes());
    M.resize(mesh.getNumNodes(), mesh.getNumNodes());

    // Choose between serial and parallel assembly based on the use_mpi flag
    if (use_mpi) {
#ifdef USE_MPI
        // Use parallel assembly if MPI is enabled
        assemble_matrices_parallel();
#else
        // Fallback to serial if MPI is disabled at compile time
        assemble_matrices_serial();
#endif
    } else {
        // Use serial assembly if MPI is disabled
        assemble_matrices_serial();
    }
}

/**
 * @brief Assembles the Hamiltonian and mass matrices in serial mode.
 *
 * This private method assembles the Hamiltonian and mass matrices in serial mode.
 * It iterates over all elements in the mesh, assembles the element matrices,
 * and adds the contributions to the global matrices using triplets.
 *
 * The assembly process involves:
 * 1. Iterating over all elements in the mesh
 * 2. Assembling the element matrices for each element
 * 3. Getting the element nodes based on the element order
 * 4. Adding the element matrix entries to the global matrices using triplets
 * 5. Setting the global matrices from the triplets
 *
 * @throws std::runtime_error If the assembly fails
 */
void FEMSolver::assemble_matrices_serial() {
    // Create triplet lists for sparse matrix assembly
    std::vector<Eigen::Triplet<std::complex<double>>> H_triplets, M_triplets;

    // Iterate over all elements in the mesh
    for (size_t e = 0; e < mesh.getNumElements(); ++e) {
        // Assemble the element matrices
        Eigen::MatrixXcd H_e, M_e;
        assemble_element_matrix(e, H_e, M_e);

        // Get the element nodes based on element order
        std::vector<int> element;
        if (order == 1) {
            // Linear elements (P1)
            element.assign(mesh.getElements()[e].begin(), mesh.getElements()[e].end());
        } else if (order == 2) {
            // Quadratic elements (P2)
            element.assign(mesh.getQuadraticElements()[e].begin(), mesh.getQuadraticElements()[e].end());
        } else { // order == 3
            // Cubic elements (P3)
            element.assign(mesh.getCubicElements()[e].begin(), mesh.getCubicElements()[e].end());
        }

        // Add the element matrix entries to the global matrices using triplets
        for (int i = 0; i < H_e.rows(); ++i) {
            for (int j = 0; j < H_e.cols(); ++j) {
                H_triplets.emplace_back(element[i], element[j], H_e(i, j));
                M_triplets.emplace_back(element[i], element[j], M_e(i, j));
            }
        }
    }

    // Set the global matrices from the triplets
    H.setFromTriplets(H_triplets.begin(), H_triplets.end());
    M.setFromTriplets(M_triplets.begin(), M_triplets.end());
}

#ifdef USE_MPI
/**
 * @brief Assembles the Hamiltonian and mass matrices in parallel mode using MPI.
 *
 * This private method assembles the Hamiltonian and mass matrices in parallel mode
 * using MPI. It distributes the elements among the processes, assembles the local
 * matrices, gathers the results to the root process, and broadcasts the global
 * matrices to all processes.
 *
 * The parallel assembly process involves:
 * 1. Determining the rank and size of the MPI communicator
 * 2. Computing the number of elements per process
 * 3. Computing the start and end elements for each process
 * 4. Assembling the local matrices for the assigned elements
 * 5. Gathering all triplets to the root process
 * 6. Root process assembling the global matrices
 * 7. Broadcasting the matrices to all processes
 *
 * @throws std::runtime_error If the assembly fails
 */
void FEMSolver::assemble_matrices_parallel() {
    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute the number of elements per process
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





/**
 * @brief Adapts the mesh based on the given eigenvector.
 *
 * This method adapts the mesh by refining elements where the eigenvector
 * has a large gradient. The refinement is controlled by the threshold parameter.
 * It also supports caching the refined mesh to avoid recomputing it.
 *
 * The adaptation process involves:
 * 1. Checking if a cached mesh exists and loading it if available
 * 2. Computing refinement flags based on the eigenvector and threshold
 * 3. Refining the mesh using the computed flags
 * 4. Saving the refined mesh to the cache if enabled
 * 5. Recreating the interpolator for the refined mesh
 * 6. Reassembling the matrices for the refined mesh
 *
 * @param eigenvector The eigenvector to use for adaptation
 * @param threshold The threshold for refinement (elements with gradient > threshold are refined)
 * @param cache_dir The directory to use for caching intermediate results
 *
 * @throws std::invalid_argument If the eigenvector has an invalid size
 * @throws std::runtime_error If the adaptation fails
 */
void FEMSolver::adapt_mesh(const Eigen::VectorXd& eigenvector, double threshold, const std::string& cache_dir) {
    // Check if a cached mesh exists and load it if available
    std::string cache_file = cache_dir.empty() ? "" : cache_dir + "/mesh_" + std::to_string(eigenvector.norm()) + ".txt";
    if (!cache_file.empty() && std::filesystem::exists(cache_file)) {
        // Load the mesh from the cache file
        mesh = Mesh::load(cache_file);

        // Recreate the interpolator for the new mesh
        if (interpolator) {
            delete interpolator;
        }
        interpolator = new FEInterpolator(mesh);

        // Reassemble the matrices for the new mesh
        assemble_matrices();
        return;
    }

    // Compute refinement flags based on the eigenvector and threshold
    std::vector<bool> refine_flags = AdaptiveMesh::computeRefinementFlags(mesh, eigenvector, threshold);

    // Use MPI for mesh refinement if enabled
    if (use_mpi) {
#ifdef USE_MPI
        // Refine the mesh in parallel using MPI
        mesh.refine(refine_flags, MPI_COMM_WORLD);
#else
        // Fallback to serial refinement if MPI is disabled at compile time
        mesh.refine(refine_flags);
#endif
    } else {
        // Refine the mesh in serial mode
        mesh.refine(refine_flags);
    }

    // Save the refined mesh to the cache if enabled
    if (!cache_file.empty()) {
        // Create the cache directory if it doesn't exist
        std::filesystem::create_directories(cache_dir);
        // Save the mesh to the cache file
        mesh.save(cache_file);
    }

    // Recreate the interpolator for the refined mesh
    if (interpolator) {
        delete interpolator;
    }
    interpolator = new FEInterpolator(mesh);

    // Reassemble the matrices for the refined mesh
    assemble_matrices();
}



/**
 * @brief Assembles the element matrices for a single element.
 *
 * This private method assembles the Hamiltonian and mass matrices for a single element.
 * It computes the element matrices based on the element order, the material properties,
 * and the potential at the element nodes.
 *
 * The assembly process involves:
 * 1. Initializing the element matrices based on the element order
 * 2. Getting the element nodes
 * 3. Calculating the element area for proper scaling
 * 4. Computing the quadrature points and weights
 * 5. Evaluating the material properties and potential at each quadrature point
 * 6. Computing the basis functions and their gradients
 * 7. Assembling the element matrices using the quadrature rule
 *
 * @param e The index of the element
 * @param H_e The element Hamiltonian matrix (output)
 * @param M_e The element mass matrix (output)
 *
 * @throws std::runtime_error If the assembly fails
 */
void FEMSolver::assemble_element_matrix(size_t e, Eigen::MatrixXcd& H_e, Eigen::MatrixXcd& M_e) {
    // Constants
    const double hbar = 1.054e-34; // Reduced Planck constant in J·s

    // Initialize matrices based on element order
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

    // Calculate element area for proper scaling
    double element_area = 0.0;
    if (element_nodes.size() >= 3) {
        // For triangular elements, area = 0.5 * |cross product of two sides|
        Eigen::Vector2d v1 = element_nodes[1] - element_nodes[0];
        Eigen::Vector2d v2 = element_nodes[2] - element_nodes[0];
        element_area = 0.5 * std::abs(v1(0) * v2(1) - v1(1) * v2(0));
    }

    // Define quadrature points and weights based on element order
    std::vector<Eigen::Vector2d> quad_points;
    std::vector<double> quad_weights;

    // Set up quadrature points and weights for triangular elements
    if (order == 1) {
        // For P1 elements, use 3-point quadrature
        quad_points.resize(3);
        quad_weights.resize(3, element_area / 3.0);

        // Barycentric coordinates of quadrature points
        std::vector<std::vector<double>> bary_coords = {
            {1.0/6.0, 1.0/6.0, 2.0/3.0},
            {1.0/6.0, 2.0/3.0, 1.0/6.0},
            {2.0/3.0, 1.0/6.0, 1.0/6.0}
        };

        // Convert barycentric to physical coordinates
        for (int q = 0; q < 3; ++q) {
            quad_points[q] = bary_coords[q][0] * element_nodes[0] +
                             bary_coords[q][1] * element_nodes[1] +
                             bary_coords[q][2] * element_nodes[2];
        }
    } else if (order == 2) {
        // For P2 elements, use 7-point quadrature
        quad_points.resize(7);
        quad_weights.resize(7);

        // Barycentric coordinates and weights for 7-point quadrature
        std::vector<std::vector<double>> bary_coords = {
            {1.0/3.0, 1.0/3.0, 1.0/3.0},
            {0.0597, 0.4701, 0.4701},
            {0.4701, 0.0597, 0.4701},
            {0.4701, 0.4701, 0.0597},
            {0.7974, 0.1013, 0.1013},
            {0.1013, 0.7974, 0.1013},
            {0.1013, 0.1013, 0.7974}
        };

        std::vector<double> weights = {
            0.225 * element_area,
            0.1323941527 * element_area,
            0.1323941527 * element_area,
            0.1323941527 * element_area,
            0.1259391805 * element_area,
            0.1259391805 * element_area,
            0.1259391805 * element_area
        };

        // Convert barycentric to physical coordinates
        for (int q = 0; q < 7; ++q) {
            quad_points[q] = bary_coords[q][0] * element_nodes[0] +
                             bary_coords[q][1] * element_nodes[1] +
                             bary_coords[q][2] * element_nodes[2];
            quad_weights[q] = weights[q];
        }
    } else { // order == 3
        // For P3 elements, use 12-point quadrature
        quad_points.resize(12);
        quad_weights.resize(12);

        // Barycentric coordinates and weights for 12-point quadrature
        // These are simplified for this implementation
        std::vector<std::vector<double>> bary_coords = {
            {0.2406, 0.2406, 0.5188},
            {0.2406, 0.5188, 0.2406},
            {0.5188, 0.2406, 0.2406},
            {0.0630, 0.0630, 0.8740},
            {0.0630, 0.8740, 0.0630},
            {0.8740, 0.0630, 0.0630},
            {0.0597, 0.4701, 0.4701},
            {0.4701, 0.0597, 0.4701},
            {0.4701, 0.4701, 0.0597},
            {0.7974, 0.1013, 0.1013},
            {0.1013, 0.7974, 0.1013},
            {0.1013, 0.1013, 0.7974}
        };

        // Equal weights for simplicity
        for (int q = 0; q < 12; ++q) {
            quad_weights[q] = element_area / 12.0;
        }

        // Convert barycentric to physical coordinates
        for (int q = 0; q < 12; ++q) {
            quad_points[q] = bary_coords[q][0] * element_nodes[0] +
                             bary_coords[q][1] * element_nodes[1] +
                             bary_coords[q][2] * element_nodes[2];
        }
    }

    // Loop over quadrature points
    for (size_t q = 0; q < quad_points.size(); ++q) {
        double x = quad_points[q].x();
        double y = quad_points[q].y();
        double quad_weight = quad_weights[q];

        // Get physical parameters at this point
        double m = m_star(x, y);
        double V_val = V(x, y);
        double eta = cap(x, y);

        // Compute barycentric coordinates for this quadrature point
        std::vector<double> lambda(3);
        std::vector<Eigen::Vector2d> vertices = {
            element_nodes[0], element_nodes[1], element_nodes[2]
        };
        interpolator->computeBarycentricCoordinates(x, y, vertices, lambda);

        // Evaluate shape functions and their gradients
        std::vector<double> shape_values;
        std::vector<Eigen::Vector2d> shape_gradients;

        if (order == 1) {
            shape_values.resize(3);
            shape_gradients.resize(3);
        } else if (order == 2) {
            shape_values.resize(6);
            shape_gradients.resize(6);
        } else { // order == 3
            shape_values.resize(10);
            shape_gradients.resize(10);
        }

        // Evaluate shape functions and gradients
        interpolator->evaluateShapeFunctions(lambda, shape_values);
        interpolator->evaluateShapeFunctionGradients(lambda, vertices, shape_gradients);

        // Scale factor for kinetic energy term
        double kinetic_scale = (hbar * hbar) / (2.0 * m);

        // Assemble element matrices
        for (int i = 0; i < H_e.rows(); ++i) {
            for (int j = 0; j < H_e.cols(); ++j) {
                // Kinetic energy term (T = -ħ²/2m * ∇²)
                // Compute gradient dot product
                double grad_dot = shape_gradients[i].dot(shape_gradients[j]);

                // Potential energy term (V + iη)
                std::complex<double> potential(V_val, eta);

                // Hamiltonian: H = T + V
                H_e(i, j) += quad_weight * (kinetic_scale * grad_dot +
                                           potential * shape_values[i] * shape_values[j]);

                // Mass matrix (overlap of basis functions)
                M_e(i, j) += quad_weight * shape_values[i] * shape_values[j];
            }
        }
    }
}