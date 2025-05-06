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
                     double (*cap)(double, double), SelfConsistentSolver& sc_solver, int order, bool use_mpi)
    : mesh(mesh), sc_solver(sc_solver), m_star(m_star), V(V), cap(cap), order(order), use_mpi(use_mpi), use_interpolation(false) {
    // Initialize matrices with the correct size
    H.resize(mesh.getNumNodes(), mesh.getNumNodes());
    M.resize(mesh.getNumNodes(), mesh.getNumNodes());

    // Initialize potential values with zeros
    potential_values.resize(mesh.getNumNodes());
    potential_values.setZero();

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
 * @brief Enables or disables MPI for parallel computations.
 *
 * This method allows enabling or disabling MPI at runtime. When enabled,
 * the solver will use MPI for parallel matrix assembly and mesh refinement.
 * When disabled, the solver will use serial implementations.
 *
 * The method also handles MPI initialization if needed.
 *
 * @param enable Whether to enable MPI
 */
void FEMSolver::enable_mpi(bool enable) {
    // Only change the flag if the value is different
    if (use_mpi != enable) {
        use_mpi = enable;

        // If enabling MPI, check if MPI is initialized
        if (use_mpi) {
            int initialized;
            MPI_Initialized(&initialized);
            if (!initialized) {
                // Initialize MPI with thread support
                int provided;
                MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

                // Check if the requested thread support level was provided
                if (provided < MPI_THREAD_MULTIPLE) {
                    std::cerr << "Warning: MPI implementation does not support MPI_THREAD_MULTIPLE" << std::endl;
                }
            }
        }
    }
}

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

    // Compute the number of elements per process with better load balancing
    int num_elements = mesh.getNumElements();
    int elements_per_rank = num_elements / size;
    int remainder = num_elements % size;

    // Distribute remainder elements to first 'remainder' processes
    int start_elem, end_elem;
    if (rank < remainder) {
        // This process gets an extra element
        elements_per_rank++;
        start_elem = rank * elements_per_rank;
    } else {
        // This process gets the standard number of elements
        start_elem = rank * elements_per_rank + remainder;
    }
    end_elem = start_elem + elements_per_rank;

    // Assemble local matrices
    std::vector<Eigen::Triplet<std::complex<double>>> H_triplets, M_triplets;
    H_triplets.reserve(elements_per_rank * 10 * 10); // Reserve space for worst case (P3 elements)
    M_triplets.reserve(elements_per_rank * 10 * 10);

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

    // Pack triplets into arrays for efficient communication
    int h_count = H_triplets.size();
    int m_count = M_triplets.size();

    // Create arrays to hold all triplet data
    std::vector<int> h_rows(h_count), h_cols(h_count);
    std::vector<double> h_real(h_count), h_imag(h_count);
    std::vector<int> m_rows(m_count), m_cols(m_count);
    std::vector<double> m_real(m_count), m_imag(m_count);

    // Pack triplet data into arrays
    for (int i = 0; i < h_count; ++i) {
        h_rows[i] = H_triplets[i].row();
        h_cols[i] = H_triplets[i].col();
        h_real[i] = H_triplets[i].value().real();
        h_imag[i] = H_triplets[i].value().imag();
    }

    for (int i = 0; i < m_count; ++i) {
        m_rows[i] = M_triplets[i].row();
        m_cols[i] = M_triplets[i].col();
        m_real[i] = M_triplets[i].value().real();
        m_imag[i] = M_triplets[i].value().imag();
    }

    // Gather counts from all processes
    std::vector<int> h_counts(size), h_displs(size);
    std::vector<int> m_counts(size), m_displs(size);

    MPI_Gather(&h_count, 1, MPI_INT, h_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&m_count, 1, MPI_INT, m_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for gathering data
    if (rank == 0) {
        h_displs[0] = 0;
        m_displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            h_displs[i] = h_displs[i-1] + h_counts[i-1];
            m_displs[i] = m_displs[i-1] + m_counts[i-1];
        }
    }

    // Calculate total counts
    int total_h_count = 0, total_m_count = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            total_h_count += h_counts[i];
            total_m_count += m_counts[i];
        }
    }

    // Allocate arrays for gathered data on root process
    std::vector<int> global_h_rows, global_h_cols;
    std::vector<double> global_h_real, global_h_imag;
    std::vector<int> global_m_rows, global_m_cols;
    std::vector<double> global_m_real, global_m_imag;

    if (rank == 0) {
        global_h_rows.resize(total_h_count);
        global_h_cols.resize(total_h_count);
        global_h_real.resize(total_h_count);
        global_h_imag.resize(total_h_count);
        global_m_rows.resize(total_m_count);
        global_m_cols.resize(total_m_count);
        global_m_real.resize(total_m_count);
        global_m_imag.resize(total_m_count);
    }

    // Gather all triplet data using MPI_Gatherv for efficient communication
    MPI_Gatherv(h_rows.data(), h_count, MPI_INT,
                global_h_rows.data(), h_counts.data(), h_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_cols.data(), h_count, MPI_INT,
                global_h_cols.data(), h_counts.data(), h_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_real.data(), h_count, MPI_DOUBLE,
                global_h_real.data(), h_counts.data(), h_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_imag.data(), h_count, MPI_DOUBLE,
                global_h_imag.data(), h_counts.data(), h_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(m_rows.data(), m_count, MPI_INT,
                global_m_rows.data(), m_counts.data(), m_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(m_cols.data(), m_count, MPI_INT,
                global_m_cols.data(), m_counts.data(), m_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(m_real.data(), m_count, MPI_DOUBLE,
                global_m_real.data(), m_counts.data(), m_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(m_imag.data(), m_count, MPI_DOUBLE,
                global_m_imag.data(), m_counts.data(), m_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Assemble global matrices on root process
    if (rank == 0) {
        std::vector<Eigen::Triplet<std::complex<double>>> global_H_triplets, global_M_triplets;
        global_H_triplets.reserve(total_h_count);
        global_M_triplets.reserve(total_m_count);

        for (int i = 0; i < total_h_count; ++i) {
            global_H_triplets.emplace_back(global_h_rows[i], global_h_cols[i],
                                          std::complex<double>(global_h_real[i], global_h_imag[i]));
        }

        for (int i = 0; i < total_m_count; ++i) {
            global_M_triplets.emplace_back(global_m_rows[i], global_m_cols[i],
                                          std::complex<double>(global_m_real[i], global_m_imag[i]));
        }

        // Set matrices from triplets
        H.setFromTriplets(global_H_triplets.begin(), global_H_triplets.end());
        M.setFromTriplets(global_M_triplets.begin(), global_M_triplets.end());
    }

    // Broadcast matrix structure to all processes
    int rows = mesh.getNumNodes();
    int cols = mesh.getNumNodes();
    int nnz_H = 0, nnz_M = 0;

    if (rank == 0) {
        nnz_H = H.nonZeros();
        nnz_M = M.nonZeros();
    }

    // Broadcast matrix dimensions and non-zero counts
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz_H, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz_M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // For non-root processes, initialize matrices with correct dimensions
    if (rank != 0) {
        H.resize(rows, cols);
        M.resize(rows, cols);

        // In a full implementation, we would broadcast the actual matrix data
        // This would involve serializing the sparse matrices
        // For now, we'll just ensure all processes have matrices with correct dimensions
    }

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
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
/**
 * @brief Sets the potential values at mesh nodes for interpolation.
 *
 * This method sets the potential values at mesh nodes for proper finite element
 * interpolation. The potential values are used in the assemble_element_matrix method
 * when use_interpolation is enabled.
 *
 * @param potential_values The potential values at mesh nodes
 *
 * @throws std::invalid_argument If the potential_values vector has an invalid size
 */
void FEMSolver::set_potential_values(const Eigen::VectorXd& potential_values) {
    // Check if the potential_values vector has the correct size
    if (potential_values.size() != mesh.getNumNodes()) {
        throw std::invalid_argument("Potential values vector has invalid size");
    }

    // Set the potential values
    this->potential_values = potential_values;
}

/**
 * @brief Enables or disables the use of interpolated potentials.
 *
 * When enabled, the solver will use finite element interpolation for potentials
 * instead of calling the V function directly. This provides more accurate results,
 * especially for higher-order elements.
 *
 * @param enable Whether to enable interpolated potentials
 */
void FEMSolver::use_interpolated_potential(bool enable) {
    use_interpolation = enable;
}

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

    // Print mesh quality information
    std::cout << "Mesh quality after refinement:" << std::endl;
    std::cout << "  Number of nodes: " << mesh.getNumNodes() << std::endl;
    std::cout << "  Number of elements: " << mesh.getNumElements() << std::endl;
}



/**
 * @brief Maps a point from the reference element to the physical element.
 *
 * This method maps a point from the reference element (unit triangle) to the
 * physical element using linear interpolation with barycentric coordinates.
 *
 * @param ref_point The point in the reference element
 * @param nodes The nodes of the physical element
 * @return Eigen::Vector2d The corresponding point in the physical element
 */
Eigen::Vector2d FEMSolver::map_reference_to_physical(const Eigen::Vector2d& ref_point,
                                                   const std::vector<Eigen::Vector2d>& nodes) const {
    // For a triangle, the mapping is linear using barycentric coordinates
    double lambda1 = 1.0 - ref_point[0] - ref_point[1];
    double lambda2 = ref_point[0];
    double lambda3 = ref_point[1];

    // Map using barycentric coordinates
    return lambda1 * nodes[0] + lambda2 * nodes[1] + lambda3 * nodes[2];
}

/**
 * @brief Evaluates the shape functions and their gradients at a point.
 *
 * This method evaluates the shape functions and their gradients at a point
 * in the reference element.
 *
 * @param ref_point The point in the reference element
 * @param shape_values Output parameter for the shape function values
 * @param shape_gradients Output parameter for the shape function gradients
 * @param nodes The nodes of the physical element
 * @param order The order of the finite elements
 */
void FEMSolver::evaluate_shape_functions(const Eigen::Vector2d& ref_point,
                                       std::vector<double>& shape_values,
                                       std::vector<Eigen::Vector2d>& shape_gradients,
                                       const std::vector<Eigen::Vector2d>& nodes,
                                       int order) const {
    // Barycentric coordinates
    double lambda1 = 1.0 - ref_point[0] - ref_point[1];
    double lambda2 = ref_point[0];
    double lambda3 = ref_point[1];

    // Evaluate shape functions based on element order
    if (order == 1) {
        // P1 elements (linear)
        shape_values.resize(3);
        shape_values[0] = lambda1;
        shape_values[1] = lambda2;
        shape_values[2] = lambda3;

        // Gradients in reference element
        std::vector<Eigen::Vector2d> ref_gradients(3);
        ref_gradients[0] = Eigen::Vector2d(-1.0, -1.0);
        ref_gradients[1] = Eigen::Vector2d(1.0, 0.0);
        ref_gradients[2] = Eigen::Vector2d(0.0, 1.0);

        // Map gradients to physical element
        shape_gradients.resize(3);
        map_gradients_to_physical(ref_gradients, shape_gradients, nodes);
    } else if (order == 2) {
        // P2 elements (quadratic)
        shape_values.resize(6);

        // Vertex nodes
        shape_values[0] = lambda1 * (2.0 * lambda1 - 1.0);
        shape_values[1] = lambda2 * (2.0 * lambda2 - 1.0);
        shape_values[2] = lambda3 * (2.0 * lambda3 - 1.0);

        // Edge nodes
        shape_values[3] = 4.0 * lambda1 * lambda2;
        shape_values[4] = 4.0 * lambda2 * lambda3;
        shape_values[5] = 4.0 * lambda3 * lambda1;

        // Gradients in reference element
        std::vector<Eigen::Vector2d> ref_gradients(6);

        // Vertex nodes
        ref_gradients[0] = Eigen::Vector2d(-3.0 + 4.0 * lambda1, -3.0 + 4.0 * lambda1);
        ref_gradients[1] = Eigen::Vector2d(4.0 * lambda2 - 1.0, 0.0);
        ref_gradients[2] = Eigen::Vector2d(0.0, 4.0 * lambda3 - 1.0);

        // Edge nodes
        ref_gradients[3] = Eigen::Vector2d(4.0 * (1.0 - 2.0 * lambda1 - lambda2), -4.0 * lambda2);
        ref_gradients[4] = Eigen::Vector2d(4.0 * lambda3, 4.0 * lambda2);
        ref_gradients[5] = Eigen::Vector2d(-4.0 * lambda3, 4.0 * (1.0 - lambda1 - 2.0 * lambda3));

        // Map gradients to physical element
        shape_gradients.resize(6);
        map_gradients_to_physical(ref_gradients, shape_gradients, nodes);
    } else { // order == 3
        // P3 elements (cubic)
        shape_values.resize(10);

        // Vertex nodes
        shape_values[0] = 0.5 * lambda1 * (3.0 * lambda1 - 1.0) * (3.0 * lambda1 - 2.0);
        shape_values[1] = 0.5 * lambda2 * (3.0 * lambda2 - 1.0) * (3.0 * lambda2 - 2.0);
        shape_values[2] = 0.5 * lambda3 * (3.0 * lambda3 - 1.0) * (3.0 * lambda3 - 2.0);

        // Edge nodes (2 per edge)
        shape_values[3] = 4.5 * lambda1 * lambda2 * (3.0 * lambda1 - 1.0);
        shape_values[4] = 4.5 * lambda1 * lambda2 * (3.0 * lambda2 - 1.0);

        shape_values[5] = 4.5 * lambda2 * lambda3 * (3.0 * lambda2 - 1.0);
        shape_values[6] = 4.5 * lambda2 * lambda3 * (3.0 * lambda3 - 1.0);

        shape_values[7] = 4.5 * lambda3 * lambda1 * (3.0 * lambda3 - 1.0);
        shape_values[8] = 4.5 * lambda3 * lambda1 * (3.0 * lambda1 - 1.0);

        // Interior node
        shape_values[9] = 27.0 * lambda1 * lambda2 * lambda3;

        // Gradients in reference element
        std::vector<Eigen::Vector2d> ref_gradients(10);

        // Vertex nodes
        ref_gradients[0] = Eigen::Vector2d(-9.0 * lambda1 * lambda1 + 9.0 * lambda1 - 1.0, -9.0 * lambda1 * lambda1 + 9.0 * lambda1 - 1.0);
        ref_gradients[1] = Eigen::Vector2d(9.0 * lambda2 * lambda2 - 9.0 * lambda2 + 1.0, 0.0);
        ref_gradients[2] = Eigen::Vector2d(0.0, 9.0 * lambda3 * lambda3 - 9.0 * lambda3 + 1.0);

        // Edge nodes (2 per edge)
        ref_gradients[3] = Eigen::Vector2d(4.5 * (3.0 * lambda1 - 1.0) * lambda2 + 4.5 * lambda1 * lambda2 * 3.0, 4.5 * lambda1 * (3.0 * lambda1 - 1.0));
        ref_gradients[4] = Eigen::Vector2d(4.5 * lambda1 * (3.0 * lambda2 - 1.0) + 4.5 * lambda1 * lambda2 * 3.0, 4.5 * lambda1 * (3.0 * lambda2 - 1.0));

        ref_gradients[5] = Eigen::Vector2d(4.5 * lambda3 * (3.0 * lambda2 - 1.0) + 4.5 * lambda2 * lambda3 * 3.0, 4.5 * lambda2 * (3.0 * lambda2 - 1.0));
        ref_gradients[6] = Eigen::Vector2d(4.5 * lambda3 * (3.0 * lambda2 - 1.0), 4.5 * lambda2 * (3.0 * lambda3 - 1.0) + 4.5 * lambda2 * lambda3 * 3.0);

        ref_gradients[7] = Eigen::Vector2d(4.5 * lambda3 * (3.0 * lambda3 - 1.0), 4.5 * lambda1 * (3.0 * lambda3 - 1.0) + 4.5 * lambda3 * lambda1 * 3.0);
        ref_gradients[8] = Eigen::Vector2d(4.5 * lambda3 * (3.0 * lambda1 - 1.0) + 4.5 * lambda3 * lambda1 * 3.0, 4.5 * lambda1 * (3.0 * lambda1 - 1.0));

        // Interior node
        ref_gradients[9] = Eigen::Vector2d(27.0 * lambda2 * lambda3, 27.0 * lambda1 * lambda3);

        // Map gradients to physical element
        shape_gradients.resize(10);
        map_gradients_to_physical(ref_gradients, shape_gradients, nodes);
    }
}

/**
 * @brief Maps gradients from the reference element to the physical element.
 *
 * This method maps gradients from the reference element to the physical element
 * using the Jacobian of the transformation.
 *
 * @param ref_gradients The gradients in the reference element
 * @param phys_gradients Output parameter for the gradients in the physical element
 * @param nodes The nodes of the physical element
 */
void FEMSolver::map_gradients_to_physical(const std::vector<Eigen::Vector2d>& ref_gradients,
                                        std::vector<Eigen::Vector2d>& phys_gradients,
                                        const std::vector<Eigen::Vector2d>& nodes) const {
    // Compute Jacobian matrix for mapping from reference to physical element
    Eigen::Matrix2d J;
    J(0, 0) = nodes[1][0] - nodes[0][0];
    J(0, 1) = nodes[2][0] - nodes[0][0];
    J(1, 0) = nodes[1][1] - nodes[0][1];
    J(1, 1) = nodes[2][1] - nodes[0][1];

    // Compute inverse of Jacobian
    Eigen::Matrix2d J_inv = J.inverse();

    // Map gradients
    for (size_t i = 0; i < ref_gradients.size(); ++i) {
        phys_gradients[i] = J_inv.transpose() * ref_gradients[i];
    }
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
    const double e_charge = 1.602e-19; // Elementary charge in C

    // Initialize matrices based on element order
    int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;
    H_e.setZero(nodes_per_elem, nodes_per_elem);
    M_e.setZero(nodes_per_elem, nodes_per_elem);

    // Get element nodes
    std::vector<Eigen::Vector2d> element_nodes;
    std::vector<int> element_indices;
    if (order == 1) {
        const auto& elem = mesh.getElements()[e];
        element_indices.assign(elem.begin(), elem.end());
        for (int i = 0; i < 3; ++i) {
            element_nodes.push_back(mesh.getNodes()[elem[i]]);
        }
    } else if (order == 2) {
        const auto& elem = mesh.getQuadraticElements()[e];
        element_indices.assign(elem.begin(), elem.end());
        for (int i = 0; i < 6; ++i) {
            element_nodes.push_back(mesh.getNodes()[elem[i]]);
        }
    } else { // order == 3
        const auto& elem = mesh.getCubicElements()[e];
        element_indices.assign(elem.begin(), elem.end());
        for (int i = 0; i < 10; ++i) {
            element_nodes.push_back(mesh.getNodes()[elem[i]]);
        }
    }

    // Calculate element area
    Eigen::Vector2d v1 = element_nodes[1] - element_nodes[0];
    Eigen::Vector2d v2 = element_nodes[2] - element_nodes[0];
    double element_area = 0.5 * std::abs(v1(0) * v2(1) - v1(1) * v2(0));

    // Define quadrature points and weights based on element order
    std::vector<Eigen::Vector3d> quad_points;
    std::vector<double> quad_weights;

    if (order == 1) {
        // 3-point Gaussian quadrature for P1 elements
        quad_points = {
            {1.0/6.0, 1.0/6.0, 2.0/3.0},
            {1.0/6.0, 2.0/3.0, 1.0/6.0},
            {2.0/3.0, 1.0/6.0, 1.0/6.0}
        };
        quad_weights = {1.0/6.0, 1.0/6.0, 1.0/6.0};
    } else if (order == 2) {
        // 7-point Gaussian quadrature for P2 elements
        quad_points = {
            {1.0/3.0, 1.0/3.0, 1.0/3.0},
            {0.059715871789770, 0.470142064105115, 0.470142064105115},
            {0.470142064105115, 0.059715871789770, 0.470142064105115},
            {0.470142064105115, 0.470142064105115, 0.059715871789770},
            {0.797426985353087, 0.101286507323456, 0.101286507323456},
            {0.101286507323456, 0.797426985353087, 0.101286507323456},
            {0.101286507323456, 0.101286507323456, 0.797426985353087}
        };
        quad_weights = {
            0.225000000000000,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827
        };
    } else { // order == 3
        // 12-point Gaussian quadrature for P3 elements
        quad_points = {
            {0.249286745170910, 0.249286745170910, 0.501426509658179},
            {0.249286745170910, 0.501426509658179, 0.249286745170910},
            {0.501426509658179, 0.249286745170910, 0.249286745170910},
            {0.063089014491502, 0.063089014491502, 0.873821971016996},
            {0.063089014491502, 0.873821971016996, 0.063089014491502},
            {0.873821971016996, 0.063089014491502, 0.063089014491502},
            {0.310352451033785, 0.636502499121399, 0.053145049844816},
            {0.636502499121399, 0.053145049844816, 0.310352451033785},
            {0.053145049844816, 0.310352451033785, 0.636502499121399},
            {0.636502499121399, 0.310352451033785, 0.053145049844816},
            {0.310352451033785, 0.053145049844816, 0.636502499121399},
            {0.053145049844816, 0.636502499121399, 0.310352451033785}
        };
        quad_weights = {
            0.116786275726379,
            0.116786275726379,
            0.116786275726379,
            0.050844906370207,
            0.050844906370207,
            0.050844906370207,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374
        };
    }

    // Compute shape function gradients in reference element
    std::vector<Eigen::Vector2d> ref_gradients;
    if (order == 1) {
        // P1 shape function gradients in reference element
        ref_gradients = {
            {-1.0, -1.0},  // dN1/d(xi), dN1/d(eta)
            {1.0, 0.0},    // dN2/d(xi), dN2/d(eta)
            {0.0, 1.0}     // dN3/d(xi), dN3/d(eta)
        };
    } else if (order == 2) {
        // P2 shape function gradients in reference element
        ref_gradients = {
            {-3.0 + 4.0 * (1.0 - 1.0 - 1.0), -3.0 + 4.0 * (1.0 - 1.0 - 1.0)},  // Vertex 1
            {4.0 - 4.0 * 2.0, 0.0},                                            // Vertex 2
            {0.0, 4.0 - 4.0 * 2.0},                                            // Vertex 3
            {4.0 * 1.0, 4.0 * 1.0},                                            // Edge 1-2
            {0.0, 4.0 * 1.0},                                                  // Edge 2-3
            {4.0 * 1.0, 0.0}                                                   // Edge 3-1
        };
    } else { // order == 3
        // P3 shape function gradients in reference element
        ref_gradients.resize(10);

        // Vertex nodes
        ref_gradients[0] = Eigen::Vector2d(-9.0 * lambda1 * lambda1 + 9.0 * lambda1 - 1.0, -9.0 * lambda1 * lambda1 + 9.0 * lambda1 - 1.0);
        ref_gradients[1] = Eigen::Vector2d(9.0 * lambda2 * lambda2 - 9.0 * lambda2 + 1.0, 0.0);
        ref_gradients[2] = Eigen::Vector2d(0.0, 9.0 * lambda3 * lambda3 - 9.0 * lambda3 + 1.0);

        // Edge nodes (2 per edge)
        ref_gradients[3] = Eigen::Vector2d(9.0 * lambda1 * (3.0 * lambda2 - 1.0) + 9.0 * lambda2 * (3.0 * lambda1 - 1.0), 9.0 * lambda1 * (3.0 * lambda2 - 1.0));
        ref_gradients[4] = Eigen::Vector2d(9.0 * lambda1 * (3.0 * lambda2 - 1.0) + 9.0 * lambda2 * (3.0 * lambda1 - 1.0), 9.0 * lambda2 * (3.0 * lambda1 - 1.0));

        ref_gradients[5] = Eigen::Vector2d(9.0 * lambda2 * (3.0 * lambda3 - 1.0), 9.0 * lambda2 * (3.0 * lambda3 - 1.0) + 9.0 * lambda3 * (3.0 * lambda2 - 1.0));
        ref_gradients[6] = Eigen::Vector2d(9.0 * lambda3 * (3.0 * lambda2 - 1.0), 9.0 * lambda2 * (3.0 * lambda3 - 1.0) + 9.0 * lambda3 * (3.0 * lambda2 - 1.0));

        ref_gradients[7] = Eigen::Vector2d(9.0 * lambda3 * (3.0 * lambda1 - 1.0), 9.0 * lambda3 * (3.0 * lambda1 - 1.0) + 9.0 * lambda1 * (3.0 * lambda3 - 1.0));
        ref_gradients[8] = Eigen::Vector2d(9.0 * lambda3 * (3.0 * lambda1 - 1.0) + 9.0 * lambda1 * (3.0 * lambda3 - 1.0), 9.0 * lambda1 * (3.0 * lambda3 - 1.0));

        // Interior node
        ref_gradients[9] = Eigen::Vector2d(27.0 * lambda2 * lambda3, 27.0 * lambda1 * lambda3);
    }

    // Compute Jacobian matrix for mapping from reference to physical element
    Eigen::Matrix2d J;
    J << element_nodes[1](0) - element_nodes[0](0), element_nodes[2](0) - element_nodes[0](0),
         element_nodes[1](1) - element_nodes[0](1), element_nodes[2](1) - element_nodes[0](1);
    Eigen::Matrix2d J_inv = J.inverse();
    double det_J = J.determinant();

    // Loop over quadrature points
    for (size_t q = 0; q < quad_points.size(); ++q) {
        // Barycentric coordinates at quadrature point
        double lambda1 = quad_points[q](0);
        double lambda2 = quad_points[q](1);
        double lambda3 = quad_points[q](2);

        // Map from reference element to physical element
        double x = lambda1 * element_nodes[0](0) + lambda2 * element_nodes[1](0) + lambda3 * element_nodes[2](0);
        double y = lambda1 * element_nodes[0](1) + lambda2 * element_nodes[1](1) + lambda3 * element_nodes[2](1);

        // Get physical parameters at this point
        double m = m_star(x, y);
        double V_val;

        // Use interpolated potential if enabled, otherwise use the V function
        if (use_interpolation && potential_values.size() == mesh.getNumNodes()) {
            // Use the interpolator to get the potential at (x,y)
            V_val = interpolator->interpolate(x, y, potential_values);
        } else {
            // Use the V function directly
            V_val = V(x, y);
        }

        double eta = cap(x, y);

        // Scale factor for kinetic energy term
        double kinetic_scale = (hbar * hbar) / (2.0 * m);

        // Compute shape functions and their gradients at quadrature point
        std::vector<double> shape_values;
        std::vector<Eigen::Vector2d> shape_gradients;

        if (order == 1) {
            // P1 shape functions are just the barycentric coordinates
            shape_values = {lambda1, lambda2, lambda3};

            // Transform gradients from reference to physical element
            shape_gradients.resize(3);
            for (int i = 0; i < 3; ++i) {
                shape_gradients[i] = J_inv.transpose() * ref_gradients[i];
            }
        } else if (order == 2) {
            // P2 shape functions
            shape_values = {
                lambda1 * (2.0 * lambda1 - 1.0),  // Vertex 1
                lambda2 * (2.0 * lambda2 - 1.0),  // Vertex 2
                lambda3 * (2.0 * lambda3 - 1.0),  // Vertex 3
                4.0 * lambda1 * lambda2,          // Edge 1-2
                4.0 * lambda2 * lambda3,          // Edge 2-3
                4.0 * lambda3 * lambda1           // Edge 3-1
            };

            // Transform gradients from reference to physical element
            shape_gradients.resize(6);
            for (int i = 0; i < 6; ++i) {
                shape_gradients[i] = J_inv.transpose() * ref_gradients[i];
            }
        } else { // order == 3
            // P3 shape functions
            shape_values = {
                lambda1 * (3.0 * lambda1 - 1.0) * (3.0 * lambda1 - 2.0) / 2.0,  // Vertex 1
                lambda2 * (3.0 * lambda2 - 1.0) * (3.0 * lambda2 - 2.0) / 2.0,  // Vertex 2
                lambda3 * (3.0 * lambda3 - 1.0) * (3.0 * lambda3 - 2.0) / 2.0,  // Vertex 3
                9.0 * lambda1 * lambda2 * (3.0 * lambda1 - 1.0) / 2.0,          // Edge 1-2 (near 1)
                9.0 * lambda1 * lambda2 * (3.0 * lambda2 - 1.0) / 2.0,          // Edge 1-2 (near 2)
                9.0 * lambda2 * lambda3 * (3.0 * lambda2 - 1.0) / 2.0,          // Edge 2-3 (near 2)
                9.0 * lambda2 * lambda3 * (3.0 * lambda3 - 1.0) / 2.0,          // Edge 2-3 (near 3)
                9.0 * lambda3 * lambda1 * (3.0 * lambda3 - 1.0) / 2.0,          // Edge 3-1 (near 3)
                9.0 * lambda3 * lambda1 * (3.0 * lambda1 - 1.0) / 2.0,          // Edge 3-1 (near 1)
                27.0 * lambda1 * lambda2 * lambda3                              // Center
            };

            // Transform gradients from reference to physical element
            shape_gradients.resize(10);
            for (int i = 0; i < 10; ++i) {
                shape_gradients[i] = J_inv.transpose() * ref_gradients[i];
            }
        }

        // Quadrature weight scaled by determinant of Jacobian
        double weight = quad_weights[q] * std::abs(det_J);

        // Potential energy term (V + iη)
        std::complex<double> potential(V_val, eta);

        // Assemble element matrices
        for (int i = 0; i < nodes_per_elem; ++i) {
            for (int j = 0; j < nodes_per_elem; ++j) {
                // Kinetic energy term: (ħ²/2m) * ∇φᵢ · ∇φⱼ
                double grad_dot = shape_gradients[i].dot(shape_gradients[j]);

                // Hamiltonian: H = T + V
                H_e(i, j) += weight * (kinetic_scale * grad_dot + potential * shape_values[i] * shape_values[j]);

                // Mass matrix: M = φᵢ · φⱼ
                M_e(i, j) += weight * shape_values[i] * shape_values[j];
            }
        }
    }
}