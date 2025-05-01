#pragma once
/**
 * @file mesh.h
 * @brief Defines the Mesh class for finite element discretization.
 *
 * This file contains the declaration of the Mesh class, which represents
 * a 2D triangular mesh for finite element simulations. The mesh supports
 * linear (P1), quadratic (P2), and cubic (P3) elements, and provides
 * methods for mesh generation, refinement, and I/O.
 *
 * Physical units:
 * - Coordinates: nanometers (nm)
 *
 * Assumptions and limitations:
 * - The mesh is 2D and rectangular
 * - The mesh is structured (regular grid of triangles)
 * - The mesh supports linear, quadratic, and cubic elements
 * - The mesh can be refined adaptively
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <vector>
#include <array>
#include <Eigen/Dense>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @class Mesh
 * @brief Represents a 2D triangular mesh for finite element simulations.
 *
 * The Mesh class provides functionality for creating, manipulating, and
 * accessing a 2D triangular mesh. It supports linear (P1), quadratic (P2),
 * and cubic (P3) elements, and provides methods for mesh generation,
 * refinement, and I/O.
 */
class Mesh {
public:
    /**
     * @brief Constructs a new Mesh object.
     *
     * @param Lx Width of the domain in nanometers (nm)
     * @param Ly Height of the domain in nanometers (nm)
     * @param nx Number of elements in the x-direction
     * @param ny Number of elements in the y-direction
     * @param element_order Order of the elements (1 for P1, 2 for P2, 3 for P3)
     *
     * @throws std::invalid_argument If the input parameters are invalid
     */
    Mesh(double Lx, double Ly, int nx, int ny, int element_order = 1);
    /**
     * @brief Get the nodes of the mesh.
     * @return A reference to the vector of node coordinates
     */
    const std::vector<Eigen::Vector2d>& getNodes() const { return nodes; }

    /**
     * @brief Get the linear (P1) elements of the mesh.
     * @return A reference to the vector of linear elements
     */
    const std::vector<std::array<int, 3>>& getElements() const { return elements; }

    /**
     * @brief Get the quadratic (P2) elements of the mesh.
     * @return A reference to the vector of quadratic elements
     */
    const std::vector<std::array<int, 6>>& getQuadraticElements() const { return quadratic_elements; }

    /**
     * @brief Get the cubic (P3) elements of the mesh.
     * @return A reference to the vector of cubic elements
     */
    const std::vector<std::array<int, 10>>& getCubicElements() const { return cubic_elements; }

    /**
     * @brief Get the number of nodes in the mesh.
     * @return The number of nodes
     */
    int getNumNodes() const { return nodes.size(); }

    /**
     * @brief Get the number of elements in the mesh.
     * @return The number of elements
     */
    int getNumElements() const { return elements.size(); }

    /**
     * @brief Get the order of the elements in the mesh.
     * @return The element order (1 for P1, 2 for P2, 3 for P3)
     */
    int getElementOrder() const { return element_order; }

    /**
     * @brief Get the width of the domain.
     * @return The width in nanometers (nm)
     */
    double get_lx() const { return Lx; }

    /**
     * @brief Get the height of the domain.
     * @return The height in nanometers (nm)
     */
    double get_ly() const { return Ly; }

    /**
     * @brief Get the number of elements in the x-direction.
     * @return The number of elements in the x-direction
     */
    int get_nx() const { return nx; }

    /**
     * @brief Get the number of elements in the y-direction.
     * @return The number of elements in the y-direction
     */
    int get_ny() const { return ny; }
    /**
     * @brief Refine the mesh based on the given flags.
     *
     * This method refines the mesh by subdividing the elements marked for refinement.
     * The refinement is performed in a way that maintains the mesh quality and
     * ensures that the resulting mesh is conforming (no hanging nodes).
     *
     * @param refine_flags A vector of boolean flags indicating which elements to refine
     *
     * @throws std::invalid_argument If the size of refine_flags does not match the number of elements
     */
    void refine(const std::vector<bool>& refine_flags);

#ifdef USE_MPI
    /**
     * @brief Refine the mesh in parallel using MPI.
     *
     * This method refines the mesh in parallel using MPI. The refinement is performed
     * in a way that maintains the mesh quality and ensures that the resulting mesh
     * is conforming (no hanging nodes) across process boundaries.
     *
     * @param refine_flags A vector of boolean flags indicating which elements to refine
     * @param comm The MPI communicator
     *
     * @throws std::invalid_argument If the size of refine_flags does not match the number of elements
     */
    void refine(const std::vector<bool>& refine_flags, MPI_Comm comm);
#endif

    /**
     * @brief Save the mesh to a file.
     *
     * This method saves the mesh to a file in a custom binary format.
     * The file contains the mesh nodes, elements, and other metadata.
     *
     * @param filename The name of the file to save the mesh to
     *
     * @throws std::runtime_error If the file cannot be opened or written to
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load a mesh from a file.
     *
     * This static method loads a mesh from a file in the custom binary format
     * created by the save method.
     *
     * @param filename The name of the file to load the mesh from
     * @return A new Mesh object loaded from the file
     *
     * @throws std::runtime_error If the file cannot be opened or read from
     * @throws std::invalid_argument If the file format is invalid
     */
    static Mesh load(const std::string& filename);
private:
    /** @brief Vector of node coordinates */
    std::vector<Eigen::Vector2d> nodes;

    /** @brief Vector of linear (P1) elements, each with 3 nodes */
    std::vector<std::array<int, 3>> elements;

    /** @brief Vector of quadratic (P2) elements, each with 6 nodes */
    std::vector<std::array<int, 6>> quadratic_elements;

    /** @brief Vector of cubic (P3) elements, each with 10 nodes */
    std::vector<std::array<int, 10>> cubic_elements;

    /** @brief Order of the elements (1 for P1, 2 for P2, 3 for P3) */
    int element_order;

    /** @brief Width and height of the domain in nanometers (nm) */
    double Lx, Ly;

    /** @brief Number of elements in the x and y directions */
    int nx, ny;

    /**
     * @brief Generate a triangular mesh.
     *
     * This private method generates a structured triangular mesh with the given
     * dimensions and number of elements.
     *
     * @param Lx Width of the domain in nanometers (nm)
     * @param Ly Height of the domain in nanometers (nm)
     * @param nx Number of elements in the x-direction
     * @param ny Number of elements in the y-direction
     */
    void generateTriangularMesh(double Lx, double Ly, int nx, int ny);
};