/**
 * @file error_visualizer.h
 * @brief Declaration of the ErrorVisualizer class for QDSim.
 *
 * This file contains the declaration of the ErrorVisualizer class, which provides
 * visualization tools for diagnosing errors in QDSim simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#ifndef QDSIM_ERROR_VISUALIZER_H
#define QDSIM_ERROR_VISUALIZER_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "mesh.h"
#include "error_handling.h"

namespace ErrorHandling {

/**
 * @class ErrorVisualizer
 * @brief Class for visualizing errors in QDSim simulations.
 *
 * The ErrorVisualizer class provides tools for visualizing errors in QDSim simulations,
 * such as mesh quality issues, solver convergence problems, and physical inconsistencies.
 */
class ErrorVisualizer {
public:
    /**
     * @brief Gets the singleton instance of the ErrorVisualizer.
     *
     * @return The singleton instance
     */
    static ErrorVisualizer& instance() {
        static ErrorVisualizer instance;
        return instance;
    }

    /**
     * @brief Visualizes mesh quality issues.
     *
     * This method generates a visualization of mesh quality issues, highlighting
     * elements with low quality.
     *
     * @param mesh The mesh to visualize
     * @param filename The output file name
     * @param quality_threshold The quality threshold (elements with quality below this threshold are highlighted)
     * @return True if the visualization was generated successfully, false otherwise
     */
    bool visualize_mesh_quality(const Mesh& mesh, const std::string& filename, double quality_threshold = 0.3);

    /**
     * @brief Visualizes solver convergence issues.
     *
     * This method generates a visualization of solver convergence issues, showing
     * the convergence history and highlighting problematic iterations.
     *
     * @param residuals The residuals from the solver
     * @param filename The output file name
     * @param tolerance The convergence tolerance
     * @return True if the visualization was generated successfully, false otherwise
     */
    bool visualize_solver_convergence(const std::vector<double>& residuals, const std::string& filename, double tolerance);

    /**
     * @brief Visualizes physical inconsistencies.
     *
     * This method generates a visualization of physical inconsistencies, such as
     * discontinuities in the solution or potential.
     *
     * @param mesh The mesh
     * @param solution The solution vector
     * @param filename The output file name
     * @return True if the visualization was generated successfully, false otherwise
     */
    bool visualize_physical_inconsistencies(const Mesh& mesh, const Eigen::VectorXd& solution, const std::string& filename);

    /**
     * @brief Visualizes matrix properties.
     *
     * This method generates a visualization of matrix properties, such as
     * sparsity pattern, condition number, and eigenvalue distribution.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @param filename The output file name
     * @return True if the visualization was generated successfully, false otherwise
     */
    bool visualize_matrix_properties(const Eigen::SparseMatrix<std::complex<double>>& H,
                                   const Eigen::SparseMatrix<std::complex<double>>& M,
                                   const std::string& filename);

    /**
     * @brief Visualizes an error location in the mesh.
     *
     * This method generates a visualization of an error location in the mesh,
     * highlighting the elements or nodes where the error occurred.
     *
     * @param mesh The mesh
     * @param error The error to visualize
     * @param filename The output file name
     * @return True if the visualization was generated successfully, false otherwise
     */
    bool visualize_error_location(const Mesh& mesh, const QDSimException& error, const std::string& filename);

    /**
     * @brief Visualizes memory usage.
     *
     * This method generates a visualization of memory usage, showing the
     * memory usage over time or by component.
     *
     * @param memory_usage A map of component names to memory usage in bytes
     * @param filename The output file name
     * @return True if the visualization was generated successfully, false otherwise
     */
    bool visualize_memory_usage(const std::map<std::string, size_t>& memory_usage, const std::string& filename);

    /**
     * @brief Visualizes performance issues.
     *
     * This method generates a visualization of performance issues, showing the
     * execution time of different components and highlighting bottlenecks.
     *
     * @param timings A map of operation names to execution times
     * @param filename The output file name
     * @return True if the visualization was generated successfully, false otherwise
     */
    bool visualize_performance(const std::map<std::string, double>& timings, const std::string& filename);

private:
    /**
     * @brief Constructs a new ErrorVisualizer object.
     */
    ErrorVisualizer() {}

    /**
     * @brief Destructor for the ErrorVisualizer object.
     */
    ~ErrorVisualizer() {}

    // Prevent copying and assignment
    ErrorVisualizer(const ErrorVisualizer&) = delete;
    ErrorVisualizer& operator=(const ErrorVisualizer&) = delete;

    /**
     * @brief Generates a color for a quality value.
     *
     * @param quality The quality value
     * @param min_quality The minimum quality value
     * @param max_quality The maximum quality value
     * @return A color in the format "#RRGGBB"
     */
    std::string quality_to_color(double quality, double min_quality, double max_quality);

    /**
     * @brief Writes an SVG file.
     *
     * @param content The SVG content
     * @param filename The output file name
     * @return True if the file was written successfully, false otherwise
     */
    bool write_svg(const std::string& content, const std::string& filename);

    /**
     * @brief Writes a PNG file.
     *
     * @param content The PNG content
     * @param filename The output file name
     * @return True if the file was written successfully, false otherwise
     */
    bool write_png(const std::vector<unsigned char>& content, const std::string& filename);
};

} // namespace ErrorHandling

// Convenience macro for error visualization
#define QDSIM_VISUALIZE_MESH_QUALITY(mesh, filename, threshold) \
    ErrorHandling::ErrorVisualizer::instance().visualize_mesh_quality(mesh, filename, threshold)

#define QDSIM_VISUALIZE_SOLVER_CONVERGENCE(residuals, filename, tolerance) \
    ErrorHandling::ErrorVisualizer::instance().visualize_solver_convergence(residuals, filename, tolerance)

#define QDSIM_VISUALIZE_ERROR_LOCATION(mesh, error, filename) \
    ErrorHandling::ErrorVisualizer::instance().visualize_error_location(mesh, error, filename)

#endif // QDSIM_ERROR_VISUALIZER_H
