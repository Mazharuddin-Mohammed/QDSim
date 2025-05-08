/**
 * @file diagnostic_manager.h
 * @brief Declaration of the DiagnosticManager class for QDSim.
 *
 * This file contains the declaration of the DiagnosticManager class, which provides
 * diagnostic tools and utilities for QDSim. The DiagnosticManager helps users
 * diagnose and fix issues in their simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#ifndef QDSIM_DIAGNOSTIC_MANAGER_H
#define QDSIM_DIAGNOSTIC_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <chrono>
#include <fstream>
#include <json/json.h>
#include "error_handling.h"
#include "mesh.h"

namespace Diagnostics {

/**
 * @enum DiagnosticLevel
 * @brief Enumeration of diagnostic levels for QDSim.
 */
enum class DiagnosticLevel {
    BASIC,      ///< Basic diagnostics (minimal information)
    STANDARD,   ///< Standard diagnostics (moderate information)
    DETAILED,   ///< Detailed diagnostics (comprehensive information)
    EXPERT      ///< Expert diagnostics (all available information)
};

/**
 * @enum DiagnosticCategory
 * @brief Enumeration of diagnostic categories for QDSim.
 */
enum class DiagnosticCategory {
    MESH,               ///< Mesh-related diagnostics
    MATRIX_ASSEMBLY,    ///< Matrix assembly diagnostics
    SOLVER,             ///< Solver diagnostics
    PHYSICS,            ///< Physics diagnostics
    SELF_CONSISTENT,    ///< Self-consistent solver diagnostics
    MEMORY,             ///< Memory diagnostics
    PERFORMANCE,        ///< Performance diagnostics
    VISUALIZATION,      ///< Visualization diagnostics
    GENERAL             ///< General diagnostics
};

/**
 * @struct DiagnosticResult
 * @brief Structure to hold the result of a diagnostic check.
 */
struct DiagnosticResult {
    bool success;                   ///< Whether the diagnostic check passed
    std::string message;            ///< Message describing the result
    std::vector<std::string> details; ///< Detailed information about the result
    std::vector<std::string> suggestions; ///< Suggestions for fixing issues
    Json::Value data;               ///< Additional data in JSON format
};

/**
 * @class DiagnosticManager
 * @brief Class for managing diagnostics in QDSim.
 *
 * The DiagnosticManager class provides tools and utilities for diagnosing
 * issues in QDSim simulations. It includes methods for checking mesh quality,
 * matrix properties, solver convergence, and more.
 */
class DiagnosticManager {
public:
    /**
     * @brief Gets the singleton instance of the DiagnosticManager.
     *
     * @return The singleton instance
     */
    static DiagnosticManager& instance() {
        static DiagnosticManager instance;
        return instance;
    }

    /**
     * @brief Sets the diagnostic level.
     *
     * @param level The diagnostic level
     */
    void set_diagnostic_level(DiagnosticLevel level) {
        diagnostic_level_ = level;
    }

    /**
     * @brief Gets the diagnostic level.
     *
     * @return The diagnostic level
     */
    DiagnosticLevel get_diagnostic_level() const {
        return diagnostic_level_;
    }

    /**
     * @brief Sets the output file for diagnostic reports.
     *
     * @param filename The output file name
     */
    void set_output_file(const std::string& filename) {
        output_file_ = filename;
    }

    /**
     * @brief Runs all diagnostic checks.
     *
     * @return A map of diagnostic results by category
     */
    std::map<DiagnosticCategory, std::vector<DiagnosticResult>> run_all_diagnostics();

    /**
     * @brief Runs diagnostic checks for a specific category.
     *
     * @param category The diagnostic category
     * @return A vector of diagnostic results
     */
    std::vector<DiagnosticResult> run_diagnostics(DiagnosticCategory category);

    /**
     * @brief Checks mesh quality.
     *
     * @param mesh The mesh to check
     * @return The diagnostic result
     */
    DiagnosticResult check_mesh_quality(const Mesh& mesh);

    /**
     * @brief Checks matrix properties.
     *
     * @param H The Hamiltonian matrix
     * @param M The mass matrix
     * @return The diagnostic result
     */
    DiagnosticResult check_matrix_properties(const Eigen::SparseMatrix<std::complex<double>>& H,
                                           const Eigen::SparseMatrix<std::complex<double>>& M);

    /**
     * @brief Checks solver convergence.
     *
     * @param residuals The residuals from the solver
     * @param tolerance The convergence tolerance
     * @return The diagnostic result
     */
    DiagnosticResult check_solver_convergence(const std::vector<double>& residuals, double tolerance);

    /**
     * @brief Checks physical parameters.
     *
     * @param m_star The effective mass function
     * @param V The potential function
     * @param mesh The mesh
     * @return The diagnostic result
     */
    DiagnosticResult check_physical_parameters(std::function<double(double, double)> m_star,
                                             std::function<double(double, double)> V,
                                             const Mesh& mesh);

    /**
     * @brief Checks memory usage.
     *
     * @return The diagnostic result
     */
    DiagnosticResult check_memory_usage();

    /**
     * @brief Checks performance.
     *
     * @param timings A map of operation names to execution times
     * @return The diagnostic result
     */
    DiagnosticResult check_performance(const std::map<std::string, double>& timings);

    /**
     * @brief Generates a diagnostic report.
     *
     * @param results The diagnostic results
     * @param filename The output file name (if empty, uses the default output file)
     */
    void generate_report(const std::map<DiagnosticCategory, std::vector<DiagnosticResult>>& results,
                        const std::string& filename = "");

    /**
     * @brief Starts a progress tracker for a long-running operation.
     *
     * @param operation_name The name of the operation
     * @param total_steps The total number of steps in the operation
     * @return A unique identifier for the progress tracker
     */
    int start_progress(const std::string& operation_name, int total_steps);

    /**
     * @brief Updates the progress of a long-running operation.
     *
     * @param progress_id The progress tracker identifier
     * @param current_step The current step
     * @param message An optional message describing the current step
     */
    void update_progress(int progress_id, int current_step, const std::string& message = "");

    /**
     * @brief Completes a progress tracker.
     *
     * @param progress_id The progress tracker identifier
     * @param success Whether the operation completed successfully
     * @param message An optional message describing the completion status
     */
    void complete_progress(int progress_id, bool success, const std::string& message = "");

    /**
     * @brief Visualizes a diagnostic result.
     *
     * @param result The diagnostic result
     * @param filename The output file name
     */
    void visualize_diagnostic(const DiagnosticResult& result, const std::string& filename);

private:
    /**
     * @brief Constructs a new DiagnosticManager object.
     */
    DiagnosticManager() : diagnostic_level_(DiagnosticLevel::STANDARD), next_progress_id_(0) {}

    /**
     * @brief Destructor for the DiagnosticManager object.
     */
    ~DiagnosticManager() {}

    // Prevent copying and assignment
    DiagnosticManager(const DiagnosticManager&) = delete;
    DiagnosticManager& operator=(const DiagnosticManager&) = delete;

    DiagnosticLevel diagnostic_level_;  ///< The current diagnostic level
    std::string output_file_;           ///< The output file for diagnostic reports

    // Progress tracking
    struct ProgressTracker {
        std::string operation_name;
        int total_steps;
        int current_step;
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point last_update_time;
        bool completed;
        bool success;
    };

    std::map<int, ProgressTracker> progress_trackers_;  ///< Map of progress trackers
    int next_progress_id_;                             ///< Next progress tracker ID
};

} // namespace Diagnostics

// Convenience macros for diagnostics
#define QDSIM_DIAGNOSTIC_MANAGER Diagnostics::DiagnosticManager::instance()
#define QDSIM_RUN_DIAGNOSTICS(category) QDSIM_DIAGNOSTIC_MANAGER.run_diagnostics(category)
#define QDSIM_RUN_ALL_DIAGNOSTICS() QDSIM_DIAGNOSTIC_MANAGER.run_all_diagnostics()
#define QDSIM_START_PROGRESS(operation_name, total_steps) QDSIM_DIAGNOSTIC_MANAGER.start_progress(operation_name, total_steps)
#define QDSIM_UPDATE_PROGRESS(progress_id, current_step, message) QDSIM_DIAGNOSTIC_MANAGER.update_progress(progress_id, current_step, message)
#define QDSIM_COMPLETE_PROGRESS(progress_id, success, message) QDSIM_DIAGNOSTIC_MANAGER.complete_progress(progress_id, success, message)

#endif // QDSIM_DIAGNOSTIC_MANAGER_H
