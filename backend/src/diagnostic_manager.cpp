/**
 * @file diagnostic_manager.cpp
 * @brief Implementation of the DiagnosticManager class for QDSim.
 *
 * This file contains the implementation of the DiagnosticManager class, which provides
 * diagnostic tools and utilities for QDSim. The DiagnosticManager helps users
 * diagnose and fix issues in their simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "diagnostic_manager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sys/resource.h>
#include "adaptive_mesh.h"

namespace Diagnostics {

std::map<DiagnosticCategory, std::vector<DiagnosticResult>> DiagnosticManager::run_all_diagnostics() {
    std::map<DiagnosticCategory, std::vector<DiagnosticResult>> results;
    
    // Run diagnostics for each category
    for (int i = static_cast<int>(DiagnosticCategory::MESH); 
         i <= static_cast<int>(DiagnosticCategory::GENERAL); ++i) {
        DiagnosticCategory category = static_cast<DiagnosticCategory>(i);
        results[category] = run_diagnostics(category);
    }
    
    // Generate a report if an output file is specified
    if (!output_file_.empty()) {
        generate_report(results);
    }
    
    return results;
}

std::vector<DiagnosticResult> DiagnosticManager::run_diagnostics(DiagnosticCategory category) {
    std::vector<DiagnosticResult> results;
    
    // Run diagnostics based on the category
    switch (category) {
        case DiagnosticCategory::MESH:
            // Mesh diagnostics would be run here
            // For now, we'll return a placeholder result
            {
                DiagnosticResult result;
                result.success = true;
                result.message = "Mesh diagnostics not implemented yet";
                results.push_back(result);
            }
            break;
            
        case DiagnosticCategory::MATRIX_ASSEMBLY:
            // Matrix assembly diagnostics would be run here
            // For now, we'll return a placeholder result
            {
                DiagnosticResult result;
                result.success = true;
                result.message = "Matrix assembly diagnostics not implemented yet";
                results.push_back(result);
            }
            break;
            
        case DiagnosticCategory::SOLVER:
            // Solver diagnostics would be run here
            // For now, we'll return a placeholder result
            {
                DiagnosticResult result;
                result.success = true;
                result.message = "Solver diagnostics not implemented yet";
                results.push_back(result);
            }
            break;
            
        case DiagnosticCategory::PHYSICS:
            // Physics diagnostics would be run here
            // For now, we'll return a placeholder result
            {
                DiagnosticResult result;
                result.success = true;
                result.message = "Physics diagnostics not implemented yet";
                results.push_back(result);
            }
            break;
            
        case DiagnosticCategory::SELF_CONSISTENT:
            // Self-consistent solver diagnostics would be run here
            // For now, we'll return a placeholder result
            {
                DiagnosticResult result;
                result.success = true;
                result.message = "Self-consistent solver diagnostics not implemented yet";
                results.push_back(result);
            }
            break;
            
        case DiagnosticCategory::MEMORY:
            // Memory diagnostics
            results.push_back(check_memory_usage());
            break;
            
        case DiagnosticCategory::PERFORMANCE:
            // Performance diagnostics would be run here
            // For now, we'll return a placeholder result
            {
                DiagnosticResult result;
                result.success = true;
                result.message = "Performance diagnostics not implemented yet";
                results.push_back(result);
            }
            break;
            
        case DiagnosticCategory::VISUALIZATION:
            // Visualization diagnostics would be run here
            // For now, we'll return a placeholder result
            {
                DiagnosticResult result;
                result.success = true;
                result.message = "Visualization diagnostics not implemented yet";
                results.push_back(result);
            }
            break;
            
        case DiagnosticCategory::GENERAL:
            // General diagnostics would be run here
            // For now, we'll return a placeholder result
            {
                DiagnosticResult result;
                result.success = true;
                result.message = "General diagnostics not implemented yet";
                results.push_back(result);
            }
            break;
    }
    
    return results;
}

DiagnosticResult DiagnosticManager::check_mesh_quality(const Mesh& mesh) {
    DiagnosticResult result;
    result.success = true;
    result.message = "Mesh quality check";
    
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    
    // Check mesh size
    if (nodes.empty() || elements.empty()) {
        result.success = false;
        result.message = "Mesh is empty";
        result.suggestions.push_back("Create a valid mesh before running simulations");
        return result;
    }
    
    // Check element quality
    std::vector<double> qualities;
    qualities.reserve(elements.size());
    
    for (size_t i = 0; i < elements.size(); ++i) {
        double quality = AdaptiveMesh::computeTriangleQuality(mesh, i);
        qualities.push_back(quality);
    }
    
    // Compute quality statistics
    double min_quality = *std::min_element(qualities.begin(), qualities.end());
    double max_quality = *std::max_element(qualities.begin(), qualities.end());
    double avg_quality = std::accumulate(qualities.begin(), qualities.end(), 0.0) / qualities.size();
    
    // Count low-quality elements
    int low_quality_count = std::count_if(qualities.begin(), qualities.end(), 
                                         [](double q) { return q < 0.3; });
    
    // Add details to the result
    result.details.push_back("Number of nodes: " + std::to_string(nodes.size()));
    result.details.push_back("Number of elements: " + std::to_string(elements.size()));
    result.details.push_back("Minimum element quality: " + std::to_string(min_quality));
    result.details.push_back("Maximum element quality: " + std::to_string(max_quality));
    result.details.push_back("Average element quality: " + std::to_string(avg_quality));
    result.details.push_back("Number of low-quality elements: " + std::to_string(low_quality_count));
    
    // Add data for visualization
    Json::Value quality_data;
    quality_data["min_quality"] = min_quality;
    quality_data["max_quality"] = max_quality;
    quality_data["avg_quality"] = avg_quality;
    quality_data["low_quality_count"] = low_quality_count;
    quality_data["qualities"] = Json::Value(Json::arrayValue);
    for (double q : qualities) {
        quality_data["qualities"].append(q);
    }
    result.data["quality"] = quality_data;
    
    // Check if mesh quality is acceptable
    if (min_quality < 0.1) {
        result.success = false;
        result.message = "Mesh contains very low-quality elements";
        result.suggestions.push_back("Refine the mesh to improve element quality");
        result.suggestions.push_back("Use mesh smoothing to improve element quality");
    } else if (low_quality_count > 0.1 * elements.size()) {
        result.success = false;
        result.message = "Mesh contains many low-quality elements";
        result.suggestions.push_back("Refine the mesh to improve element quality");
        result.suggestions.push_back("Use mesh smoothing to improve element quality");
    }
    
    return result;
}

DiagnosticResult DiagnosticManager::check_memory_usage() {
    DiagnosticResult result;
    result.success = true;
    result.message = "Memory usage check";
    
    // Get current memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    
    // Convert to MB
    double memory_usage_mb = usage.ru_maxrss / 1024.0;
    
    // Add details to the result
    result.details.push_back("Current memory usage: " + std::to_string(memory_usage_mb) + " MB");
    
    // Add data for visualization
    Json::Value memory_data;
    memory_data["memory_usage_mb"] = memory_usage_mb;
    result.data["memory"] = memory_data;
    
    // Check if memory usage is acceptable
    if (memory_usage_mb > 1024 * 10) {  // 10 GB
        result.success = false;
        result.message = "Memory usage is very high";
        result.suggestions.push_back("Reduce mesh size or use a coarser mesh");
        result.suggestions.push_back("Use out-of-core algorithms for large problems");
    } else if (memory_usage_mb > 1024 * 4) {  // 4 GB
        result.success = false;
        result.message = "Memory usage is high";
        result.suggestions.push_back("Consider reducing mesh size for better performance");
    }
    
    return result;
}

void DiagnosticManager::generate_report(const std::map<DiagnosticCategory, std::vector<DiagnosticResult>>& results,
                                      const std::string& filename) {
    // Use the specified filename or the default output file
    std::string output_filename = filename.empty() ? output_file_ : filename;
    
    // Open the output file
    std::ofstream out(output_filename);
    if (!out.is_open()) {
        QDSIM_LOG_ERROR("Failed to open diagnostic report file: " + output_filename);
        return;
    }
    
    // Write the report header
    out << "QDSim Diagnostic Report" << std::endl;
    out << "======================" << std::endl;
    out << "Generated on: " << ErrorHandling::get_timestamp() << std::endl;
    out << "Diagnostic level: " << static_cast<int>(diagnostic_level_) << std::endl;
    out << std::endl;
    
    // Write the results for each category
    for (const auto& [category, category_results] : results) {
        // Write the category header
        switch (category) {
            case DiagnosticCategory::MESH:
                out << "Mesh Diagnostics" << std::endl;
                break;
            case DiagnosticCategory::MATRIX_ASSEMBLY:
                out << "Matrix Assembly Diagnostics" << std::endl;
                break;
            case DiagnosticCategory::SOLVER:
                out << "Solver Diagnostics" << std::endl;
                break;
            case DiagnosticCategory::PHYSICS:
                out << "Physics Diagnostics" << std::endl;
                break;
            case DiagnosticCategory::SELF_CONSISTENT:
                out << "Self-Consistent Solver Diagnostics" << std::endl;
                break;
            case DiagnosticCategory::MEMORY:
                out << "Memory Diagnostics" << std::endl;
                break;
            case DiagnosticCategory::PERFORMANCE:
                out << "Performance Diagnostics" << std::endl;
                break;
            case DiagnosticCategory::VISUALIZATION:
                out << "Visualization Diagnostics" << std::endl;
                break;
            case DiagnosticCategory::GENERAL:
                out << "General Diagnostics" << std::endl;
                break;
        }
        out << std::string(30, '-') << std::endl;
        
        // Write the results for this category
        for (const auto& result : category_results) {
            out << "Result: " << (result.success ? "PASS" : "FAIL") << std::endl;
            out << "Message: " << result.message << std::endl;
            
            // Write details
            if (!result.details.empty()) {
                out << "Details:" << std::endl;
                for (const auto& detail : result.details) {
                    out << "  - " << detail << std::endl;
                }
            }
            
            // Write suggestions
            if (!result.suggestions.empty()) {
                out << "Suggestions:" << std::endl;
                for (const auto& suggestion : result.suggestions) {
                    out << "  - " << suggestion << std::endl;
                }
            }
            
            out << std::endl;
        }
        
        out << std::endl;
    }
    
    // Close the output file
    out.close();
    
    QDSIM_LOG_INFO("Diagnostic report generated: " + output_filename);
}

int DiagnosticManager::start_progress(const std::string& operation_name, int total_steps) {
    // Create a new progress tracker
    int progress_id = next_progress_id_++;
    
    ProgressTracker tracker;
    tracker.operation_name = operation_name;
    tracker.total_steps = total_steps;
    tracker.current_step = 0;
    tracker.start_time = std::chrono::steady_clock::now();
    tracker.last_update_time = tracker.start_time;
    tracker.completed = false;
    tracker.success = false;
    
    // Add the tracker to the map
    progress_trackers_[progress_id] = tracker;
    
    // Log the start of the operation
    QDSIM_LOG_INFO("Started operation: " + operation_name + " (ID: " + std::to_string(progress_id) + ")");
    
    return progress_id;
}

void DiagnosticManager::update_progress(int progress_id, int current_step, const std::string& message) {
    // Find the progress tracker
    auto it = progress_trackers_.find(progress_id);
    if (it == progress_trackers_.end()) {
        QDSIM_LOG_WARNING("Progress tracker not found: " + std::to_string(progress_id));
        return;
    }
    
    // Update the tracker
    ProgressTracker& tracker = it->second;
    tracker.current_step = current_step;
    tracker.last_update_time = std::chrono::steady_clock::now();
    
    // Calculate progress percentage
    double progress = static_cast<double>(current_step) / tracker.total_steps * 100.0;
    
    // Calculate elapsed time
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        tracker.last_update_time - tracker.start_time).count();
    
    // Calculate estimated time remaining
    double eta = 0.0;
    if (current_step > 0) {
        eta = elapsed * (tracker.total_steps - current_step) / static_cast<double>(current_step);
    }
    
    // Log the progress
    std::ostringstream oss;
    oss << "Progress: " << tracker.operation_name << " - " 
        << std::fixed << std::setprecision(1) << progress << "% "
        << "(" << current_step << "/" << tracker.total_steps << ") "
        << "Elapsed: " << elapsed << "s "
        << "ETA: " << eta << "s";
    
    if (!message.empty()) {
        oss << " - " << message;
    }
    
    QDSIM_LOG_INFO(oss.str());
}

void DiagnosticManager::complete_progress(int progress_id, bool success, const std::string& message) {
    // Find the progress tracker
    auto it = progress_trackers_.find(progress_id);
    if (it == progress_trackers_.end()) {
        QDSIM_LOG_WARNING("Progress tracker not found: " + std::to_string(progress_id));
        return;
    }
    
    // Update the tracker
    ProgressTracker& tracker = it->second;
    tracker.current_step = tracker.total_steps;
    tracker.completed = true;
    tracker.success = success;
    
    // Calculate elapsed time
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - tracker.start_time).count();
    
    // Log the completion
    std::ostringstream oss;
    oss << "Completed operation: " << tracker.operation_name 
        << " (ID: " << progress_id << ") - "
        << (success ? "SUCCESS" : "FAILURE")
        << " - Elapsed: " << elapsed << "s";
    
    if (!message.empty()) {
        oss << " - " << message;
    }
    
    if (success) {
        QDSIM_LOG_INFO(oss.str());
    } else {
        QDSIM_LOG_ERROR(oss.str());
    }
}

void DiagnosticManager::visualize_diagnostic(const DiagnosticResult& result, const std::string& filename) {
    // This is a placeholder implementation
    // In a real implementation, we would generate visualizations based on the diagnostic result
    
    // For now, we'll just write the result to a JSON file
    Json::Value root;
    root["success"] = result.success;
    root["message"] = result.message;
    
    // Add details
    root["details"] = Json::Value(Json::arrayValue);
    for (const auto& detail : result.details) {
        root["details"].append(detail);
    }
    
    // Add suggestions
    root["suggestions"] = Json::Value(Json::arrayValue);
    for (const auto& suggestion : result.suggestions) {
        root["suggestions"].append(suggestion);
    }
    
    // Add data
    root["data"] = result.data;
    
    // Write to file
    std::ofstream out(filename);
    if (out.is_open()) {
        out << root.toStyledString();
        out.close();
        QDSIM_LOG_INFO("Diagnostic visualization saved to: " + filename);
    } else {
        QDSIM_LOG_ERROR("Failed to save diagnostic visualization to: " + filename);
    }
}

} // namespace Diagnostics
