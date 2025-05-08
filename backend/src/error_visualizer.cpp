/**
 * @file error_visualizer.cpp
 * @brief Implementation of the ErrorVisualizer class for QDSim.
 *
 * This file contains the implementation of the ErrorVisualizer class, which provides
 * visualization tools for diagnosing errors in QDSim simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "error_visualizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include "adaptive_mesh.h"

namespace ErrorHandling {

bool ErrorVisualizer::visualize_mesh_quality(const Mesh& mesh, const std::string& filename, double quality_threshold) {
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    
    // Check if mesh is valid
    if (nodes.empty() || elements.empty()) {
        QDSIM_LOG_ERROR("Cannot visualize mesh quality: Mesh is empty");
        return false;
    }
    
    // Compute element qualities
    std::vector<double> qualities;
    qualities.reserve(elements.size());
    
    for (size_t i = 0; i < elements.size(); ++i) {
        double quality = AdaptiveMesh::computeTriangleQuality(mesh, i);
        qualities.push_back(quality);
    }
    
    // Compute quality statistics
    double min_quality = *std::min_element(qualities.begin(), qualities.end());
    double max_quality = *std::max_element(qualities.begin(), qualities.end());
    
    // Compute mesh bounds
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    
    for (const auto& node : nodes) {
        min_x = std::min(min_x, node[0]);
        min_y = std::min(min_y, node[1]);
        max_x = std::max(max_x, node[0]);
        max_y = std::max(max_y, node[1]);
    }
    
    // Add some padding
    double padding = 0.05 * std::max(max_x - min_x, max_y - min_y);
    min_x -= padding;
    min_y -= padding;
    max_x += padding;
    max_y += padding;
    
    // Set up SVG dimensions
    int svg_width = 800;
    int svg_height = 600;
    
    // Create SVG content
    std::ostringstream svg;
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svg << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n";
    svg << "<svg width=\"" << svg_width << "\" height=\"" << svg_height << "\" "
        << "viewBox=\"" << min_x << " " << min_y << " " << (max_x - min_x) << " " << (max_y - min_y) << "\" "
        << "xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n";
    
    // Add title
    svg << "  <title>Mesh Quality Visualization</title>\n";
    
    // Add description
    svg << "  <desc>Visualization of mesh quality for QDSim</desc>\n";
    
    // Add style
    svg << "  <style type=\"text/css\">\n";
    svg << "    .element { stroke: #000000; stroke-width: 0.01; }\n";
    svg << "    .low-quality { stroke: #FF0000; stroke-width: 0.02; }\n";
    svg << "    .text { font-family: Arial; font-size: 0.1px; fill: #000000; }\n";
    svg << "  </style>\n";
    
    // Draw elements
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        double quality = qualities[i];
        
        // Get element nodes
        const auto& n1 = nodes[elem[0]];
        const auto& n2 = nodes[elem[1]];
        const auto& n3 = nodes[elem[2]];
        
        // Compute element color based on quality
        std::string color = quality_to_color(quality, min_quality, max_quality);
        
        // Draw element
        svg << "  <polygon points=\""
            << n1[0] << "," << n1[1] << " "
            << n2[0] << "," << n2[1] << " "
            << n3[0] << "," << n3[1]
            << "\" fill=\"" << color << "\" class=\""
            << (quality < quality_threshold ? "low-quality" : "element")
            << "\" />\n";
    }
    
    // Add legend
    double legend_x = min_x + 0.05 * (max_x - min_x);
    double legend_y = min_y + 0.05 * (max_y - min_y);
    double legend_width = 0.2 * (max_x - min_x);
    double legend_height = 0.05 * (max_y - min_y);
    
    svg << "  <rect x=\"" << legend_x << "\" y=\"" << legend_y << "\" "
        << "width=\"" << legend_width << "\" height=\"" << legend_height << "\" "
        << "fill=\"url(#gradient)\" stroke=\"#000000\" stroke-width=\"0.01\" />\n";
    
    // Add gradient definition
    svg << "  <defs>\n";
    svg << "    <linearGradient id=\"gradient\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"0%\">\n";
    svg << "      <stop offset=\"0%\" style=\"stop-color:" << quality_to_color(min_quality, min_quality, max_quality) << ";stop-opacity:1\" />\n";
    svg << "      <stop offset=\"100%\" style=\"stop-color:" << quality_to_color(max_quality, min_quality, max_quality) << ";stop-opacity:1\" />\n";
    svg << "    </linearGradient>\n";
    svg << "  </defs>\n";
    
    // Add legend labels
    svg << "  <text x=\"" << legend_x << "\" y=\"" << (legend_y + legend_height + 0.03 * (max_y - min_y)) << "\" "
        << "class=\"text\">" << std::fixed << std::setprecision(2) << min_quality << "</text>\n";
    svg << "  <text x=\"" << (legend_x + legend_width) << "\" y=\"" << (legend_y + legend_height + 0.03 * (max_y - min_y)) << "\" "
        << "class=\"text\" text-anchor=\"end\">" << std::fixed << std::setprecision(2) << max_quality << "</text>\n";
    
    // Add title
    svg << "  <text x=\"" << (min_x + 0.5 * (max_x - min_x)) << "\" y=\"" << (min_y + 0.03 * (max_y - min_y)) << "\" "
        << "class=\"text\" text-anchor=\"middle\" font-size=\"0.15px\">Mesh Quality Visualization</text>\n";
    
    // Add quality threshold line
    double threshold_x = legend_x + (quality_threshold - min_quality) / (max_quality - min_quality) * legend_width;
    svg << "  <line x1=\"" << threshold_x << "\" y1=\"" << legend_y << "\" "
        << "x2=\"" << threshold_x << "\" y2=\"" << (legend_y + legend_height) << "\" "
        << "stroke=\"#FF0000\" stroke-width=\"0.01\" stroke-dasharray=\"0.02,0.01\" />\n";
    
    // Add quality threshold label
    svg << "  <text x=\"" << threshold_x << "\" y=\"" << (legend_y - 0.01 * (max_y - min_y)) << "\" "
        << "class=\"text\" text-anchor=\"middle\">Threshold: " << std::fixed << std::setprecision(2) << quality_threshold << "</text>\n";
    
    // Add statistics
    int low_quality_count = std::count_if(qualities.begin(), qualities.end(), 
                                         [quality_threshold](double q) { return q < quality_threshold; });
    double avg_quality = std::accumulate(qualities.begin(), qualities.end(), 0.0) / qualities.size();
    
    svg << "  <text x=\"" << (min_x + 0.05 * (max_x - min_x)) << "\" y=\"" << (max_y - 0.1 * (max_y - min_y)) << "\" "
        << "class=\"text\">Elements: " << elements.size() << "</text>\n";
    svg << "  <text x=\"" << (min_x + 0.05 * (max_x - min_x)) << "\" y=\"" << (max_y - 0.07 * (max_y - min_y)) << "\" "
        << "class=\"text\">Low quality elements: " << low_quality_count << " (" 
        << std::fixed << std::setprecision(1) << (100.0 * low_quality_count / elements.size()) << "%)</text>\n";
    svg << "  <text x=\"" << (min_x + 0.05 * (max_x - min_x)) << "\" y=\"" << (max_y - 0.04 * (max_y - min_y)) << "\" "
        << "class=\"text\">Average quality: " << std::fixed << std::setprecision(3) << avg_quality << "</text>\n";
    
    // Close SVG
    svg << "</svg>\n";
    
    // Write SVG to file
    return write_svg(svg.str(), filename);
}

bool ErrorVisualizer::visualize_solver_convergence(const std::vector<double>& residuals, const std::string& filename, double tolerance) {
    // Check if residuals are valid
    if (residuals.empty()) {
        QDSIM_LOG_ERROR("Cannot visualize solver convergence: No residuals provided");
        return false;
    }
    
    // Set up SVG dimensions
    int svg_width = 800;
    int svg_height = 600;
    
    // Compute residual statistics
    double min_residual = *std::min_element(residuals.begin(), residuals.end());
    double max_residual = *std::max_element(residuals.begin(), residuals.end());
    
    // Use logarithmic scale for residuals
    min_residual = std::max(min_residual, 1e-16);
    max_residual = std::max(max_residual, min_residual * 10.0);
    
    double log_min_residual = std::log10(min_residual);
    double log_max_residual = std::log10(max_residual);
    double log_tolerance = std::log10(tolerance);
    
    // Add some padding
    double padding_x = 0.05 * residuals.size();
    double padding_y = 0.5;
    
    // Compute plot bounds
    double min_x = -padding_x;
    double max_x = residuals.size() + padding_x;
    double min_y = log_min_residual - padding_y;
    double max_y = log_max_residual + padding_y;
    
    // Create SVG content
    std::ostringstream svg;
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svg << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n";
    svg << "<svg width=\"" << svg_width << "\" height=\"" << svg_height << "\" "
        << "viewBox=\"" << min_x << " " << min_y << " " << (max_x - min_x) << " " << (max_y - min_y) << "\" "
        << "xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n";
    
    // Add title
    svg << "  <title>Solver Convergence Visualization</title>\n";
    
    // Add description
    svg << "  <desc>Visualization of solver convergence for QDSim</desc>\n";
    
    // Add style
    svg << "  <style type=\"text/css\">\n";
    svg << "    .axis { stroke: #000000; stroke-width: 0.02; }\n";
    svg << "    .grid { stroke: #CCCCCC; stroke-width: 0.01; stroke-dasharray: 0.1,0.1; }\n";
    svg << "    .residual { stroke: #0000FF; stroke-width: 0.05; fill: none; }\n";
    svg << "    .tolerance { stroke: #FF0000; stroke-width: 0.02; stroke-dasharray: 0.2,0.1; }\n";
    svg << "    .text { font-family: Arial; font-size: 0.3px; fill: #000000; }\n";
    svg << "  </style>\n";
    
    // Draw axes
    svg << "  <line x1=\"0\" y1=\"" << min_y << "\" x2=\"0\" y2=\"" << max_y << "\" class=\"axis\" />\n";
    svg << "  <line x1=\"" << min_x << "\" y1=\"" << log_min_residual << "\" x2=\"" << max_x << "\" y2=\"" << log_min_residual << "\" class=\"axis\" />\n";
    
    // Draw grid lines
    for (int i = static_cast<int>(std::floor(log_min_residual)); i <= static_cast<int>(std::ceil(log_max_residual)); ++i) {
        svg << "  <line x1=\"" << min_x << "\" y1=\"" << i << "\" x2=\"" << max_x << "\" y2=\"" << i << "\" class=\"grid\" />\n";
    }
    
    for (int i = 0; i <= static_cast<int>(residuals.size()); i += 5) {
        svg << "  <line x1=\"" << i << "\" y1=\"" << min_y << "\" x2=\"" << i << "\" y2=\"" << max_y << "\" class=\"grid\" />\n";
    }
    
    // Draw tolerance line
    svg << "  <line x1=\"" << min_x << "\" y1=\"" << log_tolerance << "\" x2=\"" << max_x << "\" y2=\"" << log_tolerance << "\" class=\"tolerance\" />\n";
    
    // Draw residuals
    svg << "  <polyline points=\"";
    for (size_t i = 0; i < residuals.size(); ++i) {
        double log_residual = std::log10(std::max(residuals[i], 1e-16));
        svg << i << "," << log_residual << " ";
    }
    svg << "\" class=\"residual\" />\n";
    
    // Add axis labels
    svg << "  <text x=\"" << (max_x / 2) << "\" y=\"" << (min_y - 0.5) << "\" "
        << "class=\"text\" text-anchor=\"middle\">Iteration</text>\n";
    svg << "  <text x=\"" << (min_x - 1.0) << "\" y=\"" << (min_y + (max_y - min_y) / 2) << "\" "
        << "class=\"text\" text-anchor=\"middle\" transform=\"rotate(-90," << (min_x - 1.0) << "," << (min_y + (max_y - min_y) / 2) << ")\">log10(Residual)</text>\n";
    
    // Add tolerance label
    svg << "  <text x=\"" << (max_x - 2.0) << "\" y=\"" << (log_tolerance - 0.2) << "\" "
        << "class=\"text\">Tolerance: " << std::scientific << std::setprecision(1) << tolerance << "</text>\n";
    
    // Add title
    svg << "  <text x=\"" << (min_x + (max_x - min_x) / 2) << "\" y=\"" << (max_y - 0.5) << "\" "
        << "class=\"text\" text-anchor=\"middle\" font-size=\"0.5px\">Solver Convergence</text>\n";
    
    // Add statistics
    bool converged = residuals.back() < tolerance;
    int iterations = residuals.size();
    
    svg << "  <text x=\"" << (min_x + 1.0) << "\" y=\"" << (max_y - 2.0) << "\" "
        << "class=\"text\">Iterations: " << iterations << "</text>\n";
    svg << "  <text x=\"" << (min_x + 1.0) << "\" y=\"" << (max_y - 1.5) << "\" "
        << "class=\"text\">Final residual: " << std::scientific << std::setprecision(3) << residuals.back() << "</text>\n";
    svg << "  <text x=\"" << (min_x + 1.0) << "\" y=\"" << (max_y - 1.0) << "\" "
        << "class=\"text\">Status: " << (converged ? "Converged" : "Not converged") << "</text>\n";
    
    // Close SVG
    svg << "</svg>\n";
    
    // Write SVG to file
    return write_svg(svg.str(), filename);
}

std::string ErrorVisualizer::quality_to_color(double quality, double min_quality, double max_quality) {
    // Normalize quality to [0, 1]
    double normalized_quality = (quality - min_quality) / (max_quality - min_quality);
    normalized_quality = std::max(0.0, std::min(1.0, normalized_quality));
    
    // Use a color gradient from red (low quality) to green (high quality)
    int r = static_cast<int>(255 * (1.0 - normalized_quality));
    int g = static_cast<int>(255 * normalized_quality);
    int b = 0;
    
    // Convert to hex
    std::ostringstream color;
    color << "#" << std::hex << std::setfill('0') << std::setw(2) << r
          << std::setw(2) << g << std::setw(2) << b;
    
    return color.str();
}

bool ErrorVisualizer::write_svg(const std::string& content, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        QDSIM_LOG_ERROR("Failed to open file for writing: " + filename);
        return false;
    }
    
    out << content;
    out.close();
    
    QDSIM_LOG_INFO("Wrote SVG visualization to: " + filename);
    return true;
}

bool ErrorVisualizer::write_png(const std::vector<unsigned char>& content, const std::string& filename) {
    // This is a placeholder implementation
    // In a real implementation, we would use a library like libpng to write PNG files
    QDSIM_LOG_WARNING("PNG output not implemented yet");
    return false;
}

} // namespace ErrorHandling
