#pragma once
/**
 * @file input_validation.h
 * @brief Defines utilities for input validation in QDSim.
 *
 * This file contains utilities for validating input parameters in QDSim,
 * including range checking, type checking, and format validation.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "error_handling.h"
#include "physical_constants.h"
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <regex>
#include <fstream>
#include <limits>
#include <type_traits>

/**
 * @namespace InputValidation
 * @brief Namespace for input validation utilities.
 */
namespace InputValidation {

/**
 * @brief Validates that a value is within a specified range.
 *
 * @tparam T The type of the value
 * @param value The value to validate
 * @param min_value The minimum allowed value
 * @param max_value The maximum allowed value
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the value is out of range
 */
template <typename T>
inline void validate_range(T value, T min_value, T max_value, const std::string& parameter_name) {
    if (value < min_value || value > max_value) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' with value " << value 
            << " is out of range [" << min_value << ", " << max_value << "]";
        QDSIM_THROW(ErrorHandling::ErrorCode::OUT_OF_RANGE, oss.str());
    }
}

/**
 * @brief Validates that a value is positive.
 *
 * @tparam T The type of the value
 * @param value The value to validate
 * @param parameter_name The name of the parameter (for error messages)
 * @param allow_zero Whether to allow zero values
 *
 * @throws QDSimException if the value is not positive
 */
template <typename T>
inline void validate_positive(T value, const std::string& parameter_name, bool allow_zero = false) {
    if ((allow_zero && value < T(0)) || (!allow_zero && value <= T(0))) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' with value " << value 
            << " must be " << (allow_zero ? "non-negative" : "positive");
        QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
    }
}

/**
 * @brief Validates that a value is not NaN or infinity.
 *
 * @tparam T The type of the value
 * @param value The value to validate
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the value is NaN or infinity
 */
template <typename T>
inline void validate_finite(T value, const std::string& parameter_name) {
    if (!std::isfinite(value)) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' with value " << value 
            << " must be finite (not NaN or infinity)";
        QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
    }
}

/**
 * @brief Validates that a container is not empty.
 *
 * @tparam Container The type of the container
 * @param container The container to validate
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the container is empty
 */
template <typename Container>
inline void validate_not_empty(const Container& container, const std::string& parameter_name) {
    if (container.empty()) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' must not be empty";
        QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
    }
}

/**
 * @brief Validates that a container has a specific size.
 *
 * @tparam Container The type of the container
 * @param container The container to validate
 * @param expected_size The expected size of the container
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the container does not have the expected size
 */
template <typename Container>
inline void validate_size(const Container& container, size_t expected_size, const std::string& parameter_name) {
    if (container.size() != expected_size) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' has size " << container.size() 
            << " but expected size " << expected_size;
        QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
    }
}

/**
 * @brief Validates that a string matches a regular expression.
 *
 * @param value The string to validate
 * @param regex The regular expression to match
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the string does not match the regular expression
 */
inline void validate_regex(const std::string& value, const std::string& regex, const std::string& parameter_name) {
    std::regex pattern(regex);
    if (!std::regex_match(value, pattern)) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' with value '" << value 
            << "' does not match the required format";
        QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
    }
}

/**
 * @brief Validates that a file exists and is readable.
 *
 * @param filename The file to validate
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the file does not exist or is not readable
 */
inline void validate_file_exists(const std::string& filename, const std::string& parameter_name) {
    std::ifstream file(filename);
    if (!file.good()) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' with value '" << filename 
            << "' is not a valid file or cannot be read";
        QDSIM_THROW(ErrorHandling::ErrorCode::FILE_NOT_FOUND, oss.str());
    }
}

/**
 * @brief Validates that a file has a specific extension.
 *
 * @param filename The file to validate
 * @param extension The expected extension (without the dot)
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the file does not have the expected extension
 */
inline void validate_file_extension(const std::string& filename, const std::string& extension, const std::string& parameter_name) {
    std::string file_extension;
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        file_extension = filename.substr(dot_pos + 1);
    }
    
    if (file_extension != extension) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' with value '" << filename 
            << "' does not have the expected extension '" << extension << "'";
        QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
    }
}

/**
 * @brief Validates that a directory exists and is writable.
 *
 * @param dirname The directory to validate
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the directory does not exist or is not writable
 */
inline void validate_directory_exists(const std::string& dirname, const std::string& parameter_name) {
    // Create a temporary file in the directory to check if it's writable
    std::string temp_filename = dirname + "/qdsim_test_file.tmp";
    std::ofstream file(temp_filename);
    if (!file.good()) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' with value '" << dirname 
            << "' is not a valid directory or cannot be written to";
        QDSIM_THROW(ErrorHandling::ErrorCode::PERMISSION_DENIED, oss.str());
    }
    file.close();
    std::remove(temp_filename.c_str());
}

/**
 * @brief Validates that a function is not null.
 *
 * @tparam Func The type of the function
 * @param func The function to validate
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the function is null
 */
template <typename Func>
inline void validate_function_not_null(const Func& func, const std::string& parameter_name) {
    if (!func) {
        std::ostringstream oss;
        oss << "Parameter '" << parameter_name << "' is a null function";
        QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
    }
}

/**
 * @brief Validates that a potential function returns physically reasonable values.
 *
 * @param potential The potential function to validate
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the potential function returns unreasonable values
 */
inline void validate_potential_function(std::function<double(double, double)> potential, const std::string& parameter_name) {
    // Validate that the function is not null
    validate_function_not_null(potential, parameter_name);
    
    // Test the function at a few points
    std::vector<std::pair<double, double>> test_points = {
        {0.0, 0.0},
        {10.0, 0.0},
        {0.0, 10.0},
        {-10.0, 0.0},
        {0.0, -10.0}
    };
    
    for (const auto& point : test_points) {
        double x = point.first;
        double y = point.second;
        
        try {
            double value = potential(x, y);
            
            // Check if the value is finite
            if (!std::isfinite(value)) {
                std::ostringstream oss;
                oss << "Potential function '" << parameter_name << "' returned non-finite value " 
                    << value << " at point (" << x << ", " << y << ")";
                QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_POTENTIAL, oss.str());
            }
            
            // Check if the value is within a reasonable range
            // Convert from J to eV for comparison
            double value_eV = value / PhysicalConstants::ELECTRON_CHARGE;
            if (value_eV < PhysicalConstants::TypicalRanges::MIN_POTENTIAL || 
                value_eV > PhysicalConstants::TypicalRanges::MAX_POTENTIAL) {
                std::ostringstream oss;
                oss << "Potential function '" << parameter_name << "' returned value " 
                    << value_eV << " eV at point (" << x << ", " << y << "), which is outside the reasonable range ["
                    << PhysicalConstants::TypicalRanges::MIN_POTENTIAL << ", " 
                    << PhysicalConstants::TypicalRanges::MAX_POTENTIAL << "] eV";
                QDSIM_LOG_WARNING(oss.str());
                // Don't throw an exception, just log a warning
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "Potential function '" << parameter_name << "' threw an exception at point (" 
                << x << ", " << y << "): " << e.what();
            QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_POTENTIAL, oss.str());
        }
    }
}

/**
 * @brief Validates that an effective mass function returns physically reasonable values.
 *
 * @param m_star The effective mass function to validate
 * @param parameter_name The name of the parameter (for error messages)
 *
 * @throws QDSimException if the effective mass function returns unreasonable values
 */
inline void validate_effective_mass_function(std::function<double(double, double)> m_star, const std::string& parameter_name) {
    // Validate that the function is not null
    validate_function_not_null(m_star, parameter_name);
    
    // Test the function at a few points
    std::vector<std::pair<double, double>> test_points = {
        {0.0, 0.0},
        {10.0, 0.0},
        {0.0, 10.0},
        {-10.0, 0.0},
        {0.0, -10.0}
    };
    
    for (const auto& point : test_points) {
        double x = point.first;
        double y = point.second;
        
        try {
            double value = m_star(x, y);
            
            // Check if the value is finite
            if (!std::isfinite(value)) {
                std::ostringstream oss;
                oss << "Effective mass function '" << parameter_name << "' returned non-finite value " 
                    << value << " at point (" << x << ", " << y << ")";
                QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
            }
            
            // Check if the value is positive
            if (value <= 0.0) {
                std::ostringstream oss;
                oss << "Effective mass function '" << parameter_name << "' returned non-positive value " 
                    << value << " at point (" << x << ", " << y << ")";
                QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
            }
            
            // Check if the value is within a reasonable range
            if (value < PhysicalConstants::TypicalRanges::MIN_EFFECTIVE_MASS || 
                value > PhysicalConstants::TypicalRanges::MAX_EFFECTIVE_MASS) {
                std::ostringstream oss;
                oss << "Effective mass function '" << parameter_name << "' returned value " 
                    << value << " at point (" << x << ", " << y << "), which is outside the reasonable range ["
                    << PhysicalConstants::TypicalRanges::MIN_EFFECTIVE_MASS << ", " 
                    << PhysicalConstants::TypicalRanges::MAX_EFFECTIVE_MASS << "]";
                QDSIM_LOG_WARNING(oss.str());
                // Don't throw an exception, just log a warning
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "Effective mass function '" << parameter_name << "' threw an exception at point (" 
                << x << ", " << y << "): " << e.what();
            QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_ARGUMENT, oss.str());
        }
    }
}

/**
 * @brief Validates mesh parameters.
 *
 * @param Lx The x-dimension of the mesh
 * @param Ly The y-dimension of the mesh
 * @param nx The number of mesh points in the x-direction
 * @param ny The number of mesh points in the y-direction
 *
 * @throws QDSimException if the mesh parameters are invalid
 */
inline void validate_mesh_parameters(double Lx, double Ly, int nx, int ny) {
    // Validate that the dimensions are positive
    validate_positive(Lx, "Lx");
    validate_positive(Ly, "Ly");
    
    // Validate that the number of mesh points is reasonable
    validate_range(nx, 3, 10000, "nx");
    validate_range(ny, 3, 10000, "ny");
    
    // Validate that the mesh spacing is reasonable
    double dx = Lx / (nx - 1);
    double dy = Ly / (ny - 1);
    
    if (dx < 0.01 || dy < 0.01) {
        std::ostringstream oss;
        oss << "Mesh spacing is too small: dx = " << dx << " nm, dy = " << dy << " nm";
        QDSIM_LOG_WARNING(oss.str());
    }
    
    if (dx > 10.0 || dy > 10.0) {
        std::ostringstream oss;
        oss << "Mesh spacing is too large: dx = " << dx << " nm, dy = " << dy << " nm";
        QDSIM_LOG_WARNING(oss.str());
    }
}

/**
 * @brief Validates solver parameters.
 *
 * @param num_states The number of eigenstates to compute
 * @param tolerance The convergence tolerance
 * @param max_iterations The maximum number of iterations
 *
 * @throws QDSimException if the solver parameters are invalid
 */
inline void validate_solver_parameters(int num_states, double tolerance, int max_iterations) {
    // Validate that the number of states is positive
    validate_range(num_states, 1, 1000, "num_states");
    
    // Validate that the tolerance is positive and reasonable
    validate_positive(tolerance, "tolerance");
    validate_range(tolerance, 1e-15, 1e-1, "tolerance");
    
    // Validate that the maximum number of iterations is positive and reasonable
    validate_range(max_iterations, 1, 100000, "max_iterations");
}

/**
 * @brief Validates material parameters.
 *
 * @param material_name The name of the material
 * @param m_e The electron effective mass
 * @param m_h The hole effective mass
 * @param E_g The bandgap
 * @param epsilon_r The dielectric constant
 *
 * @throws QDSimException if the material parameters are invalid
 */
inline void validate_material_parameters(const std::string& material_name, double m_e, double m_h, double E_g, double epsilon_r) {
    // Validate that the material name is not empty
    if (material_name.empty()) {
        QDSIM_THROW(ErrorHandling::ErrorCode::INVALID_MATERIAL, "Material name cannot be empty");
    }
    
    // Validate that the effective masses are positive and reasonable
    validate_positive(m_e, "m_e");
    validate_positive(m_h, "m_h");
    validate_range(m_e, PhysicalConstants::TypicalRanges::MIN_EFFECTIVE_MASS, 
                  PhysicalConstants::TypicalRanges::MAX_EFFECTIVE_MASS, "m_e");
    validate_range(m_h, PhysicalConstants::TypicalRanges::MIN_EFFECTIVE_MASS, 
                  PhysicalConstants::TypicalRanges::MAX_EFFECTIVE_MASS, "m_h");
    
    // Validate that the bandgap is non-negative and reasonable
    validate_positive(E_g, "E_g", true);
    validate_range(E_g, 0.0, 10.0, "E_g");
    
    // Validate that the dielectric constant is positive and reasonable
    validate_positive(epsilon_r, "epsilon_r");
    validate_range(epsilon_r, 1.0, 100.0, "epsilon_r");
}

} // namespace InputValidation

// Convenience macros for input validation
#define QDSIM_INPUT_VALIDATE_RANGE(value, min_value, max_value, parameter_name) \
    InputValidation::validate_range(value, min_value, max_value, parameter_name)

#define QDSIM_VALIDATE_POSITIVE(value, parameter_name, allow_zero) \
    InputValidation::validate_positive(value, parameter_name, allow_zero)

#define QDSIM_VALIDATE_FINITE(value, parameter_name) \
    InputValidation::validate_finite(value, parameter_name)

#define QDSIM_VALIDATE_NOT_EMPTY(container, parameter_name) \
    InputValidation::validate_not_empty(container, parameter_name)

#define QDSIM_VALIDATE_SIZE(container, expected_size, parameter_name) \
    InputValidation::validate_size(container, expected_size, parameter_name)

#define QDSIM_VALIDATE_REGEX(value, regex, parameter_name) \
    InputValidation::validate_regex(value, regex, parameter_name)

#define QDSIM_INPUT_VALIDATE_FILE_EXISTS(filename, parameter_name) \
    InputValidation::validate_file_exists(filename, parameter_name)

#define QDSIM_VALIDATE_FILE_EXTENSION(filename, extension, parameter_name) \
    InputValidation::validate_file_extension(filename, extension, parameter_name)

#define QDSIM_VALIDATE_DIRECTORY_EXISTS(dirname, parameter_name) \
    InputValidation::validate_directory_exists(dirname, parameter_name)

#define QDSIM_VALIDATE_FUNCTION_NOT_NULL(func, parameter_name) \
    InputValidation::validate_function_not_null(func, parameter_name)

#define QDSIM_VALIDATE_POTENTIAL_FUNCTION(potential, parameter_name) \
    InputValidation::validate_potential_function(potential, parameter_name)

#define QDSIM_VALIDATE_EFFECTIVE_MASS_FUNCTION(m_star, parameter_name) \
    InputValidation::validate_effective_mass_function(m_star, parameter_name)

#define QDSIM_VALIDATE_MESH_PARAMETERS(Lx, Ly, nx, ny) \
    InputValidation::validate_mesh_parameters(Lx, Ly, nx, ny)

#define QDSIM_VALIDATE_SOLVER_PARAMETERS(num_states, tolerance, max_iterations) \
    InputValidation::validate_solver_parameters(num_states, tolerance, max_iterations)

#define QDSIM_VALIDATE_MATERIAL_PARAMETERS(material_name, m_e, m_h, E_g, epsilon_r) \
    InputValidation::validate_material_parameters(material_name, m_e, m_h, E_g, epsilon_r)
