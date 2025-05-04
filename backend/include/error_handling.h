#pragma once
/**
 * @file error_handling.h
 * @brief Defines the error handling framework for QDSim.
 *
 * This file contains the declaration of the error handling framework for QDSim,
 * including error codes, error messages, and utility functions for error handling.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <string>
#include <stdexcept>
#include <map>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>

/**
 * @namespace ErrorHandling
 * @brief Namespace for error handling utilities.
 */
namespace ErrorHandling {

/**
 * @enum ErrorCode
 * @brief Enumeration of error codes for QDSim.
 */
enum class ErrorCode {
    // General errors
    SUCCESS = 0,
    UNKNOWN_ERROR = 1,
    NOT_IMPLEMENTED = 2,
    INVALID_ARGUMENT = 3,
    OUT_OF_RANGE = 4,
    FILE_NOT_FOUND = 5,
    FILE_FORMAT_ERROR = 6,
    PERMISSION_DENIED = 7,
    
    // Mesh errors
    MESH_CREATION_FAILED = 100,
    MESH_REFINEMENT_FAILED = 101,
    MESH_QUALITY_ERROR = 102,
    MESH_TOPOLOGY_ERROR = 103,
    
    // Matrix assembly errors
    MATRIX_ASSEMBLY_FAILED = 200,
    MATRIX_DIMENSION_MISMATCH = 201,
    MATRIX_NOT_POSITIVE_DEFINITE = 202,
    
    // Solver errors
    SOLVER_INITIALIZATION_FAILED = 300,
    SOLVER_CONVERGENCE_FAILED = 301,
    EIGENVALUE_COMPUTATION_FAILED = 302,
    LINEAR_SYSTEM_SOLVE_FAILED = 303,
    
    // Physics errors
    INVALID_MATERIAL = 400,
    INVALID_POTENTIAL = 401,
    INVALID_BOUNDARY_CONDITION = 402,
    PHYSICAL_PARAMETER_OUT_OF_RANGE = 403,
    
    // Self-consistent solver errors
    SELF_CONSISTENT_DIVERGENCE = 500,
    POISSON_SOLVE_FAILED = 501,
    DRIFT_DIFFUSION_SOLVE_FAILED = 502,
    
    // Memory errors
    MEMORY_ALLOCATION_FAILED = 600,
    OUT_OF_MEMORY = 601,
    
    // Parallel computing errors
    MPI_ERROR = 700,
    CUDA_ERROR = 701,
    THREAD_ERROR = 702,
    
    // I/O errors
    IO_ERROR = 800,
    SERIALIZATION_ERROR = 801,
    DESERIALIZATION_ERROR = 802,
    
    // Visualization errors
    VISUALIZATION_ERROR = 900,
    PLOT_CREATION_FAILED = 901,
    
    // Python binding errors
    BINDING_ERROR = 1000,
    CALLBACK_ERROR = 1001
};

/**
 * @class QDSimException
 * @brief Exception class for QDSim errors.
 *
 * This class provides detailed error information for exceptions thrown by QDSim,
 * including error code, error message, file, line, and function where the error occurred.
 */
class QDSimException : public std::exception {
public:
    /**
     * @brief Constructs a new QDSimException object.
     *
     * @param code The error code
     * @param message The error message
     * @param file The file where the error occurred
     * @param line The line where the error occurred
     * @param function The function where the error occurred
     */
    QDSimException(ErrorCode code, const std::string& message, 
                  const std::string& file = "", int line = 0, 
                  const std::string& function = "")
        : code_(code), message_(message), file_(file), line_(line), function_(function) {
        // Build the full error message
        std::ostringstream oss;
        oss << "Error " << static_cast<int>(code_) << ": " << message_;
        if (!file_.empty()) {
            oss << " [" << file_;
            if (line_ > 0) {
                oss << ":" << line_;
            }
            if (!function_.empty()) {
                oss << ", " << function_;
            }
            oss << "]";
        }
        full_message_ = oss.str();
    }
    
    /**
     * @brief Returns the error message.
     *
     * @return The error message
     */
    const char* what() const noexcept override {
        return full_message_.c_str();
    }
    
    /**
     * @brief Gets the error code.
     *
     * @return The error code
     */
    ErrorCode code() const {
        return code_;
    }
    
    /**
     * @brief Gets the error message.
     *
     * @return The error message
     */
    const std::string& message() const {
        return message_;
    }
    
    /**
     * @brief Gets the file where the error occurred.
     *
     * @return The file name
     */
    const std::string& file() const {
        return file_;
    }
    
    /**
     * @brief Gets the line where the error occurred.
     *
     * @return The line number
     */
    int line() const {
        return line_;
    }
    
    /**
     * @brief Gets the function where the error occurred.
     *
     * @return The function name
     */
    const std::string& function() const {
        return function_;
    }
    
private:
    ErrorCode code_;          ///< The error code
    std::string message_;     ///< The error message
    std::string file_;        ///< The file where the error occurred
    int line_;                ///< The line where the error occurred
    std::string function_;    ///< The function where the error occurred
    std::string full_message_; ///< The full error message
};

/**
 * @brief Gets the error message for an error code.
 *
 * @param code The error code
 * @return The error message
 */
inline std::string get_error_message(ErrorCode code) {
    static const std::map<ErrorCode, std::string> error_messages = {
        // General errors
        {ErrorCode::SUCCESS, "Operation completed successfully"},
        {ErrorCode::UNKNOWN_ERROR, "Unknown error occurred"},
        {ErrorCode::NOT_IMPLEMENTED, "Feature not implemented"},
        {ErrorCode::INVALID_ARGUMENT, "Invalid argument provided"},
        {ErrorCode::OUT_OF_RANGE, "Argument out of valid range"},
        {ErrorCode::FILE_NOT_FOUND, "File not found"},
        {ErrorCode::FILE_FORMAT_ERROR, "Invalid file format"},
        {ErrorCode::PERMISSION_DENIED, "Permission denied"},
        
        // Mesh errors
        {ErrorCode::MESH_CREATION_FAILED, "Failed to create mesh"},
        {ErrorCode::MESH_REFINEMENT_FAILED, "Failed to refine mesh"},
        {ErrorCode::MESH_QUALITY_ERROR, "Mesh quality is too low"},
        {ErrorCode::MESH_TOPOLOGY_ERROR, "Invalid mesh topology"},
        
        // Matrix assembly errors
        {ErrorCode::MATRIX_ASSEMBLY_FAILED, "Failed to assemble matrix"},
        {ErrorCode::MATRIX_DIMENSION_MISMATCH, "Matrix dimensions do not match"},
        {ErrorCode::MATRIX_NOT_POSITIVE_DEFINITE, "Matrix is not positive definite"},
        
        // Solver errors
        {ErrorCode::SOLVER_INITIALIZATION_FAILED, "Failed to initialize solver"},
        {ErrorCode::SOLVER_CONVERGENCE_FAILED, "Solver failed to converge"},
        {ErrorCode::EIGENVALUE_COMPUTATION_FAILED, "Failed to compute eigenvalues"},
        {ErrorCode::LINEAR_SYSTEM_SOLVE_FAILED, "Failed to solve linear system"},
        
        // Physics errors
        {ErrorCode::INVALID_MATERIAL, "Invalid material properties"},
        {ErrorCode::INVALID_POTENTIAL, "Invalid potential function"},
        {ErrorCode::INVALID_BOUNDARY_CONDITION, "Invalid boundary condition"},
        {ErrorCode::PHYSICAL_PARAMETER_OUT_OF_RANGE, "Physical parameter out of valid range"},
        
        // Self-consistent solver errors
        {ErrorCode::SELF_CONSISTENT_DIVERGENCE, "Self-consistent solution diverged"},
        {ErrorCode::POISSON_SOLVE_FAILED, "Failed to solve Poisson equation"},
        {ErrorCode::DRIFT_DIFFUSION_SOLVE_FAILED, "Failed to solve drift-diffusion equations"},
        
        // Memory errors
        {ErrorCode::MEMORY_ALLOCATION_FAILED, "Failed to allocate memory"},
        {ErrorCode::OUT_OF_MEMORY, "Out of memory"},
        
        // Parallel computing errors
        {ErrorCode::MPI_ERROR, "MPI error occurred"},
        {ErrorCode::CUDA_ERROR, "CUDA error occurred"},
        {ErrorCode::THREAD_ERROR, "Thread error occurred"},
        
        // I/O errors
        {ErrorCode::IO_ERROR, "I/O error occurred"},
        {ErrorCode::SERIALIZATION_ERROR, "Failed to serialize data"},
        {ErrorCode::DESERIALIZATION_ERROR, "Failed to deserialize data"},
        
        // Visualization errors
        {ErrorCode::VISUALIZATION_ERROR, "Visualization error occurred"},
        {ErrorCode::PLOT_CREATION_FAILED, "Failed to create plot"},
        
        // Python binding errors
        {ErrorCode::BINDING_ERROR, "Python binding error occurred"},
        {ErrorCode::CALLBACK_ERROR, "Callback function error occurred"}
    };
    
    auto it = error_messages.find(code);
    if (it != error_messages.end()) {
        return it->second;
    } else {
        return "Unknown error code: " + std::to_string(static_cast<int>(code));
    }
}

/**
 * @brief Throws a QDSimException with the given error code and message.
 *
 * @param code The error code
 * @param message The error message (if empty, the default message for the code is used)
 * @param file The file where the error occurred
 * @param line The line where the error occurred
 * @param function The function where the error occurred
 *
 * @throws QDSimException with the specified error information
 */
inline void throw_error(ErrorCode code, const std::string& message = "", 
                       const std::string& file = "", int line = 0, 
                       const std::string& function = "") {
    std::string error_message = message.empty() ? get_error_message(code) : message;
    throw QDSimException(code, error_message, file, line, function);
}

/**
 * @brief Validates that a condition is true, and throws an exception if it's not.
 *
 * @param condition The condition to validate
 * @param code The error code to use if the condition is false
 * @param message The error message (if empty, the default message for the code is used)
 * @param file The file where the validation is performed
 * @param line The line where the validation is performed
 * @param function The function where the validation is performed
 *
 * @throws QDSimException if the condition is false
 */
inline void validate(bool condition, ErrorCode code, const std::string& message = "",
                    const std::string& file = "", int line = 0,
                    const std::string& function = "") {
    if (!condition) {
        throw_error(code, message, file, line, function);
    }
}

/**
 * @brief Validates that a value is within a specified range.
 *
 * @param value The value to validate
 * @param min_value The minimum allowed value
 * @param max_value The maximum allowed value
 * @param code The error code to use if the value is out of range
 * @param message The error message (if empty, a default message is generated)
 * @param file The file where the validation is performed
 * @param line The line where the validation is performed
 * @param function The function where the validation is performed
 *
 * @throws QDSimException if the value is out of range
 */
template <typename T>
inline void validate_range(T value, T min_value, T max_value, ErrorCode code = ErrorCode::OUT_OF_RANGE,
                          const std::string& message = "", const std::string& file = "", int line = 0,
                          const std::string& function = "") {
    if (value < min_value || value > max_value) {
        std::string error_message = message;
        if (error_message.empty()) {
            std::ostringstream oss;
            oss << "Value " << value << " is out of range [" << min_value << ", " << max_value << "]";
            error_message = oss.str();
        }
        throw_error(code, error_message, file, line, function);
    }
}

/**
 * @brief Validates that a pointer is not null.
 *
 * @param ptr The pointer to validate
 * @param code The error code to use if the pointer is null
 * @param message The error message (if empty, a default message is generated)
 * @param file The file where the validation is performed
 * @param line The line where the validation is performed
 * @param function The function where the validation is performed
 *
 * @throws QDSimException if the pointer is null
 */
template <typename T>
inline void validate_not_null(const T* ptr, ErrorCode code = ErrorCode::INVALID_ARGUMENT,
                             const std::string& message = "", const std::string& file = "", int line = 0,
                             const std::string& function = "") {
    if (ptr == nullptr) {
        std::string error_message = message.empty() ? "Null pointer provided" : message;
        throw_error(code, error_message, file, line, function);
    }
}

/**
 * @brief Validates that a file exists and is readable.
 *
 * @param filename The file to validate
 * @param code The error code to use if the file doesn't exist or isn't readable
 * @param message The error message (if empty, a default message is generated)
 * @param file The file where the validation is performed
 * @param line The line where the validation is performed
 * @param function The function where the validation is performed
 *
 * @throws QDSimException if the file doesn't exist or isn't readable
 */
inline void validate_file_exists(const std::string& filename, ErrorCode code = ErrorCode::FILE_NOT_FOUND,
                                const std::string& message = "", const std::string& file = "", int line = 0,
                                const std::string& function = "") {
    std::ifstream f(filename);
    if (!f.good()) {
        std::string error_message = message.empty() ? "File not found or not readable: " + filename : message;
        throw_error(code, error_message, file, line, function);
    }
}

/**
 * @class ErrorLogger
 * @brief Logger for error messages.
 *
 * This class provides logging capabilities for error messages, with support for
 * different log levels and output destinations.
 */
class ErrorLogger {
public:
    /**
     * @enum LogLevel
     * @brief Enumeration of log levels.
     */
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL
    };
    
    /**
     * @brief Gets the singleton instance of the ErrorLogger.
     *
     * @return The singleton instance
     */
    static ErrorLogger& instance() {
        static ErrorLogger instance;
        return instance;
    }
    
    /**
     * @brief Sets the log level.
     *
     * @param level The log level
     */
    void set_log_level(LogLevel level) {
        log_level_ = level;
    }
    
    /**
     * @brief Sets the log file.
     *
     * @param filename The log file name
     */
    void set_log_file(const std::string& filename) {
        log_file_ = filename;
        if (!log_file_.empty()) {
            log_stream_.open(log_file_, std::ios::out | std::ios::app);
        }
    }
    
    /**
     * @brief Logs a message.
     *
     * @param level The log level
     * @param message The message to log
     * @param file The file where the log is generated
     * @param line The line where the log is generated
     * @param function The function where the log is generated
     */
    void log(LogLevel level, const std::string& message, 
            const std::string& file = "", int line = 0, 
            const std::string& function = "") {
        if (level < log_level_) {
            return;
        }
        
        std::string level_str;
        switch (level) {
            case LogLevel::DEBUG:
                level_str = "DEBUG";
                break;
            case LogLevel::INFO:
                level_str = "INFO";
                break;
            case LogLevel::WARNING:
                level_str = "WARNING";
                break;
            case LogLevel::ERROR:
                level_str = "ERROR";
                break;
            case LogLevel::FATAL:
                level_str = "FATAL";
                break;
        }
        
        std::ostringstream oss;
        oss << "[" << level_str << "] " << message;
        if (!file.empty()) {
            oss << " [" << file;
            if (line > 0) {
                oss << ":" << line;
            }
            if (!function.empty()) {
                oss << ", " << function;
            }
            oss << "]";
        }
        
        // Log to console
        if (level >= LogLevel::WARNING) {
            std::cerr << oss.str() << std::endl;
        } else {
            std::cout << oss.str() << std::endl;
        }
        
        // Log to file
        if (log_stream_.is_open()) {
            log_stream_ << oss.str() << std::endl;
        }
    }
    
    /**
     * @brief Logs a debug message.
     *
     * @param message The message to log
     * @param file The file where the log is generated
     * @param line The line where the log is generated
     * @param function The function where the log is generated
     */
    void debug(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "") {
        log(LogLevel::DEBUG, message, file, line, function);
    }
    
    /**
     * @brief Logs an info message.
     *
     * @param message The message to log
     * @param file The file where the log is generated
     * @param line The line where the log is generated
     * @param function The function where the log is generated
     */
    void info(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "") {
        log(LogLevel::INFO, message, file, line, function);
    }
    
    /**
     * @brief Logs a warning message.
     *
     * @param message The message to log
     * @param file The file where the log is generated
     * @param line The line where the log is generated
     * @param function The function where the log is generated
     */
    void warning(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "") {
        log(LogLevel::WARNING, message, file, line, function);
    }
    
    /**
     * @brief Logs an error message.
     *
     * @param message The message to log
     * @param file The file where the log is generated
     * @param line The line where the log is generated
     * @param function The function where the log is generated
     */
    void error(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "") {
        log(LogLevel::ERROR, message, file, line, function);
    }
    
    /**
     * @brief Logs a fatal message.
     *
     * @param message The message to log
     * @param file The file where the log is generated
     * @param line The line where the log is generated
     * @param function The function where the log is generated
     */
    void fatal(const std::string& message, const std::string& file = "", int line = 0, const std::string& function = "") {
        log(LogLevel::FATAL, message, file, line, function);
    }
    
private:
    /**
     * @brief Constructs a new ErrorLogger object.
     */
    ErrorLogger() : log_level_(LogLevel::INFO) {}
    
    /**
     * @brief Destructor for the ErrorLogger object.
     */
    ~ErrorLogger() {
        if (log_stream_.is_open()) {
            log_stream_.close();
        }
    }
    
    LogLevel log_level_;      ///< The current log level
    std::string log_file_;    ///< The log file name
    std::ofstream log_stream_; ///< The log file stream
};

} // namespace ErrorHandling

// Convenience macros for error handling
#define QDSIM_THROW(code, message) \
    ErrorHandling::throw_error(code, message, __FILE__, __LINE__, __FUNCTION__)

#define QDSIM_VALIDATE(condition, code, message) \
    ErrorHandling::validate(condition, code, message, __FILE__, __LINE__, __FUNCTION__)

#define QDSIM_VALIDATE_RANGE(value, min_value, max_value, code, message) \
    ErrorHandling::validate_range(value, min_value, max_value, code, message, __FILE__, __LINE__, __FUNCTION__)

#define QDSIM_VALIDATE_NOT_NULL(ptr, code, message) \
    ErrorHandling::validate_not_null(ptr, code, message, __FILE__, __LINE__, __FUNCTION__)

#define QDSIM_VALIDATE_FILE_EXISTS(filename, code, message) \
    ErrorHandling::validate_file_exists(filename, code, message, __FILE__, __LINE__, __FUNCTION__)

// Convenience macros for logging
#define QDSIM_LOG_DEBUG(message) \
    ErrorHandling::ErrorLogger::instance().debug(message, __FILE__, __LINE__, __FUNCTION__)

#define QDSIM_LOG_INFO(message) \
    ErrorHandling::ErrorLogger::instance().info(message, __FILE__, __LINE__, __FUNCTION__)

#define QDSIM_LOG_WARNING(message) \
    ErrorHandling::ErrorLogger::instance().warning(message, __FILE__, __LINE__, __FUNCTION__)

#define QDSIM_LOG_ERROR(message) \
    ErrorHandling::ErrorLogger::instance().error(message, __FILE__, __LINE__, __FUNCTION__)

#define QDSIM_LOG_FATAL(message) \
    ErrorHandling::ErrorLogger::instance().fatal(message, __FILE__, __LINE__, __FUNCTION__)
