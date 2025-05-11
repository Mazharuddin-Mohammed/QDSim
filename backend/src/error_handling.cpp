/**
 * @file error_handling.cpp
 * @brief Implementation of the error handling framework for QDSim.
 *
 * This file contains the implementation of the error handling framework for QDSim,
 * including error codes, error messages, and utility functions for error handling.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "error_handling.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace ErrorHandling {

// Initialize the error logger
void initialize_error_logger(const std::string& log_file, ErrorLogger::LogLevel log_level) {
    ErrorLogger::instance().set_log_file(log_file);
    ErrorLogger::instance().set_log_level(log_level);

    // Log initialization message
    std::ostringstream oss;
    oss << "QDSim error logger initialized with log level " << static_cast<int>(log_level);
    ErrorLogger::instance().info(oss.str());
}

// The get_timestamp() function is already defined in the header file

// The ErrorLogger::log method is already defined in the header file

} // namespace ErrorHandling
