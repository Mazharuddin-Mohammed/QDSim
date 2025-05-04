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

// Get current timestamp as string
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_tm = std::localtime(&now_c);
    
    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Custom implementation of ErrorLogger::log to include timestamp
void ErrorLogger::log(LogLevel level, const std::string& message, 
                     const std::string& file, int line, 
                     const std::string& function) {
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
    oss << "[" << get_timestamp() << "] [" << level_str << "] " << message;
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
        log_stream_.flush();
    }
}

} // namespace ErrorHandling
