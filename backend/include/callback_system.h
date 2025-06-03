/**
 * @file callback_system.h
 * @brief Callback system for Python callbacks.
 *
 * This file contains the declarations for the callback system used to handle
 * Python callbacks from C++ code.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include <functional>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>

// Forward declaration of pybind11 types
namespace pybind11 {
    class function;
}

/**
 * @class CallbackManager
 * @brief Manages Python callbacks used by C++ code.
 *
 * This class provides thread-safe storage and retrieval of Python callback functions.
 * It ensures proper reference counting and cleanup of Python callbacks when they are
 * no longer needed. The class is implemented as a singleton to provide global access.
 */
class CallbackManager {
public:
    /**
     * @brief Get the singleton instance of the callback manager.
     *
     * @return CallbackManager& The singleton instance.
     */
    static CallbackManager& getInstance();

    /**
     * @brief Set a Python callback function.
     *
     * This method stores a Python callback function with the given name.
     * If a callback with the same name already exists, it is replaced.
     * The callback is stored as a shared_ptr to ensure proper reference counting.
     *
     * @param name The name of the callback.
     * @param callback The Python callback function.
     */
    void setCallback(const std::string& name, const pybind11::function& callback);

    /**
     * @brief Get a Python callback function.
     *
     * This method retrieves a Python callback function with the given name.
     * If no callback with the given name exists, nullptr is returned.
     *
     * @param name The name of the callback.
     * @return std::shared_ptr<pybind11::function> The Python callback function.
     */
    std::shared_ptr<pybind11::function> getCallback(const std::string& name);

    /**
     * @brief Clear all Python callback functions.
     *
     * This method removes all stored callbacks, properly releasing
     * the Python references.
     */
    void clearCallbacks();

    /**
     * @brief Clear a specific Python callback function.
     *
     * This method removes a specific callback, properly releasing
     * the Python reference.
     *
     * @param name The name of the callback to clear.
     */
    void clearCallback(const std::string& name);

    /**
     * @brief Check if a callback with the given name exists.
     *
     * @param name The name of the callback to check.
     * @return bool True if the callback exists, false otherwise.
     */
    bool hasCallback(const std::string& name);

private:
    /**
     * @brief Private constructor to enforce singleton pattern.
     */
    CallbackManager() = default;

    /**
     * @brief Destructor that ensures proper cleanup of Python references.
     */
    ~CallbackManager();

    // Delete copy constructor and assignment operator
    CallbackManager(const CallbackManager&) = delete;
    CallbackManager& operator=(const CallbackManager&) = delete;

    std::mutex mutex_;  ///< Mutex for thread-safe access
    std::unordered_map<std::string, std::shared_ptr<pybind11::function>> callbacks_;  ///< Map of callback names to functions
};

/**
 * @brief Convenience function to set a Python callback function.
 *
 * @param name The name of the callback.
 * @param callback The Python callback function.
 */
void setCallback(const std::string& name, const pybind11::function& callback);

/**
 * @brief Convenience function to get a Python callback function.
 *
 * @param name The name of the callback.
 * @return std::shared_ptr<pybind11::function> The Python callback function.
 */
std::shared_ptr<pybind11::function> getCallback(const std::string& name);

/**
 * @brief Convenience function to clear all Python callback functions.
 */
void clearCallbacks();

/**
 * @brief Convenience function to clear a specific Python callback function.
 *
 * @param name The name of the callback to clear.
 */
void clearCallback(const std::string& name);

/**
 * @brief Convenience function to check if a callback with the given name exists.
 *
 * @param name The name of the callback to check.
 * @return bool True if the callback exists, false otherwise.
 */
bool hasCallback(const std::string& name);

/**
 * @brief Exception class for callback errors.
 *
 * This class represents an exception that is thrown when a callback error occurs.
 * It provides detailed information about the error, including the callback name,
 * the position where the error occurred, and the error message.
 */
class CallbackException : public std::runtime_error {
public:
    /**
     * @brief Construct a new Callback Exception object.
     *
     * @param callback_name The name of the callback where the error occurred.
     * @param x The x-coordinate where the error occurred.
     * @param y The y-coordinate where the error occurred.
     * @param message The error message.
     */
    CallbackException(const std::string& callback_name, double x, double y, const std::string& message);

    /**
     * @brief Construct a new Callback Exception object with a simple message.
     *
     * @param message The error message.
     */
    CallbackException(const std::string& message);

    /**
     * @brief Get the name of the callback where the error occurred.
     *
     * @return const std::string& The callback name.
     */
    const std::string& getCallbackName() const;

    /**
     * @brief Get the x-coordinate where the error occurred.
     *
     * @return double The x-coordinate.
     */
    double getX() const;

    /**
     * @brief Get the y-coordinate where the error occurred.
     *
     * @return double The y-coordinate.
     */
    double getY() const;

    /**
     * @brief Get the error message.
     *
     * @return const std::string& The error message.
     */
    const std::string& getMessage() const;

private:
    /**
     * @brief Format the error message.
     *
     * @param callback_name The name of the callback where the error occurred.
     * @param x The x-coordinate where the error occurred.
     * @param y The y-coordinate where the error occurred.
     * @param message The error message.
     * @return std::string The formatted error message.
     */
    static std::string formatMessage(const std::string& callback_name, double x, double y, const std::string& message);

    std::string callback_name_;
    double x_;
    double y_;
    std::string message_;
};

/**
 * @brief Log an error message.
 *
 * This function logs an error message to the standard error stream.
 * It includes the callback name, the position where the error occurred,
 * and the error message.
 *
 * @param callback_name The name of the callback where the error occurred.
 * @param x The x-coordinate where the error occurred.
 * @param y The y-coordinate where the error occurred.
 * @param message The error message.
 */
void logError(const std::string& callback_name, double x, double y, const std::string& message);
