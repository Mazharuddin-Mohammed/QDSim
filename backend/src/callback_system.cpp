/**
 * @file callback_system.cpp
 * @brief Implementation of the callback system for Python callbacks.
 *
 * This file contains the implementation of the callback system used to handle
 * Python callbacks from C++ code.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "callback_system.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <iostream>
#include <sstream>

/**
 * @brief Get the singleton instance of the callback manager.
 *
 * @return CallbackManager& The singleton instance.
 */
CallbackManager& CallbackManager::getInstance() {
    static CallbackManager instance;
    return instance;
}

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
void CallbackManager::setCallback(const std::string& name, const pybind11::function& callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Acquire GIL before manipulating Python objects
    pybind11::gil_scoped_acquire gil;
    callbacks_[name] = std::make_shared<pybind11::function>(callback);
}

/**
 * @brief Get a Python callback function.
 *
 * This method retrieves a Python callback function with the given name.
 * If no callback with the given name exists, nullptr is returned.
 *
 * @param name The name of the callback.
 * @return std::shared_ptr<pybind11::function> The Python callback function.
 */
std::shared_ptr<pybind11::function> CallbackManager::getCallback(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = callbacks_.find(name);
    if (it != callbacks_.end()) {
        return it->second;
    }
    return nullptr;
}

/**
 * @brief Clear all Python callback functions.
 *
 * This method removes all stored callbacks, properly releasing
 * the Python references.
 */
void CallbackManager::clearCallbacks() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Only acquire GIL if Python interpreter is still running
    try {
        if (Py_IsInitialized()) {
            // Acquire GIL before manipulating Python objects
            pybind11::gil_scoped_acquire gil;
            callbacks_.clear();
        } else {
            // Python interpreter is shutting down, just clear without GIL
            callbacks_.clear();
        }
    } catch (...) {
        // If anything goes wrong, just clear the map
        try {
            callbacks_.clear();
        } catch (...) {
            // Ignore any errors during cleanup
        }
    }
}

/**
 * @brief Clear a specific Python callback function.
 *
 * This method removes a specific callback, properly releasing
 * the Python reference.
 *
 * @param name The name of the callback to clear.
 */
void CallbackManager::clearCallback(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Only acquire GIL if Python interpreter is still running
    try {
        if (Py_IsInitialized()) {
            // Acquire GIL before manipulating Python objects
            pybind11::gil_scoped_acquire gil;
            auto it = callbacks_.find(name);
            if (it != callbacks_.end()) {
                callbacks_.erase(it);
            }
        } else {
            // Python interpreter is shutting down, just clear without GIL
            auto it = callbacks_.find(name);
            if (it != callbacks_.end()) {
                callbacks_.erase(it);
            }
        }
    } catch (...) {
        // If anything goes wrong, try to clear without GIL
        try {
            auto it = callbacks_.find(name);
            if (it != callbacks_.end()) {
                callbacks_.erase(it);
            }
        } catch (...) {
            // Ignore any errors during cleanup
        }
    }
}

/**
 * @brief Check if a callback with the given name exists.
 *
 * @param name The name of the callback to check.
 * @return bool True if the callback exists, false otherwise.
 */
bool CallbackManager::hasCallback(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return callbacks_.find(name) != callbacks_.end();
}

/**
 * @brief Destructor that ensures proper cleanup of Python references.
 */
CallbackManager::~CallbackManager() {
    // Only try to acquire GIL if Python interpreter is still running
    try {
        if (Py_IsInitialized()) {
            // Acquire GIL before manipulating Python objects
            pybind11::gil_scoped_acquire gil;
            callbacks_.clear();
        } else {
            // Python interpreter is shutting down, just clear the map without GIL
            // The Python objects will be cleaned up by the interpreter
            callbacks_.clear();
        }
    } catch (...) {
        // If anything goes wrong during cleanup, just clear the map
        // This prevents crashes during interpreter shutdown
        try {
            callbacks_.clear();
        } catch (...) {
            // Ignore any errors during final cleanup
        }
    }
}

/**
 * @brief Convenience function to set a Python callback function.
 *
 * @param name The name of the callback.
 * @param callback The Python callback function.
 */
void setCallback(const std::string& name, const pybind11::function& callback) {
    CallbackManager::getInstance().setCallback(name, callback);
}

/**
 * @brief Convenience function to get a Python callback function.
 *
 * @param name The name of the callback.
 * @return std::shared_ptr<pybind11::function> The Python callback function.
 */
std::shared_ptr<pybind11::function> getCallback(const std::string& name) {
    return CallbackManager::getInstance().getCallback(name);
}

/**
 * @brief Convenience function to clear all Python callback functions.
 */
void clearCallbacks() {
    CallbackManager::getInstance().clearCallbacks();
}

/**
 * @brief Convenience function to clear a specific Python callback function.
 *
 * @param name The name of the callback to clear.
 */
void clearCallback(const std::string& name) {
    CallbackManager::getInstance().clearCallback(name);
}

/**
 * @brief Convenience function to check if a callback with the given name exists.
 *
 * @param name The name of the callback to check.
 * @return bool True if the callback exists, false otherwise.
 */
bool hasCallback(const std::string& name) {
    return CallbackManager::getInstance().hasCallback(name);
}

/**
 * @brief Construct a new Callback Exception object.
 *
 * @param callback_name The name of the callback where the error occurred.
 * @param x The x-coordinate where the error occurred.
 * @param y The y-coordinate where the error occurred.
 * @param message The error message.
 */
CallbackException::CallbackException(const std::string& callback_name, double x, double y, const std::string& message)
    : std::runtime_error(formatMessage(callback_name, x, y, message)),
      callback_name_(callback_name), x_(x), y_(y), message_(message) {}

/**
 * @brief Construct a new Callback Exception object with a simple message.
 *
 * @param message The error message.
 */
CallbackException::CallbackException(const std::string& message)
    : std::runtime_error(message),
      callback_name_("unknown"), x_(0.0), y_(0.0), message_(message) {}

/**
 * @brief Get the name of the callback where the error occurred.
 *
 * @return const std::string& The callback name.
 */
const std::string& CallbackException::getCallbackName() const { return callback_name_; }

/**
 * @brief Get the x-coordinate where the error occurred.
 *
 * @return double The x-coordinate.
 */
double CallbackException::getX() const { return x_; }

/**
 * @brief Get the y-coordinate where the error occurred.
 *
 * @return double The y-coordinate.
 */
double CallbackException::getY() const { return y_; }

/**
 * @brief Get the error message.
 *
 * @return const std::string& The error message.
 */
const std::string& CallbackException::getMessage() const { return message_; }

/**
 * @brief Format the error message.
 *
 * @param callback_name The name of the callback where the error occurred.
 * @param x The x-coordinate where the error occurred.
 * @param y The y-coordinate where the error occurred.
 * @param message The error message.
 * @return std::string The formatted error message.
 */
std::string CallbackException::formatMessage(const std::string& callback_name, double x, double y, const std::string& message) {
    std::ostringstream oss;
    oss << "Error in " << callback_name << " callback at position (" << x << ", " << y << "): " << message;
    return oss.str();
}

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
void logError(const std::string& callback_name, double x, double y, const std::string& message) {
    std::cerr << "Error in " << callback_name << " callback at position (" << x << ", " << y << "): " << message << std::endl;
}
