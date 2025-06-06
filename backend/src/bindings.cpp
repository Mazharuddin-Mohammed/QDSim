/**
 * @file bindings.cpp
 * @brief Python bindings for the C++ library.
 *
 * This file contains the Python bindings for the C++ library using pybind11.
 * It exposes the C++ classes and functions to Python, allowing them to be
 * used from Python code.
 *
 * The bindings include:
 * - Mesh class for finite element discretization
 * - FEInterpolator class for field interpolation
 * - MaterialDatabase class for material properties
 * - PoissonSolver class for electrostatic calculations
 * - FEMSolver class for quantum simulations
 * - EigenSolver class for eigenvalue problems
 * - Physics functions for material properties and potentials
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include "fem.h"
#include "solver.h"
#include "poisson.h"
#include "self_consistent.h"
#include "simple_self_consistent.h"
#include "basic_solver.h"
#include "improved_self_consistent.h"
#include "materials.h"
#include "physics.h"
#include "fe_interpolator.h"
#include "pn_junction.h"
#include "simple_mesh.h"
#include "simple_interpolator.h"
#include "schrodinger.h"
#include "gpu_accelerator.h"
#include "adaptive_mesh.h"
#include "full_poisson_dd_solver.h"
#include "callback_system.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

/**
 * @brief Callback manager for Python callbacks.
 *
 * This class manages Python callbacks used by C++ code. It ensures proper
 * reference counting and cleanup of Python callbacks when they are no longer needed.
 * It also provides thread-safe access to the callbacks.
 */
// Note: CallbackManager and CallbackException are now defined in callback_system.h
// Using the header definitions instead of duplicating them here

// Note: logError function is now defined in callback_system.cpp to avoid multiple definitions

/**
 * @brief Convert a Python material object to a C++ Material object.
 *
 * This function converts a Python material object to a C++ Material object.
 * It handles both C++ Material objects and Python objects with material properties.
 *
 * @param py_material The Python material object.
 * @return Materials::Material The C++ Material object.
 * @throws std::runtime_error If the conversion fails.
 */
Materials::Material convert_py_material_to_cpp(const pybind11::object& py_material) {
    // Check if the object is already a C++ Material
    try {
        return py_material.cast<Materials::Material>();
    } catch (const pybind11::cast_error&) {
        // Not a C++ Material, try to extract properties
    }

    // Create a default material
    Materials::Material cpp_material;

    // Try to extract properties from Python object
    try {
        // Check if the object has the expected attributes
        if (py_material.attr("__class__").attr("__name__").cast<std::string>() == "Material") {
            // Extract properties from Python Material class
            if (pybind11::hasattr(py_material, "epsilon_r")) {
                cpp_material.epsilon_r = py_material.attr("epsilon_r").cast<double>();
            }
            if (pybind11::hasattr(py_material, "band_gap") || pybind11::hasattr(py_material, "E_g")) {
                cpp_material.E_g = pybind11::hasattr(py_material, "band_gap") ?
                    py_material.attr("band_gap").cast<double>() :
                    py_material.attr("E_g").cast<double>();
            }
            if (pybind11::hasattr(py_material, "electron_mass") || pybind11::hasattr(py_material, "m_e")) {
                cpp_material.m_e = pybind11::hasattr(py_material, "electron_mass") ?
                    py_material.attr("electron_mass").cast<double>() :
                    py_material.attr("m_e").cast<double>();
            }
            if (pybind11::hasattr(py_material, "hole_mass") || pybind11::hasattr(py_material, "m_h")) {
                cpp_material.m_h = pybind11::hasattr(py_material, "hole_mass") ?
                    py_material.attr("hole_mass").cast<double>() :
                    py_material.attr("m_h").cast<double>();
            }
            if (pybind11::hasattr(py_material, "mu_n")) {
                cpp_material.mu_n = py_material.attr("mu_n").cast<double>();
            }
            if (pybind11::hasattr(py_material, "mu_p")) {
                cpp_material.mu_p = py_material.attr("mu_p").cast<double>();
            }
        } else if (py_material.is_none()) {
            // None object, use default values
            cpp_material.epsilon_r = 1.0;  // Vacuum
            cpp_material.E_g = 0.0;        // No bandgap
            cpp_material.m_e = 1.0;        // Free electron mass
            cpp_material.m_h = 1.0;        // Free hole mass
            cpp_material.mu_n = 0.0;       // No mobility
            cpp_material.mu_p = 0.0;       // No mobility
        }
    } catch (const std::exception& e) {
        std::string error_msg = "Error converting Python material to C++: ";
        error_msg += e.what();
        throw std::runtime_error(error_msg);
    }

    return cpp_material;
}

/**
 * @brief C++ wrapper function for the epsilon_r Python callback.
 *
 * This function safely calls the Python callback function for calculating
 * the relative permittivity at a given position. It handles all potential
 * exceptions and provides informative error messages.
 *
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @param p_mat The p-type material.
 * @param n_mat The n-type material.
 * @return double The relative permittivity at (x, y).
 * @throws CallbackException If an error occurs during the callback.
 */
double epsilon_r_wrapper(double x, double y, const Materials::Material& p_mat = Materials::Material(),
                         const Materials::Material& n_mat = Materials::Material()) {
    // Acquire the GIL before calling into Python
    pybind11::gil_scoped_acquire gil;

    try {
        // Get the callback function
        auto callback = CallbackManager::getInstance().getCallback("epsilon_r");
        if (!callback) {
            // No callback registered, use default value
            logError("epsilon_r", x, y, "Python callback not found. Using default value.");
            return p_mat.epsilon_r > 0.0 ? p_mat.epsilon_r : 1.0; // Use p_mat if available, otherwise vacuum
        }

        try {
            pybind11::object result;

            // Try different call signatures based on the number of arguments the callback accepts
            try {
                // First try with all parameters
                result = (*callback)(x, y, p_mat, n_mat);
            } catch (const pybind11::error_already_set& e) {
                // If that fails, try with just x and y
                pybind11::gil_scoped_release release;
                pybind11::gil_scoped_acquire acquire;

                try {
                    result = (*callback)(x, y);
                } catch (const pybind11::error_already_set& e2) {
                    // If that also fails, try with just x, y, and p_mat
                    pybind11::gil_scoped_release release2;
                    pybind11::gil_scoped_acquire acquire2;

                    try {
                        result = (*callback)(x, y, p_mat);
                    } catch (const pybind11::error_already_set& e3) {
                        // If all attempts fail, log the error and use default value
                        std::string error_msg = "Python exception in epsilon_r callback: ";
                        error_msg += e.what();

                        logError("epsilon_r", x, y, error_msg);
                        return p_mat.epsilon_r > 0.0 ? p_mat.epsilon_r : 1.0;
                    }
                }
            }

            // Check if the result is None
            if (result.is_none()) {
                logError("epsilon_r", x, y, "Python callback returned None. Using default value.");
                return p_mat.epsilon_r > 0.0 ? p_mat.epsilon_r : 1.0;
            }

            // Convert the result to double
            return result.cast<double>();

        } catch (const pybind11::cast_error& e) {
            // Type conversion error
            std::string error_msg = "Type conversion error in epsilon_r callback: ";
            error_msg += e.what();
            error_msg += " (Expected return type: double)";

            logError("epsilon_r", x, y, error_msg);
            throw CallbackException("epsilon_r", x, y, error_msg);
        }
    } catch (const CallbackException&) {
        // Re-throw CallbackException
        throw;
    } catch (const std::exception& e) {
        // Other C++ exceptions
        std::string error_msg = "C++ exception in epsilon_r callback: ";
        error_msg += e.what();

        logError("epsilon_r", x, y, error_msg);
        throw CallbackException("epsilon_r", x, y, error_msg);
    } catch (...) {
        // Unknown exceptions
        std::string error_msg = "Unknown exception in epsilon_r callback";

        logError("epsilon_r", x, y, error_msg);
        throw CallbackException("epsilon_r", x, y, error_msg);
    }

    // This line should never be reached, but is included for safety
    return p_mat.epsilon_r > 0.0 ? p_mat.epsilon_r : 1.0;
}

/**
 * @brief C++ wrapper function for the rho Python callback.
 *
 * This function safely calls the Python callback function for calculating
 * the charge density at a given position. It handles all potential
 * exceptions and provides informative error messages.
 *
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @param n The electron concentration vector.
 * @param p The hole concentration vector.
 * @return double The charge density at (x, y).
 * @throws CallbackException If an error occurs during the callback.
 */
double rho_wrapper(double x, double y, const Eigen::VectorXd& n, const Eigen::VectorXd& p) {
    // Acquire the GIL before calling into Python
    pybind11::gil_scoped_acquire gil;

    try {
        // Get the callback function
        auto callback = CallbackManager::getInstance().getCallback("rho");
        if (!callback) {
            // No callback registered, use default value
            logError("rho", x, y, "Python callback not found. Using default value.");
            return 0.0; // Default value for charge density
        }

        try {
            pybind11::object result;

            // Try different call signatures based on the number of arguments the callback accepts
            try {
                // First try with all parameters
                result = (*callback)(x, y, n, p);
            } catch (const pybind11::error_already_set& e) {
                // If that fails, try with just x and y
                pybind11::gil_scoped_release release;
                pybind11::gil_scoped_acquire acquire;

                try {
                    result = (*callback)(x, y);
                } catch (const pybind11::error_already_set& e2) {
                    // If that also fails, try with scalar n and p values (using the value at the node closest to x,y)
                    pybind11::gil_scoped_release release2;
                    pybind11::gil_scoped_acquire acquire2;

                    // Find the closest node to (x,y) to get scalar n and p values
                    double n_scalar = 0.0;
                    double p_scalar = 0.0;

                    if (n.size() > 0 && p.size() > 0) {
                        // Use the first element as a simple approximation
                        // In a real implementation, we would find the closest node
                        n_scalar = n(0);
                        p_scalar = p(0);
                    }

                    try {
                        result = (*callback)(x, y, n_scalar, p_scalar);
                    } catch (const pybind11::error_already_set& e3) {
                        // If all attempts fail, log the error and use default value
                        std::string error_msg = "Python exception in rho callback: ";
                        error_msg += e.what();

                        // Get Python traceback if available
                        if (e.trace()) {
                            error_msg += "\nTraceback: ";
                            error_msg += e.trace().cast<std::string>();
                        }

                        logError("rho", x, y, error_msg);
                        return 0.0;
                    }
                }
            }

            // Check if the result is None
            if (result.is_none()) {
                logError("rho", x, y, "Python callback returned None. Using default value.");
                return 0.0;
            }

            // Convert the result to double
            return result.cast<double>();

        } catch (const pybind11::cast_error& e) {
            // Type conversion error
            std::string error_msg = "Type conversion error in rho callback: ";
            error_msg += e.what();
            error_msg += " (Expected return type: double)";

            logError("rho", x, y, error_msg);
            throw CallbackException("rho", x, y, error_msg);
        }
    } catch (const CallbackException&) {
        // Re-throw CallbackException
        throw;
    } catch (const std::exception& e) {
        // Other C++ exceptions
        std::string error_msg = "C++ exception in rho callback: ";
        error_msg += e.what();

        logError("rho", x, y, error_msg);
        throw CallbackException("rho", x, y, error_msg);
    } catch (...) {
        // Unknown exceptions
        std::string error_msg = "Unknown exception in rho callback";

        logError("rho", x, y, error_msg);
        throw CallbackException("rho", x, y, error_msg);
    }

    // This line should never be reached, but is included for safety
    return 0.0;
}

/**
 * @brief C++ wrapper function for the n_conc Python callback.
 *
 * This function safely calls the Python callback function for calculating
 * the electron concentration at a given position. It handles all potential
 * exceptions and provides informative error messages.
 *
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @param phi The electrostatic potential.
 * @param mat The material properties.
 * @return double The electron concentration at (x, y).
 * @throws CallbackException If an error occurs during the callback.
 */
double n_conc_wrapper(double x, double y, double phi, const Materials::Material& mat) {
    // Acquire the GIL before calling into Python
    pybind11::gil_scoped_acquire gil;

    try {
        // Get the callback function
        auto callback = CallbackManager::getInstance().getCallback("n_conc");
        if (!callback) {
            // No callback registered, use Physics implementation as fallback
            logError("n_conc", x, y, "Python callback not found. Using Physics implementation.");
            return Physics::electron_concentration(x, y, phi, mat);
        }

        try {
            pybind11::object result;

            // Try different call signatures based on the number of arguments the callback accepts
            try {
                // First try with all parameters
                result = (*callback)(x, y, phi, mat);
            } catch (const pybind11::error_already_set& e) {
                // If that fails, try with just x, y, and phi
                pybind11::gil_scoped_release release;
                pybind11::gil_scoped_acquire acquire;

                try {
                    result = (*callback)(x, y, phi);
                } catch (const pybind11::error_already_set& e2) {
                    // If that also fails, try with just x and y
                    pybind11::gil_scoped_release release2;
                    pybind11::gil_scoped_acquire acquire2;

                    try {
                        result = (*callback)(x, y);
                    } catch (const pybind11::error_already_set& e3) {
                        // If all attempts fail, log the error and use Physics implementation
                        std::string error_msg = "Python exception in n_conc callback: ";
                        error_msg += e.what();

                        // Get Python traceback if available
                        if (e.trace()) {
                            error_msg += "\nTraceback: ";
                            error_msg += e.trace().cast<std::string>();
                        }

                        logError("n_conc", x, y, error_msg);
                        return Physics::electron_concentration(x, y, phi, mat);
                    }
                }
            }

            // Check if the result is None
            if (result.is_none()) {
                logError("n_conc", x, y, "Python callback returned None. Using Physics implementation.");
                return Physics::electron_concentration(x, y, phi, mat);
            }

            // Convert the result to double
            return result.cast<double>();

        } catch (const pybind11::cast_error& e) {
            // Type conversion error
            std::string error_msg = "Type conversion error in n_conc callback: ";
            error_msg += e.what();
            error_msg += " (Expected return type: double)";

            logError("n_conc", x, y, error_msg);
            throw CallbackException("n_conc", x, y, error_msg);
        }
    } catch (const CallbackException&) {
        // Re-throw CallbackException
        throw;
    } catch (const std::exception& e) {
        // Other C++ exceptions
        std::string error_msg = "C++ exception in n_conc callback: ";
        error_msg += e.what();

        logError("n_conc", x, y, error_msg);
        throw CallbackException("n_conc", x, y, error_msg);
    } catch (...) {
        // Unknown exceptions
        std::string error_msg = "Unknown exception in n_conc callback";

        logError("n_conc", x, y, error_msg);
        throw CallbackException("n_conc", x, y, error_msg);
    }

    // This line should never be reached, but is included for safety
    return Physics::electron_concentration(x, y, phi, mat);
}

/**
 * @brief C++ wrapper function for the p_conc Python callback.
 *
 * This function safely calls the Python callback function for calculating
 * the hole concentration at a given position. It handles all potential
 * exceptions and provides informative error messages.
 *
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @param phi The electrostatic potential.
 * @param mat The material properties.
 * @return double The hole concentration at (x, y).
 * @throws CallbackException If an error occurs during the callback.
 */
double p_conc_wrapper(double x, double y, double phi, const Materials::Material& mat) {
    // Acquire the GIL before calling into Python
    pybind11::gil_scoped_acquire gil;

    try {
        // Get the callback function
        auto callback = CallbackManager::getInstance().getCallback("p_conc");
        if (!callback) {
            // No callback registered, use Physics implementation as fallback
            logError("p_conc", x, y, "Python callback not found. Using Physics implementation.");
            return Physics::hole_concentration(x, y, phi, mat);
        }

        try {
            pybind11::object result;

            // Try different call signatures based on the number of arguments the callback accepts
            try {
                // First try with all parameters
                result = (*callback)(x, y, phi, mat);
            } catch (const pybind11::error_already_set& e) {
                // If that fails, try with just x, y, and phi
                pybind11::gil_scoped_release release;
                pybind11::gil_scoped_acquire acquire;

                try {
                    result = (*callback)(x, y, phi);
                } catch (const pybind11::error_already_set& e2) {
                    // If that also fails, try with just x and y
                    pybind11::gil_scoped_release release2;
                    pybind11::gil_scoped_acquire acquire2;

                    try {
                        result = (*callback)(x, y);
                    } catch (const pybind11::error_already_set& e3) {
                        // If all attempts fail, log the error and use Physics implementation
                        std::string error_msg = "Python exception in p_conc callback: ";
                        error_msg += e.what();

                        // Get Python traceback if available
                        if (e.trace()) {
                            error_msg += "\nTraceback: ";
                            error_msg += e.trace().cast<std::string>();
                        }

                        logError("p_conc", x, y, error_msg);
                        return Physics::hole_concentration(x, y, phi, mat);
                    }
                }
            }

            // Check if the result is None
            if (result.is_none()) {
                logError("p_conc", x, y, "Python callback returned None. Using Physics implementation.");
                return Physics::hole_concentration(x, y, phi, mat);
            }

            // Convert the result to double
            return result.cast<double>();

        } catch (const pybind11::cast_error& e) {
            // Type conversion error
            std::string error_msg = "Type conversion error in p_conc callback: ";
            error_msg += e.what();
            error_msg += " (Expected return type: double)";

            logError("p_conc", x, y, error_msg);
            throw CallbackException("p_conc", x, y, error_msg);
        }
    } catch (const CallbackException&) {
        // Re-throw CallbackException
        throw;
    } catch (const std::exception& e) {
        // Other C++ exceptions
        std::string error_msg = "C++ exception in p_conc callback: ";
        error_msg += e.what();

        logError("p_conc", x, y, error_msg);
        throw CallbackException("p_conc", x, y, error_msg);
    } catch (...) {
        // Unknown exceptions
        std::string error_msg = "Unknown exception in p_conc callback";

        logError("p_conc", x, y, error_msg);
        throw CallbackException("p_conc", x, y, error_msg);
    }

    // This line should never be reached, but is included for safety
    return Physics::hole_concentration(x, y, phi, mat);
}

/**
 * @brief C++ wrapper function for the mu_n Python callback.
 *
 * This function safely calls the Python callback function for calculating
 * the electron mobility at a given position. It handles all potential
 * exceptions and provides informative error messages.
 *
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @param mat The material properties.
 * @return double The electron mobility at (x, y).
 * @throws CallbackException If an error occurs during the callback.
 */
double mu_n_wrapper(double x, double y, const Materials::Material& mat) {
    // Acquire the GIL before calling into Python
    pybind11::gil_scoped_acquire gil;

    try {
        // Get the callback function
        auto callback = CallbackManager::getInstance().getCallback("mu_n");
        if (!callback) {
            // No callback registered, use Physics implementation as fallback
            logError("mu_n", x, y, "Python callback not found. Using Physics implementation.");
            return Physics::mobility_n(x, y, mat);
        }

        try {
            pybind11::object result;

            // Try different call signatures based on the number of arguments the callback accepts
            try {
                // First try with all parameters
                result = (*callback)(x, y, mat);
            } catch (const pybind11::error_already_set& e) {
                // If that fails, try with just x and y
                pybind11::gil_scoped_release release;
                pybind11::gil_scoped_acquire acquire;

                try {
                    result = (*callback)(x, y);
                } catch (const pybind11::error_already_set& e2) {
                    // If all attempts fail, log the error and use Physics implementation
                    std::string error_msg = "Python exception in mu_n callback: ";
                    error_msg += e.what();

                    // Get Python traceback if available
                    if (e.trace()) {
                        error_msg += "\nTraceback: ";
                        error_msg += e.trace().cast<std::string>();
                    }

                    logError("mu_n", x, y, error_msg);
                    return Physics::mobility_n(x, y, mat);
                }
            }

            // Check if the result is None
            if (result.is_none()) {
                logError("mu_n", x, y, "Python callback returned None. Using Physics implementation.");
                return Physics::mobility_n(x, y, mat);
            }

            // Convert the result to double
            return result.cast<double>();

        } catch (const pybind11::cast_error& e) {
            // Type conversion error
            std::string error_msg = "Type conversion error in mu_n callback: ";
            error_msg += e.what();
            error_msg += " (Expected return type: double)";

            logError("mu_n", x, y, error_msg);
            throw CallbackException("mu_n", x, y, error_msg);
        }
    } catch (const CallbackException&) {
        // Re-throw CallbackException
        throw;
    } catch (const std::exception& e) {
        // Other C++ exceptions
        std::string error_msg = "C++ exception in mu_n callback: ";
        error_msg += e.what();

        logError("mu_n", x, y, error_msg);
        throw CallbackException("mu_n", x, y, error_msg);
    } catch (...) {
        // Unknown exceptions
        std::string error_msg = "Unknown exception in mu_n callback";

        logError("mu_n", x, y, error_msg);
        throw CallbackException("mu_n", x, y, error_msg);
    }

    // This line should never be reached, but is included for safety
    return Physics::mobility_n(x, y, mat);
}

/**
 * @brief C++ wrapper function for the mu_p Python callback.
 *
 * This function safely calls the Python callback function for calculating
 * the hole mobility at a given position. It handles all potential
 * exceptions and provides informative error messages.
 *
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @param mat The material properties.
 * @return double The hole mobility at (x, y).
 * @throws CallbackException If an error occurs during the callback.
 */
double mu_p_wrapper(double x, double y, const Materials::Material& mat) {
    // Acquire the GIL before calling into Python
    pybind11::gil_scoped_acquire gil;

    try {
        // Get the callback function
        auto callback = CallbackManager::getInstance().getCallback("mu_p");
        if (!callback) {
            // No callback registered, use Physics implementation as fallback
            logError("mu_p", x, y, "Python callback not found. Using Physics implementation.");
            return Physics::mobility_p(x, y, mat);
        }

        try {
            pybind11::object result;

            // Try different call signatures based on the number of arguments the callback accepts
            try {
                // First try with all parameters
                result = (*callback)(x, y, mat);
            } catch (const pybind11::error_already_set& e) {
                // If that fails, try with just x and y
                pybind11::gil_scoped_release release;
                pybind11::gil_scoped_acquire acquire;

                try {
                    result = (*callback)(x, y);
                } catch (const pybind11::error_already_set& e2) {
                    // If all attempts fail, log the error and use Physics implementation
                    std::string error_msg = "Python exception in mu_p callback: ";
                    error_msg += e.what();

                    // Get Python traceback if available
                    if (e.trace()) {
                        error_msg += "\nTraceback: ";
                        error_msg += e.trace().cast<std::string>();
                    }

                    logError("mu_p", x, y, error_msg);
                    return Physics::mobility_p(x, y, mat);
                }
            }

            // Check if the result is None
            if (result.is_none()) {
                logError("mu_p", x, y, "Python callback returned None. Using Physics implementation.");
                return Physics::mobility_p(x, y, mat);
            }

            // Convert the result to double
            return result.cast<double>();

        } catch (const pybind11::cast_error& e) {
            // Type conversion error
            std::string error_msg = "Type conversion error in mu_p callback: ";
            error_msg += e.what();
            error_msg += " (Expected return type: double)";

            logError("mu_p", x, y, error_msg);
            throw CallbackException("mu_p", x, y, error_msg);
        }
    } catch (const CallbackException&) {
        // Re-throw CallbackException
        throw;
    } catch (const std::exception& e) {
        // Other C++ exceptions
        std::string error_msg = "C++ exception in mu_p callback: ";
        error_msg += e.what();

        logError("mu_p", x, y, error_msg);
        throw CallbackException("mu_p", x, y, error_msg);
    } catch (...) {
        // Unknown exceptions
        std::string error_msg = "Unknown exception in mu_p callback";

        logError("mu_p", x, y, error_msg);
        throw CallbackException("mu_p", x, y, error_msg);
    }

    // This line should never be reached, but is included for safety
    return Physics::mobility_p(x, y, mat);
}

/**
 * @brief Python module definition for the C++ library.
 *
 * This macro defines the Python module for the C++ library.
 * The module name is qdsim_cpp, and it exposes the C++ classes
 * and functions to Python.
 */
PYBIND11_MODULE(qdsim_cpp, m) {
    // Set module docstring
    m.doc() = "QDSim C++ backend for quantum dot simulations";

    // Mesh class for finite element discretization
    pybind11::class_<Mesh>(m, "Mesh")
        .def(pybind11::init<double, double, int, int, int>(),
             pybind11::arg("Lx"), pybind11::arg("Ly"), pybind11::arg("nx"), pybind11::arg("ny"), pybind11::arg("element_order"),
             "Construct a new Mesh object with the specified dimensions, number of elements, and element order")
        .def("get_nodes", &Mesh::getNodes, "Get the mesh nodes")
        .def("get_elements", &Mesh::getElements, "Get the mesh elements")
        .def("get_num_nodes", &Mesh::getNumNodes, "Get the number of nodes in the mesh")
        .def("get_num_elements", &Mesh::getNumElements, "Get the number of elements in the mesh")
        .def("get_element_order", &Mesh::getElementOrder, "Get the element order (1 for P1, 2 for P2, 3 for P3)")
        .def("get_lx", &Mesh::get_lx, "Get the width of the domain")
        .def("get_ly", &Mesh::get_ly, "Get the height of the domain")
        .def("get_nx", &Mesh::get_nx, "Get the number of elements in the x-direction")
        .def("get_ny", &Mesh::get_ny, "Get the number of elements in the y-direction")
        .def("getNumNodes", &Mesh::getNumNodes, "Get the number of nodes in the mesh (alias for get_num_nodes)")
        .def("getNumElements", &Mesh::getNumElements, "Get the number of elements in the mesh (alias for get_num_elements)")
        .def("getElementOrder", &Mesh::getElementOrder, "Get the element order (alias for get_element_order)");

    // FEInterpolator class for field interpolation
    pybind11::class_<FEInterpolator>(m, "FEInterpolator")
        .def(pybind11::init<const Mesh&>(),
             pybind11::arg("mesh"),
             "Construct a new FEInterpolator object for the specified mesh")
        .def("interpolate", &FEInterpolator::interpolate,
             pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("field"),
             "Interpolate a scalar field at a point (x, y)")
        .def("interpolate_with_gradient", [](FEInterpolator& self, double x, double y, const Eigen::VectorXd& field) {
                double grad_x = 0.0, grad_y = 0.0;
                double value = self.interpolateWithGradient(x, y, field, grad_x, grad_y);
                return std::make_tuple(value, grad_x, grad_y);
             },
             pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("field"),
             "Interpolate a scalar field and its gradient at a point (x, y)")
        .def("find_element", &FEInterpolator::findElement,
             pybind11::arg("x"), pybind11::arg("y"),
             "Find the element containing a point (x, y)")
        .def("compute_barycentric_coordinates", [](FEInterpolator& self, double x, double y, const std::vector<Eigen::Vector2d>& vertices) {
                std::vector<double> lambda(3);
                bool inside = self.computeBarycentricCoordinates(x, y, vertices, lambda);
                return std::make_tuple(inside, lambda);
             },
             pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("vertices"),
             "Compute the barycentric coordinates of a point in a triangle")
        .def("evaluate_shape_functions", &FEInterpolator::evaluateShapeFunctions,
             pybind11::arg("lambda"), pybind11::arg("shape_values"),
             "Evaluate the shape functions at a point given by barycentric coordinates")
        .def("evaluate_shape_function_gradients", &FEInterpolator::evaluateShapeFunctionGradients,
             pybind11::arg("lambda"), pybind11::arg("vertices"), pybind11::arg("shape_gradients"),
             "Evaluate the shape function gradients at a point given by barycentric coordinates");

    // Material class for material properties
    pybind11::class_<Materials::Material>(m, "Material")
        .def(pybind11::init<>(), "Construct a new Material object with default properties")
        .def_readwrite("m_e", &Materials::Material::m_e, "Electron effective mass (relative to free electron mass)")
        .def_readwrite("m_h", &Materials::Material::m_h, "Hole effective mass (relative to free electron mass)")
        .def_readwrite("m_lh", &Materials::Material::m_lh, "Light hole effective mass (relative to free electron mass)")
        .def_readwrite("m_hh", &Materials::Material::m_hh, "Heavy hole effective mass (relative to free electron mass)")
        .def_readwrite("m_so", &Materials::Material::m_so, "Split-off hole effective mass (relative to free electron mass)")
        .def_readwrite("E_g", &Materials::Material::E_g, "Band gap energy (eV)")
        .def_readwrite("Delta_E_c", &Materials::Material::Delta_E_c, "Conduction band offset (eV)")
        .def_readwrite("Delta_E_v", &Materials::Material::Delta_E_v, "Valence band offset (eV)")
        .def_readwrite("epsilon_r", &Materials::Material::epsilon_r, "Relative permittivity")
        .def_readwrite("mu_n", &Materials::Material::mu_n, "Electron mobility (nm²/V·s)")
        .def_readwrite("mu_p", &Materials::Material::mu_p, "Hole mobility (nm²/V·s)")
        .def_readwrite("N_c", &Materials::Material::N_c, "Effective density of states in conduction band (1/nm³)")
        .def_readwrite("N_v", &Materials::Material::N_v, "Effective density of states in valence band (1/nm³)")
        .def_readwrite("lattice_constant", &Materials::Material::lattice_constant, "Lattice constant (nm)")
        .def_readwrite("spin_orbit_splitting", &Materials::Material::spin_orbit_splitting, "Spin-orbit splitting energy (eV)")
        .def_readwrite("deformation_potential_c", &Materials::Material::deformation_potential_c, "Deformation potential for conduction band (eV)")
        .def_readwrite("deformation_potential_v", &Materials::Material::deformation_potential_v, "Deformation potential for valence band (eV)")
        .def_readwrite("elastic_c11", &Materials::Material::elastic_c11, "Elastic constant c11 (GPa)")
        .def_readwrite("elastic_c12", &Materials::Material::elastic_c12, "Elastic constant c12 (GPa)")
        .def_readwrite("elastic_c44", &Materials::Material::elastic_c44, "Elastic constant c44 (GPa)")
        .def_readwrite("varshni_alpha", &Materials::Material::varshni_alpha, "Varshni parameter alpha for temperature dependence of bandgap")
        .def_readwrite("varshni_beta", &Materials::Material::varshni_beta, "Varshni parameter beta for temperature dependence of bandgap")
        .def_readwrite("luttinger_gamma1", &Materials::Material::luttinger_gamma1, "Luttinger parameter gamma1 for k·p calculations")
        .def_readwrite("luttinger_gamma2", &Materials::Material::luttinger_gamma2, "Luttinger parameter gamma2 for k·p calculations")
        .def_readwrite("luttinger_gamma3", &Materials::Material::luttinger_gamma3, "Luttinger parameter gamma3 for k·p calculations")
        .def_readwrite("kane_parameter", &Materials::Material::kane_parameter, "Kane parameter for k·p calculations (eV·nm)")
        .def("__repr__", [](const Materials::Material& mat) {
            return "<Material: m_e=" + std::to_string(mat.m_e) +
                   ", m_h=" + std::to_string(mat.m_h) +
                   ", E_g=" + std::to_string(mat.E_g) +
                   ", epsilon_r=" + std::to_string(mat.epsilon_r) + ">";
        })
        // Add a constructor that takes a Python dictionary or object
        .def(pybind11::init([](const pybind11::object& obj) {
            return convert_py_material_to_cpp(obj);
        }), pybind11::arg("properties"), "Construct a new Material object from a Python dictionary or object");

    // MaterialDatabase class for material properties
    pybind11::class_<Materials::MaterialDatabase>(m, "MaterialDatabase")
        .def(pybind11::init<>(),
             "Construct a new MaterialDatabase object with default materials")
        .def("get_material", &Materials::MaterialDatabase::get_material,
             pybind11::arg("name"),
             "Get the properties of a material by name")
        .def("get_available_materials", &Materials::MaterialDatabase::get_available_materials,
             "Get a list of all available material names")
        .def("get_all_materials", [](const Materials::MaterialDatabase& db) {
             // Create a map of material names to Material objects
             std::unordered_map<std::string, Materials::Material> materials;
             for (const auto& name : db.get_available_materials()) {
                 materials[name] = db.get_material(name);
             }
             return materials;
         },
             "Get a map of all available materials")
        .def("add_material", &Materials::MaterialDatabase::add_material,
             pybind11::arg("name"), pybind11::arg("material"),
             "Add a new material to the database");

    // PoissonSolver class for electrostatic calculations
    pybind11::class_<PoissonSolver>(m, "PoissonSolver")
        .def(pybind11::init<Mesh&, double(*)(double, double),
                           double(*)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)>(),
             pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
             "Construct a new PoissonSolver object with the specified mesh, permittivity function, and charge density function")
        .def("solve", static_cast<void (PoissonSolver::*)(double, double)>(&PoissonSolver::solve),
             pybind11::arg("V_p"), pybind11::arg("V_n"),
             "Solve the Poisson equation with the specified boundary potentials")
        .def("solve", static_cast<void (PoissonSolver::*)(double, double, const Eigen::VectorXd&, const Eigen::VectorXd&)>(&PoissonSolver::solve),
             pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("n"), pybind11::arg("p"),
             "Solve the Poisson equation with the specified boundary potentials and carrier concentrations")
        .def("set_potential", &PoissonSolver::set_potential,
             pybind11::arg("potential"),
             "Set the potential values directly")
        .def("set_charge_density", &PoissonSolver::set_charge_density,
             pybind11::arg("charge_density"),
             "Set the charge density values directly")
        .def("get_potential", [](const PoissonSolver& solver) { return solver.phi; },
             "Get the computed electrostatic potential")
        .def("get_electric_field", [](const PoissonSolver& solver, double x, double y) {
                // Call the actual implementation
                return solver.get_electric_field(x, y);
             },
             pybind11::arg("x"), pybind11::arg("y"),
             "Get the electric field at a point (x, y)");

     // SelfConsistentSolver class for self-consistent Poisson-drift-diffusion simulations
     pybind11::class_<SelfConsistentSolver>(m, "SelfConsistentSolver")
             .def("solve", [](SelfConsistentSolver& self, double V_p, double V_n, double N_A, double N_D, double tolerance, int max_iter) {
                  // Release the GIL during long-running computation
                  pybind11::gil_scoped_release release;
                  return self.solve(V_p, V_n, N_A, N_D, tolerance, max_iter);
              }, pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("N_A"), pybind11::arg("N_D"),
                  pybind11::arg("tolerance") = 1e-6, pybind11::arg("max_iter") = 100,
                  "Solve the self-consistent Poisson-drift-diffusion equations")
             .def("get_potential", &SelfConsistentSolver::get_potential,
                  "Get the computed electrostatic potential")
             .def("get_n", &SelfConsistentSolver::get_n,
                  "Get the computed electron concentration")
             .def("get_p", &SelfConsistentSolver::get_p,
                  "Get the computed hole concentration")
             .def("get_electric_field", &SelfConsistentSolver::get_electric_field,
                  pybind11::arg("x"), pybind11::arg("y"),
                  "Get the electric field at a point (x, y)")
             .def_property("damping_factor",
                  [](SelfConsistentSolver& solver) { return solver.damping_factor; },
                  [](SelfConsistentSolver& solver, double value) { solver.damping_factor = value; },
                  "Damping factor for potential updates (0 < damping_factor <= 1)")
             .def_property("anderson_history_size",
                  [](SelfConsistentSolver& solver) { return solver.anderson_history_size; },
                  [](SelfConsistentSolver& solver, int value) { solver.anderson_history_size = value; },
                  "Number of previous iterations to use for Anderson acceleration")
             .def("set_heterojunction", [](SelfConsistentSolver& solver,
                                         const std::vector<Materials::Material>& materials,
                                         const std::vector<pybind11::function>& region_funcs) {
                 // Convert Python region functions to C++ functions
                 std::vector<std::function<bool(double, double)>> regions;
                 for (const auto& func : region_funcs) {
                     regions.push_back([func](double x, double y) {
                         pybind11::gil_scoped_acquire gil;
                         try {
                             return func(x, y).cast<bool>();
                         } catch (const std::exception& e) {
                             std::cerr << "Error in region function: " << e.what() << std::endl;
                             return false;
                         }
                     });
                 }

                 // Call the C++ method
                 solver.set_heterojunction(materials, regions);
             }, pybind11::arg("materials"), pybind11::arg("regions"),
                "Set the materials and regions for a heterojunction");

    // FEMSolver class for quantum simulations
    pybind11::class_<FEMSolver>(m, "FEMSolver")
        .def(pybind11::init<Mesh&, double(*)(double, double), double(*)(double, double), double(*)(double, double), SelfConsistentSolver&, int, bool>(),
             pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"), pybind11::arg("cap"),
             pybind11::arg("sc_solver"), pybind11::arg("order"), pybind11::arg("use_mpi") = false,
             "Construct a new FEMSolver object with the specified mesh, effective mass function, potential function, capacitance function, Self-Consistent solver, element order, and MPI flag")
        .def("assemble_matrices", &FEMSolver::assemble_matrices,
             "Assemble the Hamiltonian and mass matrices")
        .def("adapt_mesh", &FEMSolver::adapt_mesh,
             pybind11::arg("psi"), pybind11::arg("threshold"), pybind11::arg("output_file") = "",
             "Adapt the mesh based on the solution gradient")
        .def("is_using_mpi", &FEMSolver::is_using_mpi,
             "Check if the solver is using MPI")
        .def("get_H", [](const FEMSolver& solver) { return solver.get_H(); },
             "Get the Hamiltonian matrix")
        .def("get_M", [](const FEMSolver& solver) { return solver.get_M(); },
             "Get the mass matrix")
        .def("get_interpolator", [](const FEMSolver& solver) { return solver.get_interpolator(); },
             "Get the field interpolator")
        .def("set_potential_values", &FEMSolver::set_potential_values,
             pybind11::arg("potential_values"),
             "Set the potential values at mesh nodes for interpolation")
        .def("use_interpolated_potential", &FEMSolver::use_interpolated_potential,
             pybind11::arg("enable"),
             "Enable or disable the use of interpolated potentials");

    // EigenSolver class for eigenvalue problems
    pybind11::class_<EigenSolver>(m, "EigenSolver")
        .def(pybind11::init<FEMSolver&>(),
             pybind11::arg("fem_solver"),
             "Construct a new EigenSolver object for the specified FEM solver")
        .def("solve", &EigenSolver::solve,
             pybind11::arg("num_eigenpairs") = 10,
             "Solve the eigenvalue problem and compute the specified number of eigenpairs")
        .def("get_eigenvalues", &EigenSolver::get_eigenvalues,
             "Get the computed eigenvalues")
        .def("get_eigenvectors", &EigenSolver::get_eigenvectors,
             "Get the computed eigenvectors");

    // Physics functions
    m.def("effective_mass", &Physics::effective_mass,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("qd_mat"), pybind11::arg("matrix_mat"), pybind11::arg("R"),
          "Compute the effective mass at a point (x, y) based on the quantum dot and matrix materials");

    m.def("potential", static_cast<double(*)(double, double, const Materials::Material&, const Materials::Material&, double, const std::string&, const Eigen::VectorXd&, const FEInterpolator*)>(&Physics::potential),
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("qd_mat"), pybind11::arg("matrix_mat"),
          pybind11::arg("R"), pybind11::arg("type"), pybind11::arg("phi"), pybind11::arg("interpolator") = nullptr,
          "Compute the potential at a point (x, y) based on the quantum dot and matrix materials, and the electrostatic potential");

    m.def("epsilon_r", &Physics::epsilon_r,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("p_mat"), pybind11::arg("n_mat"),
          "Compute the relative permittivity at a point (x, y) based on the p-type and n-type materials");

    /*
    m.def("charge_density", &Physics::charge_density,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("N_A"), pybind11::arg("N_D"), pybind11::arg("W_d"),
          "Compute the charge density at a point (x, y) based on the doping concentrations and depletion width");
    */

    m.def("charge_density", &Physics::charge_density,
               pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("n"), pybind11::arg("p"),
               pybind11::arg("n_interpolator") = nullptr, pybind11::arg("p_interpolator") = nullptr,
               "Compute the charge density at a point (x, y) based on the carrier concentrations");

     m.def("cap", &Physics::cap,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("eta"), pybind11::arg("Lx"), pybind11::arg("Ly"), pybind11::arg("d"),
          "Compute the capacitance at a point (x, y) based on the gate geometry and dielectric properties");

     m.def("electron_concentration", &Physics::electron_concentration,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("phi"), pybind11::arg("mat"),
          "Compute the electron concentration at a point (x, y) based on the electrostatic potential and material properties");

     m.def("hole_concentration", &Physics::hole_concentration,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("phi"), pybind11::arg("mat"),
          "Compute the hole concentration at a point (x, y) based on the electrostatic potential and material properties");

     m.def("mobility_n", &Physics::mobility_n,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("mat"),
          "Compute the electron mobility at a point (x, y) based on the material properties");

     m.def("mobility_p", &Physics::mobility_p,
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("mat"),
          "Compute the hole mobility at a point (x, y) based on the material properties");

     // SimpleSelfConsistentSolver class for simplified self-consistent Poisson-drift-diffusion simulations
     pybind11::class_<SimpleSelfConsistentSolver>(m, "SimpleSelfConsistentSolver")
             .def("solve", [](SimpleSelfConsistentSolver& self, double V_p, double V_n, double N_A, double N_D, double tolerance, int max_iter) {
                  // Release the GIL during long-running computation
                  pybind11::gil_scoped_release release;
                  return self.solve(V_p, V_n, N_A, N_D, tolerance, max_iter);
              }, pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("N_A"), pybind11::arg("N_D"),
                  pybind11::arg("tolerance") = 1e-6, pybind11::arg("max_iter") = 100,
                  "Solve the self-consistent Poisson-drift-diffusion equations")
             .def("get_potential", &SimpleSelfConsistentSolver::get_potential,
                  "Get the computed electrostatic potential")
             .def("get_n", &SimpleSelfConsistentSolver::get_n,
                  "Get the computed electron concentration")
             .def("get_p", &SimpleSelfConsistentSolver::get_p,
                  "Get the computed hole concentration");

     // BasicSolver class for demonstration purposes
     pybind11::class_<BasicSolver>(m, "BasicSolver")
             .def(pybind11::init<Mesh&>(),
                  pybind11::arg("mesh"),
                  "Construct a new BasicSolver object with the specified mesh")
             .def("solve", &BasicSolver::solve,
                  pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("N_A"), pybind11::arg("N_D"),
                  "Set up simple fields without doing any actual solving")
             .def("get_potential", &BasicSolver::get_potential,
                  "Get the electrostatic potential")
             .def("get_n", &BasicSolver::get_n,
                  "Get the electron concentration")
             .def("get_p", &BasicSolver::get_p,
                  "Get the hole concentration");

     // ImprovedSelfConsistentSolver class for self-consistent Poisson-drift-diffusion simulations
     pybind11::class_<ImprovedSelfConsistentSolver>(m, "ImprovedSelfConsistentSolver")
             .def("solve", [](ImprovedSelfConsistentSolver& self, double V_p, double V_n, double N_A, double N_D, double tolerance, int max_iter) {
                  // Release the GIL during long-running computation
                  pybind11::gil_scoped_release release;
                  return self.solve(V_p, V_n, N_A, N_D, tolerance, max_iter);
              }, pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("N_A"), pybind11::arg("N_D"),
                  pybind11::arg("tolerance") = 1e-6, pybind11::arg("max_iter") = 100,
                  "Solve the self-consistent Poisson-drift-diffusion equations")
             .def("get_potential", &ImprovedSelfConsistentSolver::get_potential,
                  "Get the computed electrostatic potential")
             .def("get_n", &ImprovedSelfConsistentSolver::get_n,
                  "Get the computed electron concentration")
             .def("get_p", &ImprovedSelfConsistentSolver::get_p,
                  "Get the computed hole concentration");

     // SchrodingerSolver class for quantum simulations
     pybind11::class_<SchrodingerSolver>(m, "SchrodingerSolver")
             .def(pybind11::init<Mesh&, std::function<double(double, double)>,
                                std::function<double(double, double)>, bool>(),
                  pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"),
                  pybind11::arg("use_gpu") = false,
                  "Construct a new SchrodingerSolver object with the specified mesh, effective mass function, potential function, and GPU flag")
             .def("solve", [](SchrodingerSolver& self, int num_eigenvalues) {
                  // Release the GIL during long-running computation
                  pybind11::gil_scoped_release release;
                  return self.solve(num_eigenvalues);
              }, pybind11::arg("num_eigenvalues") = 10,
                  "Solve the Schrödinger equation and compute the specified number of eigenpairs")
             .def("get_eigenvalues", &SchrodingerSolver::get_eigenvalues,
                  "Get the computed eigenvalues")
             .def("get_eigenvectors", &SchrodingerSolver::get_eigenvectors,
                  "Get the computed eigenvectors")
             .def("get_H", &SchrodingerSolver::get_H,
                  "Get the Hamiltonian matrix")
             .def("get_M", &SchrodingerSolver::get_M,
                  "Get the mass matrix")
             .def("get_mesh", &SchrodingerSolver::get_mesh,
                  "Get the mesh")
             .def("is_gpu_enabled", &SchrodingerSolver::is_gpu_enabled,
                  "Check if GPU acceleration is enabled")
             .def("get_gpu_accelerator", &SchrodingerSolver::get_gpu_accelerator,
                  "Get the GPU accelerator");

     // Helper functions to create callbacks for SelfConsistentSolver
     m.def("create_self_consistent_solver", [](Mesh& mesh, pybind11::function epsilon_r_py, pybind11::function rho_py,
                                             pybind11::function n_conc_py, pybind11::function p_conc_py,
                                             pybind11::function mu_n_py, pybind11::function mu_p_py) {
         // Generate unique keys for this mesh instance
         std::string mesh_id = std::to_string(reinterpret_cast<uintptr_t>(&mesh));
         std::string epsilon_r_key = "epsilon_r_" + mesh_id;
         std::string rho_key = "rho_" + mesh_id;
         std::string n_conc_key = "n_conc_" + mesh_id;
         std::string p_conc_key = "p_conc_" + mesh_id;
         std::string mu_n_key = "mu_n_" + mesh_id;
         std::string mu_p_key = "mu_p_" + mesh_id;

         // Clear any existing callbacks with these keys to prevent memory leaks
         CallbackManager::getInstance().clearCallback(epsilon_r_key);
         CallbackManager::getInstance().clearCallback(rho_key);
         CallbackManager::getInstance().clearCallback(n_conc_key);
         CallbackManager::getInstance().clearCallback(p_conc_key);
         CallbackManager::getInstance().clearCallback(mu_n_key);
         CallbackManager::getInstance().clearCallback(mu_p_key);

         // Also clear the global callbacks for backward compatibility
         CallbackManager::getInstance().clearCallback("epsilon_r");
         CallbackManager::getInstance().clearCallback("rho");
         CallbackManager::getInstance().clearCallback("n_conc");
         CallbackManager::getInstance().clearCallback("p_conc");
         CallbackManager::getInstance().clearCallback("mu_n");
         CallbackManager::getInstance().clearCallback("mu_p");

         // Store the Python callbacks in the callback manager with mesh-specific keys
         CallbackManager::getInstance().setCallback(epsilon_r_key, epsilon_r_py);
         CallbackManager::getInstance().setCallback(rho_key, rho_py);
         CallbackManager::getInstance().setCallback(n_conc_key, n_conc_py);
         CallbackManager::getInstance().setCallback(p_conc_key, p_conc_py);
         CallbackManager::getInstance().setCallback(mu_n_key, mu_n_py);
         CallbackManager::getInstance().setCallback(mu_p_key, mu_p_py);

         // Also store with global keys for backward compatibility
         CallbackManager::getInstance().setCallback("epsilon_r", epsilon_r_py);
         CallbackManager::getInstance().setCallback("rho", rho_py);
         CallbackManager::getInstance().setCallback("n_conc", n_conc_py);
         CallbackManager::getInstance().setCallback("p_conc", p_conc_py);
         CallbackManager::getInstance().setCallback("mu_n", mu_n_py);
         CallbackManager::getInstance().setCallback("mu_p", mu_p_py);

         // Create the SelfConsistentSolver with the wrapper functions
         try {
             // Create a new SelfConsistentSolver
             auto solver = std::make_unique<SelfConsistentSolver>(mesh);

             // Release ownership and return the raw pointer (pybind11 will handle the memory)
             return solver.release();
         } catch (const std::exception& e) {
             // Clean up callbacks if solver creation fails
             CallbackManager::getInstance().clearCallback(epsilon_r_key);
             CallbackManager::getInstance().clearCallback(rho_key);
             CallbackManager::getInstance().clearCallback(n_conc_key);
             CallbackManager::getInstance().clearCallback(p_conc_key);
             CallbackManager::getInstance().clearCallback(mu_n_key);
             CallbackManager::getInstance().clearCallback(mu_p_key);

             // Also clear the global callbacks
             CallbackManager::getInstance().clearCallback("epsilon_r");
             CallbackManager::getInstance().clearCallback("rho");
             CallbackManager::getInstance().clearCallback("n_conc");
             CallbackManager::getInstance().clearCallback("p_conc");
             CallbackManager::getInstance().clearCallback("mu_n");
             CallbackManager::getInstance().clearCallback("mu_p");

             throw;
         }
     }, pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
        pybind11::arg("n_conc"), pybind11::arg("p_conc"), pybind11::arg("mu_n"), pybind11::arg("mu_p"),
        "Create a new SelfConsistentSolver object with the specified mesh and Python callback functions");

     // Helper function to create a SimpleSelfConsistentSolver
     m.def("create_simple_self_consistent_solver", [](Mesh& mesh, pybind11::function epsilon_r_py, pybind11::function rho_py) {
         // Generate unique keys for this mesh instance
         std::string mesh_id = std::to_string(reinterpret_cast<uintptr_t>(&mesh));
         std::string epsilon_r_key = "epsilon_r_simple_" + mesh_id;
         std::string rho_key = "rho_simple_" + mesh_id;

         // Clear any existing callbacks with these keys to prevent memory leaks
         CallbackManager::getInstance().clearCallback(epsilon_r_key);
         CallbackManager::getInstance().clearCallback(rho_key);

         // Also clear the global callbacks for backward compatibility
         CallbackManager::getInstance().clearCallback("epsilon_r");
         CallbackManager::getInstance().clearCallback("rho");

         // Store the Python callbacks in the callback manager with mesh-specific keys
         CallbackManager::getInstance().setCallback(epsilon_r_key, epsilon_r_py);
         CallbackManager::getInstance().setCallback(rho_key, rho_py);

         // Also store with global keys for backward compatibility
         CallbackManager::getInstance().setCallback("epsilon_r", epsilon_r_py);
         CallbackManager::getInstance().setCallback("rho", rho_py);

         // Create the SimpleSelfConsistentSolver with the wrapper functions
         try {
             // Use a unique_ptr to ensure proper cleanup in case of exceptions
             auto solver = std::make_unique<SimpleSelfConsistentSolver>(mesh);

             // Release ownership and return the raw pointer (pybind11 will handle the memory)
             return solver.release();
         } catch (const std::exception& e) {
             // Clean up callbacks if solver creation fails
             CallbackManager::getInstance().clearCallback(epsilon_r_key);
             CallbackManager::getInstance().clearCallback(rho_key);

             // Also clear the global callbacks
             CallbackManager::getInstance().clearCallback("epsilon_r");
             CallbackManager::getInstance().clearCallback("rho");

             throw;
         }
     }, pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
        "Create a new SimpleSelfConsistentSolver object with the specified mesh and Python callback functions");

     // Helper function to create an ImprovedSelfConsistentSolver
     m.def("create_improved_self_consistent_solver", [](Mesh& mesh, pybind11::function epsilon_r_py, pybind11::function rho_py) {
         // Generate unique keys for this mesh instance
         std::string mesh_id = std::to_string(reinterpret_cast<uintptr_t>(&mesh));
         std::string epsilon_r_key = "epsilon_r_improved_" + mesh_id;
         std::string rho_key = "rho_improved_" + mesh_id;

         // Clear any existing callbacks with these keys to prevent memory leaks
         CallbackManager::getInstance().clearCallback(epsilon_r_key);
         CallbackManager::getInstance().clearCallback(rho_key);

         // Also clear the global callbacks for backward compatibility
         CallbackManager::getInstance().clearCallback("epsilon_r");
         CallbackManager::getInstance().clearCallback("rho");

         // Store the Python callbacks in the callback manager with mesh-specific keys
         CallbackManager::getInstance().setCallback(epsilon_r_key, epsilon_r_py);
         CallbackManager::getInstance().setCallback(rho_key, rho_py);

         // Also store with global keys for backward compatibility
         CallbackManager::getInstance().setCallback("epsilon_r", epsilon_r_py);
         CallbackManager::getInstance().setCallback("rho", rho_py);

         // Create the ImprovedSelfConsistentSolver with the wrapper functions
         try {
             // Use a unique_ptr to ensure proper cleanup in case of exceptions
             auto solver = std::make_unique<ImprovedSelfConsistentSolver>(mesh);

             // Release ownership and return the raw pointer (pybind11 will handle the memory)
             return solver.release();
         } catch (const std::exception& e) {
             // Clean up callbacks if solver creation fails
             CallbackManager::getInstance().clearCallback(epsilon_r_key);
             CallbackManager::getInstance().clearCallback(rho_key);

             // Also clear the global callbacks
             CallbackManager::getInstance().clearCallback("epsilon_r");
             CallbackManager::getInstance().clearCallback("rho");

             throw;
         }
     }, pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("rho"),
        "Create a new ImprovedSelfConsistentSolver object with the specified mesh and Python callback functions");

     // Helper function to create a SchrodingerSolver
     m.def("create_schrodinger_solver", [](Mesh& mesh, pybind11::function m_star_py, pybind11::function V_py, bool use_gpu) {
         // Generate unique keys for this mesh instance
         std::string m_star_key = "m_star_" + std::to_string(reinterpret_cast<uintptr_t>(&mesh));
         std::string V_key = "V_" + std::to_string(reinterpret_cast<uintptr_t>(&mesh));

         // Clear any existing callbacks with these keys to prevent memory leaks
         CallbackManager::getInstance().clearCallback(m_star_key);
         CallbackManager::getInstance().clearCallback(V_key);

         // Store the Python callbacks in the callback manager
         CallbackManager::getInstance().setCallback(m_star_key, m_star_py);
         CallbackManager::getInstance().setCallback(V_key, V_py);

         // Create wrapper functions for the effective mass callback
         auto m_star_wrapper = [m_star_key](double x, double y) -> double {
             // Acquire the GIL before calling into Python
             pybind11::gil_scoped_acquire gil;

             try {
                 // Get the callback function
                 auto callback = CallbackManager::getInstance().getCallback(m_star_key);
                 if (!callback) {
                     throw std::runtime_error("m_star callback not found");
                 }

                 // Call the Python function and convert the result to double
                 pybind11::object result = (*callback)(x, y);

                 // Check if the result is None
                 if (result.is_none()) {
                     throw std::runtime_error("m_star callback returned None");
                 }

                 // Convert the result to double
                 return result.cast<double>();

             } catch (const pybind11::error_already_set& e) {
                 // Python exception occurred
                 std::string error_msg = "Python exception in m_star callback: ";
                 error_msg += e.what();

                 // Get Python traceback if available
                 if (e.trace()) {
                     error_msg += "\nTraceback: ";
                     error_msg += e.trace().cast<std::string>();
                 }

                 throw std::runtime_error(error_msg);

             } catch (const pybind11::cast_error& e) {
                 // Type conversion error
                 std::string error_msg = "Type conversion error in m_star callback: ";
                 error_msg += e.what();
                 error_msg += " (Expected return type: double)";

                 throw std::runtime_error(error_msg);

             } catch (const std::exception& e) {
                 // Other C++ exceptions
                 throw std::runtime_error(std::string("Error in m_star callback: ") + e.what());
             }
         };

         // Create wrapper functions for the potential callback
         auto V_wrapper = [V_key](double x, double y) -> double {
             // Acquire the GIL before calling into Python
             pybind11::gil_scoped_acquire gil;

             try {
                 // Get the callback function
                 auto callback = CallbackManager::getInstance().getCallback(V_key);
                 if (!callback) {
                     throw std::runtime_error("V callback not found");
                 }

                 // Call the Python function and convert the result to double
                 pybind11::object result = (*callback)(x, y);

                 // Check if the result is None
                 if (result.is_none()) {
                     throw std::runtime_error("V callback returned None");
                 }

                 // Convert the result to double
                 return result.cast<double>();

             } catch (const pybind11::error_already_set& e) {
                 // Python exception occurred
                 std::string error_msg = "Python exception in V callback: ";
                 error_msg += e.what();

                 // Get Python traceback if available
                 if (e.trace()) {
                     error_msg += "\nTraceback: ";
                     error_msg += e.trace().cast<std::string>();
                 }

                 throw std::runtime_error(error_msg);

             } catch (const pybind11::cast_error& e) {
                 // Type conversion error
                 std::string error_msg = "Type conversion error in V callback: ";
                 error_msg += e.what();
                 error_msg += " (Expected return type: double)";

                 throw std::runtime_error(error_msg);

             } catch (const std::exception& e) {
                 // Other C++ exceptions
                 throw std::runtime_error(std::string("Error in V callback: ") + e.what());
             }
         };

         // Create the SchrodingerSolver with the wrapper functions
         try {
             // Use a unique_ptr to ensure proper cleanup in case of exceptions
             auto solver = std::make_unique<SchrodingerSolver>(mesh, m_star_wrapper, V_wrapper, use_gpu);

             // Release ownership and return the raw pointer (pybind11 will handle the memory)
             return solver.release();
         } catch (const std::exception& e) {
             // Clean up callbacks if solver creation fails
             CallbackManager::getInstance().clearCallback(m_star_key);
             CallbackManager::getInstance().clearCallback(V_key);
             throw;
         }
     }, pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"), pybind11::arg("use_gpu") = false,
        "Create a new SchrodingerSolver object with the specified mesh and Python callback functions");

     // Helper function to clear all callbacks
     m.def("clear_callbacks", []() {
         try {
             CallbackManager::getInstance().clearCallbacks();
         } catch (...) {
             // Ignore errors during callback cleanup to prevent crashes
         }
     }, "Clear all Python callbacks to avoid memory leaks");

     // Helper function to clear a specific callback
     m.def("clear_callback", [](const std::string& name) {
         try {
             CallbackManager::getInstance().clearCallback(name);
         } catch (...) {
             // Ignore errors during callback cleanup to prevent crashes
         }
     }, pybind11::arg("name"),
        "Clear a specific Python callback to avoid memory leaks");

     // Add the clearCallbacks function for compatibility
     m.def("clearCallbacks", []() {
         try {
             CallbackManager::getInstance().clearCallbacks();
         } catch (...) {
             // Ignore errors during callback cleanup to prevent crashes
         }
     }, "Clear all Python callbacks (compatibility function)");

     // Register the CallbackException class
     pybind11::register_exception<CallbackException>(m, "CallbackException");

     // Add a function to check if a callback exists
     m.def("has_callback", &hasCallback,
         pybind11::arg("name"),
        "Check if a callback with the given name exists");

     // Add a function to set a callback
     m.def("setCallback", &setCallback,
         pybind11::arg("name"), pybind11::arg("callback"),
        "Set a callback function with the given name");

     // Add a function to clear a callback
     m.def("clearCallback", &clearCallback,
         pybind11::arg("name"),
        "Clear a callback function with the given name");

     // Add a function to clear all callbacks
     m.def("clearCallbacks", &clearCallbacks,
        "Clear all callback functions");

     // GPUAccelerator class for GPU-accelerated computations
     pybind11::class_<GPUAccelerator>(m, "GPUAccelerator")
         .def(pybind11::init<bool, int>(),
              pybind11::arg("use_gpu") = true, pybind11::arg("device_id") = 0,
              "Construct a new GPUAccelerator object with the specified parameters")
         .def("is_gpu_enabled", &GPUAccelerator::is_gpu_enabled,
              "Check if GPU acceleration is enabled")
         .def("get_device_info", &GPUAccelerator::get_device_info,
              "Get information about the GPU device")
         .def("assemble_matrices", &GPUAccelerator::assemble_matrices,
              pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"),
              pybind11::arg("order"), pybind11::arg("H"), pybind11::arg("M"),
              "Assemble matrices on GPU")
         .def("assemble_higher_order_matrices", &GPUAccelerator::assemble_higher_order_matrices,
              pybind11::arg("mesh"), pybind11::arg("m_star"), pybind11::arg("V"),
              pybind11::arg("order"), pybind11::arg("H"), pybind11::arg("M"),
              "Assemble higher-order matrices on GPU")
         .def("solve_eigen", &GPUAccelerator::solve_eigen,
              pybind11::arg("H"), pybind11::arg("M"), pybind11::arg("num_eigenvalues"),
              pybind11::arg("eigenvalues"), pybind11::arg("eigenvectors"),
              "Solve eigenvalue problem on GPU")
         .def("solve_eigen_sparse", &GPUAccelerator::solve_eigen_sparse,
              pybind11::arg("H"), pybind11::arg("M"), pybind11::arg("num_eigenvalues"),
              pybind11::arg("eigenvalues"), pybind11::arg("eigenvectors"),
              pybind11::arg("tolerance") = 1e-10, pybind11::arg("max_iterations") = 1000,
              "Solve sparse eigenvalue problem on GPU")
         .def("interpolate_field", &GPUAccelerator::interpolate_field,
              pybind11::arg("mesh"), pybind11::arg("field"), pybind11::arg("points"),
              pybind11::arg("values"),
              "Interpolate field on GPU");

     // SimpleMesh class for interpolation
     pybind11::class_<SimpleMesh>(m, "SimpleMesh")
         .def(pybind11::init<const std::vector<Eigen::Vector2d>&, const std::vector<std::array<int, 3>>&>(),
              pybind11::arg("nodes"), pybind11::arg("elements"),
              "Construct a new SimpleMesh object with the specified nodes and elements")
         .def("get_nodes", &SimpleMesh::getNodes,
              "Get the nodes of the mesh")
         .def("get_elements", &SimpleMesh::getElements,
              "Get the elements of the mesh");

     // Helper function to create a SimpleMesh from a Mesh
     m.def("create_simple_mesh", [](const Mesh& mesh) {
         return std::make_unique<SimpleMesh>(mesh.getNodes(), mesh.getElements());
     }, pybind11::arg("mesh"),
        "Create a new SimpleMesh object from a Mesh");

     // SimpleInterpolator class for interpolation
     pybind11::class_<SimpleInterpolator>(m, "SimpleInterpolator")
         .def(pybind11::init<const SimpleMesh&>(),
              pybind11::arg("mesh"),
              "Construct a new SimpleInterpolator object with the specified mesh")
         .def("interpolate", &SimpleInterpolator::interpolate,
              pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("values"),
              "Interpolate a value at a point")
         .def("find_element", &SimpleInterpolator::findElement,
              pybind11::arg("x"), pybind11::arg("y"),
              "Find the element containing a point")
         .def("compute_barycentric_coordinates", [](const SimpleInterpolator& self, double x, double y, int elem_idx) {
             std::array<double, 3> lambda;
             bool inside = self.computeBarycentricCoordinates(x, y, elem_idx, lambda);
             return std::make_tuple(inside, lambda);
         }, pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("elem_idx"),
            "Compute the barycentric coordinates of a point in an element");

     // PNJunction class for P-N junction simulations
     pybind11::class_<PNJunction>(m, "PNJunction")
         .def(pybind11::init<Mesh&, double, double, double, double, double, double>(),
              pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("N_A"),
              pybind11::arg("N_D"), pybind11::arg("T"), pybind11::arg("junction_position"),
              pybind11::arg("V_r"),
              "Construct a new PNJunction object with the specified parameters")
         .def("calculate_built_in_potential", &PNJunction::calculate_built_in_potential,
              "Calculate the built-in potential of the P-N junction")
         .def("calculate_depletion_width", &PNJunction::calculate_depletion_width,
              "Calculate the depletion width of the P-N junction")
         .def("calculate_intrinsic_carrier_concentration", &PNJunction::calculate_intrinsic_carrier_concentration,
              "Calculate the intrinsic carrier concentration")
         .def("solve", &PNJunction::solve,
              "Solve the Poisson equation for the electrostatic potential")
         .def("get_potential", &PNJunction::get_potential,
              pybind11::arg("x"), pybind11::arg("y"),
              "Get the electrostatic potential at a given position")
         .def("get_electric_field", &PNJunction::get_electric_field,
              pybind11::arg("x"), pybind11::arg("y"),
              "Get the electric field at a given position")
         .def("get_electron_concentration", &PNJunction::get_electron_concentration,
              pybind11::arg("x"), pybind11::arg("y"),
              "Get the electron concentration at a given position")
         .def("get_hole_concentration", &PNJunction::get_hole_concentration,
              pybind11::arg("x"), pybind11::arg("y"),
              "Get the hole concentration at a given position")
         .def("get_conduction_band_edge", &PNJunction::get_conduction_band_edge,
              pybind11::arg("x"), pybind11::arg("y"),
              "Get the conduction band edge at a given position")
         .def("get_valence_band_edge", &PNJunction::get_valence_band_edge,
              pybind11::arg("x"), pybind11::arg("y"),
              "Get the valence band edge at a given position")
         .def("get_quasi_fermi_level_electrons", &PNJunction::get_quasi_fermi_level_electrons,
              pybind11::arg("x"), pybind11::arg("y"),
              "Get the quasi-Fermi level for electrons at a given position")
         .def("get_quasi_fermi_level_holes", &PNJunction::get_quasi_fermi_level_holes,
              pybind11::arg("x"), pybind11::arg("y"),
              "Get the quasi-Fermi level for holes at a given position")
         .def("update_bias", &PNJunction::update_bias,
              pybind11::arg("V_r"),
              "Update the reverse bias voltage")
         .def("update_doping", &PNJunction::update_doping,
              pybind11::arg("N_A"), pybind11::arg("N_D"),
              "Update the doping concentrations")
         .def_property_readonly("V_bi", &PNJunction::get_V_bi,
              "Get the built-in potential")
         .def_property_readonly("V_r", &PNJunction::get_V_r,
              "Get the reverse bias voltage")
         .def_property_readonly("V_total", &PNJunction::get_V_total,
              "Get the total potential across the junction")
         .def_property_readonly("W", &PNJunction::get_W,
              "Get the depletion width")
         .def_property_readonly("W_p", &PNJunction::get_W_p,
              "Get the P-side depletion width")
         .def_property_readonly("W_n", &PNJunction::get_W_n,
              "Get the N-side depletion width")
         .def_property_readonly("N_A", &PNJunction::get_N_A,
              "Get the acceptor concentration")
         .def_property_readonly("N_D", &PNJunction::get_N_D,
              "Get the donor concentration")
         .def_property_readonly("n_i", &PNJunction::get_n_i,
              "Get the intrinsic carrier concentration")
         .def_property_readonly("junction_position", &PNJunction::get_junction_position,
              "Get the position of the junction");

    // Wrapper function for AdaptiveMesh::refineMesh
    auto refine_mesh_wrapper = [](Mesh& mesh, const std::vector<bool>& refine_flags) {
#ifdef USE_MPI
        AdaptiveMesh::refineMesh(mesh, refine_flags, MPI_COMM_WORLD);
#else
        // For non-MPI builds, use the overload without MPI_Comm
        AdaptiveMesh::refineMesh(mesh, refine_flags);
#endif
    };

    // AdaptiveMesh class for adaptive mesh refinement
    pybind11::class_<AdaptiveMesh>(m, "AdaptiveMesh")
        .def_static("refine_mesh", refine_mesh_wrapper,
                   pybind11::arg("mesh"), pybind11::arg("refine_flags"),
                   "Refine the mesh based on refinement flags")
        .def_static("compute_refinement_flags", &AdaptiveMesh::computeRefinementFlags,
                   pybind11::arg("mesh"), pybind11::arg("psi"), pybind11::arg("threshold"),
                   "Compute refinement flags based on solution gradients")
        .def_static("smooth_mesh", [](Mesh& mesh) {
                       AdaptiveMesh::smoothMesh(mesh, 3, 0.3);
                   },
                   pybind11::arg("mesh"),
                   "Smooth the mesh to improve element quality")
        .def_static("compute_triangle_quality", &AdaptiveMesh::computeTriangleQuality,
                   pybind11::arg("mesh"), pybind11::arg("elem_idx"),
                   "Compute the quality of a triangular element")
        .def_static("is_mesh_conforming", &AdaptiveMesh::isMeshConforming,
                   pybind11::arg("mesh"),
                   "Check if the mesh is conforming");

    // FullPoissonDriftDiffusionSolver class for comprehensive semiconductor device simulations
    pybind11::class_<FullPoissonDriftDiffusionSolver>(m, "FullPoissonDriftDiffusionSolver")
        .def(pybind11::init<Mesh&, std::function<double(double, double)>, std::function<double(double, double)>>(),
             pybind11::arg("mesh"), pybind11::arg("epsilon_r"), pybind11::arg("doping_profile"),
             "Construct a new FullPoissonDriftDiffusionSolver object")
        .def("solve", &FullPoissonDriftDiffusionSolver::solve,
             pybind11::arg("V_p"), pybind11::arg("V_n"), pybind11::arg("tolerance") = 1e-6, pybind11::arg("max_iter") = 100,
             "Solve the coupled Poisson-drift-diffusion equations")
        .def("get_potential", &FullPoissonDriftDiffusionSolver::get_potential,
             "Get the computed electrostatic potential")
        .def("get_electron_concentration", &FullPoissonDriftDiffusionSolver::get_electron_concentration,
             "Get the computed electron concentration")
        .def("get_hole_concentration", &FullPoissonDriftDiffusionSolver::get_hole_concentration,
             "Get the computed hole concentration")
        .def("get_electric_field", static_cast<Eigen::Vector2d (FullPoissonDriftDiffusionSolver::*)(double, double) const>(&FullPoissonDriftDiffusionSolver::get_electric_field),
             pybind11::arg("x"), pybind11::arg("y"),
             "Get the electric field at a given position")
        .def("get_electric_field_all", static_cast<const std::vector<Eigen::Vector2d>& (FullPoissonDriftDiffusionSolver::*)() const>(&FullPoissonDriftDiffusionSolver::get_electric_field),
             "Get the computed electric field at all mesh nodes")
        .def("get_electron_current_density", &FullPoissonDriftDiffusionSolver::get_electron_current_density,
             "Get the computed electron current density")
        .def("get_hole_current_density", &FullPoissonDriftDiffusionSolver::get_hole_current_density,
             "Get the computed hole current density")
        .def("set_heterojunction", &FullPoissonDriftDiffusionSolver::set_heterojunction,
             pybind11::arg("materials"), pybind11::arg("regions"),
             "Set the material properties for a heterojunction")
        .def("set_generation_recombination_model", &FullPoissonDriftDiffusionSolver::set_generation_recombination_model,
             pybind11::arg("g_r"),
             "Set the generation-recombination model")
        .def("set_mobility_models", &FullPoissonDriftDiffusionSolver::set_mobility_models,
             pybind11::arg("mu_n"), pybind11::arg("mu_p"),
             "Set the mobility models for electrons and holes")
        .def("set_carrier_statistics_model", &FullPoissonDriftDiffusionSolver::set_carrier_statistics_model,
             pybind11::arg("use_fermi_dirac"),
             "Set the carrier statistics model")
        .def("enable_quantum_corrections", &FullPoissonDriftDiffusionSolver::enable_quantum_corrections,
             pybind11::arg("enable"),
             "Enable or disable quantum corrections")
        .def("enable_adaptive_mesh_refinement", &FullPoissonDriftDiffusionSolver::enable_adaptive_mesh_refinement,
             pybind11::arg("enable"), pybind11::arg("refinement_threshold") = 0.1, pybind11::arg("max_refinement_level") = 3,
             "Enable or disable adaptive mesh refinement");

    // Callback system functions
    m.def("setCallback", &setCallback,
          pybind11::arg("name"), pybind11::arg("callback"),
          "Set a Python callback function");
    m.def("getCallback", [](const std::string& name) -> pybind11::object {
        auto callback = getCallback(name);
        if (callback) {
            return pybind11::object(*callback);
        }
        return pybind11::none();
    }, pybind11::arg("name"),
       "Get a Python callback function");
    m.def("clearCallbacks", &clearCallbacks,
          "Clear all Python callback functions");
    m.def("clearCallback", &clearCallback,
          pybind11::arg("name"),
          "Clear a specific Python callback function");
    m.def("hasCallback", &hasCallback,
          pybind11::arg("name"),
          "Check if a callback with the given name exists");
}