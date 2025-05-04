/**
 * @file error_estimator.cpp
 * @brief Implementation of the ErrorEstimator class for adaptive mesh refinement.
 *
 * This file contains the implementation of the ErrorEstimator class, which provides
 * methods for estimating errors in finite element solutions. The implementation
 * includes various error estimation techniques, including residual-based estimators,
 * recovery-based estimators (Zienkiewicz-Zhu), and hierarchical estimators.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "error_estimator.h"
#include <algorithm>
#include <cmath>
#include <numeric>

/**
 * @brief Constructs a new ErrorEstimator object.
 *
 * @param mesh The mesh to estimate errors on
 * @param estimator_type The type of error estimator to use
 * @param error_norm The error norm to use for error estimation
 */
ErrorEstimator::ErrorEstimator(const Mesh& mesh, EstimatorType estimator_type, ErrorNorm error_norm)
    : mesh(mesh), estimator_type(estimator_type), error_norm(error_norm) {
}

/**
 * @brief Estimates the error in the solution.
 *
 * This method estimates the error in the solution using the specified error
 * estimator and error norm. It computes local error indicators for each element
 * and returns a vector of error indicators.
 *
 * @param solution The solution vector
 * @param H The Hamiltonian matrix
 * @param M The mass matrix
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @return A vector of error indicators for each element
 */
std::vector<double> ErrorEstimator::estimateError(const Eigen::VectorXd& solution,
                                                const Eigen::SparseMatrix<std::complex<double>>& H,
                                                const Eigen::SparseMatrix<std::complex<double>>& M,
                                                std::function<double(double, double)> m_star,
                                                std::function<double(double, double)> V) {
    // Choose the appropriate error estimator based on the estimator type
    switch (estimator_type) {
        case EstimatorType::RESIDUAL:
            return computeResidualEstimator(solution, H, M, m_star, V);
        case EstimatorType::ZZ_RECOVERY:
            return computeZZEstimator(solution, H, M, m_star, V);
        case EstimatorType::HIERARCHICAL:
            return computeHierarchicalEstimator(solution, H, M, m_star, V);
        default:
            // Default to residual-based estimator
            return computeResidualEstimator(solution, H, M, m_star, V);
    }
}

/**
 * @brief Computes refinement flags based on error indicators.
 *
 * This method computes refinement flags based on the error indicators and
 * the specified marking strategy. Elements with error indicators exceeding
 * the threshold are marked for refinement.
 *
 * @param error_indicators The error indicators for each element
 * @param strategy The marking strategy to use (1: fixed fraction, 2: fixed threshold, 3: Dörfler)
 * @param parameter The parameter for the marking strategy (fraction, threshold, or Dörfler parameter)
 * @return A vector of boolean flags indicating which elements to refine
 */
std::vector<bool> ErrorEstimator::computeRefinementFlags(const std::vector<double>& error_indicators,
                                                      int strategy, double parameter) {
    // Initialize refinement flags
    std::vector<bool> refine_flags(error_indicators.size(), false);
    
    // Check if error_indicators is empty
    if (error_indicators.empty()) {
        return refine_flags;
    }
    
    // Strategy 1: Fixed fraction strategy
    // Refine a fixed fraction of elements with the largest error indicators
    if (strategy == 1) {
        // Create a vector of indices
        std::vector<size_t> indices(error_indicators.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort indices by error indicators in descending order
        std::sort(indices.begin(), indices.end(),
                 [&error_indicators](size_t i1, size_t i2) {
                     return error_indicators[i1] > error_indicators[i2];
                 });
        
        // Compute the number of elements to refine
        size_t num_to_refine = static_cast<size_t>(parameter * error_indicators.size());
        
        // Mark elements for refinement
        for (size_t i = 0; i < num_to_refine; ++i) {
            refine_flags[indices[i]] = true;
        }
    }
    // Strategy 2: Fixed threshold strategy
    // Refine elements with error indicators exceeding a fixed threshold
    else if (strategy == 2) {
        // Compute the maximum error indicator
        double max_error = *std::max_element(error_indicators.begin(), error_indicators.end());
        
        // Compute the threshold
        double threshold = parameter * max_error;
        
        // Mark elements for refinement
        for (size_t i = 0; i < error_indicators.size(); ++i) {
            if (error_indicators[i] > threshold) {
                refine_flags[i] = true;
            }
        }
    }
    // Strategy 3: Dörfler marking strategy (bulk chasing)
    // Refine elements that contribute a fixed fraction of the total error
    else if (strategy == 3) {
        // Compute the total error
        double total_error = 0.0;
        for (double error : error_indicators) {
            total_error += error * error;
        }
        
        // Create a vector of indices
        std::vector<size_t> indices(error_indicators.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort indices by error indicators in descending order
        std::sort(indices.begin(), indices.end(),
                 [&error_indicators](size_t i1, size_t i2) {
                     return error_indicators[i1] > error_indicators[i2];
                 });
        
        // Mark elements for refinement until the desired fraction of the total error is reached
        double marked_error = 0.0;
        size_t i = 0;
        while (marked_error < parameter * total_error && i < indices.size()) {
            refine_flags[indices[i]] = true;
            marked_error += error_indicators[indices[i]] * error_indicators[indices[i]];
            ++i;
        }
    }
    
    return refine_flags;
}

/**
 * @brief Computes the global error norm.
 *
 * This method computes the global error norm based on the local error indicators
 * and the specified error norm.
 *
 * @param error_indicators The error indicators for each element
 * @return The global error norm
 */
double ErrorEstimator::computeGlobalErrorNorm(const std::vector<double>& error_indicators) {
    // Compute the global error norm as the square root of the sum of squared error indicators
    double sum_squared = 0.0;
    for (double error : error_indicators) {
        sum_squared += error * error;
    }
    return std::sqrt(sum_squared);
}

/**
 * @brief Computes residual-based error indicators.
 *
 * This private method computes residual-based error indicators for each element.
 * It computes the element residual and jumps across element boundaries.
 *
 * @param solution The solution vector
 * @param H The Hamiltonian matrix
 * @param M The mass matrix
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @return A vector of error indicators for each element
 */
std::vector<double> ErrorEstimator::computeResidualEstimator(const Eigen::VectorXd& solution,
                                                          const Eigen::SparseMatrix<std::complex<double>>& H,
                                                          const Eigen::SparseMatrix<std::complex<double>>& M,
                                                          std::function<double(double, double)> m_star,
                                                          std::function<double(double, double)> V) {
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    int order = mesh.getElementOrder();
    
    // Initialize error indicators
    std::vector<double> error_indicators(elements.size(), 0.0);
    
    // Compute the residual: r = H*u - lambda*M*u
    // For simplicity, we'll use lambda = 0 (ground state)
    Eigen::VectorXd residual = (H * solution).real();
    
    // Compute element residuals and jumps
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& elem = elements[e];
        
        // Get element nodes
        std::vector<Eigen::Vector2d> elem_nodes;
        for (int i = 0; i < 3; ++i) {
            elem_nodes.push_back(nodes[elem[i]]);
        }
        
        // Compute element area
        double x1 = elem_nodes[0][0], y1 = elem_nodes[0][1];
        double x2 = elem_nodes[1][0], y2 = elem_nodes[1][1];
        double x3 = elem_nodes[2][0], y3 = elem_nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        
        // Compute element diameter (longest edge)
        double d1 = (elem_nodes[1] - elem_nodes[0]).norm();
        double d2 = (elem_nodes[2] - elem_nodes[1]).norm();
        double d3 = (elem_nodes[0] - elem_nodes[2]).norm();
        double h_e = std::max({d1, d2, d3});
        
        // Compute element centroid
        Eigen::Vector2d centroid = (elem_nodes[0] + elem_nodes[1] + elem_nodes[2]) / 3.0;
        
        // Compute effective mass at centroid
        double m = m_star(centroid[0], centroid[1]);
        
        // Compute element residual
        double elem_residual = 0.0;
        for (int i = 0; i < 3; ++i) {
            elem_residual += residual[elem[i]] * residual[elem[i]];
        }
        elem_residual = std::sqrt(elem_residual / 3.0);
        
        // Compute element error indicator
        error_indicators[e] = h_e * h_e * elem_residual * elem_residual * area;
        
        // Add jump terms across element boundaries
        // This would require information about neighboring elements
        // For simplicity, we'll omit this part in this implementation
    }
    
    return error_indicators;
}

/**
 * @brief Computes Zienkiewicz-Zhu recovery-based error indicators.
 *
 * This private method computes Zienkiewicz-Zhu recovery-based error indicators
 * for each element. It recovers a smoother gradient field and computes the
 * difference between the recovered gradient and the actual gradient.
 *
 * @param solution The solution vector
 * @param H The Hamiltonian matrix
 * @param M The mass matrix
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @return A vector of error indicators for each element
 */
std::vector<double> ErrorEstimator::computeZZEstimator(const Eigen::VectorXd& solution,
                                                    const Eigen::SparseMatrix<std::complex<double>>& H,
                                                    const Eigen::SparseMatrix<std::complex<double>>& M,
                                                    std::function<double(double, double)> m_star,
                                                    std::function<double(double, double)> V) {
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    int order = mesh.getElementOrder();
    
    // Initialize error indicators
    std::vector<double> error_indicators(elements.size(), 0.0);
    
    // Compute gradients at nodes
    std::vector<Eigen::Vector2d> gradients(nodes.size(), Eigen::Vector2d::Zero());
    std::vector<double> weights(nodes.size(), 0.0);
    
    // Compute element gradients and accumulate at nodes
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& elem = elements[e];
        
        // Get element nodes
        std::vector<Eigen::Vector2d> elem_nodes;
        for (int i = 0; i < 3; ++i) {
            elem_nodes.push_back(nodes[elem[i]]);
        }
        
        // Compute element area
        double x1 = elem_nodes[0][0], y1 = elem_nodes[0][1];
        double x2 = elem_nodes[1][0], y2 = elem_nodes[1][1];
        double x3 = elem_nodes[2][0], y3 = elem_nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        
        // Compute shape function gradients
        Eigen::Vector2d grad_N1((y2 - y3) / (2.0 * area), (x3 - x2) / (2.0 * area));
        Eigen::Vector2d grad_N2((y3 - y1) / (2.0 * area), (x1 - x3) / (2.0 * area));
        Eigen::Vector2d grad_N3((y1 - y2) / (2.0 * area), (x2 - x1) / (2.0 * area));
        
        // Compute element gradient
        Eigen::Vector2d elem_grad = solution[elem[0]] * grad_N1 + solution[elem[1]] * grad_N2 + solution[elem[2]] * grad_N3;
        
        // Accumulate gradients at nodes
        for (int i = 0; i < 3; ++i) {
            gradients[elem[i]] += elem_grad * area;
            weights[elem[i]] += area;
        }
    }
    
    // Compute recovered gradients at nodes
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (weights[i] > 0.0) {
            gradients[i] /= weights[i];
        }
    }
    
    // Compute error indicators
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& elem = elements[e];
        
        // Get element nodes
        std::vector<Eigen::Vector2d> elem_nodes;
        for (int i = 0; i < 3; ++i) {
            elem_nodes.push_back(nodes[elem[i]]);
        }
        
        // Compute element area
        double x1 = elem_nodes[0][0], y1 = elem_nodes[0][1];
        double x2 = elem_nodes[1][0], y2 = elem_nodes[1][1];
        double x3 = elem_nodes[2][0], y3 = elem_nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        
        // Compute shape function gradients
        Eigen::Vector2d grad_N1((y2 - y3) / (2.0 * area), (x3 - x2) / (2.0 * area));
        Eigen::Vector2d grad_N2((y3 - y1) / (2.0 * area), (x1 - x3) / (2.0 * area));
        Eigen::Vector2d grad_N3((y1 - y2) / (2.0 * area), (x2 - x1) / (2.0 * area));
        
        // Compute element gradient
        Eigen::Vector2d elem_grad = solution[elem[0]] * grad_N1 + solution[elem[1]] * grad_N2 + solution[elem[2]] * grad_N3;
        
        // Compute recovered gradient at element centroid
        Eigen::Vector2d recovered_grad = (gradients[elem[0]] + gradients[elem[1]] + gradients[elem[2]]) / 3.0;
        
        // Compute error indicator
        error_indicators[e] = (recovered_grad - elem_grad).squaredNorm() * area;
    }
    
    return error_indicators;
}

/**
 * @brief Computes hierarchical error indicators.
 *
 * This private method computes hierarchical error indicators for each element.
 * It uses a hierarchical basis enrichment to estimate the error.
 *
 * @param solution The solution vector
 * @param H The Hamiltonian matrix
 * @param M The mass matrix
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @return A vector of error indicators for each element
 */
std::vector<double> ErrorEstimator::computeHierarchicalEstimator(const Eigen::VectorXd& solution,
                                                              const Eigen::SparseMatrix<std::complex<double>>& H,
                                                              const Eigen::SparseMatrix<std::complex<double>>& M,
                                                              std::function<double(double, double)> m_star,
                                                              std::function<double(double, double)> V) {
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    int order = mesh.getElementOrder();
    
    // Initialize error indicators
    std::vector<double> error_indicators(elements.size(), 0.0);
    
    // For hierarchical error estimation, we need to solve a local problem
    // on each element with a higher-order basis
    // This is a simplified implementation that uses a heuristic approach
    
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& elem = elements[e];
        
        // Get element nodes
        std::vector<Eigen::Vector2d> elem_nodes;
        for (int i = 0; i < 3; ++i) {
            elem_nodes.push_back(nodes[elem[i]]);
        }
        
        // Compute element area
        double x1 = elem_nodes[0][0], y1 = elem_nodes[0][1];
        double x2 = elem_nodes[1][0], y2 = elem_nodes[1][1];
        double x3 = elem_nodes[2][0], y3 = elem_nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        
        // Compute element diameter (longest edge)
        double d1 = (elem_nodes[1] - elem_nodes[0]).norm();
        double d2 = (elem_nodes[2] - elem_nodes[1]).norm();
        double d3 = (elem_nodes[0] - elem_nodes[2]).norm();
        double h_e = std::max({d1, d2, d3});
        
        // Compute element centroid
        Eigen::Vector2d centroid = (elem_nodes[0] + elem_nodes[1] + elem_nodes[2]) / 3.0;
        
        // Compute effective mass at centroid
        double m = m_star(centroid[0], centroid[1]);
        
        // Compute solution gradient
        double u1 = solution[elem[0]], u2 = solution[elem[1]], u3 = solution[elem[2]];
        double grad_u_norm = std::sqrt(
            std::pow(u2 - u1, 2) / std::pow(d1, 2) +
            std::pow(u3 - u2, 2) / std::pow(d2, 2) +
            std::pow(u1 - u3, 2) / std::pow(d3, 2)
        );
        
        // Compute error indicator
        error_indicators[e] = h_e * h_e * grad_u_norm * grad_u_norm * area;
    }
    
    return error_indicators;
}

/**
 * @brief Computes the L2 norm of a function.
 *
 * This private method computes the L2 norm of a function over the mesh.
 *
 * @param func The function to compute the norm of
 * @return The L2 norm of the function
 */
double ErrorEstimator::computeL2Norm(std::function<double(double, double)> func) {
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    
    // Initialize norm
    double norm_squared = 0.0;
    
    // Integrate over all elements
    for (const auto& elem : elements) {
        // Get element nodes
        std::vector<Eigen::Vector2d> elem_nodes;
        for (int i = 0; i < 3; ++i) {
            elem_nodes.push_back(nodes[elem[i]]);
        }
        
        // Compute element area
        double x1 = elem_nodes[0][0], y1 = elem_nodes[0][1];
        double x2 = elem_nodes[1][0], y2 = elem_nodes[1][1];
        double x3 = elem_nodes[2][0], y3 = elem_nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        
        // Compute function values at element nodes
        double f1 = func(x1, y1);
        double f2 = func(x2, y2);
        double f3 = func(x3, y3);
        
        // Approximate integral using midpoint rule
        double f_mid = (f1 + f2 + f3) / 3.0;
        norm_squared += f_mid * f_mid * area;
    }
    
    return std::sqrt(norm_squared);
}

/**
 * @brief Computes the H1 norm of a function.
 *
 * This private method computes the H1 norm of a function over the mesh.
 * The H1 norm is the square root of the sum of the L2 norm squared and
 * the L2 norm of the gradient squared.
 *
 * @param func The function to compute the norm of
 * @param grad_func The gradient of the function
 * @return The H1 norm of the function
 */
double ErrorEstimator::computeH1Norm(std::function<double(double, double)> func,
                                   std::function<Eigen::Vector2d(double, double)> grad_func) {
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    
    // Initialize norms
    double l2_norm_squared = 0.0;
    double grad_norm_squared = 0.0;
    
    // Integrate over all elements
    for (const auto& elem : elements) {
        // Get element nodes
        std::vector<Eigen::Vector2d> elem_nodes;
        for (int i = 0; i < 3; ++i) {
            elem_nodes.push_back(nodes[elem[i]]);
        }
        
        // Compute element area
        double x1 = elem_nodes[0][0], y1 = elem_nodes[0][1];
        double x2 = elem_nodes[1][0], y2 = elem_nodes[1][1];
        double x3 = elem_nodes[2][0], y3 = elem_nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        
        // Compute function values at element nodes
        double f1 = func(x1, y1);
        double f2 = func(x2, y2);
        double f3 = func(x3, y3);
        
        // Compute gradient values at element nodes
        Eigen::Vector2d grad1 = grad_func(x1, y1);
        Eigen::Vector2d grad2 = grad_func(x2, y2);
        Eigen::Vector2d grad3 = grad_func(x3, y3);
        
        // Approximate integrals using midpoint rule
        double f_mid = (f1 + f2 + f3) / 3.0;
        Eigen::Vector2d grad_mid = (grad1 + grad2 + grad3) / 3.0;
        
        l2_norm_squared += f_mid * f_mid * area;
        grad_norm_squared += grad_mid.squaredNorm() * area;
    }
    
    return std::sqrt(l2_norm_squared + grad_norm_squared);
}

/**
 * @brief Computes the energy norm of a function.
 *
 * This private method computes the energy norm of a function over the mesh.
 * The energy norm is a weighted H1 norm based on the problem coefficients.
 *
 * @param func The function to compute the norm of
 * @param grad_func The gradient of the function
 * @param m_star Function that returns the effective mass at a given position
 * @param V Function that returns the potential at a given position
 * @return The energy norm of the function
 */
double ErrorEstimator::computeEnergyNorm(std::function<double(double, double)> func,
                                      std::function<Eigen::Vector2d(double, double)> grad_func,
                                      std::function<double(double, double)> m_star,
                                      std::function<double(double, double)> V) {
    // Get mesh data
    const auto& nodes = mesh.getNodes();
    const auto& elements = mesh.getElements();
    
    // Initialize norm
    double norm_squared = 0.0;
    
    // Integrate over all elements
    for (const auto& elem : elements) {
        // Get element nodes
        std::vector<Eigen::Vector2d> elem_nodes;
        for (int i = 0; i < 3; ++i) {
            elem_nodes.push_back(nodes[elem[i]]);
        }
        
        // Compute element area
        double x1 = elem_nodes[0][0], y1 = elem_nodes[0][1];
        double x2 = elem_nodes[1][0], y2 = elem_nodes[1][1];
        double x3 = elem_nodes[2][0], y3 = elem_nodes[2][1];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        
        // Compute centroid
        double xc = (x1 + x2 + x3) / 3.0;
        double yc = (y1 + y2 + y3) / 3.0;
        
        // Compute function value at centroid
        double f_mid = func(xc, yc);
        
        // Compute gradient at centroid
        Eigen::Vector2d grad_mid = grad_func(xc, yc);
        
        // Compute coefficients at centroid
        double m = m_star(xc, yc);
        double v = V(xc, yc);
        
        // Compute energy norm contribution
        norm_squared += (grad_mid.squaredNorm() / (2.0 * m) + v * f_mid * f_mid) * area;
    }
    
    return std::sqrt(norm_squared);
}
