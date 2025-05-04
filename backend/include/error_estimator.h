#pragma once
/**
 * @file error_estimator.h
 * @brief Defines the ErrorEstimator class for adaptive mesh refinement.
 *
 * This file contains the declaration of the ErrorEstimator class, which provides
 * methods for estimating errors in finite element solutions. The class implements
 * various error estimation techniques, including residual-based estimators,
 * recovery-based estimators (Zienkiewicz-Zhu), and hierarchical estimators.
 *
 * The error estimation process involves:
 * 1. Computing local error indicators for each element
 * 2. Computing global error norms
 * 3. Marking elements for refinement based on error thresholds
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include <Eigen/Dense>
#include <vector>
#include <functional>

/**
 * @enum ErrorNorm
 * @brief Enumeration of error norms for error estimation.
 */
enum class ErrorNorm {
    L2,     ///< L2 norm (square root of integral of squared error)
    H1,     ///< H1 norm (L2 norm of error plus L2 norm of error gradient)
    ENERGY  ///< Energy norm (weighted H1 norm based on problem coefficients)
};

/**
 * @enum EstimatorType
 * @brief Enumeration of error estimator types.
 */
enum class EstimatorType {
    RESIDUAL,     ///< Residual-based error estimator
    ZZ_RECOVERY,  ///< Zienkiewicz-Zhu recovery-based error estimator
    HIERARCHICAL  ///< Hierarchical error estimator
};

/**
 * @class ErrorEstimator
 * @brief Implements error estimation techniques for adaptive mesh refinement.
 *
 * The ErrorEstimator class provides methods for estimating errors in finite element
 * solutions and marking elements for refinement. It supports various error estimation
 * techniques and error norms.
 */
class ErrorEstimator {
public:
    /**
     * @brief Constructs a new ErrorEstimator object.
     *
     * @param mesh The mesh to estimate errors on
     * @param estimator_type The type of error estimator to use
     * @param error_norm The error norm to use for error estimation
     */
    ErrorEstimator(const Mesh& mesh, EstimatorType estimator_type = EstimatorType::RESIDUAL,
                  ErrorNorm error_norm = ErrorNorm::ENERGY);

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
    std::vector<double> estimateError(const Eigen::VectorXd& solution,
                                     const Eigen::SparseMatrix<std::complex<double>>& H,
                                     const Eigen::SparseMatrix<std::complex<double>>& M,
                                     std::function<double(double, double)> m_star,
                                     std::function<double(double, double)> V);

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
    std::vector<bool> computeRefinementFlags(const std::vector<double>& error_indicators,
                                           int strategy = 1, double parameter = 0.3);

    /**
     * @brief Computes the global error norm.
     *
     * This method computes the global error norm based on the local error indicators
     * and the specified error norm.
     *
     * @param error_indicators The error indicators for each element
     * @return The global error norm
     */
    double computeGlobalErrorNorm(const std::vector<double>& error_indicators);

private:
    const Mesh& mesh;                ///< Reference to the mesh
    EstimatorType estimator_type;    ///< Type of error estimator
    ErrorNorm error_norm;            ///< Error norm to use

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
    std::vector<double> computeResidualEstimator(const Eigen::VectorXd& solution,
                                               const Eigen::SparseMatrix<std::complex<double>>& H,
                                               const Eigen::SparseMatrix<std::complex<double>>& M,
                                               std::function<double(double, double)> m_star,
                                               std::function<double(double, double)> V);

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
    std::vector<double> computeZZEstimator(const Eigen::VectorXd& solution,
                                         const Eigen::SparseMatrix<std::complex<double>>& H,
                                         const Eigen::SparseMatrix<std::complex<double>>& M,
                                         std::function<double(double, double)> m_star,
                                         std::function<double(double, double)> V);

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
    std::vector<double> computeHierarchicalEstimator(const Eigen::VectorXd& solution,
                                                   const Eigen::SparseMatrix<std::complex<double>>& H,
                                                   const Eigen::SparseMatrix<std::complex<double>>& M,
                                                   std::function<double(double, double)> m_star,
                                                   std::function<double(double, double)> V);

    /**
     * @brief Computes the L2 norm of a function.
     *
     * This private method computes the L2 norm of a function over the mesh.
     *
     * @param func The function to compute the norm of
     * @return The L2 norm of the function
     */
    double computeL2Norm(std::function<double(double, double)> func);

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
    double computeH1Norm(std::function<double(double, double)> func,
                        std::function<Eigen::Vector2d(double, double)> grad_func);

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
    double computeEnergyNorm(std::function<double(double, double)> func,
                           std::function<Eigen::Vector2d(double, double)> grad_func,
                           std::function<double(double, double)> m_star,
                           std::function<double(double, double)> V);
};
