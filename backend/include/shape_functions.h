/**
 * @file shape_functions.h
 * @brief Defines shape functions for finite element analysis.
 *
 * This file contains the declaration of shape functions and their derivatives
 * for linear (P1), quadratic (P2), and cubic (P3) elements.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include <array>

// Define CUDA device function qualifier if CUDA is enabled
#ifdef USE_CUDA
#include <cuda_runtime.h>
#define DEVICE_FUNC __host__ __device__
#else
#define DEVICE_FUNC
#endif

namespace ShapeFunctions {

/**
 * @brief Evaluates linear (P1) shape functions at a point.
 *
 * @param xi The first barycentric coordinate
 * @param eta The second barycentric coordinate
 * @param N Output array for shape function values
 */
DEVICE_FUNC inline void evaluateP1(double xi, double eta, double* N) {
    double zeta = 1.0 - xi - eta;
    N[0] = zeta;
    N[1] = xi;
    N[2] = eta;
}

/**
 * @brief Evaluates derivatives of linear (P1) shape functions.
 *
 * @param dN_dxi Output array for derivatives with respect to xi
 * @param dN_deta Output array for derivatives with respect to eta
 */
DEVICE_FUNC inline void evaluateP1Derivatives(double* dN_dxi, double* dN_deta) {
    dN_dxi[0] = -1.0;
    dN_dxi[1] = 1.0;
    dN_dxi[2] = 0.0;

    dN_deta[0] = -1.0;
    dN_deta[1] = 0.0;
    dN_deta[2] = 1.0;
}

/**
 * @brief Evaluates quadratic (P2) shape functions at a point.
 *
 * @param xi The first barycentric coordinate
 * @param eta The second barycentric coordinate
 * @param N Output array for shape function values
 */
DEVICE_FUNC inline void evaluateP2(double xi, double eta, double* N) {
    double zeta = 1.0 - xi - eta;

    // Vertex nodes
    N[0] = zeta * (2.0 * zeta - 1.0);
    N[1] = xi * (2.0 * xi - 1.0);
    N[2] = eta * (2.0 * eta - 1.0);

    // Edge nodes
    N[3] = 4.0 * xi * zeta;
    N[4] = 4.0 * xi * eta;
    N[5] = 4.0 * eta * zeta;
}

/**
 * @brief Evaluates derivatives of quadratic (P2) shape functions.
 *
 * @param xi The first barycentric coordinate
 * @param eta The second barycentric coordinate
 * @param dN_dxi Output array for derivatives with respect to xi
 * @param dN_deta Output array for derivatives with respect to eta
 */
DEVICE_FUNC inline void evaluateP2Derivatives(double xi, double eta, double* dN_dxi, double* dN_deta) {
    double zeta = 1.0 - xi - eta;

    // Vertex nodes
    dN_dxi[0] = -4.0 * zeta + 1.0;
    dN_dxi[1] = 4.0 * xi - 1.0;
    dN_dxi[2] = 0.0;

    dN_deta[0] = -4.0 * zeta + 1.0;
    dN_deta[1] = 0.0;
    dN_deta[2] = 4.0 * eta - 1.0;

    // Edge nodes
    dN_dxi[3] = 4.0 * (zeta - xi);
    dN_dxi[4] = 4.0 * eta;
    dN_dxi[5] = -4.0 * eta;

    dN_deta[3] = -4.0 * xi;
    dN_deta[4] = 4.0 * xi;
    dN_deta[5] = 4.0 * (zeta - eta);
}

/**
 * @brief Evaluates cubic (P3) shape functions at a point.
 *
 * @param xi The first barycentric coordinate
 * @param eta The second barycentric coordinate
 * @param N Output array for shape function values
 */
DEVICE_FUNC inline void evaluateP3(double xi, double eta, double* N) {
    double zeta = 1.0 - xi - eta;

    // Vertex nodes
    N[0] = 0.5 * zeta * (3.0 * zeta - 1.0) * (3.0 * zeta - 2.0);
    N[1] = 0.5 * xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0);
    N[2] = 0.5 * eta * (3.0 * eta - 1.0) * (3.0 * eta - 2.0);

    // Edge nodes (first set)
    N[3] = 4.5 * xi * zeta * (3.0 * zeta - 1.0);
    N[4] = 4.5 * xi * zeta * (3.0 * xi - 1.0);
    N[5] = 4.5 * xi * eta * (3.0 * xi - 1.0);
    N[6] = 4.5 * xi * eta * (3.0 * eta - 1.0);
    N[7] = 4.5 * eta * zeta * (3.0 * eta - 1.0);
    N[8] = 4.5 * eta * zeta * (3.0 * zeta - 1.0);

    // Interior node
    N[9] = 27.0 * xi * eta * zeta;
}

/**
 * @brief Evaluates derivatives of cubic (P3) shape functions.
 *
 * @param xi The first barycentric coordinate
 * @param eta The second barycentric coordinate
 * @param dN_dxi Output array for derivatives with respect to xi
 * @param dN_deta Output array for derivatives with respect to eta
 */
DEVICE_FUNC inline void evaluateP3Derivatives(double xi, double eta, double* dN_dxi, double* dN_deta) {
    double zeta = 1.0 - xi - eta;

    // Vertex nodes
    dN_dxi[0] = -0.5 * (27.0 * zeta * zeta - 18.0 * zeta + 2.0);
    dN_dxi[1] = 0.5 * (27.0 * xi * xi - 18.0 * xi + 2.0);
    dN_dxi[2] = 0.0;

    dN_deta[0] = -0.5 * (27.0 * zeta * zeta - 18.0 * zeta + 2.0);
    dN_deta[1] = 0.0;
    dN_deta[2] = 0.5 * (27.0 * eta * eta - 18.0 * eta + 2.0);

    // Edge nodes (first set)
    dN_dxi[3] = 4.5 * ((3.0 * zeta - 1.0) * zeta - (3.0 * zeta - 1.0) * xi - 3.0 * xi * zeta);
    dN_dxi[4] = 4.5 * ((3.0 * xi - 1.0) * zeta + (3.0 * xi - 1.0) * xi + 3.0 * xi * zeta);
    dN_dxi[5] = 4.5 * ((3.0 * xi - 1.0) * eta + (3.0 * xi - 1.0) * xi + 3.0 * xi * eta);
    dN_dxi[6] = 4.5 * (3.0 * xi - 1.0) * eta;
    dN_dxi[7] = 0.0;
    dN_dxi[8] = -4.5 * (3.0 * zeta - 1.0) * eta;

    dN_deta[3] = -4.5 * (3.0 * zeta - 1.0) * xi;
    dN_deta[4] = -4.5 * (3.0 * xi - 1.0) * xi;
    dN_deta[5] = 4.5 * (3.0 * xi - 1.0) * xi;
    dN_deta[6] = 4.5 * ((3.0 * eta - 1.0) * xi + (3.0 * eta - 1.0) * eta + 3.0 * xi * eta);
    dN_deta[7] = 4.5 * ((3.0 * eta - 1.0) * zeta + (3.0 * eta - 1.0) * eta + 3.0 * eta * zeta);
    dN_deta[8] = 4.5 * ((3.0 * zeta - 1.0) * zeta - (3.0 * zeta - 1.0) * eta - 3.0 * eta * zeta);

    // Interior node
    dN_dxi[9] = 27.0 * eta * zeta - 27.0 * xi * eta;
    dN_deta[9] = 27.0 * xi * zeta - 27.0 * xi * eta;
}

/**
 * @brief Gets quadrature points and weights for a given order.
 *
 * @param order The order of the finite elements (1 for P1, 2 for P2, 3 for P3)
 * @param quad_points Output array for quadrature points
 * @param quad_weights Output array for quadrature weights
 * @return The number of quadrature points
 */
DEVICE_FUNC inline int getQuadraturePoints(int order, double quad_points[][2], double* quad_weights) {
    if (order == 1) {
        // 1-point quadrature for linear elements
        quad_points[0][0] = 1.0 / 3.0;
        quad_points[0][1] = 1.0 / 3.0;
        quad_weights[0] = 0.5;
        return 1;
    } else if (order == 2) {
        // 3-point quadrature for quadratic elements
        quad_points[0][0] = 1.0 / 6.0;
        quad_points[0][1] = 1.0 / 6.0;
        quad_weights[0] = 1.0 / 6.0;

        quad_points[1][0] = 2.0 / 3.0;
        quad_points[1][1] = 1.0 / 6.0;
        quad_weights[1] = 1.0 / 6.0;

        quad_points[2][0] = 1.0 / 6.0;
        quad_points[2][1] = 2.0 / 3.0;
        quad_weights[2] = 1.0 / 6.0;

        return 3;
    } else if (order == 3) {
        // 6-point quadrature for cubic elements
        quad_points[0][0] = 0.091576213509771;
        quad_points[0][1] = 0.091576213509771;
        quad_weights[0] = 0.109951743655322 / 2.0;

        quad_points[1][0] = 0.816847572980459;
        quad_points[1][1] = 0.091576213509771;
        quad_weights[1] = 0.109951743655322 / 2.0;

        quad_points[2][0] = 0.091576213509771;
        quad_points[2][1] = 0.816847572980459;
        quad_weights[2] = 0.109951743655322 / 2.0;

        quad_points[3][0] = 0.445948490915965;
        quad_points[3][1] = 0.445948490915965;
        quad_weights[3] = 0.223381589678011 / 2.0;

        quad_points[4][0] = 0.108103018168070;
        quad_points[4][1] = 0.445948490915965;
        quad_weights[4] = 0.223381589678011 / 2.0;

        quad_points[5][0] = 0.445948490915965;
        quad_points[5][1] = 0.108103018168070;
        quad_weights[5] = 0.223381589678011 / 2.0;

        return 6;
    } else {
        // Unsupported order
        return 0;
    }
}

} // namespace ShapeFunctions
