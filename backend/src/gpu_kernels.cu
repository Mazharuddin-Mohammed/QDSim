/**
 * @file gpu_kernels.cu
 * @brief CUDA kernels for GPU-accelerated computations in QDSim.
 *
 * This file contains CUDA kernels for GPU-accelerated computations in QDSim,
 * including matrix assembly, eigenvalue solving, and field interpolation.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <complex>
#include <thrust/complex.h>

// Define USE_CUDA for shape_functions.h
#ifndef USE_CUDA
#define USE_CUDA
#endif

// Include shape functions
#include "shape_functions.h"

// Constants
const double HBAR = 6.582119569e-16; // Reduced Planck constant in eV·s
const double E_CHARGE = 1.602176634e-19; // Elementary charge in C
const double KB = 8.617333262e-5; // Boltzmann constant in eV/K

/**
 * @brief CUDA kernel for assembling element matrices.
 *
 * This kernel computes the element Hamiltonian and mass matrices for a single element.
 *
 * @param element_idx The index of the element
 * @param nodes The mesh nodes (flattened array of x,y coordinates)
 * @param elements The mesh elements (flattened array of node indices)
 * @param H_e The element Hamiltonian matrix (output)
 * @param M_e The element mass matrix (output)
 * @param m_star_values The effective mass values at quadrature points
 * @param V_values The potential values at quadrature points
 * @param num_nodes The number of nodes in the mesh
 * @param order The order of the finite elements
 */
__global__ void assemble_element_matrix_kernel(
    int element_idx,
    const double* nodes,
    const int* elements,
    thrust::complex<double>* H_e,
    thrust::complex<double>* M_e,
    const double* m_star_values,
    const double* V_values,
    int num_nodes,
    int order
) {
    // Get thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Get number of nodes per element based on order
    int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;

    // Check if thread indices are valid
    if (i >= nodes_per_elem || j >= nodes_per_elem) {
        return;
    }

    // Get element node indices
    int n1 = elements[3 * element_idx];
    int n2 = elements[3 * element_idx + 1];
    int n3 = elements[3 * element_idx + 2];

    // Get node coordinates
    double x1 = nodes[2 * n1];
    double y1 = nodes[2 * n1 + 1];
    double x2 = nodes[2 * n2];
    double y2 = nodes[2 * n2 + 1];
    double x3 = nodes[2 * n3];
    double y3 = nodes[2 * n3 + 1];

    // Calculate element area
    double area = 0.5 * fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

    // Calculate shape function gradients
    double b1 = (y2 - y3) / (2.0 * area);
    double b2 = (y3 - y1) / (2.0 * area);
    double b3 = (y1 - y2) / (2.0 * area);
    double c1 = (x3 - x2) / (2.0 * area);
    double c2 = (x1 - x3) / (2.0 * area);
    double c3 = (x2 - x1) / (2.0 * area);

    // Calculate element centroid
    double xc = (x1 + x2 + x3) / 3.0;
    double yc = (y1 + y2 + y3) / 3.0;

    // Get effective mass and potential at centroid
    double m = m_star_values[element_idx];
    double V_val = V_values[element_idx];

    // For linear elements (P1), we have 3 shape functions
    if (order == 1) {
        // Calculate gradients of shape functions
        double dNi_dx, dNi_dy, dNj_dx, dNj_dy;

        if (i == 0) { dNi_dx = b1; dNi_dy = c1; }
        else if (i == 1) { dNi_dx = b2; dNi_dy = c2; }
        else { dNi_dx = b3; dNi_dy = c3; }

        if (j == 0) { dNj_dx = b1; dNj_dy = c1; }
        else if (j == 1) { dNj_dx = b2; dNj_dy = c2; }
        else { dNj_dx = b3; dNj_dy = c3; }

        // Calculate shape functions at centroid
        double Ni, Nj;

        if (i == 0) Ni = 1.0/3.0;
        else if (i == 1) Ni = 1.0/3.0;
        else Ni = 1.0/3.0;

        if (j == 0) Nj = 1.0/3.0;
        else if (j == 1) Nj = 1.0/3.0;
        else Nj = 1.0/3.0;

        // Calculate Hamiltonian matrix element
        // H_ij = ∫ (ħ²/2m) ∇Ni·∇Nj + V·Ni·Nj dΩ
        double kinetic_term = (HBAR * HBAR / (2.0 * m)) * (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * area;
        double potential_term = V_val * Ni * Nj * area;
        H_e[i * nodes_per_elem + j] = thrust::complex<double>(kinetic_term + potential_term, 0.0);

        // Calculate mass matrix element
        // M_ij = ∫ Ni·Nj dΩ
        M_e[i * nodes_per_elem + j] = thrust::complex<double>(Ni * Nj * area, 0.0);
    }
    // For quadratic elements (P2), we have 6 shape functions
    else if (order == 2) {
        // Use shared memory for better performance
        __shared__ double s_quad_points[3][2];
        __shared__ double s_quad_weights[3];
        __shared__ double s_element_nodes[6][2];

        // Initialize shared memory in the first thread of each block
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Quadrature points and weights for quadratic elements
            s_quad_points[0][0] = 0.5; s_quad_points[0][1] = 0.0;
            s_quad_points[1][0] = 0.5; s_quad_points[1][1] = 0.5;
            s_quad_points[2][0] = 0.0; s_quad_points[2][1] = 0.5;

            s_quad_weights[0] = 1.0/6.0;
            s_quad_weights[1] = 1.0/6.0;
            s_quad_weights[2] = 1.0/6.0;

            // Load element node coordinates into shared memory
            for (int k = 0; k < 6; ++k) {
                int node_idx = elements[6 * element_idx + k];
                s_element_nodes[k][0] = nodes[2 * node_idx];
                s_element_nodes[k][1] = nodes[2 * node_idx + 1];
            }
        }

        // Ensure all threads have access to shared memory
        __syncthreads();

        // Initialize matrix entries
        double h_ij = 0.0;
        double m_ij = 0.0;

        // Number of quadrature points
        const int num_quad_points = 3;

        // Integrate over quadrature points
        for (int q = 0; q < num_quad_points; ++q) {
            double xi = s_quad_points[q][0];
            double eta = s_quad_points[q][1];
            double weight = s_quad_weights[q];

            // Compute shape functions and their derivatives at quadrature point
            double N[6];
            double dN_dxi[6];
            double dN_deta[6];

            // Use the optimized shape function evaluation
            ShapeFunctions::evaluateP2(xi, eta, N);
            ShapeFunctions::evaluateP2Derivatives(xi, eta, dN_dxi, dN_deta);

            // Compute Jacobian
            double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            for (int k = 0; k < 6; ++k) {
                J[0][0] += dN_dxi[k] * s_element_nodes[k][0];
                J[0][1] += dN_dxi[k] * s_element_nodes[k][1];
                J[1][0] += dN_deta[k] * s_element_nodes[k][0];
                J[1][1] += dN_deta[k] * s_element_nodes[k][1];
            }

            // Compute determinant of Jacobian
            double det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];

            // Compute inverse of Jacobian
            double J_inv[2][2];
            J_inv[0][0] = J[1][1] / det_J;
            J_inv[0][1] = -J[0][1] / det_J;
            J_inv[1][0] = -J[1][0] / det_J;
            J_inv[1][1] = J[0][0] / det_J;

            // Compute derivatives of shape functions with respect to x and y
            double dN_dx[6];
            double dN_dy[6];
            for (int k = 0; k < 6; ++k) {
                dN_dx[k] = J_inv[0][0] * dN_dxi[k] + J_inv[0][1] * dN_deta[k];
                dN_dy[k] = J_inv[1][0] * dN_dxi[k] + J_inv[1][1] * dN_deta[k];
            }

            // Compute physical coordinates of quadrature point
            double x_q = 0.0;
            double y_q = 0.0;
            for (int k = 0; k < 6; ++k) {
                x_q += N[k] * s_element_nodes[k][0];
                y_q += N[k] * s_element_nodes[k][1];
            }

            // Get effective mass and potential at quadrature point
            double m_q = m_star_values[element_idx]; // For simplicity, use element value
            double V_q = V_values[element_idx]; // For simplicity, use element value

            // Compute contribution to element matrices
            double kinetic_term = (HBAR * HBAR / (2.0 * m_q)) * (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]);
            double potential_term = V_q * N[i] * N[j];

            h_ij += weight * det_J * (kinetic_term + potential_term);
            m_ij += weight * det_J * N[i] * N[j];
        }

        // Set matrix entries
        H_e[i * nodes_per_elem + j] = thrust::complex<double>(h_ij, 0.0);
        M_e[i * nodes_per_elem + j] = thrust::complex<double>(m_ij, 0.0);
    }
    // For cubic elements (P3), we have 10 shape functions
    else if (order == 3) {
        // Use shared memory for better performance
        __shared__ double s_quad_points[6][2];
        __shared__ double s_quad_weights[6];
        __shared__ double s_element_nodes[10][2];
        __shared__ int s_num_quad_points;

        // Initialize shared memory in the first thread of each block
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Get quadrature points and weights
            s_num_quad_points = ShapeFunctions::getQuadraturePoints(3, s_quad_points, s_quad_weights);

            // Load element node coordinates into shared memory
            for (int k = 0; k < 10; ++k) {
                int node_idx = elements[10 * element_idx + k];
                s_element_nodes[k][0] = nodes[2 * node_idx];
                s_element_nodes[k][1] = nodes[2 * node_idx + 1];
            }
        }

        // Ensure all threads have access to shared memory
        __syncthreads();

        // Initialize matrix entries
        double h_ij = 0.0;
        double m_ij = 0.0;

        // Loop over quadrature points
        for (int q = 0; q < s_num_quad_points; ++q) {
            double xi = s_quad_points[q][0];
            double eta = s_quad_points[q][1];
            double weight = s_quad_weights[q];

            // Evaluate shape functions at quadrature point
            double N[10];
            ShapeFunctions::evaluateP3(xi, eta, N);

            // Evaluate derivatives of shape functions
            double dN_dxi[10];
            double dN_deta[10];
            ShapeFunctions::evaluateP3Derivatives(xi, eta, dN_dxi, dN_deta);

            // Compute Jacobian matrix
            double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            for (int k = 0; k < 10; ++k) {
                J[0][0] += dN_dxi[k] * s_element_nodes[k][0];
                J[0][1] += dN_dxi[k] * s_element_nodes[k][1];
                J[1][0] += dN_deta[k] * s_element_nodes[k][0];
                J[1][1] += dN_deta[k] * s_element_nodes[k][1];
            }

            // Compute determinant of Jacobian
            double det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];

            // Compute inverse of Jacobian
            double J_inv[2][2];
            J_inv[0][0] = J[1][1] / det_J;
            J_inv[0][1] = -J[0][1] / det_J;
            J_inv[1][0] = -J[1][0] / det_J;
            J_inv[1][1] = J[0][0] / det_J;

            // Compute derivatives of shape functions with respect to x and y
            double dN_dx[10];
            double dN_dy[10];
            for (int k = 0; k < 10; ++k) {
                dN_dx[k] = J_inv[0][0] * dN_dxi[k] + J_inv[0][1] * dN_deta[k];
                dN_dy[k] = J_inv[1][0] * dN_dxi[k] + J_inv[1][1] * dN_deta[k];
            }

            // Compute physical coordinates of quadrature point
            double x_q = 0.0;
            double y_q = 0.0;
            for (int k = 0; k < 10; ++k) {
                x_q += N[k] * s_element_nodes[k][0];
                y_q += N[k] * s_element_nodes[k][1];
            }

            // Get effective mass and potential at quadrature point
            double m_q = m_star_values[element_idx]; // For simplicity, use element value
            double V_q = V_values[element_idx]; // For simplicity, use element value

            // Compute contribution to element matrices
            double kinetic_term = (HBAR * HBAR / (2.0 * m_q)) * (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]);
            double potential_term = V_q * N[i] * N[j];

            h_ij += weight * det_J * (kinetic_term + potential_term);
            m_ij += weight * det_J * N[i] * N[j];
        }

        // Set matrix entries
        H_e[i * nodes_per_elem + j] = thrust::complex<double>(h_ij, 0.0);
        M_e[i * nodes_per_elem + j] = thrust::complex<double>(m_ij, 0.0);
    }
    // For other orders, set to zero
    else {
        H_e[i * nodes_per_elem + j] = thrust::complex<double>(0.0, 0.0);
        M_e[i * nodes_per_elem + j] = thrust::complex<double>(0.0, 0.0);
    }
}

/**
 * @brief CUDA kernel for field interpolation.
 *
 * This kernel interpolates a field at arbitrary points in the mesh.
 *
 * @param nodes The mesh nodes (flattened array of x,y coordinates)
 * @param elements The mesh elements (flattened array of node indices)
 * @param field The field values at mesh nodes
 * @param points The points at which to interpolate the field (flattened array of x,y coordinates)
 * @param values The interpolated field values (output)
 * @param num_nodes The number of nodes in the mesh
 * @param num_elements The number of elements in the mesh
 * @param num_points The number of points at which to interpolate
 */
__global__ void interpolate_field_kernel(
    const double* nodes,
    const int* elements,
    const double* field,
    const double* points,
    double* values,
    int num_nodes,
    int num_elements,
    int num_points
) {
    // Get thread index
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is valid
    if (point_idx >= num_points) {
        return;
    }

    // Get point coordinates
    double x = points[2 * point_idx];
    double y = points[2 * point_idx + 1];

    // Find the element containing the point
    int element_idx = -1;
    double lambda1, lambda2, lambda3;

    for (int e = 0; e < num_elements; ++e) {
        // Get element node indices
        int n1 = elements[3 * e];
        int n2 = elements[3 * e + 1];
        int n3 = elements[3 * e + 2];

        // Get node coordinates
        double x1 = nodes[2 * n1];
        double y1 = nodes[2 * n1 + 1];
        double x2 = nodes[2 * n2];
        double y2 = nodes[2 * n2 + 1];
        double x3 = nodes[2 * n3];
        double y3 = nodes[2 * n3 + 1];

        // Calculate element area
        double area = 0.5 * fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Calculate barycentric coordinates
        double lambda1_e = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / (2.0 * area);
        double lambda2_e = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / (2.0 * area);
        double lambda3_e = 1.0 - lambda1_e - lambda2_e;

        // Check if point is inside the element
        if (lambda1_e >= -1e-10 && lambda2_e >= -1e-10 && lambda3_e >= -1e-10) {
            element_idx = e;
            lambda1 = lambda1_e;
            lambda2 = lambda2_e;
            lambda3 = lambda3_e;
            break;
        }
    }

    // If point is not inside any element, set value to 0
    if (element_idx == -1) {
        values[point_idx] = 0.0;
        return;
    }

    // Get element node indices
    int n1 = elements[3 * element_idx];
    int n2 = elements[3 * element_idx + 1];
    int n3 = elements[3 * element_idx + 2];

    // Interpolate field value using barycentric coordinates
    values[point_idx] = lambda1 * field[n1] + lambda2 * field[n2] + lambda3 * field[n3];
}

/**
 * @brief CUDA kernel for batched assembly of element matrices.
 *
 * This kernel computes the element Hamiltonian and mass matrices for multiple elements in parallel.
 *
 * @param batch_start The starting index of the batch
 * @param batch_size The size of the batch
 * @param nodes The mesh nodes (flattened array of x,y coordinates)
 * @param elements The mesh elements (flattened array of node indices)
 * @param H_e The element Hamiltonian matrices (output)
 * @param M_e The element mass matrices (output)
 * @param m_star_values The effective mass values at quadrature points
 * @param V_values The potential values at quadrature points
 * @param num_nodes The number of nodes in the mesh
 * @param order The order of the finite elements
 */
__global__ void assemble_element_matrices_batched_kernel(
    int batch_start,
    int batch_size,
    const double* nodes,
    const int* elements,
    thrust::complex<double>* H_e,
    thrust::complex<double>* M_e,
    const double* m_star_values,
    const double* V_values,
    int num_nodes,
    int order
) {
    // Get thread indices
    int element_offset = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    int j = blockIdx.z * blockDim.y + threadIdx.y;

    // Get element index
    int element_idx = batch_start + element_offset;

    // Check if element index is valid
    if (element_idx >= batch_start + batch_size) {
        return;
    }

    // Get number of nodes per element based on order
    int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;

    // Check if thread indices are valid
    if (i >= nodes_per_elem || j >= nodes_per_elem) {
        return;
    }

    // Get matrix index
    int matrix_idx = element_offset * nodes_per_elem * nodes_per_elem + i * nodes_per_elem + j;

    // Call appropriate kernel logic based on order
    if (order == 1) {
        // Linear elements (P1)
        // Get element node indices
        int n1 = elements[3 * element_idx];
        int n2 = elements[3 * element_idx + 1];
        int n3 = elements[3 * element_idx + 2];

        // Get node coordinates
        double x1 = nodes[2 * n1];
        double y1 = nodes[2 * n1 + 1];
        double x2 = nodes[2 * n2];
        double y2 = nodes[2 * n2 + 1];
        double x3 = nodes[2 * n3];
        double y3 = nodes[2 * n3 + 1];

        // Calculate element area
        double area = 0.5 * fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

        // Calculate shape function gradients
        double b1 = (y2 - y3) / (2.0 * area);
        double b2 = (y3 - y1) / (2.0 * area);
        double b3 = (y1 - y2) / (2.0 * area);
        double c1 = (x3 - x2) / (2.0 * area);
        double c2 = (x1 - x3) / (2.0 * area);
        double c3 = (x2 - x1) / (2.0 * area);

        // Calculate element centroid
        double xc = (x1 + x2 + x3) / 3.0;
        double yc = (y1 + y2 + y3) / 3.0;

        // Get effective mass and potential at centroid
        double m = m_star_values[element_idx];
        double V_val = V_values[element_idx];

        // Calculate gradients of shape functions
        double dNi_dx, dNi_dy, dNj_dx, dNj_dy;

        if (i == 0) { dNi_dx = b1; dNi_dy = c1; }
        else if (i == 1) { dNi_dx = b2; dNi_dy = c2; }
        else { dNi_dx = b3; dNi_dy = c3; }

        if (j == 0) { dNj_dx = b1; dNj_dy = c1; }
        else if (j == 1) { dNj_dx = b2; dNj_dy = c2; }
        else { dNj_dx = b3; dNj_dy = c3; }

        // Calculate shape functions at centroid
        double Ni, Nj;

        if (i == 0) Ni = 1.0/3.0;
        else if (i == 1) Ni = 1.0/3.0;
        else Ni = 1.0/3.0;

        if (j == 0) Nj = 1.0/3.0;
        else if (j == 1) Nj = 1.0/3.0;
        else Nj = 1.0/3.0;

        // Calculate Hamiltonian matrix element
        double kinetic_term = (HBAR * HBAR / (2.0 * m)) * (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * area;
        double potential_term = V_val * Ni * Nj * area;
        H_e[matrix_idx] = thrust::complex<double>(kinetic_term + potential_term, 0.0);

        // Calculate mass matrix element
        M_e[matrix_idx] = thrust::complex<double>(Ni * Nj * area, 0.0);
    }
    else if (order == 2) {
        // Quadratic elements (P2)
        // Use shared memory for better performance
        __shared__ double s_quad_points[6][2];
        __shared__ double s_quad_weights[6];
        __shared__ double s_element_nodes[6][2];
        __shared__ int s_num_quad_points;

        // Initialize shared memory in the first thread of each block
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Get quadrature points and weights
            s_num_quad_points = ShapeFunctions::getQuadraturePoints(2, s_quad_points, s_quad_weights);

            // Load element node coordinates into shared memory
            for (int k = 0; k < 6; ++k) {
                int node_idx = elements[6 * element_idx + k];
                s_element_nodes[k][0] = nodes[2 * node_idx];
                s_element_nodes[k][1] = nodes[2 * node_idx + 1];
            }
        }

        // Ensure all threads have access to shared memory
        __syncthreads();

        // Initialize matrix entries
        double h_ij = 0.0;
        double m_ij = 0.0;

        // Loop over quadrature points
        for (int q = 0; q < s_num_quad_points; ++q) {
            double xi = s_quad_points[q][0];
            double eta = s_quad_points[q][1];
            double weight = s_quad_weights[q];

            // Evaluate shape functions at quadrature point
            double N[6];
            ShapeFunctions::evaluateP2(xi, eta, N);

            // Evaluate derivatives of shape functions
            double dN_dxi[6];
            double dN_deta[6];
            ShapeFunctions::evaluateP2Derivatives(xi, eta, dN_dxi, dN_deta);

            // Compute Jacobian matrix
            double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            for (int k = 0; k < 6; ++k) {
                J[0][0] += dN_dxi[k] * s_element_nodes[k][0];
                J[0][1] += dN_dxi[k] * s_element_nodes[k][1];
                J[1][0] += dN_deta[k] * s_element_nodes[k][0];
                J[1][1] += dN_deta[k] * s_element_nodes[k][1];
            }

            // Compute determinant of Jacobian
            double det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];

            // Compute inverse of Jacobian
            double J_inv[2][2];
            J_inv[0][0] = J[1][1] / det_J;
            J_inv[0][1] = -J[0][1] / det_J;
            J_inv[1][0] = -J[1][0] / det_J;
            J_inv[1][1] = J[0][0] / det_J;

            // Compute derivatives of shape functions with respect to x and y
            double dN_dx[6];
            double dN_dy[6];
            for (int k = 0; k < 6; ++k) {
                dN_dx[k] = J_inv[0][0] * dN_dxi[k] + J_inv[0][1] * dN_deta[k];
                dN_dy[k] = J_inv[1][0] * dN_dxi[k] + J_inv[1][1] * dN_deta[k];
            }

            // Get effective mass and potential at quadrature point
            double m_q = m_star_values[element_idx]; // For simplicity, use element value
            double V_q = V_values[element_idx]; // For simplicity, use element value

            // Compute contribution to element matrices
            double kinetic_term = (HBAR * HBAR / (2.0 * m_q)) * (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]);
            double potential_term = V_q * N[i] * N[j];

            h_ij += weight * det_J * (kinetic_term + potential_term);
            m_ij += weight * det_J * N[i] * N[j];
        }

        // Set matrix entries
        H_e[matrix_idx] = thrust::complex<double>(h_ij, 0.0);
        M_e[matrix_idx] = thrust::complex<double>(m_ij, 0.0);
    }
    else if (order == 3) {
        // Cubic elements (P3)
        // Use shared memory for better performance
        __shared__ double s_quad_points[6][2];
        __shared__ double s_quad_weights[6];
        __shared__ double s_element_nodes[10][2];
        __shared__ int s_num_quad_points;

        // Initialize shared memory in the first thread of each block
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Get quadrature points and weights
            s_num_quad_points = ShapeFunctions::getQuadraturePoints(3, s_quad_points, s_quad_weights);

            // Load element node coordinates into shared memory
            for (int k = 0; k < 10; ++k) {
                int node_idx = elements[10 * element_idx + k];
                s_element_nodes[k][0] = nodes[2 * node_idx];
                s_element_nodes[k][1] = nodes[2 * node_idx + 1];
            }
        }

        // Ensure all threads have access to shared memory
        __syncthreads();

        // Initialize matrix entries
        double h_ij = 0.0;
        double m_ij = 0.0;

        // Loop over quadrature points
        for (int q = 0; q < s_num_quad_points; ++q) {
            double xi = s_quad_points[q][0];
            double eta = s_quad_points[q][1];
            double weight = s_quad_weights[q];

            // Evaluate shape functions at quadrature point
            double N[10];
            ShapeFunctions::evaluateP3(xi, eta, N);

            // Evaluate derivatives of shape functions
            double dN_dxi[10];
            double dN_deta[10];
            ShapeFunctions::evaluateP3Derivatives(xi, eta, dN_dxi, dN_deta);

            // Compute Jacobian matrix
            double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            for (int k = 0; k < 10; ++k) {
                J[0][0] += dN_dxi[k] * s_element_nodes[k][0];
                J[0][1] += dN_dxi[k] * s_element_nodes[k][1];
                J[1][0] += dN_deta[k] * s_element_nodes[k][0];
                J[1][1] += dN_deta[k] * s_element_nodes[k][1];
            }

            // Compute determinant of Jacobian
            double det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];

            // Compute inverse of Jacobian
            double J_inv[2][2];
            J_inv[0][0] = J[1][1] / det_J;
            J_inv[0][1] = -J[0][1] / det_J;
            J_inv[1][0] = -J[1][0] / det_J;
            J_inv[1][1] = J[0][0] / det_J;

            // Compute derivatives of shape functions with respect to x and y
            double dN_dx[10];
            double dN_dy[10];
            for (int k = 0; k < 10; ++k) {
                dN_dx[k] = J_inv[0][0] * dN_dxi[k] + J_inv[0][1] * dN_deta[k];
                dN_dy[k] = J_inv[1][0] * dN_dxi[k] + J_inv[1][1] * dN_deta[k];
            }

            // Get effective mass and potential at quadrature point
            double m_q = m_star_values[element_idx]; // For simplicity, use element value
            double V_q = V_values[element_idx]; // For simplicity, use element value

            // Compute contribution to element matrices
            double kinetic_term = (HBAR * HBAR / (2.0 * m_q)) * (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]);
            double potential_term = V_q * N[i] * N[j];

            h_ij += weight * det_J * (kinetic_term + potential_term);
            m_ij += weight * det_J * N[i] * N[j];
        }

        // Set matrix entries
        H_e[matrix_idx] = thrust::complex<double>(h_ij, 0.0);
        M_e[matrix_idx] = thrust::complex<double>(m_ij, 0.0);
    }
    else {
        // Unsupported order
        H_e[matrix_idx] = thrust::complex<double>(0.0, 0.0);
        M_e[matrix_idx] = thrust::complex<double>(0.0, 0.0);
    }
}

// Wrapper functions for CUDA kernels

// Forward declarations of CUDA kernels are not needed as they are already defined above

extern "C" void assemble_element_matrix_cuda(
    int element_idx,
    const double* nodes,
    const int* elements,
    std::complex<double>* H_e,
    std::complex<double>* M_e,
    const double* m_star_values,
    const double* V_values,
    int order) {
    // Get number of nodes per element based on order
    int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;

    // Allocate device memory for element matrices
    thrust::complex<double>* d_H_e = nullptr;
    thrust::complex<double>* d_M_e = nullptr;

    cudaError_t cuda_status = cudaMalloc(&d_H_e, nodes_per_elem * nodes_per_elem * sizeof(thrust::complex<double>));
    if (cuda_status != cudaSuccess) {
        printf("Failed to allocate GPU memory for element Hamiltonian matrix: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMalloc(&d_M_e, nodes_per_elem * nodes_per_elem * sizeof(thrust::complex<double>));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_H_e);
        printf("Failed to allocate GPU memory for element mass matrix: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Set up kernel launch configuration
    // For higher-order elements, use smaller block sizes to avoid exceeding resources
    int block_dim = (nodes_per_elem > 8) ? 8 : nodes_per_elem;
    dim3 block_size(block_dim, block_dim);
    dim3 grid_size((nodes_per_elem + block_dim - 1) / block_dim,
                  (nodes_per_elem + block_dim - 1) / block_dim);

    // Launch kernel
    assemble_element_matrix_kernel<<<grid_size, block_size>>>(
        element_idx, nodes, elements, d_H_e, d_M_e, m_star_values, V_values, nodes_per_elem, order
    );

    // Check for kernel launch errors
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_e);
        cudaFree(d_H_e);
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Copy results back to host
    cuda_status = cudaMemcpy(H_e, d_H_e, nodes_per_elem * nodes_per_elem * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_e);
        cudaFree(d_H_e);
        printf("Failed to copy element Hamiltonian matrix from GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMemcpy(M_e, d_M_e, nodes_per_elem * nodes_per_elem * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_e);
        cudaFree(d_H_e);
        printf("Failed to copy element mass matrix from GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Clean up
    cudaFree(d_M_e);
    cudaFree(d_H_e);
}

extern "C" void interpolate_field_cuda(
    const double* nodes,
    const int* elements,
    const double* field,
    const double* points,
    double* values,
    int num_nodes,
    int num_elements,
    int num_points) {
    // Allocate device memory
    double* d_nodes = nullptr;
    int* d_elements = nullptr;
    double* d_field = nullptr;
    double* d_points = nullptr;
    double* d_values = nullptr;

    // Allocate memory on GPU
    cudaError_t cuda_status = cudaMalloc(&d_nodes, 2 * num_nodes * sizeof(double));
    if (cuda_status != cudaSuccess) {
        printf("Failed to allocate GPU memory for nodes: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMalloc(&d_elements, 3 * num_elements * sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_nodes);
        printf("Failed to allocate GPU memory for elements: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMalloc(&d_field, num_nodes * sizeof(double));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to allocate GPU memory for field: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMalloc(&d_points, 2 * num_points * sizeof(double));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to allocate GPU memory for points: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMalloc(&d_values, num_points * sizeof(double));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to allocate GPU memory for values: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Copy data to GPU
    cuda_status = cudaMemcpy(d_nodes, nodes, 2 * num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to copy nodes to GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMemcpy(d_elements, elements, 3 * num_elements * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to copy elements to GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMemcpy(d_field, field, num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to copy field to GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMemcpy(d_points, points, 2 * num_points * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to copy points to GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Set up kernel launch configuration
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;

    // Launch kernel
    interpolate_field_kernel<<<grid_size, block_size>>>(
        d_nodes, d_elements, d_field, d_points, d_values, num_nodes, num_elements, num_points
    );

    // Check for kernel launch errors
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Copy results back to host
    cuda_status = cudaMemcpy(values, d_values, num_points * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_values);
        cudaFree(d_points);
        cudaFree(d_field);
        cudaFree(d_elements);
        cudaFree(d_nodes);
        printf("Failed to copy values from GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Clean up
    cudaFree(d_values);
    cudaFree(d_points);
    cudaFree(d_field);
    cudaFree(d_elements);
    cudaFree(d_nodes);
}

extern "C" void assemble_element_matrices_batched_cuda(
    int batch_start,
    int batch_size,
    const double* nodes,
    const int* elements,
    std::complex<double>* H_e,
    std::complex<double>* M_e,
    const double* m_star_values,
    const double* V_values,
    int order) {
    // Get number of nodes per element based on order
    int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;

    // Allocate device memory for element matrices
    thrust::complex<double>* d_H_e = nullptr;
    thrust::complex<double>* d_M_e = nullptr;

    cudaError_t cuda_status = cudaMalloc(&d_H_e, batch_size * nodes_per_elem * nodes_per_elem * sizeof(thrust::complex<double>));
    if (cuda_status != cudaSuccess) {
        printf("Failed to allocate GPU memory for element Hamiltonian matrices: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMalloc(&d_M_e, batch_size * nodes_per_elem * nodes_per_elem * sizeof(thrust::complex<double>));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_H_e);
        printf("Failed to allocate GPU memory for element mass matrices: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Set up kernel launch configuration
    // For higher-order elements, use smaller block sizes to avoid exceeding resources
    int block_dim = (nodes_per_elem > 8) ? 4 : 8;
    dim3 block_size(block_dim, block_dim);

    // Calculate grid dimensions to cover all elements and matrix entries
    dim3 grid_size(
        batch_size,
        (nodes_per_elem + block_dim - 1) / block_dim,
        (nodes_per_elem + block_dim - 1) / block_dim
    );

    // Launch kernel with optimized configuration
    assemble_element_matrices_batched_kernel<<<grid_size, block_size>>>(
        batch_start, batch_size, nodes, elements, d_H_e, d_M_e, m_star_values, V_values, nodes_per_elem, order
    );

    // Check for kernel launch errors
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_e);
        cudaFree(d_H_e);
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Copy results back to host
    cuda_status = cudaMemcpy(H_e, d_H_e, batch_size * nodes_per_elem * nodes_per_elem * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_e);
        cudaFree(d_H_e);
        printf("Failed to copy element Hamiltonian matrices from GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cuda_status = cudaMemcpy(M_e, d_M_e, batch_size * nodes_per_elem * nodes_per_elem * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_M_e);
        cudaFree(d_H_e);
        printf("Failed to copy element mass matrices from GPU: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // Clean up
    cudaFree(d_M_e);
    cudaFree(d_H_e);
}

