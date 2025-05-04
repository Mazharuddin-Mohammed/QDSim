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

// Constants
const double HBAR = 1.054e-34; // Reduced Planck constant in J·s
const double E_CHARGE = 1.602e-19; // Elementary charge in C

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
    // For higher-order elements, we would need to implement quadrature integration
    else {
        // Set to zero for now
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

// Wrapper functions for CUDA kernels

extern "C" void assemble_element_matrix_cuda(
    int element_idx,
    const double* nodes,
    const int* elements,
    std::complex<double>* H_e,
    std::complex<double>* M_e,
    const double* m_star_values,
    const double* V_values,
    int num_nodes,
    int order
) {
    // Convert std::complex to thrust::complex
    thrust::complex<double>* H_e_thrust = reinterpret_cast<thrust::complex<double>*>(H_e);
    thrust::complex<double>* M_e_thrust = reinterpret_cast<thrust::complex<double>*>(M_e);
    
    // Get number of nodes per element based on order
    int nodes_per_elem = (order == 1) ? 3 : (order == 2) ? 6 : 10;
    
    // Define grid and block dimensions
    dim3 block_dim(8, 8);
    dim3 grid_dim((nodes_per_elem + block_dim.x - 1) / block_dim.x,
                 (nodes_per_elem + block_dim.y - 1) / block_dim.y);
    
    // Launch kernel
    assemble_element_matrix_kernel<<<grid_dim, block_dim>>>(
        element_idx,
        nodes,
        elements,
        H_e_thrust,
        M_e_thrust,
        m_star_values,
        V_values,
        num_nodes,
        order
    );
    
    // Check for errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_status));
    }
    
    // Wait for kernel to finish
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_status));
    }
}

extern "C" void interpolate_field_cuda(
    const double* nodes,
    const int* elements,
    const double* field,
    const double* points,
    double* values,
    int num_nodes,
    int num_elements,
    int num_points
) {
    // Define grid and block dimensions
    int block_dim = 256;
    int grid_dim = (num_points + block_dim - 1) / block_dim;
    
    // Launch kernel
    interpolate_field_kernel<<<grid_dim, block_dim>>>(
        nodes,
        elements,
        field,
        points,
        values,
        num_nodes,
        num_elements,
        num_points
    );
    
    // Check for errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_status));
    }
    
    // Wait for kernel to finish
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_status));
    }
}
