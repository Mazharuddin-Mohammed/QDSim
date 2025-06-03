#include "fused_gpu_kernels.cuh"
#include <cmath>

namespace QDSim {
namespace GPU {

__global__ void fusedHamiltonianAssembly(
    const double* __restrict__ nodes,
    const int* __restrict__ elements,
    const double* __restrict__ potential,
    const double* __restrict__ effective_mass,
    cuDoubleComplex* __restrict__ hamiltonian_matrix,
    cuDoubleComplex* __restrict__ mass_matrix,
    int num_elements,
    int nodes_per_element,
    int total_nodes) {
    
    // Cooperative groups for better memory coalescing
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int element_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (element_idx >= num_elements) return;
    
    // Shared memory for element matrices (optimized for memory coalescing)
    extern __shared__ double shared_memory[];
    double* shared_nodes = shared_memory;
    cuDoubleComplex* shared_H = reinterpret_cast<cuDoubleComplex*>(
        shared_nodes + blockDim.x * nodes_per_element * 2);
    cuDoubleComplex* shared_M = shared_H + blockDim.x * nodes_per_element * nodes_per_element;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int thread_offset = threadIdx.x;
    
    // Load element nodes with coalesced access
    int element_offset = element_idx * nodes_per_element;
    for (int i = 0; i < nodes_per_element; ++i) {
        int node_id = elements[element_offset + i];
        shared_nodes[thread_offset * nodes_per_element * 2 + i * 2] = nodes[node_id * 2];
        shared_nodes[thread_offset * nodes_per_element * 2 + i * 2 + 1] = nodes[node_id * 2 + 1];
    }
    
    __syncthreads();
    
    // Compute element matrices using optimized memory access patterns
    double* element_nodes = &shared_nodes[thread_offset * nodes_per_element * 2];
    cuDoubleComplex* element_H = &shared_H[thread_offset * nodes_per_element * nodes_per_element];
    cuDoubleComplex* element_M = &shared_M[thread_offset * nodes_per_element * nodes_per_element];
    
    // Get material properties
    double m_eff = effective_mass[element_idx];
    double V = potential[element_idx];
    
    // Vectorized computation for better throughput
    for (int i = 0; i < nodes_per_element; ++i) {
        for (int j = i; j < nodes_per_element; ++j) { // Exploit symmetry
            
            // Compute shape function derivatives (optimized)
            double dNi_dx, dNi_dy, dNj_dx, dNj_dy;
            computeShapeFunctionDerivatives(element_nodes, i, j, dNi_dx, dNi_dy, dNj_dx, dNj_dy);
            
            // Compute matrix elements
            double kinetic = (HBAR_SQ_OVER_2M / m_eff) * (dNi_dx * dNj_dx + dNi_dy * dNj_dy);
            double potential_term = V * computeShapeFunctionProduct(element_nodes, i, j);
            
            element_H[i * nodes_per_element + j] = make_cuDoubleComplex(kinetic + potential_term, 0.0);
            element_M[i * nodes_per_element + j] = make_cuDoubleComplex(
                computeShapeFunctionProduct(element_nodes, i, j), 0.0);
            
            // Exploit symmetry
            if (i != j) {
                element_H[j * nodes_per_element + i] = element_H[i * nodes_per_element + j];
                element_M[j * nodes_per_element + i] = element_M[i * nodes_per_element + j];
            }
        }
    }
    
    __syncthreads();
    
    // Assemble into global matrices with coalesced writes
    for (int i = 0; i < nodes_per_element; ++i) {
        for (int j = 0; j < nodes_per_element; ++j) {
            int global_i = elements[element_offset + i];
            int global_j = elements[element_offset + j];
            int global_idx = global_i * total_nodes + global_j;
            
            // Atomic add for thread safety
            atomicAdd(&hamiltonian_matrix[global_idx].x, element_H[i * nodes_per_element + j].x);
            atomicAdd(&mass_matrix[global_idx].x, element_M[i * nodes_per_element + j].x);
        }
    }
}

__global__ void fusedSpMV(
    const cuDoubleComplex* __restrict__ matrix_values,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const cuDoubleComplex* __restrict__ input_vector,
    cuDoubleComplex* __restrict__ output_vector,
    int num_rows) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= num_rows) return;
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    
    // Vectorized loop for better performance
    for (int j = start; j < end; ++j) {
        int col = col_indices[j];
        cuDoubleComplex matrix_val = matrix_values[j];
        cuDoubleComplex vector_val = input_vector[col];
        
        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        sum.x += matrix_val.x * vector_val.x - matrix_val.y * vector_val.y;
        sum.y += matrix_val.x * vector_val.y + matrix_val.y * vector_val.x;
    }
    
    output_vector[row] = sum;
}

__global__ void fusedElectronDensity(
    const cuDoubleComplex* __restrict__ eigenvectors,
    const double* __restrict__ eigenvalues,
    const double* __restrict__ fermi_weights,
    double* __restrict__ electron_density,
    int num_nodes,
    int num_states,
    double temperature,
    double fermi_level) {
    
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node >= num_nodes) return;
    
    double density = 0.0;
    const double kB = 1.380649e-23; // Boltzmann constant
    
    for (int state = 0; state < num_states; ++state) {
        // Get wavefunction amplitude at this node
        cuDoubleComplex psi = eigenvectors[state * num_nodes + node];
        double psi_squared = psi.x * psi.x + psi.y * psi.y;
        
        // Fermi-Dirac distribution
        double energy_diff = eigenvalues[state] - fermi_level;
        double fermi_factor = 1.0 / (1.0 + exp(energy_diff / (kB * temperature)));
        
        density += fermi_factor * psi_squared;
    }
    
    electron_density[node] = density;
}

__global__ void fusedPoissonSolver(
    const double* __restrict__ charge_density,
    const double* __restrict__ permittivity,
    const double* __restrict__ boundary_values,
    const int* __restrict__ boundary_flags,
    double* __restrict__ potential,
    double* __restrict__ electric_field_x,
    double* __restrict__ electric_field_y,
    int nx, int ny,
    double dx, double dy) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= nx || j >= ny) return;
    
    int idx = j * nx + i;
    
    // Handle boundary conditions
    if (boundary_flags[idx] != 0) {
        potential[idx] = boundary_values[idx];
        electric_field_x[idx] = 0.0;
        electric_field_y[idx] = 0.0;
        return;
    }
    
    // 5-point stencil for Poisson equation
    double eps = permittivity[idx];
    double rho = charge_density[idx];
    
    double phi_center = potential[idx];
    double phi_left = (i > 0) ? potential[idx - 1] : phi_center;
    double phi_right = (i < nx - 1) ? potential[idx + 1] : phi_center;
    double phi_bottom = (j > 0) ? potential[idx - nx] : phi_center;
    double phi_top = (j < ny - 1) ? potential[idx + nx] : phi_center;
    
    // Update potential using finite difference
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double new_potential = (phi_left + phi_right) / dx2 + (phi_bottom + phi_top) / dy2;
    new_potential = new_potential / (2.0 / dx2 + 2.0 / dy2) - rho / eps;
    
    potential[idx] = new_potential;
    
    // Compute electric field: E = -∇φ
    electric_field_x[idx] = -(phi_right - phi_left) / (2.0 * dx);
    electric_field_y[idx] = -(phi_top - phi_bottom) / (2.0 * dy);
}

// Device function implementations
__device__ inline void computeShapeFunctionDerivatives(
    const double* element_nodes,
    int i, int j,
    double& dNi_dx, double& dNi_dy,
    double& dNj_dx, double& dNj_dy) {
    
    // Simplified linear triangle element derivatives
    // For a more complete implementation, this would include
    // proper coordinate transformations and Jacobian calculations
    
    // Get node coordinates
    double x1 = element_nodes[0], y1 = element_nodes[1];
    double x2 = element_nodes[2], y2 = element_nodes[3];
    double x3 = element_nodes[4], y3 = element_nodes[5];
    
    // Compute area (2 * triangle area)
    double area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    
    if (i == 0) {
        dNi_dx = (y2 - y3) / area2;
        dNi_dy = (x3 - x2) / area2;
    } else if (i == 1) {
        dNi_dx = (y3 - y1) / area2;
        dNi_dy = (x1 - x3) / area2;
    } else {
        dNi_dx = (y1 - y2) / area2;
        dNi_dy = (x2 - x1) / area2;
    }
    
    if (j == 0) {
        dNj_dx = (y2 - y3) / area2;
        dNj_dy = (x3 - x2) / area2;
    } else if (j == 1) {
        dNj_dx = (y3 - y1) / area2;
        dNj_dy = (x1 - x3) / area2;
    } else {
        dNj_dx = (y1 - y2) / area2;
        dNj_dy = (x2 - x1) / area2;
    }
}

__device__ inline double computeShapeFunctionProduct(
    const double* element_nodes,
    int i, int j) {
    
    // For linear triangular elements, the integral of Ni * Nj
    // over the element can be computed analytically
    
    // Get node coordinates
    double x1 = element_nodes[0], y1 = element_nodes[1];
    double x2 = element_nodes[2], y2 = element_nodes[3];
    double x3 = element_nodes[4], y3 = element_nodes[5];
    
    // Compute triangle area
    double area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
    
    if (i == j) {
        // Diagonal terms: ∫ Ni² dA = area/6
        return area / 6.0;
    } else {
        // Off-diagonal terms: ∫ Ni * Nj dA = area/12
        return area / 12.0;
    }
}

__device__ inline double computeJacobian(
    const double* element_nodes,
    double xi, double eta) {
    
    // Jacobian for coordinate transformation
    // This is a simplified version for linear elements
    
    double x1 = element_nodes[0], y1 = element_nodes[1];
    double x2 = element_nodes[2], y2 = element_nodes[3];
    double x3 = element_nodes[4], y3 = element_nodes[5];
    
    double J11 = x2 - x1;
    double J12 = x3 - x1;
    double J21 = y2 - y1;
    double J22 = y3 - y1;
    
    return J11 * J22 - J12 * J21;
}

} // namespace GPU
} // namespace QDSim
