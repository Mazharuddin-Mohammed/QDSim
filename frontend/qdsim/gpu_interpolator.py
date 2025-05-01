"""
GPU-accelerated implementation of finite element interpolation.
"""

import numpy as np
import os
import time

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    _has_cupy = True
    print("Using CuPy for GPU acceleration")
except ImportError:
    _has_cupy = False
    print("Warning: CuPy not available, falling back to CPU implementation")

class GPUInterpolator:
    """
    GPU-accelerated Finite Element Interpolator for scalar fields on a mesh.
    
    This class provides methods to interpolate scalar fields (like potentials)
    at arbitrary points in the mesh using GPU acceleration if available.
    """
    
    def __init__(self, mesh, use_gpu=True):
        """
        Initialize the GPU interpolator.
        
        Args:
            mesh: The mesh object
            use_gpu: Whether to use GPU acceleration if available
        """
        self.mesh = mesh
        self._use_gpu = use_gpu and _has_cupy
        
        # Get mesh data
        self.nodes = np.array(mesh.get_nodes())
        self.elements = np.array(mesh.get_elements())
        
        # Transfer mesh data to GPU if using GPU
        if self._use_gpu:
            self.nodes_gpu = cp.array(self.nodes)
            self.elements_gpu = cp.array(self.elements)
            
            # Compile the CUDA kernels
            self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile the CUDA kernels for interpolation."""
        if not self._use_gpu:
            return
            
        # Kernel for finding the element containing a point
        self.find_element_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void find_element_kernel(const float *nodes, const int *elements, const float *points,
                                int *element_indices, float *barycentric_coords,
                                int num_nodes, int num_elements, int num_points) {
            int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (point_idx >= num_points) return;
            
            float x = points[point_idx * 2];
            float y = points[point_idx * 2 + 1];
            
            // Initialize to not found
            element_indices[point_idx] = -1;
            
            // Check each element
            for (int e = 0; e < num_elements; e++) {
                int v0_idx = elements[e * 3];
                int v1_idx = elements[e * 3 + 1];
                int v2_idx = elements[e * 3 + 2];
                
                float x0 = nodes[v0_idx * 2];
                float y0 = nodes[v0_idx * 2 + 1];
                float x1 = nodes[v1_idx * 2];
                float y1 = nodes[v1_idx * 2 + 1];
                float x2 = nodes[v2_idx * 2];
                float y2 = nodes[v2_idx * 2 + 1];
                
                // Compute barycentric coordinates
                float det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
                if (fabs(det) < 1e-10f) continue;
                
                float lambda0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det;
                float lambda1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det;
                float lambda2 = 1.0f - lambda0 - lambda1;
                
                // Check if the point is inside the element
                if (lambda0 >= -1e-10f && lambda1 >= -1e-10f && lambda2 >= -1e-10f) {
                    element_indices[point_idx] = e;
                    barycentric_coords[point_idx * 3] = lambda0;
                    barycentric_coords[point_idx * 3 + 1] = lambda1;
                    barycentric_coords[point_idx * 3 + 2] = lambda2;
                    break;
                }
            }
        }
        ''', 'find_element_kernel')
        
        # Kernel for interpolating field values
        self.interpolate_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void interpolate_kernel(const int *elements, const float *field, const int *element_indices,
                               const float *barycentric_coords, float *values, int num_points) {
            int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (point_idx >= num_points) return;
            
            int element_idx = element_indices[point_idx];
            if (element_idx < 0) {
                values[point_idx] = 0.0f;  // Default value for points outside the mesh
                return;
            }
            
            int v0_idx = elements[element_idx * 3];
            int v1_idx = elements[element_idx * 3 + 1];
            int v2_idx = elements[element_idx * 3 + 2];
            
            float lambda0 = barycentric_coords[point_idx * 3];
            float lambda1 = barycentric_coords[point_idx * 3 + 1];
            float lambda2 = barycentric_coords[point_idx * 3 + 2];
            
            // Interpolate the field value
            values[point_idx] = lambda0 * field[v0_idx] + lambda1 * field[v1_idx] + lambda2 * field[v2_idx];
        }
        ''', 'interpolate_kernel')
        
        # Kernel for interpolating field values and gradients
        self.interpolate_gradient_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void interpolate_gradient_kernel(const float *nodes, const int *elements, const float *field,
                                        const int *element_indices, const float *barycentric_coords,
                                        float *values, float *grad_x, float *grad_y, int num_points) {
            int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (point_idx >= num_points) return;
            
            int element_idx = element_indices[point_idx];
            if (element_idx < 0) {
                values[point_idx] = 0.0f;
                grad_x[point_idx] = 0.0f;
                grad_y[point_idx] = 0.0f;
                return;
            }
            
            int v0_idx = elements[element_idx * 3];
            int v1_idx = elements[element_idx * 3 + 1];
            int v2_idx = elements[element_idx * 3 + 2];
            
            float lambda0 = barycentric_coords[point_idx * 3];
            float lambda1 = barycentric_coords[point_idx * 3 + 1];
            float lambda2 = barycentric_coords[point_idx * 3 + 2];
            
            // Interpolate the field value
            values[point_idx] = lambda0 * field[v0_idx] + lambda1 * field[v1_idx] + lambda2 * field[v2_idx];
            
            // Compute the gradient
            float x0 = nodes[v0_idx * 2];
            float y0 = nodes[v0_idx * 2 + 1];
            float x1 = nodes[v1_idx * 2];
            float y1 = nodes[v1_idx * 2 + 1];
            float x2 = nodes[v2_idx * 2];
            float y2 = nodes[v2_idx * 2 + 1];
            
            float det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
            
            grad_x[point_idx] = ((field[v0_idx] * (y1 - y2) +
                                 field[v1_idx] * (y2 - y0) +
                                 field[v2_idx] * (y0 - y1)) / det);
            
            grad_y[point_idx] = ((field[v0_idx] * (x2 - x1) +
                                 field[v1_idx] * (x0 - x2) +
                                 field[v2_idx] * (x1 - x0)) / det);
        }
        ''', 'interpolate_gradient_kernel')
    
    def interpolate_grid(self, x_min, x_max, y_min, y_max, nx, ny, field):
        """
        Interpolate a field on a regular grid using GPU acceleration.
        
        Args:
            x_min, x_max: Range of x coordinates
            y_min, y_max: Range of y coordinates
            nx, ny: Number of grid points in x and y directions
            field: Field values at mesh nodes
            
        Returns:
            2D array of interpolated values with shape (ny, nx)
        """
        # Create the grid
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        # Flatten the grid to a list of points
        points = np.column_stack((X.flatten(), Y.flatten()))
        
        if self._use_gpu:
            # Transfer data to GPU
            points_gpu = cp.array(points, dtype=np.float32)
            field_gpu = cp.array(field, dtype=np.float32)
            
            # Allocate memory for results
            num_points = len(points)
            element_indices_gpu = cp.empty(num_points, dtype=np.int32)
            barycentric_coords_gpu = cp.empty((num_points, 3), dtype=np.float32)
            values_gpu = cp.empty(num_points, dtype=np.float32)
            
            # Find the elements containing the points
            block_size = 256
            grid_size = (num_points + block_size - 1) // block_size
            self.find_element_kernel((grid_size,), (block_size,),
                                    (self.nodes_gpu, self.elements_gpu, points_gpu,
                                     element_indices_gpu, barycentric_coords_gpu,
                                     len(self.nodes), len(self.elements), num_points))
            
            # Interpolate the field values
            self.interpolate_kernel((grid_size,), (block_size,),
                                   (self.elements_gpu, field_gpu, element_indices_gpu,
                                    barycentric_coords_gpu, values_gpu, num_points))
            
            # Transfer results back to CPU
            values = cp.asnumpy(values_gpu)
            
            # Reshape to grid
            return values.reshape(ny, nx)
        else:
            # Fall back to CPU implementation
            values = np.zeros(len(points))
            
            # Find the elements containing the points and interpolate
            for i, (x, y) in enumerate(points):
                element_idx, lambda_coords = self._find_element_cpu(x, y)
                if element_idx >= 0:
                    # Interpolate using barycentric coordinates
                    elem_nodes = self.elements[element_idx]
                    values[i] = (lambda_coords[0] * field[elem_nodes[0]] + 
                                lambda_coords[1] * field[elem_nodes[1]] + 
                                lambda_coords[2] * field[elem_nodes[2]])
                else:
                    # Point is outside the mesh, find the nearest node
                    distances = np.sqrt((self.nodes[:, 0] - x)**2 + (self.nodes[:, 1] - y)**2)
                    nearest_node = np.argmin(distances)
                    values[i] = field[nearest_node]
            
            # Reshape to grid
            return values.reshape(ny, nx)
    
    def interpolate_grid_with_gradient(self, x_min, x_max, y_min, y_max, nx, ny, field):
        """
        Interpolate a field and its gradient on a regular grid using GPU acceleration.
        
        Args:
            x_min, x_max: Range of x coordinates
            y_min, y_max: Range of y coordinates
            nx, ny: Number of grid points in x and y directions
            field: Field values at mesh nodes
            
        Returns:
            Tuple of (values, grad_x, grad_y) where each is a 2D array with shape (ny, nx)
        """
        # Create the grid
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        # Flatten the grid to a list of points
        points = np.column_stack((X.flatten(), Y.flatten()))
        
        if self._use_gpu:
            # Transfer data to GPU
            points_gpu = cp.array(points, dtype=np.float32)
            field_gpu = cp.array(field, dtype=np.float32)
            
            # Allocate memory for results
            num_points = len(points)
            element_indices_gpu = cp.empty(num_points, dtype=np.int32)
            barycentric_coords_gpu = cp.empty((num_points, 3), dtype=np.float32)
            values_gpu = cp.empty(num_points, dtype=np.float32)
            grad_x_gpu = cp.empty(num_points, dtype=np.float32)
            grad_y_gpu = cp.empty(num_points, dtype=np.float32)
            
            # Find the elements containing the points
            block_size = 256
            grid_size = (num_points + block_size - 1) // block_size
            self.find_element_kernel((grid_size,), (block_size,),
                                    (self.nodes_gpu, self.elements_gpu, points_gpu,
                                     element_indices_gpu, barycentric_coords_gpu,
                                     len(self.nodes), len(self.elements), num_points))
            
            # Interpolate the field values and gradients
            self.interpolate_gradient_kernel((grid_size,), (block_size,),
                                           (self.nodes_gpu, self.elements_gpu, field_gpu,
                                            element_indices_gpu, barycentric_coords_gpu,
                                            values_gpu, grad_x_gpu, grad_y_gpu, num_points))
            
            # Transfer results back to CPU
            values = cp.asnumpy(values_gpu)
            grad_x = cp.asnumpy(grad_x_gpu)
            grad_y = cp.asnumpy(grad_y_gpu)
            
            # Reshape to grid
            return (values.reshape(ny, nx),
                    grad_x.reshape(ny, nx),
                    grad_y.reshape(ny, nx))
        else:
            # Fall back to CPU implementation
            values = np.zeros(len(points))
            grad_x = np.zeros(len(points))
            grad_y = np.zeros(len(points))
            
            # Find the elements containing the points and interpolate
            for i, (x, y) in enumerate(points):
                element_idx, lambda_coords = self._find_element_cpu(x, y)
                if element_idx >= 0:
                    # Get the element nodes
                    elem_nodes = self.elements[element_idx]
                    
                    # Interpolate using barycentric coordinates
                    values[i] = (lambda_coords[0] * field[elem_nodes[0]] + 
                                lambda_coords[1] * field[elem_nodes[1]] + 
                                lambda_coords[2] * field[elem_nodes[2]])
                    
                    # Compute the gradient
                    vertices = [self.nodes[elem_nodes[0]], self.nodes[elem_nodes[1]], self.nodes[elem_nodes[2]]]
                    det = (vertices[1][1] - vertices[2][1]) * (vertices[0][0] - vertices[2][0]) + \
                          (vertices[2][0] - vertices[1][0]) * (vertices[0][1] - vertices[2][1])
                    
                    grad_x[i] = ((field[elem_nodes[0]] * (vertices[1][1] - vertices[2][1]) + 
                                 field[elem_nodes[1]] * (vertices[2][1] - vertices[0][1]) + 
                                 field[elem_nodes[2]] * (vertices[0][1] - vertices[1][1])) / det)
                    
                    grad_y[i] = ((field[elem_nodes[0]] * (vertices[2][0] - vertices[1][0]) + 
                                 field[elem_nodes[1]] * (vertices[0][0] - vertices[2][0]) + 
                                 field[elem_nodes[2]] * (vertices[1][0] - vertices[0][0])) / det)
                else:
                    # Point is outside the mesh, find the nearest node
                    distances = np.sqrt((self.nodes[:, 0] - x)**2 + (self.nodes[:, 1] - y)**2)
                    nearest_node = np.argmin(distances)
                    values[i] = field[nearest_node]
                    grad_x[i] = 0.0
                    grad_y[i] = 0.0
            
            # Reshape to grid
            return (values.reshape(ny, nx),
                    grad_x.reshape(ny, nx),
                    grad_y.reshape(ny, nx))
    
    def interpolate(self, x, y, field):
        """
        Interpolate a field at a point.
        
        Args:
            x, y: Coordinates of the point
            field: Field values at mesh nodes
            
        Returns:
            Interpolated value at (x, y)
        """
        # For a single point, it's more efficient to use the CPU implementation
        element_idx, lambda_coords = self._find_element_cpu(x, y)
        if element_idx >= 0:
            # Interpolate using barycentric coordinates
            elem_nodes = self.elements[element_idx]
            return (lambda_coords[0] * field[elem_nodes[0]] + 
                   lambda_coords[1] * field[elem_nodes[1]] + 
                   lambda_coords[2] * field[elem_nodes[2]])
        else:
            # Point is outside the mesh, find the nearest node
            distances = np.sqrt((self.nodes[:, 0] - x)**2 + (self.nodes[:, 1] - y)**2)
            nearest_node = np.argmin(distances)
            return field[nearest_node]
    
    def interpolate_with_gradient(self, x, y, field):
        """
        Interpolate a field and its gradient at a point.
        
        Args:
            x, y: Coordinates of the point
            field: Field values at mesh nodes
            
        Returns:
            Tuple of (value, grad_x, grad_y)
        """
        # For a single point, it's more efficient to use the CPU implementation
        element_idx, lambda_coords = self._find_element_cpu(x, y)
        if element_idx >= 0:
            # Get the element nodes
            elem_nodes = self.elements[element_idx]
            
            # Interpolate using barycentric coordinates
            value = (lambda_coords[0] * field[elem_nodes[0]] + 
                    lambda_coords[1] * field[elem_nodes[1]] + 
                    lambda_coords[2] * field[elem_nodes[2]])
            
            # Compute the gradient
            vertices = [self.nodes[elem_nodes[0]], self.nodes[elem_nodes[1]], self.nodes[elem_nodes[2]]]
            det = (vertices[1][1] - vertices[2][1]) * (vertices[0][0] - vertices[2][0]) + \
                  (vertices[2][0] - vertices[1][0]) * (vertices[0][1] - vertices[2][1])
            
            grad_x = ((field[elem_nodes[0]] * (vertices[1][1] - vertices[2][1]) + 
                      field[elem_nodes[1]] * (vertices[2][1] - vertices[0][1]) + 
                      field[elem_nodes[2]] * (vertices[0][1] - vertices[1][1])) / det)
            
            grad_y = ((field[elem_nodes[0]] * (vertices[2][0] - vertices[1][0]) + 
                      field[elem_nodes[1]] * (vertices[0][0] - vertices[2][0]) + 
                      field[elem_nodes[2]] * (vertices[1][0] - vertices[0][0])) / det)
            
            return value, grad_x, grad_y
        else:
            # Point is outside the mesh, find the nearest node
            distances = np.sqrt((self.nodes[:, 0] - x)**2 + (self.nodes[:, 1] - y)**2)
            nearest_node = np.argmin(distances)
            return field[nearest_node], 0.0, 0.0
    
    def _find_element_cpu(self, x, y):
        """
        Find the element containing a point using the CPU.
        
        Args:
            x, y: Coordinates of the point
            
        Returns:
            Tuple of (element_index, barycentric_coordinates) or (-1, None) if not found
        """
        for e in range(len(self.elements)):
            # Get the vertices of the element
            elem_nodes = self.elements[e]
            v0 = self.nodes[elem_nodes[0]]
            v1 = self.nodes[elem_nodes[1]]
            v2 = self.nodes[elem_nodes[2]]
            
            # Compute barycentric coordinates
            det = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
            if abs(det) < 1e-10:
                continue
                
            lambda0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / det
            lambda1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / det
            lambda2 = 1.0 - lambda0 - lambda1
            
            # Check if the point is inside the element
            if lambda0 >= -1e-10 and lambda1 >= -1e-10 and lambda2 >= -1e-10:
                return e, [lambda0, lambda1, lambda2]
        
        # Point is not in any element
        return -1, None
