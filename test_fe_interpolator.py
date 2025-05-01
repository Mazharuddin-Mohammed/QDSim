#!/usr/bin/env python3
"""
Test script for the FEInterpolator class.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

# Import from qdsim
from qdsim import qdsim_cpp
from qdsim.fe_interpolator import FEInterpolator

def main():
    print("Testing FEInterpolator...")
    
    # Create a mesh
    lx, ly = 100.0, 100.0  # nm
    nx, ny = 60, 60        # number of elements
    order = 1              # linear elements
    mesh = qdsim_cpp.Mesh(lx, ly, nx, ny, order)
    print(f"Mesh created with {mesh.get_num_nodes()} nodes and {mesh.get_num_elements()} elements")
    
    # Create a field to interpolate
    nodes = np.array(mesh.get_nodes())
    field = np.zeros(mesh.get_num_nodes())
    for i in range(mesh.get_num_nodes()):
        x, y = nodes[i]
        # Create a Gaussian field
        r = np.sqrt((x - lx/2)**2 + (y - ly/2)**2)
        field[i] = np.exp(-r**2 / (2 * 20.0**2))
    
    # Create an interpolator
    interpolator = FEInterpolator(mesh, use_cpp=True)
    
    # Test interpolation
    x_test = lx / 2
    y_test = ly / 2
    value = interpolator.interpolate(x_test, y_test, field)
    print(f"Interpolated value at ({x_test}, {y_test}): {value}")
    
    # Test interpolation with gradient
    value, grad_x, grad_y = interpolator.interpolate_with_gradient(x_test, y_test, field)
    print(f"Interpolated value at ({x_test}, {y_test}): {value}")
    print(f"Gradient at ({x_test}, {y_test}): ({grad_x}, {grad_y})")
    
    # Test interpolation on a grid
    x_grid = np.linspace(0, lx, 100)
    y_grid = np.linspace(0, ly, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = interpolator.interpolate(X[i, j], Y[i, j], field)
    
    # Plot the interpolated field
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, cmap='viridis', levels=20)
    plt.colorbar(label='Field value')
    plt.scatter(nodes[:, 0], nodes[:, 1], c=field, cmap='viridis', s=1)
    plt.title('Interpolated field')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.savefig('interpolated_field.png')
    print("Interpolated field plot saved to 'interpolated_field.png'")
    
if __name__ == "__main__":
    main()
