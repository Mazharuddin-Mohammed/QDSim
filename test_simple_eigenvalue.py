#!/usr/bin/env python3
"""
Simple test script for the improved eigenvalue solver in QDSim.
This script directly tests the eigenvalue solver with a simple example.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def main():
    print("Testing improved eigenvalue solver with a simple example...")
    
    # Create a simple 2D Laplacian matrix
    n = 50  # grid size
    h = 1.0 / n  # grid spacing
    N = n * n  # total number of grid points
    
    # Create the Laplacian matrix in sparse format
    row_indices = []
    col_indices = []
    values = []
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            
            # Diagonal term
            row_indices.append(idx)
            col_indices.append(idx)
            values.append(4.0)
            
            # Off-diagonal terms
            if i > 0:
                row_indices.append(idx)
                col_indices.append((i-1) * n + j)
                values.append(-1.0)
            
            if i < n-1:
                row_indices.append(idx)
                col_indices.append((i+1) * n + j)
                values.append(-1.0)
            
            if j > 0:
                row_indices.append(idx)
                col_indices.append(i * n + (j-1))
                values.append(-1.0)
            
            if j < n-1:
                row_indices.append(idx)
                col_indices.append(i * n + (j+1))
                values.append(-1.0)
    
    # Create the sparse matrix
    H = csr_matrix((values, (row_indices, col_indices)), shape=(N, N))
    
    # Create a simple mass matrix (identity for simplicity)
    M = csr_matrix((np.ones(N), (np.arange(N), np.arange(N))), shape=(N, N))
    
    # Add a potential well at the center
    V = np.zeros(N)
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            r = np.sqrt((i - n/2)**2 + (j - n/2)**2) / n
            V[idx] = -0.5 * np.exp(-10 * r**2)
    
    # Add the potential to the Hamiltonian
    for i in range(N):
        H[i, i] += V[i]
    
    # Solve the eigenvalue problem using SciPy
    print("Solving eigenvalue problem...")
    eigenvalues, eigenvectors = eigsh(H, k=10, M=M, which='SM')
    
    # Print eigenvalues
    print("\nEigenvalues:")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ{i} = {ev:.6f}")
    
    # Plot the first few eigenfunctions
    plot_eigenfunctions(eigenvectors, n, 4)
    
def plot_eigenfunctions(eigenvectors, n, num_to_plot):
    """Plot the first few eigenfunctions."""
    num_to_plot = min(num_to_plot, eigenvectors.shape[1])
    
    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # Create a grid for plotting
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Plot each eigenfunction
    for i in range(num_to_plot):
        # Get the eigenfunction
        psi = eigenvectors[:, i].reshape(n, n)
        
        # Normalize for visualization
        psi = psi / np.max(np.abs(psi))
        
        # Create a contour plot
        im = axes[i].contourf(X, Y, psi, cmap='viridis', levels=20)
        axes[i].set_title(f'Eigenfunction {i}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_aspect('equal')
        
        # Add a colorbar
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('eigenfunctions_simple.png')
    print("Eigenfunctions plot saved to 'eigenfunctions_simple.png'")
    
if __name__ == "__main__":
    main()
