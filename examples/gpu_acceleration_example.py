#!/usr/bin/env python3
"""
GPU Acceleration Example

This example demonstrates the use of GPU acceleration for higher-order elements.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import qdsim_cpp as qdc

def main():
    """Main function."""
    # Check if GPU is available
    gpu_accelerator = qdc.GPUAccelerator(use_gpu=True)
    if not gpu_accelerator.is_gpu_enabled():
        print("GPU acceleration not available. Using CPU instead.")
    else:
        print("GPU acceleration enabled.")
        print("Device info:")
        print(gpu_accelerator.get_device_info())
    
    # Create meshes with different orders
    Lx = 100.0  # nm
    Ly = 50.0   # nm
    nx = 101
    ny = 51
    
    print("\nCreating meshes...")
    mesh_p1 = qdc.Mesh(Lx, Ly, nx, ny, 1)  # P1 elements
    mesh_p2 = qdc.Mesh(Lx, Ly, nx, ny, 2)  # P2 elements
    
    print(f"P1 mesh: {mesh_p1.getNumNodes()} nodes, {mesh_p1.getNumElements()} elements")
    print(f"P2 mesh: {mesh_p2.getNumNodes()} nodes, {mesh_p2.getNumElements()} elements")
    
    # Define potential and effective mass functions
    def V(x, y):
        # Simple harmonic oscillator potential
        return 0.5 * (x**2 + y**2) * 1e-4
    
    def m_star(x, y):
        # Constant effective mass (GaAs)
        return 0.067
    
    # Benchmark matrix assembly for different orders
    print("\nBenchmarking matrix assembly...")
    
    # P1 elements on CPU
    print("P1 elements on CPU...")
    start_time = time.time()
    solver_p1_cpu = qdc.SchrodingerSolver(mesh_p1, m_star, V, use_gpu=False)
    p1_cpu_time = time.time() - start_time
    print(f"P1 CPU time: {p1_cpu_time:.4f} seconds")
    
    # P1 elements on GPU
    if gpu_accelerator.is_gpu_enabled():
        print("P1 elements on GPU...")
        start_time = time.time()
        solver_p1_gpu = qdc.SchrodingerSolver(mesh_p1, m_star, V, use_gpu=True)
        p1_gpu_time = time.time() - start_time
        print(f"P1 GPU time: {p1_gpu_time:.4f} seconds")
        print(f"GPU speedup: {p1_cpu_time / p1_gpu_time:.2f}x")
    
    # P2 elements on CPU
    print("P2 elements on CPU...")
    start_time = time.time()
    solver_p2_cpu = qdc.SchrodingerSolver(mesh_p2, m_star, V, use_gpu=False)
    p2_cpu_time = time.time() - start_time
    print(f"P2 CPU time: {p2_cpu_time:.4f} seconds")
    
    # P2 elements on GPU
    if gpu_accelerator.is_gpu_enabled():
        print("P2 elements on GPU...")
        start_time = time.time()
        solver_p2_gpu = qdc.SchrodingerSolver(mesh_p2, m_star, V, use_gpu=True)
        p2_gpu_time = time.time() - start_time
        print(f"P2 GPU time: {p2_gpu_time:.4f} seconds")
        print(f"GPU speedup: {p2_cpu_time / p2_gpu_time:.2f}x")
    
    # Benchmark eigenvalue solving for different orders
    print("\nBenchmarking eigenvalue solving...")
    
    # P1 elements on CPU
    print("P1 elements on CPU...")
    start_time = time.time()
    eigenvalues_p1_cpu, eigenvectors_p1_cpu = solver_p1_cpu.solve(10)
    p1_cpu_solve_time = time.time() - start_time
    print(f"P1 CPU solve time: {p1_cpu_solve_time:.4f} seconds")
    
    # P1 elements on GPU
    if gpu_accelerator.is_gpu_enabled():
        print("P1 elements on GPU...")
        start_time = time.time()
        eigenvalues_p1_gpu, eigenvectors_p1_gpu = solver_p1_gpu.solve(10)
        p1_gpu_solve_time = time.time() - start_time
        print(f"P1 GPU solve time: {p1_gpu_solve_time:.4f} seconds")
        print(f"GPU speedup: {p1_cpu_solve_time / p1_gpu_solve_time:.2f}x")
    
    # P2 elements on CPU
    print("P2 elements on CPU...")
    start_time = time.time()
    eigenvalues_p2_cpu, eigenvectors_p2_cpu = solver_p2_cpu.solve(10)
    p2_cpu_solve_time = time.time() - start_time
    print(f"P2 CPU solve time: {p2_cpu_solve_time:.4f} seconds")
    
    # P2 elements on GPU
    if gpu_accelerator.is_gpu_enabled():
        print("P2 elements on GPU...")
        start_time = time.time()
        eigenvalues_p2_gpu, eigenvectors_p2_gpu = solver_p2_gpu.solve(10)
        p2_gpu_solve_time = time.time() - start_time
        print(f"P2 GPU solve time: {p2_gpu_solve_time:.4f} seconds")
        print(f"GPU speedup: {p2_cpu_solve_time / p2_gpu_solve_time:.2f}x")
    
    # Compare eigenvalues
    print("\nComparing eigenvalues...")
    print("P1 CPU eigenvalues:")
    for i, ev in enumerate(eigenvalues_p1_cpu):
        print(f"  λ{i} = {ev:.6f} eV")
    
    if gpu_accelerator.is_gpu_enabled():
        print("P1 GPU eigenvalues:")
        for i, ev in enumerate(eigenvalues_p1_gpu):
            print(f"  λ{i} = {ev:.6f} eV")
        
        # Calculate relative error
        rel_error_p1 = np.abs(np.array(eigenvalues_p1_cpu) - np.array(eigenvalues_p1_gpu)) / np.abs(np.array(eigenvalues_p1_cpu))
        print(f"P1 max relative error: {np.max(rel_error_p1):.6e}")
    
    print("P2 CPU eigenvalues:")
    for i, ev in enumerate(eigenvalues_p2_cpu):
        print(f"  λ{i} = {ev:.6f} eV")
    
    if gpu_accelerator.is_gpu_enabled():
        print("P2 GPU eigenvalues:")
        for i, ev in enumerate(eigenvalues_p2_gpu):
            print(f"  λ{i} = {ev:.6f} eV")
        
        # Calculate relative error
        rel_error_p2 = np.abs(np.array(eigenvalues_p2_cpu) - np.array(eigenvalues_p2_gpu)) / np.abs(np.array(eigenvalues_p2_cpu))
        print(f"P2 max relative error: {np.max(rel_error_p2):.6e}")
    
    # Compare P1 and P2 accuracy
    print("\nComparing P1 and P2 accuracy...")
    # Calculate relative error between P1 and P2
    rel_error_p1_p2 = np.abs(np.array(eigenvalues_p1_cpu) - np.array(eigenvalues_p2_cpu)) / np.abs(np.array(eigenvalues_p2_cpu))
    print(f"P1 vs P2 max relative error: {np.max(rel_error_p1_p2):.6e}")
    
    # Visualize wavefunctions
    print("\nVisualizing wavefunctions...")
    
    # Create grid for visualization
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create interpolators
    simple_mesh_p1 = qdc.create_simple_mesh(mesh_p1)
    interpolator_p1 = qdc.SimpleInterpolator(simple_mesh_p1)
    
    simple_mesh_p2 = qdc.create_simple_mesh(mesh_p2)
    interpolator_p2 = qdc.SimpleInterpolator(simple_mesh_p2)
    
    # Interpolate wavefunctions onto grid
    wavefunction_p1 = np.zeros((ny, nx))
    wavefunction_p2 = np.zeros((ny, nx))
    
    # Choose the ground state (index 0)
    for i in range(ny):
        for j in range(nx):
            xi, yi = X[i, j], Y[i, j]
            try:
                wavefunction_p1[i, j] = interpolator_p1.interpolate(xi, yi, eigenvectors_p1_cpu[0])
            except:
                wavefunction_p1[i, j] = 0.0
            
            try:
                wavefunction_p2[i, j] = interpolator_p2.interpolate(xi, yi, eigenvectors_p2_cpu[0])
            except:
                wavefunction_p2[i, j] = 0.0
    
    # Normalize wavefunctions for visualization
    wavefunction_p1 = wavefunction_p1 / np.max(np.abs(wavefunction_p1))
    wavefunction_p2 = wavefunction_p2 / np.max(np.abs(wavefunction_p2))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot P1 wavefunction
    im1 = axes[0].pcolormesh(X, Y, wavefunction_p1, cmap='viridis', shading='auto')
    axes[0].set_title(f'P1 Ground State (λ = {eigenvalues_p1_cpu[0]:.6f} eV)')
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('y (nm)')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot P2 wavefunction
    im2 = axes[1].pcolormesh(X, Y, wavefunction_p2, cmap='viridis', shading='auto')
    axes[1].set_title(f'P2 Ground State (λ = {eigenvalues_p2_cpu[0]:.6f} eV)')
    axes[1].set_xlabel('x (nm)')
    axes[1].set_ylabel('y (nm)')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('gpu_acceleration_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create performance comparison plot
    if gpu_accelerator.is_gpu_enabled():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data
        labels = ['P1 Assembly', 'P2 Assembly', 'P1 Solve', 'P2 Solve']
        cpu_times = [p1_cpu_time, p2_cpu_time, p1_cpu_solve_time, p2_cpu_solve_time]
        gpu_times = [p1_gpu_time, p2_gpu_time, p1_gpu_solve_time, p2_gpu_solve_time]
        
        # Bar positions
        x = np.arange(len(labels))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, cpu_times, width, label='CPU')
        ax.bar(x + width/2, gpu_times, width, label='GPU')
        
        # Add labels and title
        ax.set_ylabel('Time (seconds)')
        ax.set_title('CPU vs GPU Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add speedup annotations
        for i, (cpu_t, gpu_t) in enumerate(zip(cpu_times, gpu_times)):
            speedup = cpu_t / gpu_t
            ax.annotate(f'{speedup:.2f}x',
                        xy=(i + width/2, gpu_t),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('gpu_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
