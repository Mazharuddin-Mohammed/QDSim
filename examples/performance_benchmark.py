#!/usr/bin/env python3
"""
Performance benchmark for QDSim.

This script benchmarks the performance of QDSim with different optimization options,
including GPU acceleration, parallel eigensolvers, and memory-efficient data structures.

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path to import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qdsim

def run_benchmark(mesh_sizes, num_states=10, use_gpu=False, use_parallel=False, use_memory_efficient=False):
    """
    Run a performance benchmark with different mesh sizes.
    
    Args:
        mesh_sizes: List of mesh sizes to benchmark
        num_states: Number of eigenstates to compute
        use_gpu: Whether to use GPU acceleration
        use_parallel: Whether to use parallel eigensolvers
        use_memory_efficient: Whether to use memory-efficient data structures
        
    Returns:
        times: Dictionary of timing results
    """
    times = {
        'mesh_creation': [],
        'matrix_assembly': [],
        'eigenvalue_solving': [],
        'total': []
    }
    
    for mesh_size in mesh_sizes:
        print(f"\nRunning benchmark with mesh size {mesh_size}x{mesh_size}...")
        
        # Create simulator
        simulator = qdsim.Simulator()
        
        # Set optimization options
        simulator.use_gpu = use_gpu
        simulator.use_parallel = use_parallel
        simulator.use_memory_efficient = use_memory_efficient
        
        # Measure mesh creation time
        start_time = time.time()
        simulator.create_mesh(100.0, 100.0, mesh_size, mesh_size)
        mesh_time = time.time() - start_time
        times['mesh_creation'].append(mesh_time)
        print(f"  Mesh creation time: {mesh_time:.4f} seconds")
        
        # Define potential (simple quantum well)
        def potential(x, y):
            # Parameters
            V0 = 1.0  # Potential height (eV)
            Lx = 100.0  # Domain size (nm)
            Ly = 100.0
            
            # Convert to J
            V0_J = V0 * 1.602e-19  # Convert eV to J
            
            # Calculate potential
            if abs(x) < Lx/4 and abs(y) < Ly/4:
                return 0.0
            else:
                return V0_J
        
        # Set potential
        simulator.set_potential(potential)
        
        # Set material (GaAs)
        simulator.set_material("GaAs")
        
        # Measure matrix assembly time
        start_time = time.time()
        simulator.assemble_matrices()
        assembly_time = time.time() - start_time
        times['matrix_assembly'].append(assembly_time)
        print(f"  Matrix assembly time: {assembly_time:.4f} seconds")
        
        # Measure eigenvalue solving time
        start_time = time.time()
        simulator.solve(num_states=num_states)
        solve_time = time.time() - start_time
        times['eigenvalue_solving'].append(solve_time)
        print(f"  Eigenvalue solving time: {solve_time:.4f} seconds")
        
        # Calculate total time
        total_time = mesh_time + assembly_time + solve_time
        times['total'].append(total_time)
        print(f"  Total time: {total_time:.4f} seconds")
    
    return times

def plot_results(mesh_sizes, times_standard, times_optimized, optimization_name):
    """
    Plot benchmark results.
    
    Args:
        mesh_sizes: List of mesh sizes
        times_standard: Timing results for standard implementation
        times_optimized: Timing results for optimized implementation
        optimization_name: Name of the optimization
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot mesh creation time
    axs[0, 0].plot(mesh_sizes, times_standard['mesh_creation'], 'b-o', label='Standard')
    axs[0, 0].plot(mesh_sizes, times_optimized['mesh_creation'], 'r-o', label=optimization_name)
    axs[0, 0].set_xlabel('Mesh Size')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].set_title('Mesh Creation Time')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot matrix assembly time
    axs[0, 1].plot(mesh_sizes, times_standard['matrix_assembly'], 'b-o', label='Standard')
    axs[0, 1].plot(mesh_sizes, times_optimized['matrix_assembly'], 'r-o', label=optimization_name)
    axs[0, 1].set_xlabel('Mesh Size')
    axs[0, 1].set_ylabel('Time (seconds)')
    axs[0, 1].set_title('Matrix Assembly Time')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot eigenvalue solving time
    axs[1, 0].plot(mesh_sizes, times_standard['eigenvalue_solving'], 'b-o', label='Standard')
    axs[1, 0].plot(mesh_sizes, times_optimized['eigenvalue_solving'], 'r-o', label=optimization_name)
    axs[1, 0].set_xlabel('Mesh Size')
    axs[1, 0].set_ylabel('Time (seconds)')
    axs[1, 0].set_title('Eigenvalue Solving Time')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot total time
    axs[1, 1].plot(mesh_sizes, times_standard['total'], 'b-o', label='Standard')
    axs[1, 1].plot(mesh_sizes, times_optimized['total'], 'r-o', label=optimization_name)
    axs[1, 1].set_xlabel('Mesh Size')
    axs[1, 1].set_ylabel('Time (seconds)')
    axs[1, 1].set_title('Total Time')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Calculate speedup
    speedup = [t1/t2 for t1, t2 in zip(times_standard['total'], times_optimized['total'])]
    
    # Add speedup text
    fig.text(0.5, 0.01, f'Average Speedup: {np.mean(speedup):.2f}x', ha='center', fontsize=14)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add title
    fig.suptitle(f'Performance Benchmark: Standard vs. {optimization_name}', fontsize=16)
    
    # Save figure
    fig.savefig(f'benchmark_{optimization_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    
    print(f"\nBenchmark results saved to benchmark_{optimization_name.lower().replace(' ', '_')}.png")
    print(f"Average speedup with {optimization_name}: {np.mean(speedup):.2f}x")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Performance benchmark for QDSim')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--parallel', action='store_true', help='Use parallel eigensolvers')
    parser.add_argument('--memory-efficient', action='store_true', help='Use memory-efficient data structures')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--mesh-sizes', type=int, nargs='+', default=[51, 101, 151, 201],
                        help='Mesh sizes to benchmark')
    parser.add_argument('--num-states', type=int, default=10,
                        help='Number of eigenstates to compute')
    
    args = parser.parse_args()
    
    # Print benchmark configuration
    print("QDSim Performance Benchmark")
    print("==========================")
    print(f"Mesh sizes: {args.mesh_sizes}")
    print(f"Number of eigenstates: {args.num_states}")
    
    # Run benchmarks
    if args.all or args.gpu:
        print("\nRunning GPU acceleration benchmark...")
        times_standard = run_benchmark(args.mesh_sizes, args.num_states)
        times_gpu = run_benchmark(args.mesh_sizes, args.num_states, use_gpu=True)
        plot_results(args.mesh_sizes, times_standard, times_gpu, "GPU Acceleration")
    
    if args.all or args.parallel:
        print("\nRunning parallel eigensolver benchmark...")
        times_standard = run_benchmark(args.mesh_sizes, args.num_states)
        times_parallel = run_benchmark(args.mesh_sizes, args.num_states, use_parallel=True)
        plot_results(args.mesh_sizes, times_standard, times_parallel, "Parallel Eigensolver")
    
    if args.all or args.memory_efficient:
        print("\nRunning memory-efficient benchmark...")
        times_standard = run_benchmark(args.mesh_sizes, args.num_states)
        times_memory = run_benchmark(args.mesh_sizes, args.num_states, use_memory_efficient=True)
        plot_results(args.mesh_sizes, times_standard, times_memory, "Memory-Efficient")
    
    if not (args.all or args.gpu or args.parallel or args.memory_efficient):
        print("\nRunning standard benchmark...")
        times_standard = run_benchmark(args.mesh_sizes, args.num_states)
        
        # Plot standard benchmark results
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot mesh creation time
        axs[0, 0].plot(args.mesh_sizes, times_standard['mesh_creation'], 'b-o')
        axs[0, 0].set_xlabel('Mesh Size')
        axs[0, 0].set_ylabel('Time (seconds)')
        axs[0, 0].set_title('Mesh Creation Time')
        axs[0, 0].grid(True)
        
        # Plot matrix assembly time
        axs[0, 1].plot(args.mesh_sizes, times_standard['matrix_assembly'], 'b-o')
        axs[0, 1].set_xlabel('Mesh Size')
        axs[0, 1].set_ylabel('Time (seconds)')
        axs[0, 1].set_title('Matrix Assembly Time')
        axs[0, 1].grid(True)
        
        # Plot eigenvalue solving time
        axs[1, 0].plot(args.mesh_sizes, times_standard['eigenvalue_solving'], 'b-o')
        axs[1, 0].set_xlabel('Mesh Size')
        axs[1, 0].set_ylabel('Time (seconds)')
        axs[1, 0].set_title('Eigenvalue Solving Time')
        axs[1, 0].grid(True)
        
        # Plot total time
        axs[1, 1].plot(args.mesh_sizes, times_standard['total'], 'b-o')
        axs[1, 1].set_xlabel('Mesh Size')
        axs[1, 1].set_ylabel('Time (seconds)')
        axs[1, 1].set_title('Total Time')
        axs[1, 1].grid(True)
        
        # Adjust layout
        fig.tight_layout()
        
        # Add title
        fig.suptitle('Standard Performance Benchmark', fontsize=16)
        
        # Save figure
        fig.savefig('benchmark_standard.png', dpi=300, bbox_inches='tight')
        
        print("\nBenchmark results saved to benchmark_standard.png")

if __name__ == "__main__":
    main()
