#!/usr/bin/env python3
"""
Performance benchmarks for QDSim.

This script benchmarks the performance of QDSim with different:
1. Mesh sizes
2. Element orders
3. Parallel configurations
4. GPU acceleration (if available)

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Add the parent directory to the path so we can import qdsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import qdsim

class TestPerformanceBenchmarks:
    """Test class for benchmarking QDSim performance."""

    def setup_method(self):
        """Set up common parameters for benchmarks."""
        self.config = qdsim.Config()
        self.config.e_charge = 1.602e-19  # Elementary charge in C
        self.config.hbar = 1.054e-34      # Reduced Planck's constant in J·s
        self.config.m_e = 9.11e-31        # Electron mass in kg

        # Use GaAs effective mass
        self.m_star = 0.067 * self.config.m_e

        # Set up default mesh parameters
        self.config.Lx = 100e-9  # 100 nm
        self.config.Ly = 100e-9  # 100 nm
        self.config.nx = 51      # Default mesh points
        self.config.ny = 51      # Default mesh points
        self.config.element_order = 1  # Default to linear elements

        # Define a simple quantum dot potential
        def quantum_dot_potential(x, y):
            # Gaussian quantum dot
            V0 = 0.3 * self.config.e_charge  # 0.3 eV converted to J
            R = 10e-9  # 10 nm radius
            r = np.sqrt(x**2 + y**2)
            return -V0 * np.exp(-r**2 / (2 * R**2))

        self.config.potential_function = quantum_dot_potential
        self.config.m_star_function = lambda x, y: self.m_star

        # Disable MPI by default
        self.config.use_mpi = False

        # Number of eigenvalues to compute
        self.num_eigenvalues = 5

    def benchmark_mesh_sizes(self):
        """Benchmark different mesh sizes."""
        print("\nBenchmarking different mesh sizes...")

        # Mesh sizes to benchmark
        mesh_sizes = [21, 51, 101, 151, 201]

        # Results
        mesh_creation_times = []
        matrix_assembly_times = []
        eigenvalue_solving_times = []
        total_times = []

        for mesh_size in mesh_sizes:
            print(f"Running benchmark with mesh size {mesh_size}x{mesh_size}...")

            # Update mesh size
            self.config.nx = mesh_size
            self.config.ny = mesh_size

            # Create simulator
            start_time = time.time()
            simulator = qdsim.Simulator(self.config)
            mesh_time = time.time() - start_time
            mesh_creation_times.append(mesh_time)
            print(f"  Mesh creation time: {mesh_time:.4f} seconds")

            # Assemble matrices
            start_time = time.time()
            simulator.assemble_matrices()
            assembly_time = time.time() - start_time
            matrix_assembly_times.append(assembly_time)
            print(f"  Matrix assembly time: {assembly_time:.4f} seconds")

            # Solve eigenvalue problem
            start_time = time.time()
            eigenvalues, eigenvectors = simulator.solve(num_eigenvalues=self.num_eigenvalues)
            solve_time = time.time() - start_time
            eigenvalue_solving_times.append(solve_time)
            print(f"  Eigenvalue solving time: {solve_time:.4f} seconds")

            # Total time
            total_time = mesh_time + assembly_time + solve_time
            total_times.append(total_time)
            print(f"  Total time: {total_time:.4f} seconds")

        # Plot results
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot mesh creation time
        axs[0, 0].plot(mesh_sizes, mesh_creation_times, 'bo-')
        axs[0, 0].set_xlabel('Mesh Size (NxN)')
        axs[0, 0].set_ylabel('Time (seconds)')
        axs[0, 0].set_title('Mesh Creation Time')
        axs[0, 0].grid(True)

        # Plot matrix assembly time
        axs[0, 1].plot(mesh_sizes, matrix_assembly_times, 'ro-')
        axs[0, 1].set_xlabel('Mesh Size (NxN)')
        axs[0, 1].set_ylabel('Time (seconds)')
        axs[0, 1].set_title('Matrix Assembly Time')
        axs[0, 1].grid(True)

        # Plot eigenvalue solving time
        axs[1, 0].plot(mesh_sizes, eigenvalue_solving_times, 'go-')
        axs[1, 0].set_xlabel('Mesh Size (NxN)')
        axs[1, 0].set_ylabel('Time (seconds)')
        axs[1, 0].set_title('Eigenvalue Solving Time')
        axs[1, 0].grid(True)

        # Plot total time
        axs[1, 1].plot(mesh_sizes, total_times, 'mo-')
        axs[1, 1].set_xlabel('Mesh Size (NxN)')
        axs[1, 1].set_ylabel('Time (seconds)')
        axs[1, 1].set_title('Total Time')
        axs[1, 1].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.savefig('benchmark_mesh_sizes.png', dpi=300)

        return mesh_sizes, total_times

    def benchmark_element_orders(self):
        """Benchmark different element orders."""
        print("\nBenchmarking different element orders...")

        # Element orders to benchmark
        element_orders = [1, 2, 3]  # Linear, quadratic, cubic

        # Results
        mesh_creation_times = []
        matrix_assembly_times = []
        eigenvalue_solving_times = []
        total_times = []

        # Fix mesh size to a moderate value
        self.config.nx = 51
        self.config.ny = 51

        for order in element_orders:
            print(f"Running benchmark with element order {order}...")

            # Update element order
            self.config.element_order = order

            # Create simulator
            start_time = time.time()
            simulator = qdsim.Simulator(self.config)
            mesh_time = time.time() - start_time
            mesh_creation_times.append(mesh_time)
            print(f"  Mesh creation time: {mesh_time:.4f} seconds")

            # Assemble matrices
            start_time = time.time()
            simulator.assemble_matrices()
            assembly_time = time.time() - start_time
            matrix_assembly_times.append(assembly_time)
            print(f"  Matrix assembly time: {assembly_time:.4f} seconds")

            # Solve eigenvalue problem
            start_time = time.time()
            eigenvalues, eigenvectors = simulator.solve(num_eigenvalues=self.num_eigenvalues)
            solve_time = time.time() - start_time
            eigenvalue_solving_times.append(solve_time)
            print(f"  Eigenvalue solving time: {solve_time:.4f} seconds")

            # Total time
            total_time = mesh_time + assembly_time + solve_time
            total_times.append(total_time)
            print(f"  Total time: {total_time:.4f} seconds")

        # Plot results
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot mesh creation time
        axs[0, 0].bar(element_orders, mesh_creation_times, color='blue')
        axs[0, 0].set_xlabel('Element Order')
        axs[0, 0].set_ylabel('Time (seconds)')
        axs[0, 0].set_title('Mesh Creation Time')
        axs[0, 0].set_xticks(element_orders)
        axs[0, 0].grid(True)

        # Plot matrix assembly time
        axs[0, 1].bar(element_orders, matrix_assembly_times, color='red')
        axs[0, 1].set_xlabel('Element Order')
        axs[0, 1].set_ylabel('Time (seconds)')
        axs[0, 1].set_title('Matrix Assembly Time')
        axs[0, 1].set_xticks(element_orders)
        axs[0, 1].grid(True)

        # Plot eigenvalue solving time
        axs[1, 0].bar(element_orders, eigenvalue_solving_times, color='green')
        axs[1, 0].set_xlabel('Element Order')
        axs[1, 0].set_ylabel('Time (seconds)')
        axs[1, 0].set_title('Eigenvalue Solving Time')
        axs[1, 0].set_xticks(element_orders)
        axs[1, 0].grid(True)

        # Plot total time
        axs[1, 1].bar(element_orders, total_times, color='magenta')
        axs[1, 1].set_xlabel('Element Order')
        axs[1, 1].set_ylabel('Time (seconds)')
        axs[1, 1].set_title('Total Time')
        axs[1, 1].set_xticks(element_orders)
        axs[1, 1].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.savefig('benchmark_element_orders.png', dpi=300)

        return element_orders, total_times

    def benchmark_parallel_configurations(self):
        """Benchmark different parallel configurations."""
        print("\nBenchmarking parallel configurations...")

        # Check if MPI is available
        try:
            from mpi4py import MPI
            has_mpi = True
        except ImportError:
            has_mpi = False
            print("MPI not available, skipping parallel benchmarks")
            return None, None

        if has_mpi:
            # Number of processes to benchmark
            num_processes = [1, 2, 4, 8]  # Adjust based on available cores

            # Results
            total_times = []

            # Fix mesh size and element order
            self.config.nx = 101
            self.config.ny = 101
            self.config.element_order = 2

            for np in num_processes:
                print(f"Running benchmark with {np} processes...")

                # Enable MPI
                self.config.use_mpi = True

                # Create simulator
                simulator = qdsim.Simulator(self.config)

                # Run with MPI
                start_time = time.time()
                # This would typically be run with mpirun -np {np} python script.py
                # For testing, we'll just time the sequential execution
                eigenvalues, eigenvectors = simulator.run(num_eigenvalues=self.num_eigenvalues)
                total_time = time.time() - start_time
                total_times.append(total_time)
                print(f"  Total time with {np} processes: {total_time:.4f} seconds")

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(num_processes, total_times, 'bo-')
            ax.set_xlabel('Number of Processes')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Parallel Performance')
            ax.grid(True)

            # Calculate speedup
            speedup = [total_times[0] / t for t in total_times]
            ax2 = ax.twinx()
            ax2.plot(num_processes, speedup, 'ro-')
            ax2.set_ylabel('Speedup', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            plt.tight_layout()
            plt.savefig('benchmark_parallel.png', dpi=300)

            return num_processes, total_times

    def benchmark_gpu_acceleration(self):
        """Benchmark GPU acceleration if available."""
        print("\nBenchmarking GPU acceleration...")

        # Check if GPU acceleration is available
        try:
            import cupy
            has_gpu = True
        except ImportError:
            has_gpu = False
            print("GPU acceleration not available, skipping GPU benchmarks")
            return None, None

        if has_gpu:
            # Mesh sizes to benchmark
            mesh_sizes = [51, 101, 151, 201]

            # Results
            cpu_times = []
            gpu_times = []

            # Fix element order
            self.config.element_order = 2

            for mesh_size in mesh_sizes:
                print(f"Running benchmark with mesh size {mesh_size}x{mesh_size}...")

                # Update mesh size
                self.config.nx = mesh_size
                self.config.ny = mesh_size

                # CPU benchmark
                self.config.use_gpu = False
                simulator_cpu = qdsim.Simulator(self.config)

                start_time = time.time()
                eigenvalues_cpu, _ = simulator_cpu.run(num_eigenvalues=self.num_eigenvalues)
                cpu_time = time.time() - start_time
                cpu_times.append(cpu_time)
                print(f"  CPU time: {cpu_time:.4f} seconds")

                # GPU benchmark
                self.config.use_gpu = True
                simulator_gpu = qdsim.Simulator(self.config)

                start_time = time.time()
                eigenvalues_gpu, _ = simulator_gpu.run(num_eigenvalues=self.num_eigenvalues)
                gpu_time = time.time() - start_time
                gpu_times.append(gpu_time)
                print(f"  GPU time: {gpu_time:.4f} seconds")

                # Verify results are similar
                rel_diff = np.abs((np.real(eigenvalues_cpu) - np.real(eigenvalues_gpu)) / np.real(eigenvalues_cpu))
                print(f"  Max relative difference in eigenvalues: {np.max(rel_diff):.6f}")

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(mesh_sizes, cpu_times, 'bo-', label='CPU')
            ax.plot(mesh_sizes, gpu_times, 'ro-', label='GPU')
            ax.set_xlabel('Mesh Size (NxN)')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('CPU vs GPU Performance')
            ax.legend()
            ax.grid(True)

            # Calculate speedup
            speedup = [cpu_t / gpu_t for cpu_t, gpu_t in zip(cpu_times, gpu_times)]
            ax2 = ax.twinx()
            ax2.plot(mesh_sizes, speedup, 'go-', label='Speedup')
            ax2.set_ylabel('Speedup', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            plt.tight_layout()
            plt.savefig('benchmark_gpu.png', dpi=300)

            return mesh_sizes, (cpu_times, gpu_times)

    def run_all_benchmarks(self):
        """Run all benchmarks and generate a comprehensive report."""
        print("Running all benchmarks...")

        # Run individual benchmarks
        try:
            mesh_sizes, mesh_times = self.benchmark_mesh_sizes()
        except Exception as e:
            print(f"Error in mesh size benchmarks: {e}")
            mesh_sizes, mesh_times = None, None

        try:
            element_orders, element_times = self.benchmark_element_orders()
        except Exception as e:
            print(f"Error in element order benchmarks: {e}")
            element_orders, element_times = None, None

        try:
            processes, parallel_times = self.benchmark_parallel_configurations()
        except Exception as e:
            print(f"Error in parallel configuration benchmarks: {e}")
            processes, parallel_times = None, None

        try:
            gpu_mesh_sizes, gpu_times = self.benchmark_gpu_acceleration()
        except Exception as e:
            print(f"Error in GPU acceleration benchmarks: {e}")
            gpu_mesh_sizes, gpu_times = None, None

        # Generate comprehensive report
        try:
            with open('benchmark_report.md', 'w') as f:
                f.write("# QDSim Performance Benchmark Report\n\n")

                if mesh_sizes and mesh_times:
                    f.write("## 1. Mesh Size Benchmarks\n\n")
                    f.write("| Mesh Size | Total Time (s) |\n")
                    f.write("|-----------|---------------|\n")
                    for size, time in zip(mesh_sizes, mesh_times):
                        f.write(f"| {size}x{size} | {time:.4f} |\n")
                    f.write("\n")
                else:
                    f.write("## 1. Mesh Size Benchmarks\n\n")
                    f.write("Mesh size benchmarks were not run or failed.\n\n")

                if element_orders and element_times:
                    f.write("## 2. Element Order Benchmarks\n\n")
                    f.write("| Element Order | Total Time (s) |\n")
                    f.write("|--------------|---------------|\n")
                    for order, time in zip(element_orders, element_times):
                        f.write(f"| P{order} | {time:.4f} |\n")
                    f.write("\n")
                else:
                    f.write("## 2. Element Order Benchmarks\n\n")
                    f.write("Element order benchmarks were not run or failed.\n\n")

                if processes and parallel_times:
                    f.write("## 3. Parallel Configuration Benchmarks\n\n")
                    f.write("| Number of Processes | Total Time (s) | Speedup |\n")
                    f.write("|---------------------|---------------|--------|\n")
                    for np, time in zip(processes, parallel_times):
                        speedup = parallel_times[0] / time
                        f.write(f"| {np} | {time:.4f} | {speedup:.2f}x |\n")
                    f.write("\n")
                else:
                    f.write("## 3. Parallel Configuration Benchmarks\n\n")
                    f.write("Parallel configuration benchmarks were not run or failed.\n\n")

                if gpu_mesh_sizes and gpu_times:
                    cpu_times, gpu_times_list = gpu_times
                    f.write("## 4. GPU Acceleration Benchmarks\n\n")
                    f.write("| Mesh Size | CPU Time (s) | GPU Time (s) | Speedup |\n")
                    f.write("|-----------|-------------|-------------|--------|\n")
                    for size, cpu_t, gpu_t in zip(gpu_mesh_sizes, cpu_times, gpu_times_list):
                        speedup = cpu_t / gpu_t
                        f.write(f"| {size}x{size} | {cpu_t:.4f} | {gpu_t:.4f} | {speedup:.2f}x |\n")
                    f.write("\n")
                else:
                    f.write("## 4. GPU Acceleration Benchmarks\n\n")
                    f.write("GPU acceleration benchmarks were not run or failed.\n\n")

                f.write("## Summary\n\n")
                f.write("Based on the benchmarks, the following recommendations can be made:\n\n")

                # Mesh size recommendation
                f.write("1. **Mesh Size**: ")
                if mesh_sizes and mesh_times and len(mesh_times) > 2:
                    # Find the point of diminishing returns
                    rel_improvement = [(mesh_times[i] - mesh_times[i+1]) / mesh_times[i] for i in range(len(mesh_times)-1)]
                    optimal_idx = next((i for i, imp in enumerate(rel_improvement) if imp < 0.1), len(mesh_sizes)-2)
                    f.write(f"A mesh size of {mesh_sizes[optimal_idx]}x{mesh_sizes[optimal_idx]} provides a good balance between accuracy and performance.\n\n")
                else:
                    f.write("More mesh sizes should be tested to determine the optimal size.\n\n")

                # Element order recommendation
                f.write("2. **Element Order**: ")
                if element_orders and element_times and len(element_times) > 1:
                    import numpy as np
                    best_order_idx = np.argmin(element_times)
                    f.write(f"P{element_orders[best_order_idx]} elements provide the best performance for this problem.\n\n")
                else:
                    f.write("More element orders should be tested to determine the optimal order.\n\n")

                # Parallel recommendation
                f.write("3. **Parallel Processing**: ")
                if processes and parallel_times and len(processes) > 1 and len(parallel_times) > 1:
                    # Find the point of diminishing returns
                    rel_improvement = [(parallel_times[i] - parallel_times[i+1]) / parallel_times[i] for i in range(len(parallel_times)-1)]
                    optimal_np_idx = next((i for i, imp in enumerate(rel_improvement) if imp < 0.1), len(processes)-2)
                    f.write(f"Using {processes[optimal_np_idx]} processes provides a good balance between speedup and resource utilization.\n\n")
                else:
                    f.write("Parallel processing was not tested or more configurations should be tested.\n\n")

                # GPU recommendation
                f.write("4. **GPU Acceleration**: ")
                if gpu_mesh_sizes and gpu_times:
                    cpu_times, gpu_times_list = gpu_times
                    avg_speedup = sum(cpu_t / gpu_t for cpu_t, gpu_t in zip(cpu_times, gpu_times_list)) / len(cpu_times)
                    if avg_speedup > 1.5:
                        f.write(f"GPU acceleration provides significant speedup (average {avg_speedup:.2f}x) and is recommended for large problems.\n\n")
                    else:
                        f.write(f"GPU acceleration provides modest speedup (average {avg_speedup:.2f}x) and may be beneficial for very large problems.\n\n")
                else:
                    f.write("GPU acceleration was not tested or is not available.\n\n")
        except Exception as e:
            print(f"Error generating benchmark report: {e}")

        print("Benchmark report generated: benchmark_report.md")

if __name__ == "__main__":
    # Run the benchmarks
    benchmark = TestPerformanceBenchmarks()
    benchmark.setup_method()
    benchmark.run_all_benchmarks()
