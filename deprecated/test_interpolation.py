#!/usr/bin/env python3
"""
Test script to verify that the finite element interpolation is working correctly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from qdsim import Simulator, Config
from qdsim.visualization import plot_potential
from qdsim.parallel_interpolator import ParallelInterpolator
from qdsim.gpu_interpolator import GPUInterpolator
from qdsim.fe_interpolator import FEInterpolator
from qdsim.cached_interpolator import CachedInterpolator

# Create output directory
os.makedirs('test_results_interpolation', exist_ok=True)

def test_interpolation():
    """Test the finite element interpolation of potentials."""
    print("\nTesting finite element interpolation:")

    # Create config
    config = Config()
    config.use_mpi = False
    config.potential_type = "gaussian"
    config.V_0 = 0.5  # eV
    config.R = 15e-9  # m
    config.nx = 30
    config.ny = 30
    config.element_order = 2  # Use quadratic elements for better interpolation
    config.max_refinements = 3
    config.adaptive_threshold = 0.01

    # Create simulator
    sim = Simulator(config)

    # Solve the Poisson equation
    sim.solve_poisson(0.0, 0.0)

    # Get the initial mesh statistics
    initial_nodes = sim.mesh.get_num_nodes()
    initial_elements = sim.mesh.get_num_elements()
    print(f"  Initial mesh: {initial_nodes} nodes, {initial_elements} elements")

    # Create a new simulator with higher resolution
    print("  Creating higher resolution mesh...")
    config_refined = Config()
    config_refined.use_mpi = False
    config_refined.potential_type = "gaussian"
    config_refined.V_0 = 0.5  # eV
    config_refined.R = 15e-9  # m
    config_refined.nx = 50  # Higher resolution
    config_refined.ny = 50  # Higher resolution
    config_refined.element_order = 2

    sim_refined = Simulator(config_refined)

    # Solve the Poisson equation on the refined mesh
    print("  Solving Poisson equation on refined mesh...")
    sim_refined.solve_poisson(0.0, 0.0)

    # Get the refined mesh statistics
    refined_nodes = sim_refined.mesh.get_num_nodes()
    refined_elements = sim_refined.mesh.get_num_elements()
    print(f"  Refined mesh: {refined_nodes} nodes, {refined_elements} elements")
    print(f"  Refinement ratio: {refined_nodes / initial_nodes:.2f}x nodes, {refined_elements / initial_elements:.2f}x elements")

    # Use the refined simulator for the rest of the test
    sim = sim_refined

    # Get mesh data
    nodes = np.array(sim.mesh.get_nodes())

    # Create a grid for visualization
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    fig.suptitle("Finite Element Interpolation Test", fontsize=16)

    # Plot mesh
    ax_mesh = fig.add_subplot(gs[0, 0])
    elements = np.array(sim.mesh.get_elements())
    ax_mesh.triplot(nodes[:, 0]*1e9, nodes[:, 1]*1e9, elements, 'k-', lw=0.5, alpha=0.5)
    ax_mesh.set_xlabel('x (nm)')
    ax_mesh.set_ylabel('y (nm)')
    ax_mesh.set_title('Mesh')
    ax_mesh.set_aspect('equal')

    # Plot potential at mesh nodes
    ax_pot_nodes = fig.add_subplot(gs[0, 1])
    plot_potential(ax_pot_nodes, sim.mesh, sim.phi,
                  title='Potential at Mesh Nodes',
                  convert_to_eV=True,
                  use_nm=True)

    # Create interpolators
    parallel_interp = ParallelInterpolator(sim)
    gpu_interp = GPUInterpolator(sim.mesh)
    cached_interp = CachedInterpolator(sim.interpolator, cache_size=10000)

    # Measure time for serial interpolation
    start_time = time.time()
    Z_serial = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            # Use our custom interpolation method
            Z_serial[i, j] = sim.interpolate(x, y, sim.phi)
    serial_time = time.time() - start_time
    print(f"  Serial interpolation time: {serial_time:.3f} seconds")

    # Measure time for parallel interpolation
    start_time = time.time()
    Z_parallel = parallel_interp.interpolate_grid(x_min, x_max, y_min, y_max, X.shape[1], X.shape[0], sim.phi)
    parallel_time = time.time() - start_time
    print(f"  Parallel interpolation time: {parallel_time:.3f} seconds")
    print(f"  Speedup: {serial_time / parallel_time:.2f}x")

    # Measure time for GPU interpolation
    start_time = time.time()
    Z = gpu_interp.interpolate_grid(x_min, x_max, y_min, y_max, X.shape[1], X.shape[0], sim.phi)
    gpu_time = time.time() - start_time
    print(f"  GPU interpolation time: {gpu_time:.3f} seconds")
    print(f"  Speedup: {serial_time / gpu_time:.2f}x")

    # Test cached interpolation
    print("\n  Testing cached interpolation:")

    # First run (cold cache)
    start_time = time.time()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            cached_interp.interpolate(x, y, sim.phi)
    cold_time = time.time() - start_time
    print(f"    Cold cache time: {cold_time:.3f} seconds")

    # Second run (warm cache)
    start_time = time.time()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            cached_interp.interpolate(x, y, sim.phi)
    warm_time = time.time() - start_time
    print(f"    Warm cache time: {warm_time:.3f} seconds")
    print(f"    Speedup: {cold_time / warm_time:.2f}x")

    # Print cache statistics
    stats = cached_interp.get_cache_stats()
    print(f"    Cache hit rate: {stats['value_hit_rate']:.2%}")
    print(f"    Cache size: {stats['value_cache_size']} / {stats['max_cache_size']}")

    # Convert to eV
    Z = Z / config.e_charge

    # Plot interpolated potential
    ax_pot_interp = fig.add_subplot(gs[1, 0])
    im = ax_pot_interp.contourf(X*1e9, Y*1e9, Z, cmap='viridis', levels=50)
    plt.colorbar(im, ax=ax_pot_interp, label='Potential (eV)')
    ax_pot_interp.set_xlabel('x (nm)')
    ax_pot_interp.set_ylabel('y (nm)')
    ax_pot_interp.set_title('Interpolated Potential')
    ax_pot_interp.set_aspect('equal')

    # Plot 1D slice of potential
    ax_slice = fig.add_subplot(gs[1, 1])

    # Get a slice along y=0
    y_idx = np.argmin(np.abs(y_grid))
    x_slice = x_grid * 1e9  # Convert to nm
    pot_slice = Z[y_idx, :]

    # Plot the slice
    ax_slice.plot(x_slice, pot_slice, 'o-', label='Interpolated')

    # Also plot the potential at mesh nodes near y=0 for comparison
    y_tolerance = (y_max - y_min) / 50
    central_nodes = [i for i, node in enumerate(nodes) if abs(node[1]) < y_tolerance]
    central_nodes.sort(key=lambda i: nodes[i][0])
    x_nodes = [nodes[i][0] * 1e9 for i in central_nodes]  # Convert to nm
    pot_nodes = [sim.phi[i] / config.e_charge for i in central_nodes]  # Convert to eV

    ax_slice.plot(x_nodes, pot_nodes, 'rx', label='Mesh Nodes')

    ax_slice.set_xlabel('x (nm)')
    ax_slice.set_ylabel('Potential (eV)')
    ax_slice.set_title('Potential (1D slice at y≈0)')
    ax_slice.grid(True)
    ax_slice.legend()

    # Save figure
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('test_results_interpolation/interpolation_test.png', dpi=300)
    plt.close(fig)

    # Test interpolation with gradient
    fig_grad = plt.figure(figsize=(15, 10))
    gs_grad = GridSpec(2, 2, figure=fig_grad)
    fig_grad.suptitle("Finite Element Interpolation with Gradient Test", fontsize=16)

    # Interpolate potential and gradient on a regular grid
    # Measure time for serial interpolation with gradient
    start_time = time.time()
    Z_grad_serial = np.zeros_like(X)
    grad_x_serial = np.zeros_like(X)
    grad_y_serial = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            # Use our custom interpolation method with gradient
            Z_grad_serial[i, j], grad_x_serial[i, j], grad_y_serial[i, j] = sim.interpolate_with_gradient(x, y, sim.phi, 0.0, 0.0)

    serial_time = time.time() - start_time
    print(f"  Serial gradient interpolation time: {serial_time:.3f} seconds")

    # Measure time for parallel interpolation with gradient
    start_time = time.time()
    Z_grad_parallel, grad_x_parallel, grad_y_parallel = parallel_interp.interpolate_grid_with_gradient(
        x_min, x_max, y_min, y_max, X.shape[1], X.shape[0], sim.phi
    )
    parallel_time = time.time() - start_time
    print(f"  Parallel gradient interpolation time: {parallel_time:.3f} seconds")
    print(f"  Speedup: {serial_time / parallel_time:.2f}x")

    # Measure time for GPU interpolation with gradient
    start_time = time.time()
    Z_grad, grad_x, grad_y = gpu_interp.interpolate_grid_with_gradient(
        x_min, x_max, y_min, y_max, X.shape[1], X.shape[0], sim.phi
    )
    gpu_time = time.time() - start_time
    print(f"  GPU gradient interpolation time: {gpu_time:.3f} seconds")
    print(f"  Speedup: {serial_time / gpu_time:.2f}x")

    # Test cached gradient interpolation
    print("\n  Testing cached gradient interpolation:")

    # Clear the cache
    cached_interp.clear_caches()

    # First run (cold cache)
    start_time = time.time()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            cached_interp.interpolate_with_gradient(x, y, sim.phi)
    cold_time = time.time() - start_time
    print(f"    Cold cache time: {cold_time:.3f} seconds")

    # Second run (warm cache)
    start_time = time.time()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            cached_interp.interpolate_with_gradient(x, y, sim.phi)
    warm_time = time.time() - start_time
    print(f"    Warm cache time: {warm_time:.3f} seconds")
    print(f"    Speedup: {cold_time / warm_time:.2f}x")

    # Print cache statistics
    stats = cached_interp.get_cache_stats()
    print(f"    Cache hit rate: {stats['gradient_hit_rate']:.2%}")
    print(f"    Cache size: {stats['gradient_cache_size']} / {stats['max_cache_size']}")

    # Convert to eV
    Z_grad = Z_grad / config.e_charge
    grad_x = grad_x / config.e_charge
    grad_y = grad_y / config.e_charge

    # Plot interpolated potential
    ax_pot_grad = fig_grad.add_subplot(gs_grad[0, 0])
    im = ax_pot_grad.contourf(X*1e9, Y*1e9, Z_grad, cmap='viridis', levels=50)
    plt.colorbar(im, ax=ax_pot_grad, label='Potential (eV)')
    ax_pot_grad.set_xlabel('x (nm)')
    ax_pot_grad.set_ylabel('y (nm)')
    ax_pot_grad.set_title('Interpolated Potential')
    ax_pot_grad.set_aspect('equal')

    # Plot gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    ax_grad_mag = fig_grad.add_subplot(gs_grad[0, 1])
    im = ax_grad_mag.contourf(X*1e9, Y*1e9, grad_mag, cmap='viridis', levels=50)
    plt.colorbar(im, ax=ax_grad_mag, label='Gradient Magnitude (eV/m)')
    ax_grad_mag.set_xlabel('x (nm)')
    ax_grad_mag.set_ylabel('y (nm)')
    ax_grad_mag.set_title('Potential Gradient Magnitude')
    ax_grad_mag.set_aspect('equal')

    # Plot gradient vectors
    ax_grad_vec = fig_grad.add_subplot(gs_grad[1, 0:])
    # Downsample for clarity
    skip = 5
    ax_grad_vec.quiver(X[::skip, ::skip]*1e9, Y[::skip, ::skip]*1e9,
                      grad_x[::skip, ::skip], grad_y[::skip, ::skip],
                      grad_mag[::skip, ::skip], cmap='viridis', scale=50)
    ax_grad_vec.set_xlabel('x (nm)')
    ax_grad_vec.set_ylabel('y (nm)')
    ax_grad_vec.set_title('Potential Gradient Vectors')
    ax_grad_vec.set_aspect('equal')

    # Save gradient figure
    fig_grad.tight_layout(rect=[0, 0, 1, 0.95])
    fig_grad.savefig('test_results_interpolation/interpolation_gradient_test.png', dpi=300)
    plt.close(fig_grad)

    print("  Interpolation test completed. Results saved to 'test_results_interpolation' directory.")

if __name__ == "__main__":
    test_interpolation()
    print("\nAll tests completed.")
