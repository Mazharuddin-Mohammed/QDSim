"""
Parallel implementation of finite element interpolation.

This module provides a parallel implementation of finite element interpolation
for efficient computation of field values and gradients at multiple points.
It uses multithreading to distribute the workload across multiple CPU cores.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import os
import threading

class ParallelInterpolator:
    """
    A class for parallel interpolation of scalar fields on a mesh.

    This class provides methods to interpolate scalar fields (like potentials)
    at multiple points in parallel, which is useful for large meshes and
    many interpolation points.
    """

    def __init__(self, simulator, num_workers=None, chunk_size=1000):
        """
        Initialize the parallel interpolator.

        Args:
            simulator: The simulator object containing the mesh and interpolation methods
            num_workers: Number of worker threads to use (default: number of CPU cores)
            chunk_size: Number of points to process in each chunk (default: 1000)
                        Larger chunks reduce overhead but may cause load imbalance
        """
        self.simulator = simulator
        self.num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
        self.chunk_size = chunk_size

        # Create a lock for thread safety
        self.lock = threading.Lock()

    def _interpolate_chunk(self, points, field):
        """
        Interpolate a field at a chunk of points.

        Args:
            points: List of (x, y) coordinates
            field: Field values at mesh nodes

        Returns:
            List of interpolated values
        """
        return [self.simulator.interpolate(x, y, field) for x, y in points]

    def _interpolate_with_gradient_chunk(self, points, field):
        """
        Interpolate a field and its gradient at a chunk of points.

        Args:
            points: List of (x, y) coordinates
            field: Field values at mesh nodes

        Returns:
            List of (value, grad_x, grad_y) tuples
        """
        return [self.simulator.interpolate_with_gradient(x, y, field, 0.0, 0.0) for x, y in points]

    def interpolate_grid(self, x_min, x_max, y_min, y_max, nx, ny, field):
        """
        Interpolate a field on a regular grid in parallel.

        This method uses a combination of NumPy vectorization and multithreading
        to efficiently interpolate the field on a regular grid.

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
        total_points = len(points)

        # Create output array
        values = np.zeros(total_points)

        # Define a worker function that processes a chunk of points
        def worker_func(start_idx, end_idx):
            # Process a chunk of points
            for i in range(start_idx, end_idx):
                x, y = points[i]
                values[i] = self.simulator.interpolate(x, y, field)

        # Split the work among threads
        threads = []
        chunk_size = max(1, total_points // self.num_workers)

        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_points)

            if start_idx >= end_idx:
                break

            thread = threading.Thread(target=worker_func, args=(start_idx, end_idx))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Reshape the results to the grid
        return values.reshape(ny, nx)

    def interpolate_grid_with_gradient(self, x_min, x_max, y_min, y_max, nx, ny, field):
        """
        Interpolate a field and its gradient on a regular grid in parallel.

        This method uses a combination of NumPy vectorization and multithreading
        to efficiently interpolate the field and its gradient on a regular grid.

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
        total_points = len(points)

        # Create output arrays
        values = np.zeros(total_points)
        grad_x_values = np.zeros(total_points)
        grad_y_values = np.zeros(total_points)

        # Define a worker function that processes a chunk of points
        def worker_func(start_idx, end_idx):
            # Process a chunk of points
            for i in range(start_idx, end_idx):
                x, y = points[i]
                val, gx, gy = self.simulator.interpolate_with_gradient(x, y, field, 0.0, 0.0)
                values[i] = val
                grad_x_values[i] = gx
                grad_y_values[i] = gy

        # Split the work among threads
        threads = []
        chunk_size = max(1, total_points // self.num_workers)

        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_points)

            if start_idx >= end_idx:
                break

            thread = threading.Thread(target=worker_func, args=(start_idx, end_idx))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Reshape the results to the grid
        return (values.reshape(ny, nx),
                grad_x_values.reshape(ny, nx),
                grad_y_values.reshape(ny, nx))

    def interpolate_points(self, points, field):
        """
        Interpolate a field at arbitrary points in parallel.

        This method uses multithreading to efficiently interpolate the field
        at arbitrary points.

        Args:
            points: List or array of (x, y) coordinates
            field: Field values at mesh nodes

        Returns:
            Array of interpolated values
        """
        # Convert points to numpy array if it's not already
        points = np.asarray(points)
        total_points = len(points)

        # Create output array
        values = np.zeros(total_points)

        # Define a worker function that processes a chunk of points
        def worker_func(start_idx, end_idx):
            # Process a chunk of points
            for i in range(start_idx, end_idx):
                x, y = points[i]
                values[i] = self.simulator.interpolate(x, y, field)

        # Split the work among threads
        threads = []
        chunk_size = max(1, total_points // self.num_workers)

        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_points)

            if start_idx >= end_idx:
                break

            thread = threading.Thread(target=worker_func, args=(start_idx, end_idx))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return values

    def interpolate_points_with_gradient(self, points, field):
        """
        Interpolate a field and its gradient at arbitrary points in parallel.

        This method uses multithreading to efficiently interpolate the field
        and its gradient at arbitrary points.

        Args:
            points: List or array of (x, y) coordinates
            field: Field values at mesh nodes

        Returns:
            Tuple of (values, grad_x, grad_y) arrays
        """
        # Convert points to numpy array if it's not already
        points = np.asarray(points)
        total_points = len(points)

        # Create output arrays
        values = np.zeros(total_points)
        grad_x_values = np.zeros(total_points)
        grad_y_values = np.zeros(total_points)

        # Define a worker function that processes a chunk of points
        def worker_func(start_idx, end_idx):
            # Process a chunk of points
            for i in range(start_idx, end_idx):
                x, y = points[i]
                val, gx, gy = self.simulator.interpolate_with_gradient(x, y, field, 0.0, 0.0)
                values[i] = val
                grad_x_values[i] = gx
                grad_y_values[i] = gy

        # Split the work among threads
        threads = []
        chunk_size = max(1, total_points // self.num_workers)

        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_points)

            if start_idx >= end_idx:
                break

            thread = threading.Thread(target=worker_func, args=(start_idx, end_idx))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return values, grad_x_values, grad_y_values
