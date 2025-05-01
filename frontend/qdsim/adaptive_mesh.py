"""
Adaptive mesh refinement based on error estimates.
"""

import numpy as np
from qdsim import qdsim_cpp

class AdaptiveMesh:
    """
    Adaptive mesh refinement based on error estimates.

    This class provides methods to refine a mesh based on error estimates
    to improve accuracy in regions with rapid field variations.
    """

    def __init__(self, simulator):
        """
        Initialize the adaptive mesh.

        Args:
            simulator: The simulator object containing the mesh and physics
        """
        self.simulator = simulator
        self.mesh = simulator.mesh
        self.config = simulator.config

    def refine(self, field, max_refinements=None, threshold=None):
        """
        Refine the mesh based on error estimates.

        Args:
            field: Field values at mesh nodes (e.g., potential)
            max_refinements: Maximum number of refinement iterations
            threshold: Error threshold for refinement

        Returns:
            New mesh after refinement
        """
        max_refinements = max_refinements if max_refinements is not None else self.config.max_refinements
        threshold = threshold if threshold is not None else self.config.adaptive_threshold

        # Get mesh data
        nodes = np.array(self.mesh.get_nodes())
        elements = np.array(self.mesh.get_elements())

        # Compute error estimates for each element
        errors = self._compute_error_estimates(field, nodes, elements)

        # Refine the mesh
        for i in range(max_refinements):
            # Check if any element needs refinement
            if np.max(errors) < threshold:
                print(f"  Refinement converged after {i} iterations")
                break

            # Mark elements for refinement
            elements_to_refine = np.where(errors > threshold)[0]
            if len(elements_to_refine) == 0:
                print(f"  No elements to refine after {i} iterations")
                break

            print(f"  Refinement iteration {i+1}: refining {len(elements_to_refine)} elements")

            # Refine the mesh
            self.mesh = self._refine_mesh(elements_to_refine)

            # Update the simulator's mesh
            self.simulator.mesh = self.mesh

            # Recompute the field on the new mesh
            self.simulator.solve_poisson(0.0, 0.0)
            field = self.simulator.phi

            # Recompute error estimates
            nodes = np.array(self.mesh.get_nodes())
            elements = np.array(self.mesh.get_elements())
            errors = self._compute_error_estimates(field, nodes, elements)

        return self.mesh

    def _compute_error_estimates(self, field, nodes, elements):
        """
        Compute error estimates for each element.

        Args:
            field: Field values at mesh nodes
            nodes: Node coordinates
            elements: Element indices

        Returns:
            Array of error estimates for each element
        """
        # Compute the gradient of the field for each element
        gradients = np.zeros((len(elements), 2))
        for i, element in enumerate(elements):
            # Get the vertices of the element
            v0 = nodes[element[0]]
            v1 = nodes[element[1]]
            v2 = nodes[element[2]]

            # Compute the area of the element
            area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))

            # Compute the gradient of the field
            det = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])

            grad_x = ((field[element[0]] * (v1[1] - v2[1]) +
                      field[element[1]] * (v2[1] - v0[1]) +
                      field[element[2]] * (v0[1] - v1[1])) / det)

            grad_y = ((field[element[0]] * (v2[0] - v1[0]) +
                      field[element[1]] * (v0[0] - v2[0]) +
                      field[element[2]] * (v1[0] - v0[0])) / det)

            gradients[i] = [grad_x, grad_y]

        # Compute the error estimate for each element
        errors = np.zeros(len(elements))
        for i, element in enumerate(elements):
            # Get the vertices of the element
            v0 = nodes[element[0]]
            v1 = nodes[element[1]]
            v2 = nodes[element[2]]

            # Compute the area of the element
            area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))

            # Compute the error estimate based on the gradient and element size
            grad_norm = np.linalg.norm(gradients[i])
            errors[i] = grad_norm * np.sqrt(area)

        return errors

    def _refine_mesh(self, elements_to_refine):
        """
        Refine the mesh by splitting elements.

        Args:
            elements_to_refine: Indices of elements to refine

        Returns:
            New mesh after refinement
        """
        # Check if the C++ implementation is available
        if hasattr(qdsim_cpp, 'adapt_mesh'):
            return qdsim_cpp.adapt_mesh(self.mesh, elements_to_refine)

        # Fall back to Python implementation
        print("    Using Python implementation for mesh refinement")

        # Since we can't directly modify the mesh, we'll create a new one with finer resolution
        # This is a simplified approach that doesn't do true adaptive refinement
        # but increases the resolution uniformly

        # Increase the number of elements based on the number of elements to refine
        refinement_ratio = 1 + len(elements_to_refine) / self.mesh.get_num_elements()
        new_nx = int(self.config.nx * np.sqrt(refinement_ratio))
        new_ny = int(self.config.ny * np.sqrt(refinement_ratio))

        print(f"    Creating new mesh with {new_nx}x{new_ny} elements (refinement ratio: {refinement_ratio:.2f})")

        # Create a new mesh with higher resolution
        new_mesh = qdsim_cpp.Mesh(
            self.config.Lx, self.config.Ly,
            new_nx, new_ny,
            self.config.element_order
        )

        return new_mesh
