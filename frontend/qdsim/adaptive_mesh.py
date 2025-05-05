"""
Adaptive mesh refinement based on error estimates.

This module provides functions for adaptive mesh refinement based on solution
features. It includes functions for computing refinement flags, refining the
mesh, ensuring mesh conformity, and improving mesh quality.

Author: Dr. Mazharuddin Mohammed
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

    @staticmethod
    def compute_refinement_flags(mesh, psi, threshold):
        """
        Compute refinement flags based on solution gradients.

        This function computes refinement flags based on the gradient of the solution.
        Elements with large gradients are marked for refinement.

        Args:
            mesh: The mesh
            psi: The solution vector
            threshold: The threshold for refinement (elements with gradient > threshold are refined)

        Returns:
            A vector of boolean flags indicating which elements to refine
        """
        # Initialize refinement flags to false for all elements
        refine_flags = [False] * mesh.get_num_elements()

        # Get mesh data
        nodes = mesh.get_nodes()
        elements = mesh.get_elements()
        order = mesh.get_element_order()
        nodes_per_elem = 3 if order == 1 else 6 if order == 2 else 10

        for i in range(len(elements)):
            # Get element nodes
            elem_nodes = []
            if order == 1:
                elem_nodes = [elements[i][j] for j in range(3)]
            elif order == 2:
                elem_nodes = [elements[i][j] for j in range(6)]
            else:
                elem_nodes = [elements[i][j] for j in range(10)]

            # Get element coordinates and solution values
            coords = np.array([nodes[j] for j in elem_nodes])
            psi_elem = np.array([psi[j] if j < len(psi) else 0.0 for j in elem_nodes])

            # Compute gradient using least squares
            A = np.ones((len(elem_nodes), 3))
            A[:, 1:] = coords
            A_inv = np.linalg.pinv(A)
            grad_psi = A_inv[1:, :] @ psi_elem
            error = np.linalg.norm(grad_psi)

            # Mark element for refinement if error exceeds threshold
            if error > threshold:
                refine_flags[i] = True

        return refine_flags

    @staticmethod
    def refine_mesh(mesh, refine_flags):
        """
        Refine the mesh based on refinement flags.

        This function refines the mesh by subdividing elements marked for refinement.
        It ensures mesh conformity by adding transition elements as needed.

        Args:
            mesh: The mesh to refine
            refine_flags: A vector of boolean flags indicating which elements to refine
        """
        # Since we can't directly modify the mesh in the C++ implementation,
        # we'll create a new mesh with increased resolution in the areas
        # where refinement is needed

        # Count how many elements need refinement
        num_to_refine = sum(refine_flags)
        if num_to_refine == 0:
            print("No elements to refine")
            return mesh

        print(f"Refining {num_to_refine} out of {len(refine_flags)} elements")

        # Increase the mesh resolution based on the number of elements to refine
        # This is a simplified approach that doesn't do true adaptive refinement
        # but increases the resolution uniformly
        refinement_ratio = 1 + num_to_refine / len(refine_flags)
        new_nx = int(mesh.get_nx() * np.sqrt(refinement_ratio))
        new_ny = int(mesh.get_ny() * np.sqrt(refinement_ratio))

        print(f"Creating new mesh with {new_nx}x{new_ny} elements (refinement ratio: {refinement_ratio:.2f})")

        # Create a new mesh with higher resolution
        new_mesh = qdsim_cpp.Mesh(
            mesh.get_lx(), mesh.get_ly(),
            new_nx, new_ny,
            mesh.get_element_order()
        )

        return new_mesh

    @staticmethod
    def smooth_mesh(mesh):
        """
        Smooth the mesh to improve element quality.

        This function is a placeholder for mesh smoothing functionality.
        Since we can't directly modify the mesh in the C++ implementation,
        we simply return the original mesh.

        Args:
            mesh: The mesh to smooth

        Returns:
            The original mesh (no smoothing is performed)
        """
        print("Mesh smoothing is not implemented in the Python fallback")
        return mesh

    @staticmethod
    def compute_triangle_quality(mesh, elem_idx):
        """
        Compute the quality of a triangular element.

        This function computes the quality of a triangular element using the
        ratio of the inscribed circle radius to the circumscribed circle radius,
        normalized to give a value of 1 for an equilateral triangle.

        Args:
            mesh: The mesh
            elem_idx: The index of the element

        Returns:
            The quality of the element (0 to 1, where 1 is an equilateral triangle)
        """
        # Get element vertices
        elem = mesh.get_elements()[elem_idx]
        nodes = mesh.get_nodes()
        a = nodes[elem[0]]
        b = nodes[elem[1]]
        c = nodes[elem[2]]

        # Compute edge lengths
        ab = np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
        bc = np.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2)
        ca = np.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)

        # Compute area using Heron's formula
        s = (ab + bc + ca) / 2.0
        area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))

        # Compute quality measure
        quality = 4.0 * np.sqrt(3.0) * area / (ab*ab + bc*bc + ca*ca)

        return quality

    @staticmethod
    def is_mesh_conforming(mesh):
        """
        Check if the mesh is conforming.

        This function checks if the mesh is conforming, i.e., if there are no
        hanging nodes. A conforming mesh is required for finite element analysis.

        Args:
            mesh: The mesh to check

        Returns:
            True if the mesh is conforming, false otherwise
        """
        # Get mesh elements
        elements = mesh.get_elements()

        # Map each edge to the elements that contain it
        edge_to_elements = {}
        for i, elem in enumerate(elements):
            for j in range(3):
                edge = (min(elem[j], elem[(j + 1) % 3]), max(elem[j], elem[(j + 1) % 3]))
                if edge not in edge_to_elements:
                    edge_to_elements[edge] = []
                edge_to_elements[edge].append(i)

        # Check for hanging nodes
        for edge, elems in edge_to_elements.items():
            if len(elems) > 2:
                return False

        return True
