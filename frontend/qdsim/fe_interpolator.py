"""
Python wrapper for the FEInterpolator C++ module.

This module provides a Python interface to the C++ FEInterpolator module,
which implements finite element interpolation for scalar fields on a mesh.
It falls back to a pure Python implementation if the C++ module is not available.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np

try:
    # Try to import the C++ extension module
    from . import fe_interpolator_ext
    _has_cpp_module = True
    print("Successfully imported fe_interpolator_ext")
except ImportError as e:
    # Fall back to Python implementation
    _has_cpp_module = False
    print(f"Warning: Could not import fe_interpolator_ext ({e}), falling back to Python implementation")

class FEInterpolator:
    """
    Finite Element Interpolator for scalar fields on a mesh.

    This class provides methods to interpolate scalar fields (like potentials)
    at arbitrary points in the mesh. It uses the C++ implementation if available,
    otherwise it falls back to a Python implementation.

    The interpolation is based on linear finite elements, using barycentric coordinates
    for interpolation within each triangular element. The gradient is also computed
    using the finite element shape functions.

    Physical units:
    - Coordinates (x, y): nanometers (nm)
    - Field values: arbitrary (depends on the field being interpolated)
    - Gradient: field units per nanometer (field units/nm)

    Assumptions and limitations:
    - The mesh is assumed to be a triangular mesh with linear elements
    - The field is assumed to be defined at the mesh nodes
    - The interpolation is linear within each element
    - Points outside the mesh are handled by returning the value at the nearest node

    Author: Dr. Mazharuddin Mohammed
    """

    def __init__(self, mesh, use_cpp=True):
        """
        Initialize the interpolator.

        Args:
            mesh: The mesh object, must have get_nodes() and get_elements() methods
            use_cpp: Whether to use the C++ implementation if available

        Raises:
            ValueError: If the mesh is invalid or cannot be used for interpolation
        """
        self.mesh = mesh
        self._use_cpp = use_cpp and _has_cpp_module

        if self._use_cpp:
            try:
                # Create the C++ interpolator
                try:
                    # Get nodes and elements from the mesh
                    try:
                        nodes = np.array(mesh.get_nodes()).tolist()
                    except Exception as e:
                        print(f"Warning: Failed to get nodes from mesh: {e}")
                        nodes = []

                    try:
                        elements = np.array(mesh.get_elements()).tolist()
                    except Exception as e:
                        print(f"Warning: Failed to get elements from mesh: {e}")
                        elements = []

                    # Check if we have valid nodes and elements
                    if not nodes or not elements:
                        raise ValueError("Invalid mesh: empty nodes or elements")

                    # Create the SimpleMesh
                    try:
                        # First try to create a SimpleMesh directly
                        cpp_mesh = fe_interpolator_ext.SimpleMesh(nodes, elements)

                        # Then create the SimpleInterpolator
                        self._cpp_interpolator = fe_interpolator_ext.SimpleInterpolator(cpp_mesh)
                    except Exception as e:
                        print(f"Warning: Failed to create SimpleMesh directly: {e}")
                        # Fall back to Python implementation
                        raise ValueError("Failed to create SimpleMesh")
                    print("Using C++ FEInterpolator")
                except Exception as e:
                    raise ValueError(f"Failed to create SimpleMesh: {e}")
            except Exception as e:
                print(f"Warning: Could not create C++ FEInterpolator ({e}), falling back to Python implementation")
                self._use_cpp = False

        if not self._use_cpp:
            # Create a quadtree for efficient element lookup
            from .quadtree import Quadtree
            nodes = np.array(mesh.get_nodes())
            elements = np.array(mesh.get_elements())
            self._quadtree = Quadtree(nodes, elements)

    def interpolate(self, x, y, field):
        """
        Interpolate a field at a point.

        Args:
            x, y: Coordinates of the point in nanometers (nm)
            field: Field values at mesh nodes, must have the same length as the number of nodes

        Returns:
            Interpolated value at (x, y)

        Raises:
            ValueError: If the field has an invalid length
            RuntimeError: If the interpolation fails
        """
        if self._use_cpp:
            return self._cpp_interpolator.interpolate(x, y, field)
        else:
            return self._interpolate_python(x, y, field)

    def interpolate_with_gradient(self, x, y, field):
        """
        Interpolate a field and its gradient at a point.

        Args:
            x, y: Coordinates of the point in nanometers (nm)
            field: Field values at mesh nodes, must have the same length as the number of nodes

        Returns:
            Tuple of (value, grad_x, grad_y), where grad_x and grad_y are the components
            of the gradient in field units per nanometer (field units/nm)

        Raises:
            ValueError: If the field has an invalid length
            RuntimeError: If the interpolation fails
        """
        if self._use_cpp:
            return self._cpp_interpolator.interpolate_with_gradient(x, y, field)
        else:
            return self._interpolate_with_gradient_python(x, y, field)

    def find_element(self, x, y):
        """
        Find the element containing a point.

        Args:
            x, y: Coordinates of the point in nanometers (nm)

        Returns:
            Element index or -1 if the point is outside the mesh

        Notes:
            This method is useful for determining if a point is inside the mesh,
            and for finding the element containing a point for further processing.
        """
        if self._use_cpp:
            return self._cpp_interpolator.find_element(x, y)
        else:
            return self._quadtree.find_element(x, y)

    def _interpolate_python(self, x, y, field):
        """
        Python implementation of interpolate.

        This method is used when the C++ implementation is not available.
        It uses barycentric coordinates for interpolation within each triangular element.

        Args:
            x, y: Coordinates of the point in nanometers (nm)
            field: Field values at mesh nodes

        Returns:
            Interpolated value at (x, y)
        """
        # Find the element containing the point
        element_idx, lambda_coords = self._quadtree.find_element(x, y, compute_barycentric=True)

        if element_idx < 0:
            # Point is outside the mesh
            # Find the nearest node
            nodes = np.array(self.mesh.get_nodes())
            distances = np.sqrt((nodes[:, 0] - x)**2 + (nodes[:, 1] - y)**2)
            nearest_node = np.argmin(distances)
            return field[nearest_node]

        # Get the element nodes
        elements = np.array(self.mesh.get_elements())
        elem_nodes = elements[element_idx]

        # Interpolate using barycentric coordinates
        return (lambda_coords[0] * field[elem_nodes[0]] +
                lambda_coords[1] * field[elem_nodes[1]] +
                lambda_coords[2] * field[elem_nodes[2]])

    def _interpolate_with_gradient_python(self, x, y, field):
        """
        Python implementation of interpolate_with_gradient.

        This method is used when the C++ implementation is not available.
        It uses barycentric coordinates for interpolation within each triangular element,
        and computes the gradient using the finite element shape functions.

        Args:
            x, y: Coordinates of the point in nanometers (nm)
            field: Field values at mesh nodes

        Returns:
            Tuple of (value, grad_x, grad_y), where grad_x and grad_y are the components
            of the gradient in field units per nanometer (field units/nm)
        """
        # Find the element containing the point
        element_idx, lambda_coords = self._quadtree.find_element(x, y, compute_barycentric=True)

        if element_idx < 0:
            # Point is outside the mesh
            # Find the nearest node
            nodes = np.array(self.mesh.get_nodes())
            distances = np.sqrt((nodes[:, 0] - x)**2 + (nodes[:, 1] - y)**2)
            nearest_node = np.argmin(distances)
            return field[nearest_node], 0.0, 0.0

        # Get the element nodes
        elements = np.array(self.mesh.get_elements())
        elem_nodes = elements[element_idx]

        # Get the vertices
        nodes = np.array(self.mesh.get_nodes())
        vertices = [nodes[elem_nodes[0]], nodes[elem_nodes[1]], nodes[elem_nodes[2]]]

        # Interpolate using barycentric coordinates
        value = (lambda_coords[0] * field[elem_nodes[0]] +
                 lambda_coords[1] * field[elem_nodes[1]] +
                 lambda_coords[2] * field[elem_nodes[2]])

        # Compute the gradient
        # For linear elements, the gradient is constant within each element
        det = (vertices[1][1] - vertices[2][1]) * (vertices[0][0] - vertices[2][0]) + \
              (vertices[2][0] - vertices[1][0]) * (vertices[0][1] - vertices[2][1])

        grad_x = ((field[elem_nodes[0]] * (vertices[1][1] - vertices[2][1]) +
                   field[elem_nodes[1]] * (vertices[2][1] - vertices[0][1]) +
                   field[elem_nodes[2]] * (vertices[0][1] - vertices[1][1])) / det)

        grad_y = ((field[elem_nodes[0]] * (vertices[2][0] - vertices[1][0]) +
                   field[elem_nodes[1]] * (vertices[0][0] - vertices[2][0]) +
                   field[elem_nodes[2]] * (vertices[1][0] - vertices[0][0])) / det)

        return value, grad_x, grad_y
