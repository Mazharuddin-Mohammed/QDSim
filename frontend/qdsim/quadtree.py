"""
Quadtree implementation for efficient spatial queries in the mesh.

This module provides a quadtree data structure for efficient spatial queries
in finite element meshes. It is optimized for adaptive meshes with varying
element sizes, and provides methods for finding the element containing a point.

The implementation uses a hierarchical tree structure to partition the space,
which reduces the search complexity from O(n) to O(log n) for point queries.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np

class QuadtreeNode:
    """A node in the quadtree."""

    def __init__(self, x_min, y_min, x_max, y_max, max_elements=10, max_depth=10):
        """
        Initialize a quadtree node.

        Args:
            x_min, y_min, x_max, y_max: Boundaries of this node
            max_elements: Maximum number of elements before splitting
            max_depth: Maximum depth of the tree
        """
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.max_elements = max_elements
        self.max_depth = max_depth
        self.elements = []  # List of (element_index, vertices) tuples
        self.children = None  # NW, NE, SW, SE quadrants
        self.depth = 0

    def insert(self, element_index, vertices, depth=0):
        """
        Insert an element into the quadtree.

        Args:
            element_index: Index of the element
            vertices: List of vertex coordinates [(x0, y0), (x1, y1), (x2, y2)]
            depth: Current depth in the tree
        """
        self.depth = depth

        # If we've already split, insert into children
        if self.children is not None:
            self._insert_into_children(element_index, vertices)
            return

        # Add the element to this node
        self.elements.append((element_index, vertices))

        # Split if we have too many elements and haven't reached max depth
        if len(self.elements) > self.max_elements and depth < self.max_depth:
            self._split()

    def _split(self):
        """Split this node into four children."""
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2

        # Create four children (NW, NE, SW, SE)
        self.children = [
            QuadtreeNode(self.x_min, y_mid, x_mid, self.y_max, self.max_elements, self.max_depth),  # NW
            QuadtreeNode(x_mid, y_mid, self.x_max, self.y_max, self.max_elements, self.max_depth),  # NE
            QuadtreeNode(self.x_min, self.y_min, x_mid, y_mid, self.max_elements, self.max_depth),  # SW
            QuadtreeNode(x_mid, self.y_min, self.x_max, y_mid, self.max_elements, self.max_depth)   # SE
        ]

        # Move existing elements to children
        elements = self.elements
        self.elements = []

        for element_index, vertices in elements:
            self._insert_into_children(element_index, vertices)

    def _insert_into_children(self, element_index, vertices):
        """Insert an element into all children that it overlaps."""
        for child in self.children:
            if self._element_overlaps_node(vertices, child):
                child.insert(element_index, vertices, self.depth + 1)

    def _element_overlaps_node(self, vertices, node):
        """
        Check if an element overlaps a node.

        This method is optimized for adaptive meshes with varying element sizes.
        It uses a more robust algorithm to handle small elements near boundaries.
        """
        # Check if any vertex is inside the node
        for x, y in vertices:
            if (node.x_min <= x <= node.x_max and
                node.y_min <= y <= node.y_max):
                return True

        # Check if any edge intersects the node boundary
        for i in range(3):
            j = (i + 1) % 3
            if self._line_intersects_box(
                vertices[i][0], vertices[i][1],
                vertices[j][0], vertices[j][1],
                node.x_min, node.y_min, node.x_max, node.y_max
            ):
                return True

        # Check if any corner of the node is inside the element
        corners = [
            (node.x_min, node.y_min),
            (node.x_max, node.y_min),
            (node.x_min, node.y_max),
            (node.x_max, node.y_max)
        ]

        for corner_x, corner_y in corners:
            if self._point_in_triangle(
                corner_x, corner_y,
                vertices[0][0], vertices[0][1],
                vertices[1][0], vertices[1][1],
                vertices[2][0], vertices[2][1]
            ):
                return True

        # Special case: very small elements might be completely inside the node
        # without any vertices being inside or edges intersecting
        # Compute the bounding box of the element
        elem_x_min = min(v[0] for v in vertices)
        elem_y_min = min(v[1] for v in vertices)
        elem_x_max = max(v[0] for v in vertices)
        elem_y_max = max(v[1] for v in vertices)

        # Check if the element's bounding box is completely inside the node
        if (node.x_min <= elem_x_min and elem_x_max <= node.x_max and
            node.y_min <= elem_y_min and elem_y_max <= node.y_max):
            return True

        return False

    def _line_intersects_box(self, x1, y1, x2, y2, box_x_min, box_y_min, box_x_max, box_y_max):
        """Check if a line segment intersects a box."""
        # Check if the line is completely outside the box
        if ((x1 < box_x_min and x2 < box_x_min) or
            (x1 > box_x_max and x2 > box_x_max) or
            (y1 < box_y_min and y2 < box_y_min) or
            (y1 > box_y_max and y2 > box_y_max)):
            return False

        # Check if either endpoint is inside the box
        if ((box_x_min <= x1 <= box_x_max and box_y_min <= y1 <= box_y_max) or
            (box_x_min <= x2 <= box_x_max and box_y_min <= y2 <= box_y_max)):
            return True

        # Check intersection with each edge of the box
        if self._line_intersects_line(x1, y1, x2, y2, box_x_min, box_y_min, box_x_max, box_y_min):
            return True  # Bottom edge
        if self._line_intersects_line(x1, y1, x2, y2, box_x_max, box_y_min, box_x_max, box_y_max):
            return True  # Right edge
        if self._line_intersects_line(x1, y1, x2, y2, box_x_min, box_y_max, box_x_max, box_y_max):
            return True  # Top edge
        if self._line_intersects_line(x1, y1, x2, y2, box_x_min, box_y_min, box_x_min, box_y_max):
            return True  # Left edge

        return False

    def _line_intersects_line(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Check if two line segments intersect."""
        # Compute the direction vectors
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3

        # Compute the cross product of the direction vectors
        cross = dx1 * dy2 - dy1 * dx2

        # If cross product is zero, lines are parallel
        if abs(cross) < 1e-10:
            return False

        # Compute the parameters of intersection
        s = ((x1 - x3) * dy2 - (y1 - y3) * dx2) / cross
        t = ((x1 - x3) * dy1 - (y1 - y3) * dx1) / cross

        # Check if the intersection point is within both line segments
        return 0 <= s <= 1 and 0 <= t <= 1

    def _point_in_triangle(self, px, py, x1, y1, x2, y2, x3, y3):
        """Check if a point is inside a triangle using barycentric coordinates."""
        # Compute barycentric coordinates
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) < 1e-10:
            return False

        lambda1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
        lambda2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
        lambda3 = 1 - lambda1 - lambda2

        # Check if the point is inside the triangle
        return (lambda1 >= 0 and lambda2 >= 0 and lambda3 >= 0)

    def query_point(self, x, y):
        """
        Find all elements that might contain the point (x, y).

        This method is optimized for adaptive meshes with varying element sizes.
        It prioritizes smaller elements which are more likely to contain the point
        in regions with high mesh refinement.

        Args:
            x, y: Coordinates of the point

        Returns:
            List of (element_index, vertices) tuples, sorted by element size (smallest first)
        """
        # Check if the point is outside this node
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return []

        # If we have children, query them
        if self.children is not None:
            result = []
            for child in self.children:
                if (child.x_min <= x <= child.x_max and
                    child.y_min <= y <= child.y_max):
                    result.extend(child.query_point(x, y))
            return result

        # Otherwise, return all elements in this node, sorted by size
        # This prioritizes smaller elements which are more likely to be the correct ones
        # in regions with high mesh refinement
        if not self.elements:
            return []

        # Compute the size (area) of each element
        elements_with_size = []
        for element_index, vertices in self.elements:
            # Compute the area of the triangle using the cross product
            v0, v1, v2 = vertices
            area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
            elements_with_size.append((element_index, vertices, area))

        # Sort by area (smallest first)
        elements_with_size.sort(key=lambda x: x[2])

        # Return the elements without the area
        return [(element_index, vertices) for element_index, vertices, _ in elements_with_size]


class Quadtree:
    """A quadtree for efficient spatial queries in the mesh."""

    def __init__(self, nodes, elements, max_elements=10, max_depth=10):
        """
        Initialize the quadtree.

        This implementation is optimized for adaptive meshes with varying element sizes.
        It uses a more balanced tree structure and handles small elements better.

        Args:
            nodes: List of node coordinates [(x0, y0), (x1, y1), ...]
            elements: List of element indices [[n0, n1, n2], [n3, n4, n5], ...]
            max_elements: Maximum number of elements in a leaf node
            max_depth: Maximum depth of the tree
        """
        # Find the bounding box of the mesh
        x_min = min(node[0] for node in nodes)
        y_min = min(node[1] for node in nodes)
        x_max = max(node[0] for node in nodes)
        y_max = max(node[1] for node in nodes)

        # Add a small margin to ensure all elements are inside
        margin = 0.01 * max(x_max - x_min, y_max - y_min)
        x_min -= margin
        y_min -= margin
        x_max += margin
        y_max += margin

        # Create the root node
        self.root = QuadtreeNode(x_min, y_min, x_max, y_max, max_elements, max_depth)

        # Analyze the mesh to determine element size distribution
        element_sizes = []
        for i, element in enumerate(elements):
            vertices = [nodes[element[j]] for j in range(3)]
            # Compute the area of the triangle using the cross product
            v0, v1, v2 = vertices
            area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
            element_sizes.append((i, vertices, area))

        # Sort elements by size (largest first) to ensure larger elements are inserted first
        # This helps create a more balanced tree for adaptive meshes
        element_sizes.sort(key=lambda x: x[2], reverse=True)

        # Insert all elements
        for i, vertices, _ in element_sizes:
            self.root.insert(i, vertices)

    def find_element(self, x, y, compute_barycentric=False):
        """
        Find the element containing the point (x, y).

        This method is optimized for adaptive meshes with varying element sizes.
        It prioritizes smaller elements which are more likely to contain the point
        in regions with high mesh refinement.

        Args:
            x, y: Coordinates of the point
            compute_barycentric: If True, also compute barycentric coordinates

        Returns:
            If compute_barycentric is False:
                element_index or -1 if not found
            If compute_barycentric is True:
                (element_index, [lambda0, lambda1, lambda2]) or (-1, None) if not found
        """
        # Query the quadtree for candidate elements
        # The query_point method already sorts elements by size (smallest first)
        # which is optimal for adaptive meshes
        candidates = self.root.query_point(x, y)

        # Check each candidate
        for element_index, vertices in candidates:
            # Compute barycentric coordinates
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]

            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if abs(det) < 1e-10:
                continue

            lambda1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
            lambda2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
            lambda3 = 1 - lambda1 - lambda2

            # Check if the point is inside the element
            # Use a slightly larger tolerance for points near the boundary
            # This is especially important for adaptive meshes where elements
            # can vary greatly in size
            eps = 1e-10
            if lambda1 >= -eps and lambda2 >= -eps and lambda3 >= -eps:
                if compute_barycentric:
                    return element_index, [lambda1, lambda2, lambda3]
                else:
                    return element_index

        # If no exact match is found, try to find the closest element
        # This is useful for points that are slightly outside the mesh
        # due to numerical precision issues
        if candidates:
            # Find the element with the closest barycentric coordinates
            best_element = -1
            best_distance = float('inf')
            best_lambdas = None

            for element_index, vertices in candidates:
                x1, y1 = vertices[0]
                x2, y2 = vertices[1]
                x3, y3 = vertices[2]

                det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                if abs(det) < 1e-10:
                    continue

                lambda1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
                lambda2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
                lambda3 = 1 - lambda1 - lambda2

                # Compute the distance to the closest edge or vertex
                # This is the minimum of the barycentric coordinates
                # (negative values indicate distance outside the element)
                distance = min(
                    abs(min(0, lambda1)),
                    abs(min(0, lambda2)),
                    abs(min(0, lambda3))
                )

                if distance < best_distance:
                    best_distance = distance
                    best_element = element_index
                    best_lambdas = [lambda1, lambda2, lambda3]

            # If the best element is very close (within a larger tolerance),
            # return it as a fallback
            if best_distance < 1e-6:
                if compute_barycentric:
                    return best_element, best_lambdas
                else:
                    return best_element

        # Point is not in any element
        if compute_barycentric:
            return -1, None
        else:
            return -1
