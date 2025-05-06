# QDSim Documentation Guide

This guide outlines the documentation standards for the QDSim project. Following these standards ensures that the codebase is well-documented, maintainable, and accessible to new contributors.

## Table of Contents

1. [General Principles](#general-principles)
2. [C++ Documentation](#c-documentation)
3. [Python Documentation](#python-documentation)
4. [Markdown Documentation](#markdown-documentation)
5. [Documentation Generation](#documentation-generation)
6. [Examples and Tutorials](#examples-and-tutorials)
7. [Version Control and Documentation](#version-control-and-documentation)

## General Principles

- **Completeness**: Document all public APIs, classes, functions, and parameters.
- **Clarity**: Use clear, concise language that is easy to understand.
- **Consistency**: Follow the same documentation style throughout the codebase.
- **Examples**: Include examples where appropriate to illustrate usage.
- **Updates**: Keep documentation up-to-date with code changes.
- **Attribution**: Include author attribution in all source files.

## C++ Documentation

### File Headers

Every C++ file should begin with a file header that includes:

- File name
- Brief description
- Detailed description
- Author
- Date

Example:

```cpp
/**
 * @file mesh.h
 * @brief Defines the mesh data structure for finite element simulations.
 *
 * This file contains the declaration of the Mesh class, which represents
 * a 2D triangular mesh used for finite element simulations. It provides
 * methods for mesh generation, refinement, and manipulation.
 *
 * @author Dr. Mazharuddin Mohammed
 * @date 2023-05-15
 */
```

### Class Documentation

Document classes with:

- Brief description
- Detailed description
- Template parameters (if applicable)
- See also (related classes or functions)

Example:

```cpp
/**
 * @class Mesh
 * @brief Represents a 2D triangular mesh for finite element simulations.
 *
 * The Mesh class provides a data structure for storing and manipulating
 * a 2D triangular mesh. It includes methods for mesh generation, refinement,
 * and manipulation, as well as access to mesh elements and nodes.
 *
 * @see FEMSolver
 * @see AdaptiveMesh
 */
```

### Method Documentation

Document methods with:

- Brief description
- Detailed description
- Parameters
- Return value
- Exceptions
- Notes, warnings, or see also (if applicable)

Example:

```cpp
/**
 * @brief Refines the mesh based on refinement flags.
 *
 * This method refines the mesh by splitting elements marked for refinement.
 * It updates the mesh data structure to include the new nodes and elements.
 *
 * @param refinement_flags Vector of boolean flags indicating which elements to refine
 * @return Number of new elements created during refinement
 * @throws std::runtime_error If the mesh is invalid or cannot be refined
 *
 * @note This method preserves the mesh quality by adjusting neighboring elements.
 * @see AdaptiveMesh::computeRefinementFlags()
 */
int refine(const std::vector<bool>& refinement_flags);
```

### Member Variable Documentation

Document member variables with:

- Brief description

Example:

```cpp
int num_nodes;  ///< Number of nodes in the mesh
int num_elements;  ///< Number of elements in the mesh
```

### Namespace Documentation

Document namespaces with:

- Brief description
- Detailed description

Example:

```cpp
/**
 * @namespace Physics
 * @brief Contains physics-related functions and constants.
 *
 * This namespace contains functions and constants related to the physics
 * of semiconductor quantum dots, including effective mass, potential,
 * and other physical properties.
 */
```

### Templates

Document template parameters with:

- Brief description for each parameter

Example:

```cpp
/**
 * @brief A generic container for mesh data.
 *
 * @tparam T The data type stored in the container
 * @tparam Allocator The allocator type (default: std::allocator<T>)
 */
template <typename T, typename Allocator = std::allocator<T>>
class MeshData {
    // ...
};
```

## Python Documentation

### Module Headers

Every Python module should begin with a module header that includes:

- Module name
- Brief description
- Detailed description
- Author
- Date

Example:

```python
#!/usr/bin/env python3
"""
Module: simulator

Brief description of the simulator module.

Detailed description of the simulator module, including its purpose,
the main components it contains, and how it fits into the overall system.

Author: Dr. Mazharuddin Mohammed
Date: 2023-05-15
"""
```

### Class Documentation

Document classes with:

- Brief description
- Detailed description
- Attributes

Example:

```python
class Simulator:
    """
    Quantum dot simulator for 2D simulations.
    
    The Simulator class provides a high-level interface for running
    quantum dot simulations. It encapsulates the mesh generation,
    finite element assembly, and eigenvalue solution process.
    
    Attributes:
        config (Config): The simulation configuration.
        mesh (Mesh): The simulation mesh.
        solver (Solver): The eigenvalue solver.
    """
```

### Method Documentation

Document methods with:

- Brief description
- Detailed description
- Parameters
- Return value
- Exceptions
- Notes, warnings, or see also (if applicable)

Example:

```python
def run(self, num_eigenvalues=10):
    """
    Run the quantum dot simulation.
    
    This method runs the quantum dot simulation by setting up the mesh,
    assembling the finite element matrices, and solving the eigenvalue problem.
    
    Args:
        num_eigenvalues (int, optional): Number of eigenvalues to compute. Defaults to 10.
        
    Returns:
        tuple: A tuple containing (eigenvalues, eigenvectors).
        
    Raises:
        SimulationError: If the simulation fails to run.
        
    Note:
        The eigenvalues are returned in ascending order.
    """
```

### Function Documentation

Document functions with:

- Brief description
- Detailed description
- Parameters
- Return value
- Exceptions
- Notes, warnings, or see also (if applicable)

Example:

```python
def plot_wavefunction(wavefunction, mesh, title=None):
    """
    Plot a wavefunction on a mesh.
    
    This function plots a wavefunction on a mesh using matplotlib.
    It creates a 2D color plot of the wavefunction magnitude.
    
    Args:
        wavefunction (numpy.ndarray): The wavefunction values at mesh nodes.
        mesh (Mesh): The mesh on which the wavefunction is defined.
        title (str, optional): The plot title. Defaults to None.
        
    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
```

## Markdown Documentation

### README Files

README files should include:

- Project name and brief description
- Installation instructions
- Basic usage examples
- Dependencies
- License information
- Contact information

### User Guide

The user guide should include:

- Detailed installation instructions
- Configuration options
- Usage examples
- Troubleshooting
- FAQ

### Theory Documentation

The theory documentation should include:

- Mathematical background
- Physical models
- Numerical methods
- References to relevant literature

## Documentation Generation

### Doxygen

We use Doxygen to generate documentation for the C++ codebase. The Doxyfile is located in the root directory of the project.

To generate the documentation:

```bash
doxygen Doxyfile
```

This will generate HTML and LaTeX documentation in the `docs/html` and `docs/latex` directories, respectively.

### Sphinx

We use Sphinx to generate documentation for the Python codebase. The Sphinx configuration is located in the `docs` directory.

To generate the documentation:

```bash
cd docs
make html
```

This will generate HTML documentation in the `docs/_build/html` directory.

## Examples and Tutorials

### Example Scripts

Example scripts should be well-documented with:

- Script purpose
- Usage instructions
- Expected output
- Dependencies

### Tutorials

Tutorials should be written in Markdown or Jupyter Notebook format and should include:

- Step-by-step instructions
- Code snippets
- Explanations of key concepts
- Expected results

## Version Control and Documentation

### Commit Messages

Commit messages should be clear and descriptive, and should reference documentation updates when applicable:

```
Add documentation for Mesh class

- Add class documentation
- Add method documentation
- Add examples
```

### Pull Requests

Pull requests should include documentation updates for any new features or changes:

```
This PR adds a new feature for adaptive mesh refinement.

- Add AdaptiveMesh class
- Add documentation for AdaptiveMesh class
- Add example script for adaptive mesh refinement
- Update user guide with adaptive mesh refinement section
```

## Conclusion

Following these documentation standards will help ensure that the QDSim project is well-documented, maintainable, and accessible to new contributors. If you have any questions or suggestions for improving these standards, please open an issue on the project repository.
