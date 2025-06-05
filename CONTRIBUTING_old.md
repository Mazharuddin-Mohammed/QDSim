# Contributing to QDSim

Thank you for your interest in contributing to QDSim! This document outlines the process and guidelines for contributing to this project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Documentation Guidelines](#documentation-guidelines)
6. [Testing Guidelines](#testing-guidelines)
7. [Pull Request Process](#pull-request-process)
8. [Attribution](#attribution)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read and follow it to ensure a positive and respectful environment for all contributors.

- Be respectful and inclusive
- Be patient and welcoming
- Be considerate
- Be collaborative
- Be careful in the words you choose
- When we disagree, try to understand why

## Getting Started

### Prerequisites
- C++17-compatible compiler (e.g., GCC 7+, Clang 5+)
- CMake 3.10+
- MPI (e.g., OpenMPI, MPICH)
- Eigen3 (linear algebra library)
- Spectra (eigenvalue solver)
- Catch2 (testing framework)
- Python 3.8+
- Pybind11, NumPy, Matplotlib, PySide6, pytest

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/QDSim.git
   cd QDSim
   ```
3. Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/Mazharuddin-Mohammed/QDSim.git
   ```
4. Create a build directory and build the project:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make -j4
   ```
5. Install the Python package in development mode:
   ```bash
   cd ..
   pip install -e .
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   or
   ```bash
   git checkout -b fix/your-bugfix-name
   ```

2. Make your changes, following the coding standards and guidelines below.

3. Run tests to ensure your changes don't break existing functionality:
   ```bash
   cd build
   ctest
   cd ../frontend
   pytest
   ```

4. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: description of your feature"
   ```

5. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a pull request from your branch to the main repository.

## Coding Standards

### C++ Code Style

- Follow the Google C++ Style Guide with the following exceptions:
  - Use 4 spaces for indentation, not tabs
  - Line length limit is 100 characters
  - Use camelCase for method names, not snake_case
  - Use snake_case for variable names

- Always include proper header guards:
  ```cpp
  #ifndef QDSIM_MODULE_HEADER_H_
  #define QDSIM_MODULE_HEADER_H_

  // Code here

  #endif  // QDSIM_MODULE_HEADER_H_
  ```

- Use forward declarations when possible to minimize include dependencies
- Use const-correctness throughout the codebase
- Use smart pointers (std::unique_ptr, std::shared_ptr) instead of raw pointers
- Use C++17 features when appropriate (e.g., std::optional, structured bindings)
- Avoid using exceptions for control flow; use them only for exceptional conditions

### Python Code Style

- Follow PEP 8 style guide
- Use 4 spaces for indentation, not tabs
- Line length limit is 88 characters (compatible with Black formatter)
- Use snake_case for function and variable names
- Use CamelCase for class names
- Use docstrings for all public functions, classes, and methods
- Type hints are encouraged for function parameters and return values

### General Guidelines

- Write self-documenting code with clear variable and function names
- Keep functions small and focused on a single task
- Avoid global variables
- Minimize code duplication; use functions or classes to encapsulate repeated logic
- Use meaningful comments to explain why, not what
- Always add proper author attribution to Dr. Mazharuddin Mohammed in file headers

## Documentation Guidelines

### Code Documentation

- All public APIs must be documented
- Use Doxygen-style comments for C++ code:
  ```cpp
  /**
   * @brief Brief description of the function
   *
   * Detailed description of the function, including any important notes
   * or implementation details.
   *
   * @param param1 Description of first parameter
   * @param param2 Description of second parameter
   * @return Description of return value
   * @throws Description of exceptions that might be thrown
   */
  ```

- Use Google-style docstrings for Python code:
  ```python
  def function_name(param1, param2):
      """Brief description of the function.

      Detailed description of the function, including any important notes
      or implementation details.

      Args:
          param1: Description of first parameter
          param2: Description of second parameter

      Returns:
          Description of return value

      Raises:
          ExceptionType: Description of when this exception is raised
      """
  ```

### README and Other Documentation

- Keep the README.md up-to-date with the current state of the project
- Document all major features and components
- Include examples and usage instructions
- Use proper Markdown formatting for headings, code blocks, and lists
- Use LaTeX for mathematical equations (GitHub-flavored Markdown supports this)

## Testing Guidelines

QDSim requires comprehensive testing to ensure correctness, reliability, and physical accuracy. The test suite is organized into several categories:

### Unit Tests

Unit tests verify that individual components work correctly in isolation:

- Use Catch2 for C++ unit tests
- Use pytest for Python unit tests
- Aim for at least 90% code coverage
- Test both normal and edge cases
- Use descriptive test names that explain what is being tested
- Organize C++ tests in a logical hierarchy using TEST_CASE and SECTION
- Use pytest fixtures for setup and teardown in Python tests
- Use parametrized tests for testing multiple inputs
- Mock external dependencies when necessary

Example C++ unit test:
```cpp
TEST_CASE("Mesh refinement", "[mesh]") {
    // Arrange
    Mesh mesh(10, 10);
    std::vector<bool> refinement_flags(mesh.getNumElements(), false);
    refinement_flags[0] = true;

    // Act
    int new_elements = mesh.refine(refinement_flags);

    // Assert
    REQUIRE(new_elements > 0);
    REQUIRE(mesh.getNumElements() > 10);

    SECTION("Refinement preserves mesh quality") {
        // Additional tests for mesh quality
        REQUIRE(mesh.getMinQuality() > 0.1);
    }
}
```

Example Python unit test:
```python
@pytest.fixture
def mesh():
    """Create a test mesh."""
    return Mesh(10, 10)

def test_refine_creates_new_elements(mesh):
    # Arrange
    refinement_flags = [False] * mesh.get_num_elements()
    refinement_flags[0] = True

    # Act
    new_elements = mesh.refine(refinement_flags)

    # Assert
    assert new_elements > 0
    assert mesh.get_num_elements() > 10
```

### Integration Tests

Integration tests verify that multiple components work correctly together:

- Test interactions between components
- Use real dependencies where possible
- Verify end-to-end workflows
- Test error handling and edge cases
- Include tests for MPI and GPU acceleration

Example integration test:
```cpp
TEST_CASE("FEM solver with adaptive mesh refinement", "[integration]") {
    // Arrange
    Simulator simulator;
    simulator.setDomain(-10.0, 10.0, -10.0, 10.0);
    simulator.setMeshSize(20, 20);
    simulator.setPotential(PotentialType::HARMONIC);
    simulator.setAdaptiveMeshRefinement(true);

    // Act
    auto result = simulator.run(5);

    // Assert
    REQUIRE(result.eigenvalues.size() == 5);
    REQUIRE(simulator.getMesh().getNumElements() > 400);  // Mesh should be refined

    // Check eigenvalues against analytical solution for harmonic oscillator
    REQUIRE(result.eigenvalues[0] == Approx(1.0).epsilon(0.05));
    REQUIRE(result.eigenvalues[1] == Approx(2.0).epsilon(0.05));
    REQUIRE(result.eigenvalues[2] == Approx(2.0).epsilon(0.05));
}
```

### Validation Tests

Validation tests verify that the simulation results match expected physical behavior:

- Compare simulation results with analytical solutions
- Verify physical accuracy
- Test edge cases and boundary conditions
- Include tests for different material parameters
- Validate against experimental data when available

Example validation test:
```cpp
TEST_CASE("Harmonic oscillator eigenvalues", "[validation]") {
    // Arrange
    Simulator simulator;
    simulator.setPotential(PotentialType::HARMONIC);
    simulator.setDomain(-10.0, 10.0, -10.0, 10.0);
    simulator.setMeshSize(100, 100);

    // Act
    auto result = simulator.run(10);

    // Assert
    // For a 2D harmonic oscillator, eigenvalues should be E_n,m = (n + m + 1) * hbar * omega
    double hbar_omega = 1.0;  // Assuming normalized units
    REQUIRE(result.eigenvalues[0] == Approx(1.0 * hbar_omega).epsilon(0.01));
    REQUIRE(result.eigenvalues[1] == Approx(2.0 * hbar_omega).epsilon(0.01));
    REQUIRE(result.eigenvalues[2] == Approx(2.0 * hbar_omega).epsilon(0.01));
    REQUIRE(result.eigenvalues[3] == Approx(3.0 * hbar_omega).epsilon(0.01));
}
```

### Performance Tests

Performance tests verify that the code meets performance requirements:

- Measure execution time
- Measure memory usage
- Compare against baseline performance
- Test scalability with problem size
- Test parallel performance with different numbers of processes/threads
- Test GPU acceleration

Example performance test:
```cpp
TEST_CASE("Mesh refinement performance", "[performance]") {
    // Arrange
    Mesh mesh = generateLargeMesh();
    std::vector<bool> refinement_flags(mesh.getNumElements(), false);
    for (int i = 0; i < refinement_flags.size(); i += 10) {
        refinement_flags[i] = true;
    }

    // Act
    auto start = std::chrono::high_resolution_clock::now();
    mesh.refine(refinement_flags);
    auto end = std::chrono::high_resolution_clock::now();

    // Assert
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    REQUIRE(duration.count() < 1000);  // Should complete in less than 1 second
}
```

### Test Coverage

To generate test coverage reports:

```bash
# C++ test coverage
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make
make test
make coverage

# Python test coverage
cd frontend
pytest --cov=qdsim tests/
```

Aim for at least 90% code coverage for critical components and at least 80% overall.

### Continuous Integration

QDSim uses GitHub Actions for continuous integration. The CI pipeline runs all tests on every push and pull request, ensuring that the code remains reliable and correct. The CI configuration is located in the `.github/workflows` directory.

When adding new features or fixing bugs, make sure to add appropriate tests to verify the changes.

## Pull Request Process

1. Ensure your code follows the coding standards and passes all tests
2. Update the documentation, including the README.md if necessary
3. Add or update tests as appropriate
4. Make sure your branch is up-to-date with the main branch:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```
5. Create a pull request with a clear title and description
6. Address any feedback from code reviews
7. Once approved, your pull request will be merged by a maintainer

### Pull Request Template

When creating a pull request, please use the following template:

```markdown
## Description
[Provide a brief description of the changes in this pull request]

## Related Issue
[Reference any related issue(s) using #issue_number]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code cleanup or refactoring

## How Has This Been Tested?
[Describe the tests that you ran to verify your changes]

## Checklist
- [ ] My code follows the coding standards of this project
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the documentation accordingly
- [ ] I have added proper author attribution to Dr. Mazharuddin Mohammed
- [ ] I have checked that my changes don't break existing functionality
```

## Attribution

All contributions to QDSim must include proper attribution to Dr. Mazharuddin Mohammed as the original author. This should be included in the file header comments as follows:

For C++ files:
```cpp
/**
 * @file filename.h
 * @brief Brief description of the file
 * @author Dr. Mazharuddin Mohammed
 * @author Your Name (if you've made significant changes)
 */
```

For Python files:
```python
"""
Brief description of the file

Author: Dr. Mazharuddin Mohammed
Contributors: Your Name (if you've made significant changes)
"""
```

---

Thank you for contributing to QDSim! Your efforts help make this project better for everyone.
