# QDSim Testing Guide

This guide outlines the testing standards for the QDSim project. Following these standards ensures that the codebase is well-tested, reliable, and maintainable.

## Table of Contents

1. [General Principles](#general-principles)
2. [Test Types](#test-types)
3. [C++ Testing](#c-testing)
4. [Python Testing](#python-testing)
5. [Test Coverage](#test-coverage)
6. [Continuous Integration](#continuous-integration)
7. [Test Documentation](#test-documentation)
8. [Test Data](#test-data)
9. [Performance Testing](#performance-testing)
10. [Validation Testing](#validation-testing)

## General Principles

- **Completeness**: Test all public APIs, classes, and functions.
- **Isolation**: Tests should be independent and not rely on the state of other tests.
- **Readability**: Tests should be easy to understand and maintain.
- **Reliability**: Tests should produce consistent results.
- **Speed**: Tests should run quickly to encourage frequent testing.
- **Coverage**: Aim for high test coverage, especially for critical components.

## Test Types

### Unit Tests

Unit tests verify that individual components (functions, methods, classes) work correctly in isolation. They should:

- Test a single unit of functionality
- Mock or stub dependencies
- Run quickly
- Be deterministic

### Integration Tests

Integration tests verify that multiple components work correctly together. They should:

- Test interactions between components
- Use real dependencies where possible
- Verify end-to-end workflows

### Validation Tests

Validation tests verify that the simulation results match expected physical behavior. They should:

- Compare simulation results with analytical solutions
- Verify physical accuracy
- Test edge cases and boundary conditions

## C++ Testing

We use Google Test for C++ testing. Test files should be placed in the `tests/cpp` directory and follow the naming convention `test_*.cpp`.

### Test Fixture

Use test fixtures to set up common test data and utilities:

```cpp
class MeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        mesh = Mesh(10, 10);
    }

    void TearDown() override {
        // Clean up test data
    }

    Mesh mesh;
};
```

### Test Cases

Write test cases using the `TEST_F` macro for fixture-based tests or the `TEST` macro for simple tests:

```cpp
TEST_F(MeshTest, RefineCreatesNewElements) {
    // Arrange
    std::vector<bool> refinement_flags(mesh.getNumElements(), false);
    refinement_flags[0] = true;
    
    // Act
    int new_elements = mesh.refine(refinement_flags);
    
    // Assert
    EXPECT_GT(new_elements, 0);
    EXPECT_GT(mesh.getNumElements(), 10);
}

TEST(VectorTest, DotProductIsCorrect) {
    // Arrange
    Vector v1(1.0, 2.0, 3.0);
    Vector v2(4.0, 5.0, 6.0);
    
    // Act
    double dot = v1.dot(v2);
    
    // Assert
    EXPECT_DOUBLE_EQ(dot, 32.0);
}
```

### Mocking

Use Google Mock to mock dependencies:

```cpp
class MockSolver : public Solver {
public:
    MOCK_METHOD(void, solve, (const Matrix&, const Vector&, Vector&), (override));
};

TEST(FEMTest, SolveCallsSolver) {
    // Arrange
    MockSolver solver;
    FEM fem(&solver);
    Matrix A(10, 10);
    Vector b(10), x(10);
    
    EXPECT_CALL(solver, solve(A, b, testing::_))
        .Times(1);
    
    // Act
    fem.solve(A, b, x);
    
    // Assert
    // No need for explicit assertions, EXPECT_CALL handles it
}
```

## Python Testing

We use pytest for Python testing. Test files should be placed in the `tests/python` directory and follow the naming convention `test_*.py`.

### Test Fixtures

Use pytest fixtures to set up common test data and utilities:

```python
import pytest
from qdsim import Mesh

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

### Parametrized Tests

Use parametrized tests to test multiple inputs:

```python
import pytest
from qdsim.physics import potential

@pytest.mark.parametrize("x,y,expected", [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 2.0),
])
def test_harmonic_potential(x, y, expected):
    # Arrange
    k = 1.0
    
    # Act
    result = potential.harmonic(x, y, k)
    
    # Assert
    assert result == pytest.approx(expected)
```

### Mocking

Use unittest.mock to mock dependencies:

```python
from unittest.mock import Mock, patch
from qdsim import Simulator

def test_run_calls_solver():
    # Arrange
    mock_solver = Mock()
    simulator = Simulator(solver=mock_solver)
    
    # Act
    simulator.run()
    
    # Assert
    mock_solver.solve.assert_called_once()
```

## Test Coverage

We use gcov/lcov for C++ test coverage and coverage.py for Python test coverage.

### C++ Coverage

To generate C++ test coverage:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make
make test
make coverage
```

This will generate coverage reports in the `build/coverage` directory.

### Python Coverage

To generate Python test coverage:

```bash
pytest --cov=qdsim tests/python
```

To generate an HTML coverage report:

```bash
pytest --cov=qdsim --cov-report=html tests/python
```

This will generate coverage reports in the `htmlcov` directory.

## Continuous Integration

We use GitHub Actions for continuous integration. The CI pipeline runs:

- C++ and Python tests
- Code coverage analysis
- Static code analysis
- Documentation generation

The CI configuration is located in the `.github/workflows` directory.

## Test Documentation

### Test File Headers

Every test file should begin with a header that includes:

- File name
- Brief description
- Author
- Date

Example:

```cpp
/**
 * @file test_mesh.cpp
 * @brief Tests for the Mesh class.
 *
 * @author Dr. Mazharuddin Mohammed
 * @date 2023-05-15
 */
```

### Test Case Documentation

Document test cases with:

- Brief description
- Test scenario
- Expected outcome

Example:

```cpp
/**
 * @brief Test mesh refinement.
 *
 * This test verifies that refining a mesh with a single element
 * marked for refinement creates new elements.
 *
 * @test Mesh::refine
 * @see Mesh::getNumElements
 */
TEST_F(MeshTest, RefineCreatesNewElements) {
    // ...
}
```

## Test Data

### Test Data Files

Test data files should be placed in the `tests/data` directory and follow a clear naming convention.

### Test Data Generation

Document how test data is generated, especially for complex or large datasets.

Example:

```cpp
/**
 * @brief Generate a test mesh.
 *
 * This function generates a test mesh with a specified number of elements.
 * The mesh is a square domain with uniform triangular elements.
 *
 * @param nx Number of elements in the x direction
 * @param ny Number of elements in the y direction
 * @return A test mesh
 */
Mesh generateTestMesh(int nx, int ny) {
    // ...
}
```

## Performance Testing

Performance tests verify that the code meets performance requirements. They should:

- Measure execution time
- Measure memory usage
- Compare against baseline performance
- Run on consistent hardware

Example:

```cpp
TEST(PerformanceTest, MeshRefinementPerformance) {
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
    EXPECT_LT(duration.count(), 1000);  // Should complete in less than 1 second
}
```

## Validation Testing

Validation tests verify that the simulation results match expected physical behavior. They should:

- Compare simulation results with analytical solutions
- Verify physical accuracy
- Test edge cases and boundary conditions

Example:

```cpp
TEST(ValidationTest, HarmonicOscillatorEigenvalues) {
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
    EXPECT_NEAR(result.eigenvalues[0], 1.0 * hbar_omega, 1e-2);
    EXPECT_NEAR(result.eigenvalues[1], 2.0 * hbar_omega, 1e-2);
    EXPECT_NEAR(result.eigenvalues[2], 2.0 * hbar_omega, 1e-2);
    EXPECT_NEAR(result.eigenvalues[3], 3.0 * hbar_omega, 1e-2);
}
```

## Conclusion

Following these testing standards will help ensure that the QDSim project is well-tested, reliable, and maintainable. If you have any questions or suggestions for improving these standards, please open an issue on the project repository.
