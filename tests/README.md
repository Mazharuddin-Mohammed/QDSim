# QDSim Test Suite

This directory contains a comprehensive test suite for QDSim, including unit tests, validation tests, and performance benchmarks.

## Directory Structure

- `unit/`: Unit tests for individual components of QDSim
- `validation/`: Validation tests comparing QDSim results with analytical solutions and realistic test cases
- `benchmarks/`: Performance benchmarks for different mesh sizes, element orders, and parallel configurations

## Running Tests

You can run the tests using the `run_tests.py` script in the root directory:

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run only validation tests
python run_tests.py --validation

# Run only benchmarks
python run_tests.py --benchmarks

# Show test output in real-time
python run_tests.py --verbose
```

## Test Descriptions

### Unit Tests

- `test_fem_implementation.py`: Tests the FEM implementation, including element matrix assembly, finite element interpolation, and MPI data transfer efficiency.

### Validation Tests

- `test_analytical_solutions.py`: Validates QDSim against known analytical solutions for the infinite square well, harmonic oscillator, and hydrogen-like atom.
- `test_chromium_qd_algaas_pn_diode.py`: Validates QDSim with a realistic test case of a chromium quantum dot embedded in an AlGaAs P-N junction diode under reverse bias.

### Performance Benchmarks

- `test_performance_benchmarks.py`: Benchmarks the performance of QDSim with different mesh sizes, element orders, parallel configurations, and GPU acceleration (if available).

## Test Results

The tests generate various output files, including:

- PNG images of wavefunctions, potentials, and probability densities
- Performance benchmark plots
- A comprehensive benchmark report (`benchmark_report.md`)

## Adding New Tests

To add a new test:

1. Create a new Python file in the appropriate directory (`unit/`, `validation/`, or `benchmarks/`)
2. Add the test file to the corresponding list in `run_tests.py`
3. Run the tests to verify that the new test works correctly

## Author

Dr. Mazharuddin Mohammed
