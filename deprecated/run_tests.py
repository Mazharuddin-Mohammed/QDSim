#!/usr/bin/env python3
"""
Main test runner script for QDSim.

This script runs all the tests in the test suite, including:
1. C++ unit tests
2. Python unit tests
3. Integration tests
4. Validation tests
5. Performance benchmarks

Author: Dr. Mazharuddin Mohammed
Date: 2023-07-15
"""

import os
import sys
import argparse
import subprocess
import time
import platform
import multiprocessing
import json
from pathlib import Path

def run_cpp_unit_tests(args):
    """Run C++ unit tests."""
    print("\n=== Running C++ Unit Tests ===\n")

    # Check if build directory exists
    if not os.path.exists("build"):
        print("Build directory not found. Creating and configuring...")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")

        cmake_args = ["cmake", ".."]
        if args.coverage:
            cmake_args.extend(["-DCMAKE_BUILD_TYPE=Debug", "-DENABLE_COVERAGE=ON"])

        result = subprocess.run(cmake_args, capture_output=not args.verbose)
        if result.returncode != 0:
            print("❌ CMake configuration failed")
            if not args.verbose:
                print("\nOutput:")
                print(result.stdout.decode())
                print(result.stderr.decode())
            os.chdir("..")
            return False

        # Build the tests
        make_args = ["make", f"-j{args.jobs}"]
        result = subprocess.run(make_args, capture_output=not args.verbose)
        if result.returncode != 0:
            print("❌ Build failed")
            if not args.verbose:
                print("\nOutput:")
                print(result.stdout.decode())
                print(result.stderr.decode())
            os.chdir("..")
            return False

        os.chdir("..")

    # Run the tests
    os.chdir("build")
    result = subprocess.run(["ctest", "--output-on-failure"], capture_output=not args.verbose)
    os.chdir("..")

    if result.returncode == 0:
        print("✅ C++ unit tests passed")
        return True
    else:
        print("❌ C++ unit tests failed")
        if not args.verbose:
            print("\nOutput:")
            print(result.stdout.decode())
            print(result.stderr.decode())
        return False

def run_python_unit_tests(args):
    """Run Python unit tests."""
    print("\n=== Running Python Unit Tests ===\n")

    # Find all Python test files
    unit_test_files = [
        os.path.join("tests", "unit", "test_fem_implementation.py"),
        os.path.join("frontend", "tests", "test_config.py"),
        os.path.join("frontend", "tests", "test_simulator.py"),
        os.path.join("frontend", "tests", "test_visualization.py")
    ]

    all_passed = True

    for test_file in unit_test_files:
        if not os.path.exists(test_file):
            print(f"⚠️ {test_file} not found, skipping")
            continue

        print(f"\nRunning {test_file}...")

        pytest_args = [sys.executable, "-m", "pytest", test_file]
        if args.coverage:
            pytest_args.extend(["--cov=frontend.qdsim", "--cov-report=term"])
        if args.verbose:
            pytest_args.append("-v")

        result = subprocess.run(pytest_args, capture_output=not args.verbose)

        if result.returncode == 0:
            print(f"✅ {test_file} passed")
        else:
            print(f"❌ {test_file} failed")
            if not args.verbose:
                print("\nOutput:")
                print(result.stdout.decode())
                print(result.stderr.decode())
            all_passed = False

    return all_passed

def run_integration_tests(args):
    """Run integration tests."""
    print("\n=== Running Integration Tests ===\n")

    integration_test_files = [
        os.path.join("tests", "integration", "test_fem_solver_integration.py"),
        os.path.join("tests", "integration", "test_poisson_drift_diffusion_integration.py")
    ]

    all_passed = True

    for test_file in integration_test_files:
        if not os.path.exists(test_file):
            print(f"⚠️ {test_file} not found, skipping")
            continue

        print(f"\nRunning {test_file}...")
        result = subprocess.run([sys.executable, test_file], capture_output=not args.verbose)

        if result.returncode == 0:
            print(f"✅ {test_file} passed")
        else:
            print(f"❌ {test_file} failed")
            if not args.verbose:
                print("\nOutput:")
                print(result.stdout.decode())
                print(result.stderr.decode())
            all_passed = False

    return all_passed

def run_validation_tests(args):
    """Run validation tests."""
    print("\n=== Running Validation Tests ===\n")

    validation_test_files = [
        os.path.join("tests", "validation", "test_analytical_solutions.py"),
        os.path.join("tests", "validation", "test_chromium_qd_algaas_pn_diode.py")
    ]

    all_passed = True

    for test_file in validation_test_files:
        if not os.path.exists(test_file):
            print(f"⚠️ {test_file} not found, skipping")
            continue

        print(f"\nRunning {test_file}...")
        result = subprocess.run([sys.executable, test_file], capture_output=not args.verbose)

        if result.returncode == 0:
            print(f"✅ {test_file} passed")
        else:
            print(f"❌ {test_file} failed")
            if not args.verbose:
                print("\nOutput:")
                print(result.stdout.decode())
                print(result.stderr.decode())
            all_passed = False

    return all_passed

def run_benchmarks(args):
    """Run performance benchmarks."""
    print("\n=== Running Performance Benchmarks ===\n")

    benchmark_files = [
        os.path.join("tests", "benchmarks", "test_performance_benchmarks.py")
    ]

    all_passed = True

    for benchmark_file in benchmark_files:
        if not os.path.exists(benchmark_file):
            print(f"⚠️ {benchmark_file} not found, skipping")
            continue

        print(f"\nRunning {benchmark_file}...")
        result = subprocess.run([sys.executable, benchmark_file], capture_output=not args.verbose)

        if result.returncode == 0:
            print(f"✅ {benchmark_file} completed successfully")
        else:
            print(f"❌ {benchmark_file} failed")
            if not args.verbose:
                print("\nOutput:")
                print(result.stdout.decode())
                print(result.stderr.decode())
            all_passed = False

    return all_passed

def generate_coverage_report(args):
    """Generate coverage report."""
    print("\n=== Generating Coverage Report ===\n")

    # C++ coverage
    if os.path.exists("build"):
        os.chdir("build")
        result = subprocess.run(["make", "coverage"], capture_output=not args.verbose)
        os.chdir("..")

        if result.returncode == 0:
            print("✅ C++ coverage report generated")
        else:
            print("❌ Failed to generate C++ coverage report")
            if not args.verbose:
                print("\nOutput:")
                print(result.stdout.decode())
                print(result.stderr.decode())

    # Python coverage
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "--cov=frontend.qdsim",
        "--cov-report=html:coverage_html",
        "frontend/tests"
    ], capture_output=not args.verbose)

    if result.returncode == 0:
        print("✅ Python coverage report generated")
    else:
        print("❌ Failed to generate Python coverage report")
        if not args.verbose:
            print("\nOutput:")
            print(result.stdout.decode())
            print(result.stderr.decode())

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run QDSim tests and benchmarks.")
    parser.add_argument("--cpp", action="store_true", help="Run C++ unit tests only")
    parser.add_argument("--python", action="store_true", help="Run Python unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--validation", action="store_true", help="Run validation tests only")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmarks only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage reports")
    parser.add_argument("--verbose", action="store_true", help="Show test output in real-time")
    parser.add_argument("--jobs", type=int, default=multiprocessing.cpu_count(), help="Number of parallel jobs for building")
    parser.add_argument("--output", type=str, default="test_results.json", help="Output file for test results")
    args = parser.parse_args()

    # If no specific tests are requested, run all tests
    run_all = not (args.cpp or args.python or args.integration or args.validation or args.benchmarks)

    start_time = time.time()
    results = {}

    if args.cpp or run_all:
        results["cpp_unit_tests"] = run_cpp_unit_tests(args)

    if args.python or run_all:
        results["python_unit_tests"] = run_python_unit_tests(args)

    if args.integration or run_all:
        results["integration_tests"] = run_integration_tests(args)

    if args.validation or run_all:
        results["validation_tests"] = run_validation_tests(args)

    if args.benchmarks or run_all:
        results["benchmarks"] = run_benchmarks(args)

    if args.coverage:
        generate_coverage_report(args)

    end_time = time.time()
    duration = end_time - start_time

    # Print summary
    print("\n=== Test Summary ===\n")
    all_passed = True
    for test_type, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_type}: {status}")
        all_passed = all_passed and passed

    print(f"\nAll tests completed in {duration:.2f} seconds")

    # Save results to file
    with open(args.output, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "duration": duration,
            "results": results
        }, f, indent=2)

    print(f"Test results saved to {args.output}")

    # Return exit code
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
