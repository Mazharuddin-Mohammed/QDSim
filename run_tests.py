#!/usr/bin/env python3
"""
Main test runner script for QDSim.

This script runs all the tests in the test suite, including:
1. Unit tests
2. Validation tests
3. Performance benchmarks

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import argparse
import subprocess
import time

def run_unit_tests(args):
    """Run unit tests."""
    print("\n=== Running Unit Tests ===\n")
    
    unit_test_files = [
        os.path.join("tests", "unit", "test_fem_implementation.py")
    ]
    
    for test_file in unit_test_files:
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

def run_validation_tests(args):
    """Run validation tests."""
    print("\n=== Running Validation Tests ===\n")
    
    validation_test_files = [
        os.path.join("tests", "validation", "test_analytical_solutions.py"),
        os.path.join("tests", "validation", "test_chromium_qd_algaas_pn_diode.py")
    ]
    
    for test_file in validation_test_files:
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

def run_benchmarks(args):
    """Run performance benchmarks."""
    print("\n=== Running Performance Benchmarks ===\n")
    
    benchmark_files = [
        os.path.join("tests", "benchmarks", "test_performance_benchmarks.py")
    ]
    
    for benchmark_file in benchmark_files:
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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run QDSim tests and benchmarks.")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--validation", action="store_true", help="Run validation tests only")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmarks only")
    parser.add_argument("--verbose", action="store_true", help="Show test output in real-time")
    args = parser.parse_args()
    
    # If no specific tests are requested, run all tests
    run_all = not (args.unit or args.validation or args.benchmarks)
    
    start_time = time.time()
    
    if args.unit or run_all:
        run_unit_tests(args)
    
    if args.validation or run_all:
        run_validation_tests(args)
    
    if args.benchmarks or run_all:
        run_benchmarks(args)
    
    end_time = time.time()
    
    print(f"\nAll tests completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
