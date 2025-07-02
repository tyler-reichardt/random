#!/usr/bin/env python
"""
Test runner script for TPC ML Pipeline tests.
This script sets up the environment and runs all tests with proper configuration.
"""

import sys
import os
import pytest
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run TPC ML Pipeline tests")
    
    parser.add_argument(
        "--module",
        choices=["datapreprocessing", "featureengineering", "training", "all"],
        default="all",
        help="Which test module to run"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--failfast",
        "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--markers",
        "-m",
        help="Run tests matching given mark expression"
    )
    
    parser.add_argument(
        "--keyword",
        "-k",
        help="Run tests matching given keyword expression"
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Add test file(s)
    if args.module == "all":
        pytest_args.append(str(project_root / "tests"))
    else:
        pytest_args.append(str(project_root / "tests" / f"test_{args.module}.py"))
    
    # Add coverage if requested
    if args.coverage:
        pytest_args.extend([
            "--cov=notebooks",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add verbose flag
    if args.verbose:
        pytest_args.append("-v")
    
    # Add failfast flag
    if args.failfast:
        pytest_args.append("-x")
    
    # Add markers
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    
    # Add keyword
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])
    
    # Add some default options
    pytest_args.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Treat unregistered markers as errors
        "--disable-warnings",  # Disable warnings for cleaner output
    ])
    
    # Print configuration
    print("=" * 70)
    print("TPC ML Pipeline Test Runner")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Test module: {args.module}")
    print(f"Coverage: {'Yes' if args.coverage else 'No'}")
    print(f"Pytest args: {' '.join(pytest_args)}")
    print("=" * 70)
    print()
    
    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    # Print summary
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 70)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())