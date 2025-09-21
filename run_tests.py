#!/usr/bin/env python3
"""
Convenient test runner for CausalAlign package.
"""

import argparse
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f" {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"{description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run CausalAlign tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "cli", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--html", action="store_true", help="Generate HTML coverage report"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--markers",
        "-m",
        help="Run tests with specific markers (e.g., 'unit', 'not slow')",
    )

    args = parser.parse_args()

    # Build pytest command
    cmd = ["pytest"]

    # Add test directory based on type
    if args.type == "unit":
        cmd.append("tests/unit/")
    elif args.type == "integration":
        cmd.append("tests/integration/")
    elif args.type == "cli":
        cmd.append("tests/cli/")
    else:  # all
        cmd.append("tests/")

    # Add coverage options
    if args.coverage or args.html:
        cmd.extend(["--cov=src/causalign", "--cov-report=term-missing"])

        if args.html:
            cmd.append("--cov-report=html:htmlcov")

    # Add verbose option
    if args.verbose:
        cmd.append("-v")

    # Add markers
    if args.markers:
        cmd.extend(["-m", args.markers])

    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")

    if args.html and success:
        print("\n HTML coverage report generated in: htmlcov/index.html")
        print("Open with: open htmlcov/index.html")

    # Summary
    print("\n" + "=" * 50)
    if success:
        print(" All tests completed successfully!")
    else:
        print(" Some tests failed. Check output above.")
    print("=" * 50)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
