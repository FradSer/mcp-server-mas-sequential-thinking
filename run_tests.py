#!/usr/bin/env python3
"""Test runner script for MCP Sequential Thinking Server.

This script provides a convenient way to run tests with different
configurations and options.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return its exit code."""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
    else:
        print(f"❌ {description} failed with exit code {result.returncode}")

    return result.returncode


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for MCP Sequential Thinking Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --security         # Run only security tests
  python run_tests.py --coverage         # Run tests with coverage report
  python run_tests.py --fast             # Run tests without coverage
  python run_tests.py --debug            # Run tests in debug mode
        """
    )

    # Test selection options
    parser.add_argument(
        "--unit", action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", action="store_true",
        help="Run only integration tests"
    )
    parser.add_argument(
        "--security", action="store_true",
        help="Run only security tests"
    )
    parser.add_argument(
        "--config", action="store_true",
        help="Run only configuration tests"
    )

    # Test execution options
    parser.add_argument(
        "--coverage", action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run tests without coverage (faster)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run tests in debug mode with verbose output"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Generate HTML coverage report"
    )

    # Output options
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--junit", action="store_true",
        help="Generate JUnit XML report"
    )

    args = parser.parse_args()

    # Build pytest command
    cmd = ["uv", "run", "pytest"]

    # Test selection
    if args.unit:
        cmd.append("tests/unit")
    elif args.integration:
        cmd.append("tests/integration")
    elif args.security:
        cmd.extend(["-m", "security"])
    elif args.config:
        cmd.extend(["-m", "config"])
    else:
        cmd.append("tests")

    # Coverage options
    if args.coverage or (not args.fast and not args.debug):
        cmd.extend([
            "--cov=src/mcp_server_mas_sequential_thinking",
            "--cov-report=term-missing"
        ])

        if args.html:
            cmd.append("--cov-report=html")

    # Output options
    if args.debug:
        cmd.extend(["-v", "-s", "--tb=long"])
    elif args.quiet:
        cmd.extend(["-q", "--tb=short"])
    else:
        cmd.extend(["-v", "--tb=short"])

    # Parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])

    # JUnit reporting
    if args.junit:
        cmd.append("--junit-xml=test-results.xml")

    # Run the tests
    description = "Running tests"
    if args.unit:
        description = "Running unit tests"
    elif args.integration:
        description = "Running integration tests"
    elif args.security:
        description = "Running security tests"
    elif args.config:
        description = "Running configuration tests"

    exit_code = run_command(cmd, description)

    # Additional commands based on options
    if args.coverage and exit_code == 0:
        print("\n📊 Coverage report generated!")
        if args.html:
            print("📄 HTML coverage report: htmlcov/index.html")

    if exit_code == 0:
        print("\n🎉 All tests completed successfully!")

        # Run additional quality checks if all tests pass
        print("\n🔍 Running additional quality checks...")

        # Type checking
        type_check_result = run_command(
            ["uv", "run", "mypy", "src", "--ignore-missing-imports"],
            "Type checking with mypy"
        )

        # Linting
        lint_result = run_command(
            ["uv", "run", "ruff", "check", "src", "tests"],
            "Linting with ruff"
        )

        if type_check_result == 0 and lint_result == 0:
            print("\n✨ All quality checks passed!")
        else:
            print("\n⚠️  Some quality checks failed")
            exit_code = max(exit_code, type_check_result, lint_result)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()