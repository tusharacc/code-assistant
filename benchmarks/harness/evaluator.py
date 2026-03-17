"""
Post-generation evaluator.

Counts files, checks syntax with py_compile, and runs pytest if test files exist.
"""
from __future__ import annotations

import py_compile
import re
import subprocess
from pathlib import Path

from .metrics import BenchmarkResult


def evaluate(result: BenchmarkResult) -> BenchmarkResult:
    """Run syntax checks and tests on the generated output; mutate result in place."""
    output_dir = Path(result.output_dir)

    # Syntax check every .py file
    syntax_errors = 0
    for py_file in sorted(output_dir.rglob("*.py")):
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError:
            syntax_errors += 1

    result.syntax_errors = syntax_errors

    # Run pytest if test files exist
    test_files = list(output_dir.rglob("test_*.py")) + list(output_dir.rglob("*_test.py"))
    if test_files:
        proc = subprocess.run(
            ["python", "-m", "pytest", "--tb=short", "-q", str(output_dir)],
            capture_output=True,
            text=True,
            cwd=output_dir,
        )
        output = proc.stdout + proc.stderr
        result.test_output = output[:4000]  # cap for report
        passed, failed = _parse_pytest_summary(output)
        result.tests_passed = passed
        result.tests_failed = failed
    else:
        result.test_output = "(no test files found)"

    return result


def _parse_pytest_summary(output: str) -> tuple[int, int]:
    """Parse pytest -q summary line: '5 passed, 2 failed in 0.12s'"""
    passed = failed = 0
    m = re.search(r"(\d+)\s+passed", output)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", output)
    if m:
        failed = int(m.group(1))
    return passed, failed
