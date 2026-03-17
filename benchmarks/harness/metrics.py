"""Shared dataclasses for benchmark results."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseMetrics:
    name: str
    tokens_in: int
    tokens_out: int
    api_calls: int
    elapsed: float

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class BenchmarkResult:
    runner: str             # "code_assistant" | "claude_api"
    model: str
    requirement_file: str
    output_dir: str
    phases: list[PhaseMetrics]
    files_written: list[str]
    total_bytes: int
    total_lines: int
    elapsed_total: float
    timestamp: str = ""
    # Populated by evaluator after run
    syntax_errors: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    test_output: str = ""

    @property
    def total_tokens_in(self) -> int:
        return sum(p.tokens_in for p in self.phases)

    @property
    def total_tokens_out(self) -> int:
        return sum(p.tokens_out for p in self.phases)

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out

    @property
    def total_api_calls(self) -> int:
        return sum(p.api_calls for p in self.phases)

    def to_dict(self) -> dict:
        return {
            "runner": self.runner,
            "model": self.model,
            "requirement_file": self.requirement_file,
            "output_dir": self.output_dir,
            "timestamp": self.timestamp,
            "elapsed_total": self.elapsed_total,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_api_calls": self.total_api_calls,
            "files_written": self.files_written,
            "total_bytes": self.total_bytes,
            "total_lines": self.total_lines,
            "syntax_errors": self.syntax_errors,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "phases": [
                {
                    "name": p.name,
                    "tokens_in": p.tokens_in,
                    "tokens_out": p.tokens_out,
                    "api_calls": p.api_calls,
                    "elapsed": p.elapsed,
                }
                for p in self.phases
            ],
        }
