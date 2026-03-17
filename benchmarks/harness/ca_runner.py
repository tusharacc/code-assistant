"""
Code-assistant pipeline runner for the benchmark harness.

Wraps the existing Pipeline, changes cwd to the output directory so
write_file tool calls land in the right place, then reads pipeline.metrics
to build a BenchmarkResult.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .metrics import BenchmarkResult, PhaseMetrics


def run(req_file: Path, output_dir: Path) -> BenchmarkResult:
    """Run the CA pipeline on `req_file`, writing outputs to `output_dir`."""
    from datetime import datetime

    # Must import after sys.path is set up (called from __main__)
    from code_assistant.agents.base import Message
    from code_assistant.agents.pipeline import Pipeline
    from code_assistant.config import config

    req_file = req_file.resolve()   # absolute — survives cwd change
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()

    try:
        # Change cwd so relative write_file paths land in output_dir
        os.chdir(output_dir)

        requirement = req_file.read_text(encoding="utf-8")
        history = [Message(role="user", content=requirement)]

        pipeline = Pipeline(initial_history=history)

        t0 = time.perf_counter()
        pipeline.run(requirement)
        elapsed = time.perf_counter() - t0

    finally:
        os.chdir(original_cwd)

    # Build PhaseMetrics from pipeline.metrics dict
    phases: list[PhaseMetrics] = []
    for name, m in pipeline.metrics.items():
        if name == "elapsed_total":
            continue
        phases.append(PhaseMetrics(
            name=name,
            tokens_in=m["tokens_in"],
            tokens_out=m["tokens_out"],
            api_calls=m["api_calls"],
            elapsed=m["elapsed"],
        ))

    files = sorted(
        str(p.relative_to(output_dir))
        for p in output_dir.rglob("*")
        if p.is_file()
    )
    total_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    total_lines = sum(
        len(p.read_text(errors="replace").splitlines())
        for p in output_dir.rglob("*.py")
        if p.is_file()
    )

    return BenchmarkResult(
        runner="code_assistant",
        model=f"arch={config.architect_model} impl={config.implementer_model}",
        requirement_file=str(req_file),
        output_dir=str(output_dir),
        phases=phases,
        files_written=files,
        total_bytes=total_bytes,
        total_lines=total_lines,
        elapsed_total=elapsed,
        timestamp=datetime.now().isoformat(),
    )
