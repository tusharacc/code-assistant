"""
Report generator — Rich terminal table + JSON + Markdown file output.
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from .metrics import BenchmarkResult

console = Console()

# Sonnet 4.6 pricing (USD per million tokens, as of 2026)
_CLAUDE_PRICE_IN  = 3.00   # per M input tokens
_CLAUDE_PRICE_OUT = 15.00  # per M output tokens


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _fmt_tokens(n: int) -> str:
    if n == 0:
        return "—"
    return f"{n:,}"


def _estimate_cost(result: BenchmarkResult) -> str:
    if result.runner == "code_assistant":
        return "local (free)"
    cost = (
        result.total_tokens_in  / 1_000_000 * _CLAUDE_PRICE_IN
        + result.total_tokens_out / 1_000_000 * _CLAUDE_PRICE_OUT
    )
    return f"~${cost:.4f}"


def print_report(ca: BenchmarkResult | None, claude: BenchmarkResult | None) -> None:
    """Print side-by-side comparison to the terminal."""
    results = [r for r in (ca, claude) if r is not None]

    console.print()
    console.rule("[bold cyan]Benchmark Comparison Report[/bold cyan]")
    console.print()

    # ── Summary table ──────────────────────────────────────────────────
    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    tbl.add_column("Metric", style="dim", min_width=24)
    for r in results:
        tbl.add_column(r.runner.replace("_", " ").title(), justify="right", min_width=20)

    def row(label: str, *values: str) -> None:
        tbl.add_row(label, *values)

    row("Runner / model",     *[r.runner for r in results])
    row("Model",              *[r.model for r in results])
    tbl.add_section()
    row("Total time",         *[_fmt_time(r.elapsed_total) for r in results])
    row("API calls",          *[str(r.total_api_calls)     for r in results])
    row("Tokens in",          *[_fmt_tokens(r.total_tokens_in)  for r in results])
    row("Tokens out",         *[_fmt_tokens(r.total_tokens_out) for r in results])
    row("Tokens total",       *[_fmt_tokens(r.total_tokens)     for r in results])
    row("Est. cost",          *[_estimate_cost(r)               for r in results])
    tbl.add_section()
    row("Files written",      *[str(len(r.files_written))  for r in results])
    row("Total lines (py)",   *[str(r.total_lines)          for r in results])
    row("Total bytes",        *[f"{r.total_bytes:,}"        for r in results])
    tbl.add_section()
    row("Syntax errors",      *[str(r.syntax_errors)  for r in results])
    row("Tests passed",       *[str(r.tests_passed)   for r in results])
    row("Tests failed",       *[str(r.tests_failed)   for r in results])

    console.print(tbl)

    # ── Per-phase breakdown ────────────────────────────────────────────
    # Collect all unique phase names across both runners
    all_phases: list[str] = []
    seen: set[str] = set()
    for r in results:
        for p in r.phases:
            if p.name not in seen:
                all_phases.append(p.name)
                seen.add(p.name)

    if all_phases:
        console.print()
        console.print("[bold]Phase breakdown[/bold]")
        phase_tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        phase_tbl.add_column("Phase", style="dim", min_width=22)
        for r in results:
            phase_tbl.add_column(f"{r.runner[:14]} tok_in", justify="right")
            phase_tbl.add_column("tok_out", justify="right")
            phase_tbl.add_column("calls", justify="right")
            phase_tbl.add_column("time", justify="right")

        for phase_name in all_phases:
            cells = [phase_name]
            for r in results:
                pm = next((p for p in r.phases if p.name == phase_name), None)
                if pm:
                    cells += [
                        _fmt_tokens(pm.tokens_in),
                        _fmt_tokens(pm.tokens_out),
                        str(pm.api_calls),
                        _fmt_time(pm.elapsed),
                    ]
                else:
                    cells += ["—", "—", "—", "—"]
            phase_tbl.add_row(*cells)

        console.print(phase_tbl)
    console.print()


def save_report(
    results_dir: Path,
    ca: BenchmarkResult | None,
    claude: BenchmarkResult | None,
) -> None:
    """Write report.json and report.md to results_dir."""
    data = {}
    if ca:
        data["code_assistant"] = ca.to_dict()
    if claude:
        data["claude_api"] = claude.to_dict()

    (results_dir / "report.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )

    # Markdown table — works for single-runner dirs too
    md_lines = ["# Benchmark Report\n"]
    results_list = [r for r in (ca, claude) if r is not None]
    if results_list:
        req_file = results_list[0].requirement_file
        md_lines += [f"**Requirement:** {req_file}", ""]

    if ca and claude:
        # Side-by-side comparison (both runners present)
        md_lines += [
            "| Metric | Code Assistant | Claude API |",
            "|--------|---------------|-----------|",
            f"| Model | `{ca.model}` | `{claude.model}` |",
            f"| Total time | {_fmt_time(ca.elapsed_total)} | {_fmt_time(claude.elapsed_total)} |",
            f"| API calls | {ca.total_api_calls} | {claude.total_api_calls} |",
            f"| Tokens in | {_fmt_tokens(ca.total_tokens_in)} | {_fmt_tokens(claude.total_tokens_in)} |",
            f"| Tokens out | {_fmt_tokens(ca.total_tokens_out)} | {_fmt_tokens(claude.total_tokens_out)} |",
            f"| Est. cost | {_estimate_cost(ca)} | {_estimate_cost(claude)} |",
            f"| Files written | {len(ca.files_written)} | {len(claude.files_written)} |",
            f"| Total lines | {ca.total_lines} | {claude.total_lines} |",
            f"| Syntax errors | {ca.syntax_errors} | {claude.syntax_errors} |",
            f"| Tests passed | {ca.tests_passed} | {claude.tests_passed} |",
            f"| Tests failed | {ca.tests_failed} | {claude.tests_failed} |",
        ]
    elif results_list:
        # Single-runner — write a single-column table
        r = results_list[0]
        label = "Code Assistant" if r.runner == "code_assistant" else "Claude API"
        md_lines += [
            f"| Metric | {label} |",
            "|--------|---------|",
            f"| Model | `{r.model}` |",
            f"| Total time | {_fmt_time(r.elapsed_total)} |",
            f"| API calls | {r.total_api_calls} |",
            f"| Tokens in | {_fmt_tokens(r.total_tokens_in)} |",
            f"| Tokens out | {_fmt_tokens(r.total_tokens_out)} |",
            f"| Est. cost | {_estimate_cost(r)} |",
            f"| Files written | {len(r.files_written)} |",
            f"| Total lines | {r.total_lines} |",
            f"| Syntax errors | {r.syntax_errors} |",
            f"| Tests passed | {r.tests_passed} |",
            f"| Tests failed | {r.tests_failed} |",
        ]
        if r.phases:
            md_lines += ["", "### Phase breakdown", "", "| Phase | Tok-in | Tok-out | Calls | Time |",
                         "|-------|--------|---------|-------|------|"]
            for p in r.phases:
                md_lines.append(
                    f"| {p.name} | {_fmt_tokens(p.tokens_in)} | {_fmt_tokens(p.tokens_out)} | {p.api_calls} | {_fmt_time(p.elapsed)} |"
                )

    (results_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
