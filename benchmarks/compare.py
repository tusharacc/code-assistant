#!/usr/bin/env python3
"""
Cross-requirement comparative report.

Discovers the most-recent result directories in benchmarks/results/,
reads their report.json files, and prints + saves a combined comparison.

Usage:
    python benchmarks/compare.py                      # auto-discover latest results
    python benchmarks/compare.py --dirs r1 r2 r3 ...  # explicit dirs
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

console = Console()

_CLAUDE_PRICE_IN  = 3.00   # USD per million input tokens
_CLAUDE_PRICE_OUT = 15.00  # USD per million output tokens


def _fmt_time(s: float) -> str:
    if s <= 0:
        return "—"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s"


def _fmt_n(n: int) -> str:
    return f"{n:,}" if n else "—"


def _cost(data: dict) -> str:
    if data.get("runner") == "code_assistant":
        return "local"
    ti = data.get("total_tokens_in", 0)
    to = data.get("total_tokens_out", 0)
    c = ti / 1_000_000 * _CLAUDE_PRICE_IN + to / 1_000_000 * _CLAUDE_PRICE_OUT
    return f"${c:.3f}"


def _req_name(path: str) -> str:
    return Path(path).stem.replace("req_", "").replace("_", " ")


def load_results(dirs: list[Path]) -> list[dict]:
    """Load all report.json files from the given directories."""
    results = []
    for d in dirs:
        rj = d / "report.json"
        if not rj.exists():
            continue
        data = json.loads(rj.read_text())
        for runner_key in ("code_assistant", "claude_api"):
            if runner_key in data:
                results.append(data[runner_key])
    return results


def discover_dirs(base: Path, n: int = 6) -> list[Path]:
    """Return the n most-recently modified result directories."""
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return dirs[:n]


def print_comparison(results: list[dict]) -> None:
    if not results:
        console.print("[red]No results found.[/red]")
        return

    console.print()
    console.rule("[bold cyan]Cross-Requirement Comparison[/bold cyan]")
    console.print()

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    tbl.add_column("Requirement",   style="dim", min_width=16)
    tbl.add_column("Runner",        min_width=16)
    tbl.add_column("Time",          justify="right", min_width=8)
    tbl.add_column("API calls",     justify="right", min_width=8)
    tbl.add_column("Tok in",        justify="right", min_width=9)
    tbl.add_column("Tok out",       justify="right", min_width=9)
    tbl.add_column("Est. cost",     justify="right", min_width=9)
    tbl.add_column("Files",         justify="right", min_width=6)
    tbl.add_column("Lines",         justify="right", min_width=7)
    tbl.add_column("Syn.err",       justify="right", min_width=7)
    tbl.add_column("Tests ✓/✗",    justify="right", min_width=9)

    # Group by requirement
    by_req: dict[str, list[dict]] = {}
    for r in results:
        req = _req_name(r.get("requirement_file", "unknown"))
        by_req.setdefault(req, []).append(r)

    for i, (req, entries) in enumerate(sorted(by_req.items())):
        if i > 0:
            tbl.add_section()
        for j, r in enumerate(sorted(entries, key=lambda x: x.get("runner", ""))):
            runner_label = r.get("runner", "?").replace("_", " ")
            runner_color = "cyan" if "assistant" in runner_label else "magenta"
            tests = f"{r.get('tests_passed', 0)}/{r.get('tests_failed', 0)}"
            tbl.add_row(
                req if j == 0 else "",
                f"[{runner_color}]{runner_label}[/{runner_color}]",
                _fmt_time(r.get("elapsed_total", 0)),
                str(r.get("total_api_calls", 0)),
                _fmt_n(r.get("total_tokens_in", 0)),
                _fmt_n(r.get("total_tokens_out", 0)),
                _cost(r),
                str(len(r.get("files_written", []))),
                _fmt_n(r.get("total_lines", 0)),
                str(r.get("syntax_errors", 0)),
                tests,
            )

    console.print(tbl)

    # ── Summary: winner counts ─────────────────────────────────────
    metrics = {
        "Faster":            lambda r: r.get("elapsed_total", float("inf")),
        "Fewer tokens":      lambda r: r.get("total_tokens_in", 0) + r.get("total_tokens_out", 0),
        "More files":        lambda r: -len(r.get("files_written", [])),
        "More lines":        lambda r: -r.get("total_lines", 0),
        "Fewer syntax errs": lambda r: r.get("syntax_errors", float("inf")),
        "More tests pass":   lambda r: -r.get("tests_passed", 0),
    }

    ca_wins: dict[str, int] = {m: 0 for m in metrics}
    claude_wins: dict[str, int] = {m: 0 for m in metrics}
    compared = 0

    for req, entries in by_req.items():
        ca = next((e for e in entries if e.get("runner") == "code_assistant"), None)
        cl = next((e for e in entries if e.get("runner") == "claude_api"), None)
        if not ca or not cl:
            continue
        compared += 1
        for name, fn in metrics.items():
            if fn(ca) < fn(cl):
                ca_wins[name] += 1
            elif fn(cl) < fn(ca):
                claude_wins[name] += 1

    if compared:
        console.print()
        lines = [f"[bold]Summary — winner by metric[/bold] (out of {compared} requirements)\n"]
        for m in metrics:
            cw = ca_wins[m]
            klw = claude_wins[m]
            if cw > klw:
                winner = f"[cyan]Code Assistant  {cw}/{compared}[/cyan]"
            elif klw > cw:
                winner = f"[magenta]Claude API      {klw}/{compared}[/magenta]"
            else:
                winner = f"[dim]Tie             {cw}/{compared}[/dim]"
            lines.append(f"  {m:<22} {winner}")
        console.print(Panel("\n".join(lines), border_style="dim", padding=(0, 1)))
    console.print()


def save_comparison(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    (out_dir / f"comparison_{stamp}.json").write_text(
        json.dumps({"results": results}, indent=2), encoding="utf-8"
    )

    # Markdown
    md = ["# Cross-Requirement Comparison\n",
          f"Generated: {datetime.now().isoformat()}\n",
          "| Requirement | Runner | Time | Tok-in | Tok-out | Cost | Files | Lines | Syn.Err | Tests ✓/✗ |",
          "|------------|--------|------|--------|---------|------|-------|-------|---------|-----------|"]
    for r in sorted(results, key=lambda x: (x.get("requirement_file", ""), x.get("runner", ""))):
        md.append(
            f"| {_req_name(r.get('requirement_file','?'))} "
            f"| {r.get('runner','?')} "
            f"| {_fmt_time(r.get('elapsed_total',0))} "
            f"| {_fmt_n(r.get('total_tokens_in',0))} "
            f"| {_fmt_n(r.get('total_tokens_out',0))} "
            f"| {_cost(r)} "
            f"| {len(r.get('files_written',[]))} "
            f"| {r.get('total_lines',0)} "
            f"| {r.get('syntax_errors',0)} "
            f"| {r.get('tests_passed',0)}/{r.get('tests_failed',0)} |"
        )

    md_path = out_dir / f"comparison_{stamp}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    console.print(f"[dim]Saved:[/dim] [cyan]{md_path}[/cyan]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-requirement benchmark comparison")
    ap.add_argument("--dirs", nargs="+", type=Path,
                    help="Explicit result directories (default: auto-discover latest 6)")
    ap.add_argument("--base", type=Path,
                    default=Path(__file__).parent / "results",
                    help="Base results directory for auto-discovery")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs] if args.dirs else discover_dirs(args.base, n=6)

    if not dirs:
        console.print(f"[red]No result directories found in {args.base}[/red]")
        return

    console.print(f"[dim]Loading results from {len(dirs)} director{'y' if len(dirs)==1 else 'ies'}...[/dim]")
    for d in dirs:
        console.print(f"  [dim]{d}[/dim]")

    results = load_results(dirs)
    if not results:
        console.print("[red]No report.json files found.[/red]")
        return

    print_comparison(results)
    save_comparison(results, args.base)


if __name__ == "__main__":
    main()
