"""
Benchmark harness CLI.

Usage:
  python -m benchmarks.harness --req benchmarks/req_01_calculator.txt
  python -m benchmarks.harness --req benchmarks/req_01_calculator.txt --skip-ca
  python -m benchmarks.harness --req benchmarks/req_01_calculator.txt --skip-claude
  python -m benchmarks.harness --req benchmarks/req_01_calculator.txt --out /tmp/bench --model claude-sonnet-4-6
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    req: Path = typer.Option(..., "--req", "-r", help="Requirement file to benchmark",
                             exists=True, file_okay=True, dir_okay=False, readable=True),
    out: Path = typer.Option(None, "--out", "-o", help="Base output directory (default: benchmarks/results/)"),
    model: str = typer.Option("claude-sonnet-4-6", "--model", "-m", help="Claude model to use for API runner"),
    skip_ca: bool = typer.Option(False, "--skip-ca", help="Skip the code-assistant pipeline run"),
    skip_claude: bool = typer.Option(False, "--skip-claude", help="Skip the Claude API run"),
) -> None:
    """Compare code-assistant pipeline vs Claude API on the same requirement."""

    if skip_ca and skip_claude:
        console.print("[red]Error:[/red] --skip-ca and --skip-claude cannot both be set.")
        raise typer.Exit(1)

    # Check API key early
    if not skip_claude and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]Error:[/red] ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it or use --skip-claude to run only the code-assistant."
        )
        raise typer.Exit(1)

    # Resolve output directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out or (Path(__file__).parent.parent / "results")
    results_dir = base / stamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Copy requirement into results dir for reference
    import shutil
    shutil.copy(req, results_dir / "requirement.txt")

    ca_result = None
    claude_result = None

    # ── Code-assistant run ─────────────────────────────────────────────
    if not skip_ca:
        console.print(f"\n[bold cyan]▶ Running Code Assistant pipeline...[/bold cyan]")
        console.print(f"  Output → {results_dir / 'ca'}\n")
        try:
            from benchmarks.harness import ca_runner
            ca_result = ca_runner.run(req_file=req, output_dir=results_dir / "ca")
            console.print("\n[green]✓ Code Assistant run complete.[/green]")
        except Exception as e:
            console.print(f"[red]✗ Code Assistant run failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    # ── Claude API run ─────────────────────────────────────────────────
    if not skip_claude:
        console.print(f"\n[bold magenta]▶ Running Claude API ({model})...[/bold magenta]")
        console.print(f"  Output → {results_dir / 'claude'}\n")
        try:
            from benchmarks.harness.claude_runner import ClaudeRunner
            requirement = req.read_text(encoding="utf-8")
            runner = ClaudeRunner(
                model=model,
                output_dir=results_dir / "claude",
                requirement=requirement,
                req_file=req,
            )
            claude_result = runner.run()
            console.print("\n[green]✓ Claude API run complete.[/green]")
        except Exception as e:
            console.print(f"[red]✗ Claude API run failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    # ── Evaluation ─────────────────────────────────────────────────────
    from benchmarks.harness import evaluator

    if ca_result:
        console.print("\n[dim]Evaluating code-assistant output...[/dim]")
        ca_result = evaluator.evaluate(ca_result)

    if claude_result:
        console.print("[dim]Evaluating Claude API output...[/dim]")
        claude_result = evaluator.evaluate(claude_result)

    # ── Report ─────────────────────────────────────────────────────────
    from benchmarks.harness import report

    report.print_report(ca_result, claude_result)
    report.save_report(results_dir, ca_result, claude_result)

    console.print(f"[dim]Results saved to:[/dim] [cyan]{results_dir}[/cyan]")


if __name__ == "__main__":
    app()
