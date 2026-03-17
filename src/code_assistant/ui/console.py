"""
Rich-based console utilities — streaming output, panels, code blocks.
"""
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme
from rich.markdown import Markdown
from rich.rule import Rule

# Shared console instance — import this everywhere
_theme = Theme({
    "architect":   "bold cyan",
    "implementer": "bold green",
    "tool":        "bold yellow",
    "tool.result": "dim white",
    "user":        "bold white",
    "error":       "bold red",
    "warning":     "yellow",
    "info":        "dim cyan",
    "dim":         "dim white",
    "confirm":     "bold magenta",
})

console = Console(theme=_theme, highlight=True)


def print_rule(title: str = "", style: str = "dim") -> None:
    console.print(Rule(title, style=style))


def print_agent_header(role: str) -> None:
    """Print a colored role header before streaming output."""
    icon = "🏛" if role == "architect" else "⚙"
    label = f"[{role}] {icon} {role.upper()}"
    console.print(f"\n[{role}]{label}[/{role}]")


def stream_token(token: str) -> None:
    """Print a single streamed token without newline."""
    console.print(token, end="", markup=False, highlight=False)


def print_tool_call(name: str, args: dict) -> None:
    args_str = "  " + "\n  ".join(f"{k}: {v}" for k, v in args.items())
    console.print(Panel(
        f"[bold]{name}[/bold]\n[tool.result]{args_str}[/tool.result]",
        title="[tool]⚙ tool call[/tool]",
        border_style="yellow",
        padding=(0, 1),
    ))


_ERROR_PREFIXES = ("error:", "error ", "cancelled", "timed out", "permission denied")


def print_tool_result(name: str, result: str, error: bool = False) -> None:
    # Auto-detect error results by content so callers don't have to track this
    is_error = error or result.lower().startswith(_ERROR_PREFIXES)
    style = "red" if is_error else "tool.result"
    icon = "✗" if is_error else "✓"
    console.print(
        f"[{style}]{icon} {name}:[/{style}] [dim]{result[:300]}{'…' if len(result) > 300 else ''}[/dim]"
    )


def print_code(code: str, language: str = "python") -> None:
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)


def print_error(msg: str) -> None:
    console.print(f"[error]✗ {msg}[/error]")


def print_warning(msg: str) -> None:
    console.print(f"[warning]⚠ {msg}[/warning]")


def print_info(msg: str) -> None:
    console.print(f"[info]{msg}[/info]")


def print_success(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green] {msg}")


def confirm(prompt: str, default: bool = False) -> bool:
    """Ask a y/n question, return bool.

    Returns True immediately when:
    - config.auto_approve is True (default) — approve all tool calls silently, OR
    - stdin is not a TTY (pipeline / background / benchmark runs).

    Use --no-auto-approve on the CLI (or CA_AUTO_APPROVE=false) to be prompted
    for every write_file / edit_file / run_shell call.
    """
    import sys
    from ..config import config
    if config.auto_approve:
        return True
    if not sys.stdin.isatty():
        console.print(f"[dim]{prompt} → auto-confirmed (non-interactive)[/dim]")
        return True

    hint = "[Y/n]" if default else "[y/N]"
    while True:
        console.print(f"[confirm]{prompt} {hint}:[/confirm] ", end="")
        answer = input().strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        console.print("[dim]Please enter y or n.[/dim]")


def print_markdown(text: str) -> None:
    console.print(Markdown(text))


def print_debate_separator() -> None:
    console.print(Rule("[dim]debate[/dim]", style="dim cyan"))
