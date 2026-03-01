"""
Colored unified diff display for file edits.
"""
import difflib
from rich.text import Text
from .console import console


def make_diff(path: str, original: str, updated: str) -> str:
    """Return a unified diff string."""
    return "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    ))


def print_diff(path: str, original: str, updated: str) -> bool:
    """
    Print a colored diff and return True if there are actual changes.
    """
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    ))

    if not diff_lines:
        return False

    console.print(f"\n[bold]Changes to [cyan]{path}[/cyan]:[/bold]")
    text = Text()
    for line in diff_lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("+++") or stripped.startswith("---"):
            text.append(stripped + "\n", style="bold")
        elif stripped.startswith("+"):
            text.append(stripped + "\n", style="green")
        elif stripped.startswith("-"):
            text.append(stripped + "\n", style="red")
        elif stripped.startswith("@@"):
            text.append(stripped + "\n", style="cyan")
        else:
            text.append(stripped + "\n", style="dim")

    console.print(text)
    return True
