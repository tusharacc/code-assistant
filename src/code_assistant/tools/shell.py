"""
Shell execution tool — always asks for user confirmation before running.
"""
import subprocess
import shlex
from typing import Any

from ..logger import get_logger
from ..ui.console import console, confirm, print_error

log = get_logger(__name__)


def run_shell(command: str, working_dir: str = ".") -> str:
    """
    Execute a shell command. Always prompts for confirmation first.
    Returns stdout + stderr combined.
    """
    log.info("run_shell | command=%r cwd=%s", command, working_dir)

    console.print(f"\n[bold yellow]Shell command:[/bold yellow] [white]{command}[/white]")
    if working_dir and working_dir != ".":
        console.print(f"[dim]  cwd: {working_dir}[/dim]")

    if not confirm("Run this command?"):
        log.info("run_shell | user cancelled: %r", command)
        return "Cancelled — command not executed."

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir or None,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")

        combined = "\n".join(output_parts).strip()
        exit_info = f"\n[exit code: {result.returncode}]"

        if result.returncode != 0:
            log.error(
                "run_shell | exit %d | command=%r | stderr=%s",
                result.returncode, command,
                result.stderr[:500] if result.stderr else "",
            )
            console.print(f"[bold red]✗ exit {result.returncode}[/bold red]")
        else:
            log.info("run_shell | exit 0 | command=%r", command)
            console.print(f"[bold green]✓ exit 0[/bold green]")

        log.debug("run_shell | output | %s", (combined[:1000] if combined else "(empty)"))
        return (combined + exit_info) if combined else exit_info

    except subprocess.TimeoutExpired:
        log.error("run_shell | timeout after 120s | command=%r", command)
        return "Error: command timed out after 120 seconds."
    except Exception as e:
        log.error("run_shell | unexpected error | command=%r | %s", command, e, exc_info=True)
        return f"Error running command: {e}"


TOOL_HANDLERS: dict[str, Any] = {
    "run_shell": run_shell,
}
