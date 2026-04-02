"""
File operation tools: read, write, edit, list, glob.
All edit operations show a diff and require user confirmation.
"""
import itertools
import os
from pathlib import Path
from typing import Any

from ..logger import get_logger
from ..ui.console import console, print_error, confirm, print_success
from ..ui.diff import print_diff

log = get_logger(__name__)


def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    log.info("read_file | path=%s", path)
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            log.error("read_file | not found: %s", path)
            return f"Error: file not found: {path}"
        if not p.is_file():
            log.error("read_file | not a file: %s", path)
            return f"Error: not a file: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        log.debug("read_file | %s — %d lines, %d chars", path, len(lines), len(content))
        # Return with line numbers so the model can reference specific lines
        numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return f"```{p.suffix.lstrip('.') or 'text'}\n{numbered}\n```"
    except PermissionError:
        log.error("read_file | permission denied: %s", path)
        return f"Error: permission denied: {path}"
    except Exception as e:
        log.error("read_file | unexpected error: %s", e, exc_info=True)
        return f"Error reading {path}: {e}"


def _regression_guard(path: str, original: str, new_content: str) -> str | None:
    """Return an error string if the new content is a suspicious regression.

    Triggers when BOTH of the following are true:
      - The file shrinks by more than SHRINK_THRESHOLD (60%)
      - The original file was at least MIN_ORIGINAL_LINES lines long

    This catches the most common pipeline failure mode: a write_file call that
    replaces a 200-line working file with a 1-line stub or blank import.

    Returns None if the write looks safe.
    Returns an Error string (starts with "Error:") if the guard fires.
    Pass force_overwrite=True to bypass the guard for legitimate large rewrites
    (e.g. generated files, complete refactors).
    """
    SHRINK_THRESHOLD   = 0.60   # 60% character reduction triggers the guard
    MIN_ORIGINAL_LINES = 10     # only guard files that had meaningful content

    orig_chars = len(original)
    new_chars  = len(new_content)

    if orig_chars == 0:
        return None  # new file — nothing to protect

    orig_lines = original.count("\n") + 1
    if orig_lines < MIN_ORIGINAL_LINES:
        return None  # original was a stub itself — no protection needed

    shrink = (orig_chars - new_chars) / orig_chars
    if shrink > SHRINK_THRESHOLD:
        pct = int(shrink * 100)
        log.error(
            "write_file | REGRESSION GUARD fired: %s shrinks %d→%d chars (%d%%) "
            "— use force_overwrite=True to bypass",
            path, orig_chars, new_chars, pct,
        )
        return (
            f"Error: regression guard blocked write to {path}.\n"
            f"  Original: {orig_chars:,} chars ({orig_lines} lines)\n"
            f"  New content: {new_chars:,} chars ({new_content.count(chr(10)) + 1} lines)\n"
            f"  Reduction: {pct}% — exceeds the 60% safety threshold.\n\n"
            f"This usually means write_file is being called with an incomplete or stub "
            f"implementation that would destroy working code.\n\n"
            f"If this is intentional (e.g. complete refactor, generated file), "
            f"call write_file again with force_overwrite=true to bypass the guard."
        )
    return None


def write_file(path: str, content: str, force_overwrite: bool = False) -> str:
    """Create or overwrite a file. Shows diff if file exists, asks for confirmation.

    The regression guard blocks any write that would shrink an existing file by
    more than 60%, preventing pipeline phases from replacing working code with
    stubs. Pass force_overwrite=True to bypass for legitimate large rewrites.
    """
    log.info(
        "write_file | path=%s chars=%d force_overwrite=%s",
        path, len(content), force_overwrite,
    )
    try:
        p = Path(path).expanduser().resolve()
        original = ""
        verb = "create"

        if p.exists():
            original = p.read_text(encoding="utf-8", errors="replace")
            verb = "overwrite"

            # ── Regression guard ─────────────────────────────────────────────
            if not force_overwrite:
                guard_error = _regression_guard(path, original, content)
                if guard_error:
                    from ..ui.console import print_error as _pe
                    _pe(guard_error.replace("Error: ", ""))
                    return guard_error

            changed = print_diff(str(p), original, content)
            if not changed:
                log.debug("write_file | no changes to %s", path)
                return "No changes — file already has that content."
        else:
            console.print(f"\n[bold]Create new file:[/bold] [cyan]{path}[/cyan]")
            # Show first 20 lines preview — split once and reuse
            lines = content.splitlines()
            preview = "\n".join(lines[:20])
            if len(lines) > 20:
                preview += f"\n[dim]… ({len(lines)} lines total)[/dim]"
            console.print(preview)

        if not confirm(f"OK to {verb} {path}?"):
            log.info("write_file | user cancelled: %s", path)
            return "Cancelled — file not written."

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        log.info("write_file | success: %s (%d chars)", path, len(content))
        print_success(f"Written: {path}")
        return f"Success: wrote {len(content)} chars to {path}"

    except PermissionError:
        log.error("write_file | permission denied: %s", path)
        return f"Error: permission denied: {path}"
    except Exception as e:
        log.error("write_file | unexpected error: %s", e, exc_info=True)
        return f"Error writing {path}: {e}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """
    Replace the first occurrence of old_string with new_string in a file.
    Shows a diff and asks for confirmation before applying.
    """
    log.info(
        "edit_file | path=%s old_chars=%d new_chars=%d",
        path, len(old_string), len(new_string),
    )
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            log.error("edit_file | not found: %s", path)
            return f"Error: file not found: {path}"

        original = p.read_text(encoding="utf-8", errors="replace")

        if old_string not in original:
            log.error("edit_file | old_string not found in %s", path)
            return (
                f"Error: the exact string was not found in {path}. "
                "Make sure old_string matches the file contents exactly, including whitespace."
            )

        updated = original.replace(old_string, new_string, 1)

        changed = print_diff(path, original, updated)
        if not changed:
            log.debug("edit_file | no effective changes in %s", path)
            return "No changes after replacement."

        if not confirm(f"Apply edit to {path}?"):
            log.info("edit_file | user cancelled: %s", path)
            return "Cancelled — file not modified."

        p.write_text(updated, encoding="utf-8")
        log.info("edit_file | success: %s", path)
        print_success(f"Edited: {path}")
        return f"Success: applied edit to {path}"

    except PermissionError:
        log.error("edit_file | permission denied: %s", path)
        return f"Error: permission denied: {path}"
    except Exception as e:
        log.error("edit_file | unexpected error: %s", e, exc_info=True)
        return f"Error editing {path}: {e}"


def list_dir(path: str = ".", max_depth: int = 3) -> str:
    """List directory contents as an indented tree (max_depth levels deep)."""
    log.debug("list_dir | path=%s max_depth=%d", path, max_depth)
    try:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            log.error("list_dir | not found: %s", path)
            return f"Error: path not found: {path}"
        if not root.is_dir():
            log.error("list_dir | not a directory: %s", path)
            return f"Error: not a directory: {path}"

        lines: list[str] = [str(root)]
        _walk_tree(root, lines, prefix="", depth=0, max_depth=max_depth)
        log.debug("list_dir | %s returned %d lines", path, len(lines))
        return "\n".join(lines)
    except Exception as e:
        log.error("list_dir | error: %s", e, exc_info=True)
        return f"Error listing {path}: {e}"


def _walk_tree(
    directory: Path,
    lines: list[str],
    prefix: str,
    depth: int,
    max_depth: int,
    skip_dirs: frozenset = frozenset({".git", "__pycache__", ".chroma", "node_modules", ".venv", "venv", "dist", "build"}),
) -> None:
    if depth >= max_depth:
        lines.append(f"{prefix}    [… truncated at depth {max_depth}]")
        return

    try:
        entries = sorted(directory.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir() and entry.name not in skip_dirs:
            extension = "    " if is_last else "│   "
            _walk_tree(entry, lines, prefix + extension, depth + 1, max_depth)


def compute_file_sha256(path: str) -> str:
    """Compute and return the SHA-256 hash of a file on disk.

    Agents call this after writing a file to obtain the hash they will include
    in their ## Handoff section.  The verifier independently recomputes the
    hash and compares — a mismatch means the file was not written correctly.
    """
    import hashlib
    log.debug("compute_file_sha256 | path=%s", path)
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: file not found: {path}"
        if not p.is_file():
            return f"Error: not a file: {path}"
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        log.debug("compute_file_sha256 | %s → %s", path, digest[:12])
        return f"sha256:{digest}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error computing sha256 for {path}: {e}"


def glob_files(pattern: str, root: str = ".") -> str:
    """Find files matching a glob pattern under root."""
    log.debug("glob_files | pattern=%s root=%s", pattern, root)
    try:
        base = Path(root).expanduser().resolve()
        matches = list(itertools.islice(base.glob(pattern), 100))
        log.debug("glob_files | matched %d files for pattern %s", len(matches), pattern)
        if not matches:
            return f"No files matched: {pattern}"
        return "\n".join(str(m.relative_to(base)) for m in matches)
    except Exception as e:
        log.error("glob_files | error: %s", e, exc_info=True)
        return f"Error globbing {pattern}: {e}"


# Dispatch table — used by the tool executor
TOOL_HANDLERS: dict[str, Any] = {
    "read_file":           read_file,
    "write_file":          write_file,
    "edit_file":           edit_file,
    "list_dir":            list_dir,
    "glob_files":          glob_files,
    "compute_file_sha256": compute_file_sha256,
}
