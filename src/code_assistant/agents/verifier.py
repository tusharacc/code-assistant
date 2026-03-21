"""
PhaseVerifier — file integrity and traceability checker for the pipeline.

PROTOCOL
--------
Every agent phase must hand off by producing a ## Handoff section in its
final message.  The format is:

    ## Handoff
    - /abs/path/to/file.py sha256:abc123...
    - /abs/path/to/other.md sha256:def456...

Agents obtain each sha256 by calling compute_file_sha256(path) immediately
after writing a file — they cannot compute SHA-256 themselves so they must
use the tool.

After each phase the pipeline calls verify_phase(), which:
  1. Parses the ## Handoff block from the agent's final message.
  2. For each claimed file: reads the file from disk and recomputes SHA-256.
  3. Compares agent-claimed SHA vs on-disk SHA — any mismatch is a failure.
  4. Falls back to scanning tool calls when no ## Handoff block is present
     (backward-compatible with phases that haven't been updated yet).

Additionally the pipeline writes a `.ca_pipeline/<N>_<phase>.md` artifact
after every phase.  These artifact files give a complete audit trail of what
each agent produced, independent of the chat history.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..logger import get_logger
from ..ui.console import console, print_error, print_warning

if TYPE_CHECKING:
    from .base import Message

log = get_logger(__name__)

# Matches a ## Handoff block at the end of an agent's response
_HANDOFF_BLOCK = re.compile(
    r"##\s*Handoff\s*\n(.*?)(?=\n##|\Z)", re.DOTALL | re.IGNORECASE
)
# Matches a single handoff line: "- /path/to/file sha256:hexdigest"
_HANDOFF_LINE = re.compile(
    r"-\s*(\S+)\s+sha256:([0-9a-f]{8,64})", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    path: str
    op: str                     # "write" | "edit" | "artifact"
    sha_claimed: str | None     # SHA the agent reported (from handoff or tool call content)
    sha_actual: str | None      # SHA computed by verifier from disk
    exists: bool

    @property
    def ok(self) -> bool:
        if not self.exists:
            return False
        if self.sha_claimed and self.sha_actual:
            return self.sha_claimed == self.sha_actual
        # No claimed SHA (e.g. edit_file without compute_file_sha256 call) — existence is enough
        return self.exists


@dataclass
class VerificationResult:
    phase: str
    records: list[FileRecord] = field(default_factory=list)
    handoff_parsed: bool = False    # True when ## Handoff block was found and used

    @property
    def passed(self) -> bool:
        return bool(self.records) and all(r.ok for r in self.records)

    @property
    def missing(self) -> list[str]:
        return [r.path for r in self.records if not r.exists]

    @property
    def mismatched(self) -> list[str]:
        return [r.path for r in self.records if r.exists and not r.ok]

    @property
    def file_count(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# SHA helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_str(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8", errors="replace"))


def _sha256_file(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Handoff parser
# ---------------------------------------------------------------------------

def _parse_handoff(text: str) -> list[tuple[str, str]]:
    """
    Extract (path, sha256) pairs from an agent's ## Handoff block.
    Returns [] if no block found.
    """
    m = _HANDOFF_BLOCK.search(text)
    if not m:
        return []
    claims: list[tuple[str, str]] = []
    for line_match in _HANDOFF_LINE.finditer(m.group(1)):
        path_str = line_match.group(1).strip()
        sha = line_match.group(2).strip().lower()
        claims.append((path_str, sha))
    return claims


# ---------------------------------------------------------------------------
# Core verifier
# ---------------------------------------------------------------------------

def verify_phase(phase_name: str, messages: list["Message"]) -> VerificationResult:
    """
    Verify all files an agent claimed to produce.

    Strategy (in priority order):
    1. Parse ## Handoff block from final assistant message — uses agent-reported SHAs.
    2. Fall back to scanning write_file/edit_file tool calls when no handoff found.

    For every claimed file the verifier independently reads the file from disk
    and recomputes SHA-256.  A mismatch → the write never happened or was corrupted.
    """
    result = VerificationResult(phase=phase_name)

    # ── 1. Try handoff block first ───────────────────────────────────────
    final_text = ""
    for msg in reversed(messages):
        if msg.role == "assistant" and msg.content:
            final_text = msg.content
            break

    handoff_claims = _parse_handoff(final_text)
    if handoff_claims:
        result.handoff_parsed = True
        log.info(
            "verify_phase | phase=%s source=handoff_block claims=%d",
            phase_name, len(handoff_claims),
        )
        for path_str, sha_claimed in handoff_claims:
            p = Path(path_str).expanduser().resolve()
            exists = p.exists() and p.is_file()
            sha_actual = _sha256_file(p) if exists else None
            record = FileRecord(
                path=path_str,
                op="write",
                sha_claimed=sha_claimed,
                sha_actual=sha_actual,
                exists=exists,
            )
            result.records.append(record)
            if not record.ok:
                log.warning(
                    "verify_phase | FAIL phase=%s path=%s exists=%s "
                    "claimed=%s actual=%s",
                    phase_name, path_str, exists,
                    sha_claimed[:12], str(sha_actual)[:12] if sha_actual else "none",
                )

    else:
        # ── 2. Fall back: scan tool calls ────────────────────────────────
        log.info(
            "verify_phase | phase=%s source=tool_calls (no handoff block found)",
            phase_name,
        )
        # Collect the LAST write_file / edit_file for each path.
        # The model may overwrite the same file multiple times across rounds
        # (e.g. initial write → Q&A → corrective rewrite).  Only the final
        # write represents the intended on-disk state.
        last_write: dict[str, tuple[str, dict]] = {}   # path → (op_name, args)
        for msg in messages:
            for tc in msg.tool_calls:
                fn = tc.get("function", {})
                op_name = fn.get("name", "")
                if op_name not in ("write_file", "edit_file"):
                    continue
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        continue
                path_str = args.get("path", "").strip()
                if path_str:
                    last_write[path_str] = (op_name, args)

        # Build a map: path → LAST sha256 from compute_file_sha256 tool results.
        # Used only for edit_file operations where we don't have the full content.
        # For write_file we always derive sha_claimed from the write content itself
        # so that a stale mid-stream compute_file_sha256 call never causes a false
        # positive when the model overwrote the file in a later round.
        sha_from_tool: dict[str, str] = {}
        for i, msg in enumerate(messages):
            for j, tc in enumerate(msg.tool_calls):
                fn = tc.get("function", {})
                if fn.get("name") != "compute_file_sha256":
                    continue
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        continue
                path_str = args.get("path", "").strip()
                if not path_str:
                    continue
                # Tool results follow in order: one tool-role message per call
                # in the same assistant message.  The result for call j is at
                # messages[i + 1 + j] (first call → i+1, second call → i+2, …).
                result_idx = i + 1 + j
                if result_idx < len(messages):
                    result_msg = messages[result_idx]
                    if result_msg.role == "tool" and result_msg.content.startswith("sha256:"):
                        # Overwrite with the latest SHA — last call wins
                        sha_from_tool[path_str] = result_msg.content[7:]

        for path_str, (op_name, args) in last_write.items():
                p = Path(path_str).expanduser().resolve()
                exists = p.exists() and p.is_file()
                sha_actual = _sha256_file(p) if exists else None

                if op_name == "write_file":
                    # Always derive claimed SHA from the write content — this
                    # reflects the final intended state regardless of when (or
                    # whether) compute_file_sha256 was called.
                    content = args.get("content", "")
                    sha_claimed = _sha256_str(content)
                else:
                    # edit_file: we don't have the full final content, so fall
                    # back to the compute_file_sha256 result if available.
                    sha_claimed = sha_from_tool.get(path_str)

                record = FileRecord(
                    path=path_str,
                    op="write" if op_name == "write_file" else "edit",
                    sha_claimed=sha_claimed,
                    sha_actual=sha_actual,
                    exists=exists,
                )
                result.records.append(record)
                if not record.ok:
                    log.warning(
                        "verify_phase | FAIL phase=%s path=%s exists=%s "
                        "claimed=%s actual=%s",
                        phase_name, path_str, exists,
                        str(sha_claimed)[:12] if sha_claimed else "none",
                        str(sha_actual)[:12] if sha_actual else "none",
                    )

    log.info(
        "verify_phase | phase=%s files=%d passed=%s missing=%s mismatched=%s "
        "handoff=%s",
        phase_name, result.file_count, result.passed,
        result.missing, result.mismatched, result.handoff_parsed,
    )
    return result


# ---------------------------------------------------------------------------
# Artifact writing — pipeline side
# ---------------------------------------------------------------------------

def write_pipeline_artifact(
    artifacts_dir: Path,
    filename: str,
    content: str,
) -> tuple[Path, str]:
    """
    Write a pipeline artifact and return (path, sha256).

    Called by the pipeline after each phase to persist the agent's text output
    as a file in .ca_pipeline/.  The returned SHA is used by verify_artifact()
    to confirm the write succeeded.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / filename
    sha = _sha256_str(content)
    try:
        path.write_text(content, encoding="utf-8")
        log.info(
            "pipeline_artifact | written path=%s chars=%d sha=%s",
            path, len(content), sha[:12],
        )
    except Exception as e:
        log.error("pipeline_artifact | write failed path=%s error=%s", path, e)
    return path, sha


def verify_artifact(
    phase_name: str,
    path: Path,
    sha_expected: str,
    label: str = "",
) -> bool:
    """
    Verify a pipeline-written artifact exists on disk with the expected SHA.

    Returns True when the file exists and SHA matches.
    """
    if not path.exists():
        log.error(
            "artifact_verify | MISSING phase=%s label=%s path=%s",
            phase_name, label, path,
        )
        return False
    sha_actual = _sha256_file(path)
    ok = sha_actual == sha_expected
    if ok:
        log.info(
            "artifact_verify | OK phase=%s label=%s sha=%s",
            phase_name, label, sha_actual[:12],
        )
    else:
        log.error(
            "artifact_verify | SHA MISMATCH phase=%s label=%s "
            "expected=%s actual=%s",
            phase_name, label, sha_expected[:12], str(sha_actual)[:12],
        )
    return ok


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_verification(result: VerificationResult) -> None:
    """Print a compact per-file verification table to the console."""
    src = "handoff" if result.handoff_parsed else "tool-calls"
    status = "[bold green]PASS[/bold green]" if result.passed else "[bold red]FAIL[/bold red]"
    console.print(
        f"\n[dim]── verification · {result.phase} · source={src}[/dim] · {status}"
    )

    if not result.records:
        # Zero writes is always a hard failure — agent did nothing
        console.print(
            "  [bold red]✗ FAIL — agent wrote zero files.[/bold red]\n"
            "  [red]No write_file or edit_file calls found in phase history.\n"
            "  Review and test phases cannot proceed without code on disk.[/red]"
        )
        console.print()
        return

    for r in result.records:
        if r.ok:
            sha_disp = f"sha256:{r.sha_actual[:12]}…" if r.sha_actual else "exists"
            match_tag = "[green]✓ match[/green]" if r.sha_claimed else "[green]✓ exists[/green]"
            console.print(
                f"  {match_tag}  {r.path}  [dim]{sha_disp}[/dim]"
            )
        elif not r.exists:
            console.print(f"  [bold red]✗ MISSING[/bold red]  {r.path}")
        elif r.sha_claimed and r.sha_actual and r.sha_claimed != r.sha_actual:
            console.print(
                f"  [bold red]✗ SHA MISMATCH[/bold red]  {r.path}\n"
                f"      agent claimed: {r.sha_claimed[:16]}…\n"
                f"      on disk:       {r.sha_actual[:16]}…"
            )
        else:
            console.print(f"  [bold red]✗ EDIT NOT APPLIED[/bold red]  {r.path}")
    console.print()


def print_artifact_verification(
    phase_name: str, path: Path, ok: bool, label: str = ""
) -> None:
    tag = label or path.name
    if ok:
        console.print(
            f"  [green]✓ artifact[/green]  {tag}  [dim]{path}[/dim]"
        )
    else:
        console.print(
            f"  [bold red]✗ artifact MISSING/CORRUPT[/bold red]  {tag}  {path}"
        )
