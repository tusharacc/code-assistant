"""
FeedbackCollector — extracts (mistake → correction) pairs from pipeline runs.

Three extraction strategies:
  1. tool_error   — assistant made a tool call that returned "Error:..."
  2. review_issue — reviewer found HIGH/MEDIUM issues; implementer then fixed them
  3. test_failure — tester reported FAIL; implementer then applied a fix

Records are saved as JSON-lines to ~/.code-assistant/feedback/feedback.jsonl.
On subsequent runs the FewShotEnricher reads them back and injects the most
relevant examples into the implementer/tester system prompts.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.pipeline import Pipeline


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    """One mistake → correction training example."""
    id: str                   # uuid4 hex
    timestamp: str            # ISO-8601 UTC
    model: str                # e.g. "qwen2.5-coder:14b"
    phase: str                # "implementer" | "tester"
    mistake_type: str         # "tool_error" | "review_issue" | "test_failure"
    context: str              # assistant text that led to the mistake (≤400 chars)
    error_signal: str         # tool result / reviewer / tester feedback (≤400 chars)
    correction: str           # next assistant message that fixed it (≤600 chars)
    tags: list[str] = field(default_factory=list)   # ["edit_file","syntax","lambda"]

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Tag extraction helpers
# ------------------------------------------------------------------

_SYNTAX_KEYWORDS = ("syntaxerror", "invalid syntax", "unexpected token", "indentation")
_TYPE_KEYWORDS   = ("typeerror", "attributeerror", "nameerror")
_IMPORT_KEYWORDS = ("importerror", "modulenotfounderror", "no module named")


def _tags_from_tool(tool_name: str, error_text: str) -> list[str]:
    """Infer tags from the tool name and error message text."""
    tags: list[str] = []
    tl = tool_name.lower()
    el = error_text.lower()

    if "edit_file" in tl:
        tags.append("edit_file")
    elif "write_file" in tl:
        tags.append("write_file")
    elif "run_shell" in tl:
        tags.append("run_shell")

    if any(k in el for k in _SYNTAX_KEYWORDS):
        tags.append("syntax")
    if "lambda" in el:
        tags.append("lambda")
    if any(k in el for k in _TYPE_KEYWORDS):
        tags.append("type_error")
    if any(k in el for k in _IMPORT_KEYWORDS):
        tags.append("import_error")
    if "not found" in el or "no such file" in el:
        tags.append("file_not_found")
    if "old_string not found" in el or "exact string" in el:
        tags.append("edit_patch")

    return tags or ["tool_error"]


# ------------------------------------------------------------------
# Extractor 1 — tool errors in impl_history
# ------------------------------------------------------------------

def extract_tool_errors(messages: list, model: str) -> list[FeedbackRecord]:
    """
    Scan impl_history for:
        assistant (tool_calls) → tool ("Error:...") → assistant (correction)

    One FeedbackRecord is emitted per (assistant, error, correction) triplet.
    """
    records: list[FeedbackRecord] = []
    n = len(messages)

    i = 0
    while i < n:
        msg = messages[i]
        # Looking for an assistant message that made tool calls
        if msg.role != "assistant" or not msg.tool_calls:
            i += 1
            continue

        # Collect all immediately following tool result messages
        j = i + 1
        error_entries: list[tuple[str, str]] = []   # (tool_name, error_text)
        while j < n and messages[j].role == "tool":
            result = messages[j].content or ""
            if result.startswith("Error") or "Error:" in result[:80]:
                # Figure out which tool call this result belongs to by position
                tool_idx = j - (i + 1)
                if tool_idx < len(msg.tool_calls):
                    tc = msg.tool_calls[tool_idx]
                    tool_name = tc.get("function", {}).get("name", "unknown_tool")
                else:
                    tool_name = "unknown_tool"
                error_entries.append((tool_name, result))
            j += 1

        if not error_entries:
            i += 1
            continue

        # Look for the next assistant message (the correction)
        correction_idx = None
        for k in range(j, n):
            if messages[k].role == "assistant":
                correction_idx = k
                break

        if correction_idx is None:
            # No correction found — the run ended after the error
            i = j
            continue

        correction_msg = messages[correction_idx]

        for tool_name, error_text in error_entries:
            records.append(FeedbackRecord(
                id=uuid.uuid4().hex,
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=model,
                phase="implementer",
                mistake_type="tool_error",
                context=_trunc(msg.content or "", 400),
                error_signal=_trunc(error_text, 400),
                correction=_trunc(correction_msg.content or "", 600),
                tags=_tags_from_tool(tool_name, error_text),
            ))

        i = correction_idx + 1   # advance past the correction

    return records


# ------------------------------------------------------------------
# Extractor 2 — reviewer-triggered fix cycles
# ------------------------------------------------------------------

_REVIEW_TRIGGER = "code reviewer found these issues"


def extract_review_cycles(messages: list, review_findings: str, model: str) -> list[FeedbackRecord]:
    """
    Find user messages containing the review trigger phrase → next assistant response.

    Each such pair represents: (review complaint) → (implementer fix).
    """
    records: list[FeedbackRecord] = []
    n = len(messages)

    for i, msg in enumerate(messages):
        if msg.role != "user":
            continue
        if _REVIEW_TRIGGER not in (msg.content or "").lower():
            continue

        # Look for the next assistant message
        for j in range(i + 1, n):
            if messages[j].role == "assistant":
                correction = messages[j].content or ""
                # Determine severity tag from the review findings
                findings_lower = (review_findings or "").lower()
                tags = ["review"]
                if "## high priority" in findings_lower:
                    tags.append("HIGH")
                if "## medium priority" in findings_lower:
                    tags.append("MEDIUM")

                records.append(FeedbackRecord(
                    id=uuid.uuid4().hex,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model=model,
                    phase="implementer",
                    mistake_type="review_issue",
                    context=_trunc(msg.content or "", 400),
                    error_signal=_trunc(review_findings or "", 400),
                    correction=_trunc(correction, 600),
                    tags=tags,
                ))
                break

    return records


# ------------------------------------------------------------------
# Extractor 3 — tester-triggered fix cycles
# ------------------------------------------------------------------

_TEST_TRIGGER = "failing acceptance criteria"


def extract_test_cycles(messages: list, test_results: str, model: str) -> list[FeedbackRecord]:
    """
    Find user messages containing the test-failure trigger phrase → next assistant response.

    Captures up to _MAX_TEST_FIX_ROUNDS fix cycles.
    """
    records: list[FeedbackRecord] = []
    n = len(messages)

    for i, msg in enumerate(messages):
        if msg.role != "user":
            continue
        content_lower = (msg.content or "").lower()
        if _TEST_TRIGGER not in content_lower and "the tester found" not in content_lower:
            continue

        for j in range(i + 1, n):
            if messages[j].role == "assistant":
                correction = messages[j].content or ""
                tags = ["test"]
                # Try to extract AC number from the trigger message
                import re as _re
                m = _re.search(r"ac[\s-]*(\d+)", content_lower)
                if m:
                    tags.append(f"AC-{m.group(1)}")

                records.append(FeedbackRecord(
                    id=uuid.uuid4().hex,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model=model,
                    phase="tester",
                    mistake_type="test_failure",
                    context=_trunc(msg.content or "", 400),
                    error_signal=_trunc(test_results or "", 400),
                    correction=_trunc(correction, 600),
                    tags=tags,
                ))
                break

    return records


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def collect(pipeline: "Pipeline", requirement_text: str) -> list[FeedbackRecord]:
    """
    Extract all feedback records from a completed Pipeline run.

    Requires pipeline.last_state to be set (done at end of pipeline.run()).
    Returns an empty list if last_state is None or no examples were found.
    """
    state = getattr(pipeline, "last_state", None)
    if state is None:
        return []

    # Use the configured implementer model as the source model label
    from ..config import config
    model = config.implementer_model

    records: list[FeedbackRecord] = []

    records.extend(extract_tool_errors(state.impl_history, model))
    records.extend(extract_review_cycles(
        state.impl_history, state.review_findings, model
    ))
    records.extend(extract_test_cycles(
        state.impl_history, state.test_results, model
    ))

    # Deduplicate by (context, error_signal) — prevent double-counting on re-runs
    seen: set[tuple[str, str]] = set()
    unique: list[FeedbackRecord] = []
    for r in records:
        key = (r.context[:80], r.error_signal[:80])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def save(records: list[FeedbackRecord], feedback_dir: Path) -> None:
    """Append records to feedback_dir/feedback.jsonl (creates dir if needed)."""
    feedback_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = feedback_dir / "feedback.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _trunc(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"
