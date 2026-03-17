"""
FewShotEnricher — loads past feedback records and injects them into system prompts.

Usage (called at the start of Pipeline.run()):

    from code_assistant.feedback.enricher import enrich_impl_system, enrich_tester_system
    impl.system_prompt = enrich_impl_system(impl.system_prompt, feedback_dir, max_n)
    tester.system_prompt = enrich_tester_system(tester.system_prompt, feedback_dir, max_n)

The few-shot block is appended at the end of the system prompt so it does not
disrupt the existing persona / tool instructions.

On the very first run (feedback.jsonl doesn't exist yet) the base prompt is
returned unchanged, so nothing breaks.
"""
from __future__ import annotations

import json
from pathlib import Path

from .collector import FeedbackRecord


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_examples(
    feedback_dir: Path,
    mistake_types: list[str] | None = None,
    max_n: int = 3,
) -> list[FeedbackRecord]:
    """
    Read feedback.jsonl and return the most recent `max_n` records.

    Parameters
    ----------
    feedback_dir : Path
        Directory that contains feedback.jsonl.
    mistake_types : list[str] | None
        If given, only return records whose `mistake_type` is in this list.
        Pass None to return all types.
    max_n : int
        Maximum number of records to return.
    """
    jsonl = feedback_dir / "feedback.jsonl"
    if not jsonl.exists():
        return []

    records: list[FeedbackRecord] = []
    try:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if mistake_types and data.get("mistake_type") not in mistake_types:
            continue

        # Re-construct dataclass from dict
        try:
            records.append(FeedbackRecord(
                id=data.get("id", ""),
                timestamp=data.get("timestamp", ""),
                model=data.get("model", ""),
                phase=data.get("phase", ""),
                mistake_type=data.get("mistake_type", ""),
                context=data.get("context", ""),
                error_signal=data.get("error_signal", ""),
                correction=data.get("correction", ""),
                tags=data.get("tags", []),
            ))
        except Exception:
            continue

    # Return the most recent max_n (JSONL is append-only, so last = newest)
    return records[-max_n:] if len(records) > max_n else records


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------

def format_few_shot(records: list[FeedbackRecord]) -> str:
    """
    Format a list of FeedbackRecords as a human-readable few-shot block.

    Example output:

        ## Common mistakes to avoid (learned from past runs)

        ### Mistake 1 — tool_error [edit_file, syntax]
        WRONG — caused this error:
          Error: the exact string was not found in /tmp/proj/app.py.

        The assistant tried:
          'fact': lambda x: math.factorial(x) if x >= 0 else raise ValueError(...)

        CORRECT fix applied afterwards:
          def _fact(n):
              if n < 0: raise ValueError("factorial of negative")
              return math.factorial(n)
          ...
          'fact': _fact,

        ---
    """
    if not records:
        return ""

    lines = [
        "",
        "## Common mistakes to avoid (learned from past runs)",
        "",
    ]
    for i, r in enumerate(records, start=1):
        tags_str = ", ".join(r.tags) if r.tags else r.mistake_type
        lines.append(f"### Mistake {i} — {r.mistake_type} [{tags_str}]")

        if r.error_signal:
            lines.append("WRONG — caused this error:")
            for el in r.error_signal.splitlines()[:6]:
                lines.append(f"  {el}")

        if r.context:
            lines.append("")
            lines.append("The assistant tried:")
            for cl in r.context.splitlines()[:8]:
                lines.append(f"  {cl}")

        if r.correction:
            lines.append("")
            lines.append("CORRECT fix applied afterwards:")
            for fl in r.correction.splitlines()[:12]:
                lines.append(f"  {fl}")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public injection helpers
# ------------------------------------------------------------------

def enrich_impl_system(
    base_prompt: str,
    feedback_dir: Path,
    max_n: int = 3,
) -> str:
    """
    Append a few-shot block of tool_error and review_issue examples to the
    implementer's system prompt.

    Returns the original base_prompt unchanged if no examples exist yet.
    """
    records = load_examples(
        feedback_dir,
        mistake_types=["tool_error", "review_issue"],
        max_n=max_n,
    )
    if not records:
        return base_prompt
    return base_prompt + format_few_shot(records)


def enrich_tester_system(
    base_prompt: str,
    feedback_dir: Path,
    max_n: int = 3,
) -> str:
    """
    Append a few-shot block of test_failure examples to the tester's system prompt.

    Returns the original base_prompt unchanged if no examples exist yet.
    """
    records = load_examples(
        feedback_dir,
        mistake_types=["test_failure"],
        max_n=max_n,
    )
    if not records:
        return base_prompt
    return base_prompt + format_few_shot(records)
