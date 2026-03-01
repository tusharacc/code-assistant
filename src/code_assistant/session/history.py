"""
Conversation history management.

Keeps a flat list of Message objects and handles compaction when the
accumulated text grows too large for the context window.
"""
from __future__ import annotations

from ..agents.base import Message
from ..config import config
from ..ui.console import print_info


class History:
    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._total_chars: int = 0   # running counter — maintained on every mutation

    # ── Public API ────────────────────────────────────────────────────

    def append(self, messages: list[Message]) -> None:
        for m in messages:
            self._messages.append(m)
            self._total_chars += len(m.content)
        self._maybe_compact()

    def all(self) -> list[Message]:
        # Return the live reference — callers must not mutate this list.
        # Orchestrator and Pipeline only read it, so this is safe.
        return self._messages

    def clear(self) -> None:
        self._messages = []
        self._total_chars = 0

    def total_chars(self) -> int:
        return self._total_chars   # O(1) — maintained incrementally

    def compact(self) -> None:
        """Public wrapper for external callers (e.g. /compact REPL command)."""
        self._compact()

    def add_context_file(self, path: str, content: str) -> None:
        """Prepend a file's content as a system-style context message pair."""
        user_msg = Message(
            role="user",
            content=f"[Context file: {path}]\n```\n{content}\n```",
        )
        ack_msg = Message(
            role="assistant",
            content=f"Understood. I have read {path} and will use it as context.",
        )
        # Single O(n) list creation instead of two O(n) list.insert(0) calls
        self._messages = [user_msg, ack_msg] + self._messages
        self._total_chars += len(user_msg.content) + len(ack_msg.content)

    # ── Compaction ────────────────────────────────────────────────────

    def _maybe_compact(self) -> None:
        if self._total_chars < config.max_history_chars:
            return
        self._compact()

    def _compact(self) -> None:
        """
        Summarise the older half of the conversation into a single context message.
        We keep the most recent messages intact so the model has immediate context.
        """
        if len(self._messages) < 4:
            return  # Too short to compact

        midpoint = len(self._messages) // 2
        older = self._messages[:midpoint]
        newer = self._messages[midpoint:]

        # Build a plain-text summary of the older messages
        summary_lines = ["[Compacted conversation history]\n"]
        for m in older:
            prefix = m.role.upper()
            snippet = m.content[:300].replace("\n", " ")
            if len(m.content) > 300:
                snippet += "…"
            summary_lines.append(f"{prefix}: {snippet}")

        summary = "\n".join(summary_lines)
        compacted_msg = Message(role="user", content=summary)
        ack_msg = Message(
            role="assistant",
            content="Understood. I have the summary of our earlier conversation.",
        )

        self._messages = [compacted_msg, ack_msg] + newer
        # Recompute from scratch after structural rebuild (one-time cost at compaction)
        self._total_chars = sum(len(m.content) for m in self._messages)
        print_info(f"History compacted ({self._total_chars} chars).")
