"""
Claude API 4-phase implementation runner for the benchmark harness.

Mirrors the CA pipeline structure (Architect → Implementer → Reviewer → Tester)
using the Anthropic Python SDK with a write_file tool for file output.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import httpx
import anthropic

from .metrics import BenchmarkResult, PhaseMetrics

# ── Tool schema ───────────────────────────────────────────────────────────────

_WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": (
        "Write content to a file. The path is relative to the project root. "
        "Creates parent directories automatically. "
        "You MUST use exactly the field names 'path' and 'content'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative file path, e.g. 'src/main.py' or 'tests/test_main.py'",
            },
            "content": {
                "type": "string",
                "description": "Complete file content to write. Use the key name 'content' exactly.",
            },
        },
        "required": ["path", "content"],
    },
}

# ── System prompts ────────────────────────────────────────────────────────────

_ARCH_SYSTEM = """\
You are a senior software architect. Given a requirement document, produce a concise but complete
implementation plan. Cover: module layout, key data structures, algorithms, error handling strategy,
and test approach. Do NOT write full code — short pseudocode snippets are fine.
Output plain text, no JSON."""

_IMPL_SYSTEM = """\
You are an expert Python implementer. You will receive an architect's plan and a requirement document.
Your job: write the complete, production-ready implementation.
- Use the write_file tool to create EVERY source file and test file.
- Write all files needed for a working project — no stubs, no placeholders.
- Follow the plan faithfully; deviate only to fix clear errors in it.
- After writing all files, summarise what you created (file list and brief description)."""

_REVIEW_SYSTEM = """\
You are a rigorous code reviewer. Given a requirement and an implementation summary, identify issues.
Output structured findings with ## HIGH Priority, ## MEDIUM Priority, ## LOW Priority sections.
HIGH = bugs / missing required features. MEDIUM = code quality / edge cases. LOW = style."""

_TESTER_SYSTEM = """\
You are a quality assurance engineer. Given a requirement, an implementation summary, and code review
findings, write any missing test files using write_file, then list the exact shell commands needed
to install dependencies and run the tests.
Focus on: edge cases, error paths, and each acceptance criterion."""


class ClaudeRunner:
    def __init__(self, model: str, output_dir: Path, requirement: str, req_file: Path):
        # 300s per-call timeout; retry loop handles transient failures
        self.client = anthropic.Anthropic(timeout=httpx.Timeout(300, connect=10))
        self.model = model
        self.output_dir = output_dir
        self.requirement = requirement
        self.req_file = req_file
        self._usage: dict[str, int] = {"tokens_in": 0, "tokens_out": 0, "api_calls": 0}
        self._files_written: list[str] = []

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> BenchmarkResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        phases: list[PhaseMetrics] = []
        t0 = time.perf_counter()

        # Phase 1 — Architect
        t = time.perf_counter()
        plan = self._architect()
        phases.append(PhaseMetrics("architect", **self._flush_usage(), elapsed=time.perf_counter() - t))

        # Phase 2 — Implementer
        t = time.perf_counter()
        impl_summary = self._implementer(plan)
        phases.append(PhaseMetrics("implementer", **self._flush_usage(), elapsed=time.perf_counter() - t))

        # Phase 3 — Reviewer
        t = time.perf_counter()
        findings = self._reviewer(impl_summary)
        phases.append(PhaseMetrics("reviewer", **self._flush_usage(), elapsed=time.perf_counter() - t))

        # Phase 4 — Tester
        t = time.perf_counter()
        self._tester(impl_summary, findings)
        phases.append(PhaseMetrics("tester", **self._flush_usage(), elapsed=time.perf_counter() - t))

        elapsed_total = time.perf_counter() - t0

        files = sorted(
            str(p.relative_to(self.output_dir))
            for p in self.output_dir.rglob("*")
            if p.is_file()
        )
        total_bytes = sum(p.stat().st_size for p in self.output_dir.rglob("*") if p.is_file())
        total_lines = sum(
            len(p.read_text(errors="replace").splitlines())
            for p in self.output_dir.rglob("*.py")
            if p.is_file()
        )

        return BenchmarkResult(
            runner="claude_api",
            model=self.model,
            requirement_file=str(self.req_file),
            output_dir=str(self.output_dir),
            phases=phases,
            files_written=files,
            total_bytes=total_bytes,
            total_lines=total_lines,
            elapsed_total=elapsed_total,
            timestamp=datetime.now().isoformat(),
        )

    # ── Phases ────────────────────────────────────────────────────────────────

    def _architect(self) -> str:
        resp = self._call(
            system=_ARCH_SYSTEM,
            messages=[{"role": "user", "content": f"## Requirement\n\n{self.requirement}\n\nCreate an implementation plan."}],
            use_tools=False,
        )
        return self._text(resp)

    def _implementer(self, plan: str) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    f"## Requirement\n\n{self.requirement}\n\n"
                    f"## Architect's Plan\n\n{plan}\n\n"
                    "Implement all files using write_file. Write production-ready code — no stubs."
                ),
            }
        ]
        return self._tool_loop(system=_IMPL_SYSTEM, messages=messages)

    def _reviewer(self, impl_summary: str) -> str:
        resp = self._call(
            system=_REVIEW_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"## Requirement\n\n{self.requirement}\n\n"
                        f"## Implementation Summary\n\n{impl_summary}\n\n"
                        "Review the implementation and provide structured findings."
                    ),
                }
            ],
            use_tools=False,
        )
        return self._text(resp)

    def _tester(self, impl_summary: str, findings: str) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    f"## Requirement\n\n{self.requirement}\n\n"
                    f"## Implementation Summary\n\n{impl_summary}\n\n"
                    f"## Review Findings\n\n{findings}\n\n"
                    "Write any missing test files using write_file, then summarise how to run them."
                ),
            }
        ]
        return self._tool_loop(system=_TESTER_SYSTEM, messages=messages)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _call(
        self,
        system: str,
        messages: list[dict],
        use_tools: bool = False,
    ) -> anthropic.types.Message:
        import time as _time
        from rich.console import Console as _Console
        _con = _Console()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 8192,
            "system": system,
            "messages": messages,
        }
        if use_tools:
            kwargs["tools"] = [_WRITE_FILE_TOOL]
        max_retries = 8
        for attempt in range(max_retries):
            try:
                resp = self.client.messages.create(**kwargs)
                self._usage["tokens_in"]  += resp.usage.input_tokens
                self._usage["tokens_out"] += resp.usage.output_tokens
                self._usage["api_calls"]  += 1
                return resp
            except (anthropic.RateLimitError, anthropic.APITimeoutError,
                    anthropic.APIStatusError) as exc:
                # Retry on rate-limit, timeout, AND 529 overloaded
                if isinstance(exc, anthropic.APIStatusError) and exc.status_code not in (429, 529):
                    raise
                if attempt == max_retries - 1:
                    raise
                if isinstance(exc, anthropic.RateLimitError):
                    kind = "Rate limit"
                elif isinstance(exc, anthropic.APITimeoutError):
                    kind = "Timeout"
                else:
                    kind = f"API {exc.status_code}"
                wait = min(30 * (2 ** attempt), 240)  # 30s, 60s, 120s, 240s …
                _con.print(
                    f"[yellow]{kind} — waiting {wait}s before retry "
                    f"({attempt + 1}/{max_retries - 1})…[/yellow]"
                )
                _time.sleep(wait)
        raise RuntimeError("unreachable")

    def _tool_loop(self, system: str, messages: list[dict]) -> str:
        """Call model in a loop until stop_reason == end_turn, running write_file on the way."""
        final_text = ""
        while True:
            resp = self._call(system=system, messages=messages, use_tools=True)
            final_text = self._text(resp)

            # Append assistant turn (with content blocks intact for Anthropic API format)
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                break

            # Execute tool calls, collect results
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    inp = block.input
                    # Defensively accept common alternative key names the model may use
                    path = inp.get("path") or inp.get("file_path") or inp.get("filename", "unknown")
                    content = (
                        inp.get("content")
                        or inp.get("file_content")
                        or inp.get("code")
                        or inp.get("body")
                        or inp.get("text")
                        or ""
                    )
                    result = self._write_file(path, content)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break  # tool_use stop but no tool_use blocks — shouldn't happen

        return final_text

    def _write_file(self, path: str, content: str) -> str:
        full = self.output_dir / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        if path not in self._files_written:
            self._files_written.append(path)
        return f"Written: {path} ({len(content)} chars)"

    @staticmethod
    def _text(resp: anthropic.types.Message) -> str:
        parts = [b.text for b in resp.content if hasattr(b, "text") and b.text]
        return "\n".join(parts)

    def _flush_usage(self) -> dict[str, int]:
        m = dict(self._usage)
        self._usage = {"tokens_in": 0, "tokens_out": 0, "api_calls": 0}
        return m
