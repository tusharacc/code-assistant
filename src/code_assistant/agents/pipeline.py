"""
4-persona pipeline:
  Architect → Implementer → Reviewer → Implementer (fix) → Tester [→ Implementer (fix) → Tester]*
  → Documentation writer.

Each persona runs with keep_alive=0 so Ollama unloads the model from RAM
immediately after every response. Only one model is in memory at a time.
Per-persona conversation histories are maintained across phases for coherent context.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..config import config
from ..logger import get_logger
from ..ui.console import console, print_rule, print_info, print_warning
from .base import Agent, Message
from .architect import make_architect
from .implementer import make_implementer
from .reviewer import make_reviewer
from .tester import make_tester

log = get_logger(__name__)

# Tag that implementer uses to ask architect a question
_QA_TAG = re.compile(r"@@QUESTION_FOR_ARCHITECT:\s*(.*?)@@", re.DOTALL)

# Section headers the reviewer is expected to produce
_HIGH_SECTION = re.compile(
    r"##\s*HIGH\s*Priority\s*\n(.*?)(?=##|\Z)", re.DOTALL | re.IGNORECASE
)
_MEDIUM_SECTION = re.compile(
    r"##\s*MEDIUM\s*Priority\s*\n(.*?)(?=##|\Z)", re.DOTALL | re.IGNORECASE
)

# Max Q&A rounds between implementer and architect
_MAX_QA_ROUNDS = 5

# Tester output patterns — written to match all common LLM formatting variants:
#   ## Overall Verdict\nFAIL ...        (Markdown heading, our template)
#   **Overall Verdict:** FAIL           (bold inline, common LLM variant)
#   Overall Verdict: FAIL               (plain text)
_VERDICT_FAIL = re.compile(
    r"(?:##\s*|\*\*)?Overall\s+Verdict(?:\*\*)?[:\*\s]*\n?\s*(FAIL|PARTIAL)",
    re.IGNORECASE,
)
_FAIL_ROW = re.compile(r"\|\s*(.+?)\s*\|\s*FAIL\s*\|\s*(.+?)\s*\|", re.IGNORECASE)

# Max rounds of tester → implementer-fix → re-test
_MAX_TEST_FIX_ROUNDS = 3


@dataclass
class PipelineState:
    """Shared state that accumulates across pipeline phases."""
    # Per-persona conversation histories
    arch_history: list[Message] = field(default_factory=list)
    impl_history: list[Message] = field(default_factory=list)
    review_history: list[Message] = field(default_factory=list)
    test_history: list[Message] = field(default_factory=list)
    docs_history: list[Message] = field(default_factory=list)

    # Artifacts produced during the run
    arch_plan: str = ""
    acceptance_criteria: str = ""
    review_findings: str = ""
    run_instructions: str = ""
    test_results: str = ""
    doc_output: str = ""


def _snap(agent: "Agent") -> tuple[int, int, int]:
    """Snapshot agent token/call counters for delta calculation."""
    return agent.token_in, agent.token_out, agent.api_calls


def _phase_metrics(agent: "Agent", before: tuple[int, int, int], elapsed: float) -> dict:
    """Compute delta counters since snapshot and return as a metrics dict."""
    ti, to, nc = before
    return {
        "tokens_in":  agent.token_in  - ti,
        "tokens_out": agent.token_out - to,
        "api_calls":  agent.api_calls - nc,
        "elapsed":    elapsed,
    }


class Pipeline:
    """
    Drives the 4-persona sequential pipeline.

    Each call to run() produces a list[Message] suitable for appending
    to the main REPL history. The per-persona histories live inside
    PipelineState and are discarded at the end of the run.

    Parameters
    ----------
    rag_context : str | None
        Relevant code chunks from the RAG index.
    initial_history : list[Message] | None
        Prior conversation messages (e.g. a loaded requirement document).
        Seeded into the architect's context so it can see the full spec.
    """

    def __init__(
        self,
        rag_context: str | None = None,
        initial_history: list[Message] | None = None,
    ) -> None:
        self.rag_context = rag_context
        self.initial_history: list[Message] = initial_history or []
        # Populated by run() — keyed by phase name, plus "elapsed_total"
        self.metrics: dict = {}
        # Preserved after run() for feedback collection; None before first run
        self.last_state: PipelineState | None = None

    def run(self, user_task: str) -> list[Message]:
        """Execute all pipeline phases. Returns all produced messages."""
        state = PipelineState()
        all_messages: list[Message] = [Message(role="user", content=user_task)]

        log.info("Pipeline start | task_chars=%d", len(user_task))

        # Create all agents once — each phase receives the same objects.
        # keep_alive=0 ensures Ollama unloads each model after every response
        # so only one model occupies RAM at a time.
        arch = make_architect(keep_alive=0)
        impl = make_implementer(keep_alive=0)
        reviewer = make_reviewer(keep_alive=0)
        tester = make_tester(keep_alive=0)

        # ── Few-shot enrichment (inject past mistakes into system prompts) ──
        # Runs before any phase so the model benefits from the first turn onward.
        # Silently skipped if feedback is disabled or no examples exist yet.
        try:
            if config.feedback_enabled and config.few_shot_max > 0:
                from ..feedback.enricher import enrich_impl_system, enrich_tester_system
                from pathlib import Path as _Path
                _fdir = _Path(config.feedback_dir)
                impl.system_prompt = enrich_impl_system(
                    impl.system_prompt, _fdir, config.few_shot_max
                )
                tester.system_prompt = enrich_tester_system(
                    tester.system_prompt, _fdir, config.few_shot_max
                )
                log.debug(
                    "Few-shot enrichment done | impl_prompt_chars=%d tester_prompt_chars=%d",
                    len(impl.system_prompt), len(tester.system_prompt),
                )
        except Exception as _e:
            log.warning("Few-shot enrichment failed (non-fatal): %s", _e)

        t0 = time.perf_counter()

        # ── Phase 1: Architect ───────────────────────────────────────────
        print_rule("pipeline · phase 1 · architect", style="dim cyan")
        t = time.perf_counter()
        before = _snap(arch)
        all_messages.extend(self._phase_architect(user_task, state, arch))
        self.metrics["architect"] = _phase_metrics(arch, before, time.perf_counter() - t)

        # ── Phase 2: Implementer (with optional Q&A with architect) ──────
        print_rule("pipeline · phase 2 · implementer", style="dim green")
        t = time.perf_counter()
        before_impl = _snap(impl)
        before_arch = _snap(arch)  # arch may be called during Q&A
        all_messages.extend(self._phase_implementer(user_task, state, arch, impl))
        self.metrics["implementer"] = _phase_metrics(impl, before_impl, time.perf_counter() - t)
        self.metrics["implementer_qa_arch"] = _phase_metrics(arch, before_arch, 0)

        # ── Phase 3: Reviewer ────────────────────────────────────────────
        print_rule("pipeline · phase 3 · reviewer", style="dim yellow")
        t = time.perf_counter()
        before = _snap(reviewer)
        all_messages.extend(self._phase_reviewer(user_task, state, reviewer))
        self.metrics["reviewer"] = _phase_metrics(reviewer, before, time.perf_counter() - t)

        # ── Phase 4: Implementer fix (HIGH + MEDIUM issues only) ─────────
        high, medium = _parse_findings(state.review_findings)
        if high or medium:
            print_rule("pipeline · phase 4 · implementer fix", style="dim green")
            t = time.perf_counter()
            before = _snap(impl)
            all_messages.extend(self._phase_fix(state, high, medium, impl))
            self.metrics["implementer_fix"] = _phase_metrics(impl, before, time.perf_counter() - t)
        else:
            print_info("No HIGH/MEDIUM issues — skipping fix phase.")
            log.info("Pipeline: no HIGH/MEDIUM findings, skipping fix phase")

        # ── Phase 5: Gather run info from implementer & architect ────────
        self._gather_run_info(state, arch, impl)

        # ── Phase 6+: Tester with fix-loop ───────────────────────────────
        for test_round in range(_MAX_TEST_FIX_ROUNDS):
            round_label = f"pipeline · tester · round {test_round + 1}/{_MAX_TEST_FIX_ROUNDS}"
            print_rule(round_label, style="dim magenta")
            t = time.perf_counter()
            before = _snap(tester)
            all_messages.extend(self._phase_tester(state, tester, round_num=test_round))
            key = f"tester_round{test_round + 1}"
            self.metrics[key] = _phase_metrics(tester, before, time.perf_counter() - t)

            failures = _parse_test_failures(state.test_results)
            if not failures:
                print_info("✓ All acceptance criteria passed.")
                log.info("Tester: all criteria passed in round %d", test_round + 1)
                break

            log.info(
                "Tester round %d: %d failure(s) — handing to implementer",
                test_round + 1, len(failures),
            )

            if test_round < _MAX_TEST_FIX_ROUNDS - 1:
                fix_label = f"pipeline · tester-fix · round {test_round + 1}"
                print_rule(fix_label, style="dim green")
                t = time.perf_counter()
                before = _snap(impl)
                all_messages.extend(self._phase_test_fix(state, failures, impl))
                self.metrics[f"test_fix_round{test_round + 1}"] = _phase_metrics(
                    impl, before, time.perf_counter() - t
                )
            else:
                print_warning(
                    f"Max test-fix rounds ({_MAX_TEST_FIX_ROUNDS}) reached — "
                    f"{len(failures)} criterion/criteria still failing."
                )

        # ── Phase 7: Documentation writer ────────────────────────────────
        print_rule("pipeline · phase 7 · documentation", style="dim blue")
        t = time.perf_counter()
        before = _snap(impl)
        all_messages.extend(self._phase_docs(user_task, state, impl))
        self.metrics["docs"] = _phase_metrics(impl, before, time.perf_counter() - t)

        self.metrics["elapsed_total"] = time.perf_counter() - t0

        # Preserve state for callers (e.g. feedback collector, tests)
        self.last_state = state

        log.info(
            "Pipeline complete | total_messages=%d arch_plan_chars=%d "
            "findings_chars=%d test_results_chars=%d doc_chars=%d",
            len(all_messages),
            len(state.arch_plan),
            len(state.review_findings),
            len(state.test_results),
            len(state.doc_output),
        )

        # ── Feedback collection (save mistake→correction pairs) ──────────
        # Errors are non-fatal — a feedback write failure must never abort a run.
        try:
            if config.feedback_enabled:
                from pathlib import Path as _Path
                from ..feedback.collector import collect as _collect, save as _save_fb
                _fdir = _Path(config.feedback_dir)
                _records = _collect(self, user_task)
                if _records:
                    _save_fb(_records, _fdir)
                    log.info("feedback: saved %d new example(s) → %s", len(_records), _fdir)
                    print_info(
                        f"Feedback: {len(_records)} new example(s) saved → "
                        f"{_fdir / 'feedback.jsonl'}"
                    )
                else:
                    log.debug("feedback: no new examples this run")
        except Exception as _e:
            log.warning("Feedback collection failed (non-fatal): %s", _e)

        # ── Project context update (regenerate code_assistant.md) ────────────
        # Fully regenerates the per-project AI memory file with updated
        # architecture summary, key files, acceptance criteria outcomes, and
        # work history.  Errors are non-fatal.
        try:
            if config.project_context_enabled:
                from pathlib import Path as _Path
                from ..project_context import ProjectContext
                ProjectContext().update_from_pipeline(state, user_task)
                log.info("project_context: code_assistant.md updated after pipeline run")
        except Exception as _e:
            log.warning("project_context update failed (non-fatal): %s", _e)

        return all_messages

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_architect(
        self, user_task: str, state: PipelineState, arch: Agent
    ) -> list[Message]:
        """Architect produces the initial plan.

        Seeds arch_history with any initial_history (e.g. a loaded requirement
        document) so the architect can read the full specification.
        """
        # Prepend prior context (requirement doc, etc.) if provided
        if self.initial_history:
            state.arch_history = list(self.initial_history)
            log.info(
                "Phase 1 (architect): seeded %d prior messages from initial_history",
                len(self.initial_history),
            )

        prompt = (
            "Please analyse this task and propose a detailed implementation plan. "
            "Focus on design decisions, module layout, data structures, and algorithms. "
            "Do NOT write full code — short pseudocode snippets to illustrate are fine.\n\n"
            f"Task: {user_task}"
        )
        state.arch_history.append(Message(role="user", content=prompt))

        arch_text, new_msgs = arch.run(state.arch_history, rag_context=self.rag_context)
        state.arch_history.extend(new_msgs)
        state.arch_plan = arch_text

        log.info("Phase 1 (architect) complete | plan_chars=%d", len(arch_text))
        return new_msgs

    def _phase_implementer(
        self, user_task: str, state: PipelineState, arch: Agent, impl: Agent
    ) -> list[Message]:
        """
        Implementer codes the plan. If it asks the architect a question
        (via @@QUESTION_FOR_ARCHITECT: ...@@), we route the Q to the architect
        and return the answer, then let the implementer continue.
        """
        all_new: list[Message] = []

        initial_prompt = (
            f"The architect has proposed this plan:\n\n{state.arch_plan}\n\n"
            f"Original task: {user_task}\n\n"
            "Now implement it completely. Use your tools to create and modify files. "
            "Write production-ready code — no stubs or placeholders."
        )
        state.impl_history.append(Message(role="user", content=initial_prompt))

        for round_num in range(_MAX_QA_ROUNDS + 1):
            impl_text, new_msgs = impl.run(
                state.impl_history, rag_context=self.rag_context
            )
            state.impl_history.extend(new_msgs)
            all_new.extend(new_msgs)

            # Check for a Q directed at the architect
            m = _QA_TAG.search(impl_text)
            if not m:
                log.info(
                    "Phase 2 (implementer) complete | qa_rounds=%d impl_chars=%d",
                    round_num, len(impl_text),
                )
                break

            question = m.group(1).strip()
            log.info(
                "Implementer Q&A round %d | question_chars=%d", round_num + 1, len(question)
            )
            console.print(
                f"\n[dim cyan]↳ Implementer asks architect: {question[:120]}…[/dim cyan]"
            )

            # Route question to architect
            arch_prompt = (
                f"The implementer is asking: {question}\n\n"
                "Please answer concisely and directly."
            )
            state.arch_history.append(Message(role="user", content=arch_prompt))
            arch_text, arch_new = arch.run(state.arch_history, rag_context=self.rag_context)
            state.arch_history.extend(arch_new)

            console.print(f"[dim cyan]↳ Architect answers: {arch_text[:120]}…[/dim cyan]\n")

            # Feed architect's answer back into implementer history
            answer_msg = Message(
                role="user",
                content=f"Architect's answer to your question: {arch_text}\n\nPlease continue implementing.",
            )
            state.impl_history.append(answer_msg)
        else:
            log.warning("Implementer Q&A exceeded max rounds (%d), stopping", _MAX_QA_ROUNDS)

        return all_new

    def _phase_reviewer(
        self, user_task: str, state: PipelineState, reviewer: Agent
    ) -> list[Message]:
        """Reviewer inspects the implementation and produces structured findings."""
        impl_summary = _last_assistant_text(state.impl_history)

        prompt = (
            f"Task that was given to the implementer:\n{user_task}\n\n"
            f"Architect's plan:\n{state.arch_plan}\n\n"
            f"Implementer's summary:\n{impl_summary}\n\n"
            "Please review the implementation. Use your read tools (read_file, glob_files, "
            "list_dir) to inspect the actual files. Do NOT modify any files.\n\n"
            "Produce structured findings with ## HIGH Priority, ## MEDIUM Priority, "
            "## LOW Priority, and ## Summary sections."
        )
        state.review_history.append(Message(role="user", content=prompt))

        review_text, new_msgs = reviewer.run(
            state.review_history, rag_context=self.rag_context
        )
        state.review_history.extend(new_msgs)
        state.review_findings = review_text

        log.info("Phase 3 (reviewer) complete | findings_chars=%d", len(review_text))
        return new_msgs

    def _phase_fix(
        self, state: PipelineState, high: str, medium: str, impl: Agent
    ) -> list[Message]:
        """Implementer fixes HIGH and MEDIUM priority issues from the review."""
        findings_text = ""
        if high:
            findings_text += f"## HIGH Priority\n{high}\n\n"
        if medium:
            findings_text += f"## MEDIUM Priority\n{medium}\n"

        fix_prompt = (
            f"The code reviewer found these issues:\n\n{findings_text}\n"
            "Please fix all HIGH and MEDIUM priority issues now. "
            "Read the relevant files first, then apply the fixes."
        )
        state.impl_history.append(Message(role="user", content=fix_prompt))

        fix_text, new_msgs = impl.run(
            state.impl_history, rag_context=self.rag_context
        )
        state.impl_history.extend(new_msgs)

        log.info("Phase 4 (implementer fix) complete | response_chars=%d", len(fix_text))
        return new_msgs

    def _gather_run_info(
        self, state: PipelineState, arch: Agent, impl: Agent
    ) -> None:
        """
        Ask the implementer how to run the project and ask the architect
        for acceptance criteria. Results are stored in state for the tester.
        """
        run_q = Message(
            role="user",
            content=(
                "Before handing over to the tester, describe exactly how to run and test "
                "this project. List the precise shell commands (install, run, test) in order. "
                "Be concrete — no placeholders."
            ),
        )
        state.impl_history.append(run_q)
        run_text, run_msgs = impl.run(state.impl_history, rag_context=self.rag_context)
        state.impl_history.extend(run_msgs)
        state.run_instructions = run_text
        log.info("Gathered run instructions | chars=%d", len(run_text))

        ac_q = Message(
            role="user",
            content=(
                "What are the acceptance criteria for this task? "
                "List each criterion on a new bullet so the tester can verify them one by one."
            ),
        )
        state.arch_history.append(ac_q)
        ac_text, ac_msgs = arch.run(state.arch_history, rag_context=self.rag_context)
        state.arch_history.extend(ac_msgs)
        state.acceptance_criteria = ac_text
        log.info("Gathered acceptance criteria | chars=%d", len(ac_text))

    def _phase_tester(
        self, state: PipelineState, tester: Agent, round_num: int = 0
    ) -> list[Message]:
        """Tester runs the project, checks each acceptance criterion, reports results."""
        if round_num == 0:
            prompt = (
                f"## Run Instructions (from implementer)\n{state.run_instructions}\n\n"
                f"## Acceptance Criteria (from architect)\n{state.acceptance_criteria}\n\n"
                "Use run_shell to execute the relevant commands and verify each criterion. "
                "Do not run interactive programs — use unit tests or `python -c '...'` checks instead. "
                "Report PASS, FAIL, or MANUAL for each criterion with evidence from the command output."
            )
        else:
            prompt = (
                "The implementer has applied fixes for the previously failing criteria. "
                "Re-run the same acceptance criteria checks now and produce a fresh result table. "
                "Focus especially on the criteria that were FAIL in the previous round."
            )
        state.test_history.append(Message(role="user", content=prompt))

        test_text, new_msgs = tester.run(state.test_history, rag_context=self.rag_context)
        state.test_history.extend(new_msgs)
        state.test_results = test_text

        log.info(
            "Tester round %d complete | results_chars=%d",
            round_num + 1, len(test_text),
        )
        return new_msgs

    def _phase_test_fix(
        self,
        state: PipelineState,
        failures: list[tuple[str, str]],
        impl: Agent,
    ) -> list[Message]:
        """Implementer fixes criteria that the tester marked as FAIL."""
        failure_lines = "\n".join(
            f"  - Criterion: {criterion}\n    Evidence: {evidence}"
            for criterion, evidence in failures
        )
        fix_prompt = (
            f"The tester found {len(failures)} FAILING acceptance criteria:\n\n"
            f"{failure_lines}\n\n"
            "Read the relevant files, identify the root cause of each failure, "
            "and fix the code. After fixing, confirm exactly what you changed."
        )
        state.impl_history.append(Message(role="user", content=fix_prompt))

        fix_text, new_msgs = impl.run(state.impl_history, rag_context=self.rag_context)
        state.impl_history.extend(new_msgs)

        log.info(
            "Test-fix round complete | failures=%d response_chars=%d",
            len(failures), len(fix_text),
        )
        return new_msgs

    def _phase_docs(
        self, user_task: str, state: PipelineState, impl: Agent
    ) -> list[Message]:
        """
        Documentation writer phase.

        Seeds the docs history with the full implementer history so the writer
        already knows every file path that was created, then asks it to write
        README.md in the project root.
        """
        test_verdict = "unknown"
        m = _VERDICT_FAIL.search(state.test_results)
        if m:
            test_verdict = m.group(1).upper()
        elif state.test_results:
            test_verdict = "PASS"

        # Seed with the implementer's full history so the writer knows the
        # exact directory structure and file paths without having to guess.
        state.docs_history = list(state.impl_history)

        # Determine the absolute path for README.md from tool call history
        project_dir = _extract_project_dir(state.impl_history)
        if project_dir:
            readme_path = f"{project_dir}/README.md"
            path_instruction = (
                f"The project root is `{project_dir}` (extracted from tool call history).\n"
                f"Write the README to the EXACT absolute path: `{readme_path}`\n"
                "Do NOT use relative paths like `./README.md` — they resolve to the wrong directory."
            )
        else:
            readme_path = "README.md"
            path_instruction = (
                "You must determine the project root from your tool call history above "
                "(look for write_file calls and use that directory). "
                "Use the ABSOLUTE path for README.md, not `./README.md`."
            )
        log.info("Docs phase: project_dir=%s readme_path=%s", project_dir, readme_path)

        prompt = (
            "You are now acting as a technical writer.\n\n"
            "You have full context of everything that was implemented above. "
            f"{path_instruction}\n\n"
            "## Steps\n"
            f"1. Run `ls {project_dir or '.'}` with run_shell to confirm the layout.\n"
            f"2. Write README.md to `{readme_path}` using write_file.\n\n"
            "## Context\n"
            f"**Original task:**\n{user_task}\n\n"
            f"**Run instructions:**\n{state.run_instructions}\n\n"
            f"**Acceptance criteria:**\n{state.acceptance_criteria}\n\n"
            f"**Test verdict:** {test_verdict}\n\n"
            "## README.md must include these sections (Markdown headings):\n"
            "1. **Project title and one-line description**\n"
            "2. **Features** — bullet list of every operation / capability\n"
            "3. **Requirements** — Python version, no external dependencies\n"
            "4. **Installation** — clone / copy, no pip install needed\n"
            "5. **Usage**\n"
            "   - CLI one-shot mode: `python -m <pkg> \"expr\"`\n"
            "   - Interactive REPL mode: `python -m <pkg>` (with example session)\n"
            "   - Built-in REPL commands: help, history, clear, exit/quit\n"
            "6. **Plugin / extension API** — register_operation example\n"
            "7. **Running the tests** — exact pytest command from project root\n"
            "8. **Acceptance criteria status** — Markdown table of each criterion "
            "and its PASS/FAIL result from the tester\n"
        )
        state.docs_history.append(Message(role="user", content=prompt))

        doc_text, new_msgs = impl.run(state.docs_history, rag_context=self.rag_context)
        state.docs_history.extend(new_msgs)
        state.doc_output = doc_text

        log.info("Phase 7 (docs) complete | doc_chars=%d", len(doc_text))
        return new_msgs


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _extract_project_dir(impl_history: list[Message]) -> str | None:
    """
    Scan the implementer's history for write_file tool calls and return
    the absolute directory of the first file written — this is the project root.
    Falls back to None if nothing is found.
    """
    import json as _json
    from pathlib import Path as _Path

    for msg in impl_history:
        for tc in msg.tool_calls:
            fn = tc.get("function", {})
            if fn.get("name") != "write_file":
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    continue
            path_str = args.get("path", "")
            if path_str:
                p = _Path(path_str).expanduser().resolve()
                return str(p.parent)
    return None


def _parse_test_failures(test_text: str) -> list[tuple[str, str]]:
    """
    Return (criterion, evidence) pairs for every FAIL row in the tester's
    markdown table, but only when the Overall Verdict is FAIL or PARTIAL.

    If the verdict is FAIL but the tester didn't produce a table (used bold
    text or plain prose instead), return a synthetic entry so the fix loop
    is still triggered with the full tester output as evidence.
    """
    if not _VERDICT_FAIL.search(test_text):
        return []
    rows = _FAIL_ROW.findall(test_text)
    if rows:
        return rows
    # Verdict is FAIL but no parseable table rows — hand the full output to
    # the implementer so it can read the tester's prose and fix the issues.
    return [("(see tester output)", test_text[:1000])]


def _parse_findings(review_text: str) -> tuple[str, str]:
    """Extract HIGH and MEDIUM priority sections from reviewer output."""
    high = ""
    medium = ""
    m = _HIGH_SECTION.search(review_text)
    if m:
        high = m.group(1).strip()
        if high.lower() == "none.":
            high = ""
    m = _MEDIUM_SECTION.search(review_text)
    if m:
        medium = m.group(1).strip()
        if medium.lower() == "none.":
            medium = ""
    return high, medium


def _last_assistant_text(history: list[Message]) -> str:
    """Return the most recent assistant message content from a history list."""
    for msg in reversed(history):
        if msg.role == "assistant" and msg.content:
            return msg.content
    return "(no summary available)"
