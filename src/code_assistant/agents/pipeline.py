"""
4-persona pipeline: Architect → Implementer → Reviewer → Implementer (fix) → Tester.

Each persona runs with keep_alive=0 so Ollama unloads the model from RAM
immediately after every response. Only one model is in memory at a time.
Per-persona conversation histories are maintained across phases for coherent context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..logger import get_logger
from ..ui.console import console, print_rule, print_info
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


@dataclass
class PipelineState:
    """Shared state that accumulates across pipeline phases."""
    # Per-persona conversation histories
    arch_history: list[Message] = field(default_factory=list)
    impl_history: list[Message] = field(default_factory=list)
    review_history: list[Message] = field(default_factory=list)
    test_history: list[Message] = field(default_factory=list)

    # Artifacts produced during the run
    arch_plan: str = ""
    acceptance_criteria: str = ""
    review_findings: str = ""
    run_instructions: str = ""
    test_results: str = ""


class Pipeline:
    """
    Drives the 4-persona sequential pipeline.

    Each call to run() produces a list[Message] suitable for appending
    to the main REPL history. The per-persona histories live inside
    PipelineState and are discarded at the end of the run.
    """

    def __init__(self, rag_context: str | None = None) -> None:
        self.rag_context = rag_context

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

        # ── Phase 1: Architect ───────────────────────────────────────────
        print_rule("pipeline · phase 1 · architect", style="dim cyan")
        all_messages.extend(self._phase_architect(user_task, state, arch))

        # ── Phase 2: Implementer (with optional Q&A with architect) ──────
        print_rule("pipeline · phase 2 · implementer", style="dim green")
        all_messages.extend(self._phase_implementer(user_task, state, arch, impl))

        # ── Phase 3: Reviewer ────────────────────────────────────────────
        print_rule("pipeline · phase 3 · reviewer", style="dim yellow")
        all_messages.extend(self._phase_reviewer(user_task, state, reviewer))

        # ── Phase 4: Implementer fix (HIGH + MEDIUM issues only) ─────────
        high, medium = _parse_findings(state.review_findings)
        if high or medium:
            print_rule("pipeline · phase 4 · implementer fix", style="dim green")
            all_messages.extend(self._phase_fix(state, high, medium, impl))
        else:
            print_info("No HIGH/MEDIUM issues — skipping fix phase.")
            log.info("Pipeline: no HIGH/MEDIUM findings, skipping fix phase")

        # ── Phase 5: Gather run info from implementer & architect ────────
        self._gather_run_info(state, arch, impl)

        # ── Phase 6: Tester ──────────────────────────────────────────────
        print_rule("pipeline · phase 5 · tester", style="dim magenta")
        all_messages.extend(self._phase_tester(state, tester))

        log.info(
            "Pipeline complete | total_messages=%d arch_plan_chars=%d "
            "findings_chars=%d test_results_chars=%d",
            len(all_messages),
            len(state.arch_plan),
            len(state.review_findings),
            len(state.test_results),
        )
        return all_messages

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_architect(
        self, user_task: str, state: PipelineState, arch: Agent
    ) -> list[Message]:
        """Architect produces the initial plan."""
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

    def _phase_tester(self, state: PipelineState, tester: Agent) -> list[Message]:
        """Tester runs the project, checks each acceptance criterion, reports results."""
        prompt = (
            f"## Run Instructions (from implementer)\n{state.run_instructions}\n\n"
            f"## Acceptance Criteria (from architect)\n{state.acceptance_criteria}\n\n"
            "Use run_shell to execute the relevant commands and verify each criterion. "
            "Do not run interactive programs — use unit tests or `python -c '...'` checks instead. "
            "Report PASS, FAIL, or MANUAL for each criterion with evidence from the command output."
        )
        state.test_history.append(Message(role="user", content=prompt))

        test_text, new_msgs = tester.run(state.test_history, rag_context=self.rag_context)
        state.test_history.extend(new_msgs)
        state.test_results = test_text

        log.info("Phase 5 (tester) complete | results_chars=%d", len(test_text))
        return new_msgs


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

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
