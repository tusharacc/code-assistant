"""
Orchestrator — decides whether to run single-agent, dual-agent debate, or full
4-persona pipeline mode, then drives the conversation.
"""
from __future__ import annotations

from ..config import config
from ..logger import get_logger
from ..ui.console import console, print_debate_separator, print_rule, print_info
from .base import Agent, Message
from .architect import make_architect
from .implementer import make_implementer

log = get_logger(__name__)


# Heuristic: words that suggest a complex coding task worth debating
_COMPLEX_SIGNALS = {
    "implement", "write", "create", "build", "design", "refactor",
    "architecture", "system", "module", "class", "algorithm", "optimize",
    "restructure", "rewrite",
}

_SIMPLE_SIGNALS = {
    "explain", "what", "why", "how does", "show me", "read", "list",
    "find", "search", "tell me", "describe",
}


def _is_complex_task(text: str) -> bool:
    """Rough heuristic: should we invoke dual-agent debate?"""
    lowered = text.lower()
    # If the request is short and looks explanatory, skip debate
    if any(lowered.startswith(s) for s in _SIMPLE_SIGNALS):
        return False
    # If it contains complexity signals, debate
    return any(signal in lowered for signal in _COMPLEX_SIGNALS)


class Orchestrator:
    """
    Drives one turn of the assistant.

    Parameters
    ----------
    history : list[Message]
        The full conversation history so far (excluding system prompts).
    debate_enabled : bool
        Whether dual-agent mode is allowed for complex tasks.
    pipeline_enabled : bool
        Whether 4-persona pipeline mode is allowed for complex tasks.
        Takes precedence over debate_enabled when True.
    rag_context : str | None
        Relevant code chunks retrieved from the RAG index, if any.
    """

    def __init__(
        self,
        history: list[Message],
        debate_enabled: bool | None = None,
        pipeline_enabled: bool | None = None,
        rag_context: str | None = None,
    ) -> None:
        self.history = history
        self.debate_enabled = debate_enabled if debate_enabled is not None else config.debate_enabled
        self.pipeline_enabled = pipeline_enabled if pipeline_enabled is not None else config.use_pipeline
        self.rag_context = rag_context
        self._architect: Agent | None = None
        self._implementer: Agent | None = None

    # Lazy-init agents so we don't pay startup cost on simple queries
    @property
    def architect(self) -> Agent:
        if self._architect is None:
            self._architect = make_architect()
        return self._architect

    @property
    def implementer(self) -> Agent:
        if self._implementer is None:
            self._implementer = make_implementer()
        return self._implementer

    def run(self, user_input: str) -> list[Message]:
        """
        Process one user turn. Returns new messages produced this turn
        (to be appended to history by the caller).

        Mode selection (in priority order):
          1. pipeline — if pipeline_enabled and complex task
          2. debate   — if debate_enabled and complex task
          3. single   — implementer only
        """
        is_complex = _is_complex_task(user_input)
        use_pipeline = self.pipeline_enabled and is_complex
        use_debate = not use_pipeline and self.debate_enabled and is_complex

        mode = "pipeline" if use_pipeline else ("debate" if use_debate else "single")
        log.info(
            "Turn start | mode=%s pipeline_enabled=%s debate_enabled=%s input_chars=%d",
            mode, self.pipeline_enabled, self.debate_enabled, len(user_input),
        )
        log.debug("USER INPUT | %s", user_input)

        if use_pipeline:
            return self._run_pipeline(user_input)
        elif use_debate:
            return self._run_debate([Message(role="user", content=user_input)])
        else:
            return self._run_single([Message(role="user", content=user_input)])

    # ------------------------------------------------------------------
    # Single-agent mode — implementer only
    # ------------------------------------------------------------------

    def _run_single(self, messages_this_turn: list[Message]) -> list[Message]:
        full_history = self.history + messages_this_turn
        _, new_msgs = self.implementer.run(full_history, rag_context=self.rag_context)
        return messages_this_turn + new_msgs

    # ------------------------------------------------------------------
    # Dual-agent debate mode
    # ------------------------------------------------------------------

    def _run_debate(self, messages_this_turn: list[Message]) -> list[Message]:
        """
        Debate flow:
          1. Architect proposes a plan (no tools).
          2. Implementer reviews and may push back.
          3. (Optional) Architect revises if there was pushback.
          4. Implementer executes the agreed plan using tools.
        """
        print_rule("dual-agent debate", style="dim cyan")
        log.info("Debate mode activated")

        all_new: list[Message] = list(messages_this_turn)
        full_history = self.history + messages_this_turn

        # ── Round 1: Architect proposes ──────────────────────────────
        log.debug("Debate round 1: architect proposing plan")
        arch_prompt = (
            "Please analyse this task and propose a detailed implementation plan. "
            "Do NOT write the full code yet — focus on design decisions."
        )
        arch_input = Message(role="user", content=arch_prompt)
        arch_msgs = full_history + [arch_input]

        _, arch_response_msgs = self.architect.run(arch_msgs, rag_context=self.rag_context)
        arch_plan = arch_response_msgs[0].content if arch_response_msgs else ""
        all_new.extend(arch_response_msgs)

        print_debate_separator()

        # ── Round 2: Implementer reviews ────────────────────────────
        review_prompt = (
            f"The architect proposed this plan:\n\n{arch_plan}\n\n"
            "Review it critically. Do you agree? Any corrections or improvements? "
            "Reply with AGREE if you are satisfied, or REVISE: <your critique> if not."
        )
        impl_review_msgs = full_history + [
            Message(role="user", content=messages_this_turn[-1].content if messages_this_turn else ""),
            Message(role="assistant", content=arch_plan),
            Message(role="user", content=review_prompt),
        ]

        _, impl_review_response = self.implementer.run(
            impl_review_msgs, rag_context=self.rag_context, silent=False
        )
        review_text = impl_review_response[0].content if impl_review_response else ""
        all_new.extend(impl_review_response)

        needs_revision = "REVISE" in review_text.upper()
        log.info(
            "Debate round 2 complete | implementer_verdict=%s",
            "REVISE" if needs_revision else "AGREE",
        )

        if needs_revision and config.debate_rounds >= 2:
            print_debate_separator()
            log.debug("Debate round 3: architect revising plan")

            # ── Round 3: Architect revises ───────────────────────────
            revision_prompt = (
                f"The implementer raised these concerns:\n\n{review_text}\n\n"
                "Please revise your plan to address them."
            )
            arch_revision_msgs = full_history + [
                Message(role="user", content=messages_this_turn[-1].content if messages_this_turn else ""),
                Message(role="assistant", content=arch_plan),
                Message(role="user", content=revision_prompt),
            ]

            _, arch_revision_response = self.architect.run(
                arch_revision_msgs, rag_context=self.rag_context
            )
            revised_plan = (
                arch_revision_response[0].content if arch_revision_response else arch_plan
            )
            all_new.extend(arch_revision_response)
            final_plan = revised_plan
        else:
            final_plan = arch_plan

        print_debate_separator()
        print_rule("implementing", style="dim green")
        log.info("Debate complete — implementer executing final plan")

        # ── Final: Implementer executes ──────────────────────────────
        execute_prompt = (
            f"The agreed plan is:\n\n{final_plan}\n\n"
            f"Original task: {messages_this_turn[-1].content if messages_this_turn else ''}\n\n"
            "Now implement it completely. Use your tools to create and modify files."
        )
        execute_msgs = full_history + [Message(role="user", content=execute_prompt)]
        _, execute_response = self.implementer.run(
            execute_msgs, rag_context=self.rag_context
        )
        all_new.extend(execute_response)

        return all_new

    # ------------------------------------------------------------------
    # 4-persona pipeline mode
    # ------------------------------------------------------------------

    def _run_pipeline(self, user_input: str) -> list[Message]:
        """Delegate to the Pipeline class for the full 4-persona flow."""
        from .pipeline import Pipeline

        log.info("Pipeline mode activated")
        pipeline = Pipeline(rag_context=self.rag_context)
        return pipeline.run(user_input)
