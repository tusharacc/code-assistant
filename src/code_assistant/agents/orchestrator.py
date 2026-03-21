"""
Orchestrator — decides whether to run single-agent, dual-agent debate, or full
4-persona pipeline mode, then drives the conversation.
"""
from __future__ import annotations

import re

from ..config import config
from ..logger import get_logger
from ..ui.console import console, print_debate_separator, print_rule, print_info
from .base import Agent, Message
from .architect import make_architect
from .implementer import make_implementer

log = get_logger(__name__)

# Same Q&A tag the implementer uses to ask the architect a question
_QA_TAG = re.compile(r"@@QUESTION_FOR_ARCHITECT:\s*(.*?)@@", re.DOTALL)
_MAX_QA_ROUNDS = 5


_INTENT_SYSTEM = (
    "Classify this coding-assistant request into exactly one word:\n"
    "  conversational — questions, explanations, or analysis of existing code\n"
    "  implementation — writing, fixing, or editing code or files\n"
    "  complex        — large new systems, complete features, or major refactors\n"
    "Reply with only one word: conversational, implementation, or complex"
)


def _classify_intent(text: str) -> str:
    """
    LLM-based intent classification. Returns one of:
      'conversational' — route to Architect (Q&A, fast 7B, no tools)
      'implementation' — route to Implementer (code gen, 14B, tools)
      'complex'        — route to debate or pipeline

    Falls back to 'implementation' on any error so the REPL never breaks.
    """
    import ollama as _ollama
    try:
        resp = _ollama.chat(
            model=config.effective_classification_model(),
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user",   "content": text},
            ],
            stream=False,
            options={"temperature": 0.0, "num_predict": 10},
        )
        raw = resp.message.content.strip().lower().split()[0].rstrip(".,:")
        for valid in ("conversational", "implementation", "complex"):
            if raw.startswith(valid[:6]):
                log.debug("Intent classified | label=%s input=%s", valid, text[:80])
                return valid
        log.warning("Intent classification | unexpected label=%r, defaulting to implementation", raw)
        return "implementation"
    except Exception as e:
        log.warning("Intent classification failed (%s), defaulting to implementation", e)
        return "implementation"


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
          1. pipeline       — if pipeline_enabled and complex task
          2. debate         — if debate_enabled and complex task
          3. conversational — question / explanation request → Architect
          4. single         — implementation task → Implementer
        """
        intent = _classify_intent(user_input)

        is_complex         = intent == "complex"
        # Pipeline runs for any non-conversational task when explicitly enabled.
        # The intent classifier uses "complex" for heavy multi-file work and
        # "implementation" for simpler tasks — both belong in the pipeline.
        # Only pure Q&A ("conversational") bypasses it so quick questions stay fast.
        use_pipeline       = self.pipeline_enabled and intent != "conversational"
        use_debate         = not use_pipeline and self.debate_enabled and is_complex
        use_conversational = intent == "conversational" and not use_pipeline and not use_debate

        if use_pipeline:         mode = "pipeline"
        elif use_debate:         mode = "debate"
        elif use_conversational: mode = "conversational"
        else:                    mode = "single"

        log.info(
            "Turn start | mode=%s pipeline_enabled=%s debate_enabled=%s input_chars=%d",
            mode, self.pipeline_enabled, self.debate_enabled, len(user_input),
        )
        log.debug("USER INPUT | %s", user_input)

        if use_pipeline:
            return self._run_pipeline(user_input)
        elif use_debate:
            return self._run_debate([Message(role="user", content=user_input)])
        elif use_conversational:
            return self._run_architect_only([Message(role="user", content=user_input)])
        else:
            return self._run_single([Message(role="user", content=user_input)])

    # ------------------------------------------------------------------
    # Conversational mode — architect only, no tools
    # ------------------------------------------------------------------

    def _run_architect_only(self, messages_this_turn: list[Message]) -> list[Message]:
        """Handle informational / Q&A queries with the Architect persona.

        The Architect is faster (7B), uses no tools, and its system prompt is
        tuned for reasoning and explanation rather than code generation.
        """
        working_history = self.history + messages_this_turn
        _, new_msgs = self.architect.run(working_history, rag_context=self.rag_context)
        return messages_this_turn + new_msgs

    # ------------------------------------------------------------------
    # Single-agent mode — implementer only
    # ------------------------------------------------------------------

    def _run_single(self, messages_this_turn: list[Message]) -> list[Message]:
        """
        Single-agent mode. If the implementer emits @@QUESTION_FOR_ARCHITECT:...@@
        the question is routed to the architect, whose answer is fed back so the
        implementer can continue. The architect is always available.
        """
        all_new: list[Message] = []
        working_history = self.history + messages_this_turn
        arch_history: list[Message] = list(self.history)   # separate context for architect

        for round_num in range(_MAX_QA_ROUNDS + 1):
            impl_text, new_msgs = self.implementer.run(
                working_history, rag_context=self.rag_context
            )
            working_history.extend(new_msgs)
            all_new.extend(new_msgs)

            m = _QA_TAG.search(impl_text)
            if not m:
                break   # clean response — done

            question = m.group(1).strip()
            log.info(
                "Single mode Q&A round %d — routing to architect | q=%s",
                round_num + 1, question[:120],
            )
            console.print(
                f"\n[dim cyan]↳ Implementer asks architect: {question[:120]}…[/dim cyan]"
            )

            arch_q = Message(
                role="user",
                content=(
                    f"The implementer is asking: {question}\n\n"
                    "Please answer concisely and directly."
                ),
            )
            arch_history.append(arch_q)
            arch_ans_text, arch_ans_msgs = self.architect.run(
                arch_history, rag_context=self.rag_context
            )
            arch_history.extend(arch_ans_msgs)

            console.print(
                f"[dim cyan]↳ Architect answers: {arch_ans_text[:120]}…[/dim cyan]\n"
            )

            working_history.append(Message(
                role="user",
                content=(
                    f"Architect's answer: {arch_ans_text}\n\n"
                    "Please continue implementing."
                ),
            ))
        else:
            log.warning("Single mode: Q&A rounds exceeded %d, stopping", _MAX_QA_ROUNDS)

        return messages_this_turn + all_new

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

        # ── Final: Implementer executes (with Q&A routing to architect) ──
        execute_prompt = (
            f"The agreed plan is:\n\n{final_plan}\n\n"
            f"Original task: {messages_this_turn[-1].content if messages_this_turn else ''}\n\n"
            "Now implement it completely. Use your tools to create and modify files."
        )
        exec_history = full_history + [Message(role="user", content=execute_prompt)]
        arch_history_exec = list(full_history)   # separate copy for architect Q&A

        for round_num in range(_MAX_QA_ROUNDS + 1):
            impl_text, exec_response = self.implementer.run(
                exec_history, rag_context=self.rag_context
            )
            exec_history.extend(exec_response)
            all_new.extend(exec_response)

            m = _QA_TAG.search(impl_text)
            if not m:
                break   # clean — implementation done

            question = m.group(1).strip()
            log.info(
                "Debate execute Q&A round %d | question=%s", round_num + 1, question[:120]
            )
            console.print(
                f"\n[dim cyan]↳ Implementer asks architect: {question[:120]}…[/dim cyan]"
            )

            arch_q = Message(
                role="user",
                content=(
                    f"During implementation the implementer is asking: {question}\n\n"
                    "Please answer concisely and directly."
                ),
            )
            arch_history_exec.append(arch_q)
            arch_ans_text, arch_ans_msgs = self.architect.run(
                arch_history_exec, rag_context=self.rag_context
            )
            arch_history_exec.extend(arch_ans_msgs)

            console.print(
                f"[dim cyan]↳ Architect answers: {arch_ans_text[:120]}…[/dim cyan]\n"
            )

            exec_history.append(Message(
                role="user",
                content=(
                    f"Architect's answer: {arch_ans_text}\n\n"
                    "Please continue implementing."
                ),
            ))
        else:
            log.warning("Debate execute Q&A exceeded %d rounds, stopping", _MAX_QA_ROUNDS)

        return all_new

    # ------------------------------------------------------------------
    # 4-persona pipeline mode
    # ------------------------------------------------------------------

    def _run_pipeline(self, user_input: str) -> list[Message]:
        """Delegate to the Pipeline class for the full 4-persona flow."""
        from .pipeline import Pipeline

        log.info("Pipeline mode activated")
        pipeline = Pipeline(
            rag_context=self.rag_context,
            initial_history=self.history,   # carries loaded req-file into architect context
        )
        return pipeline.run(user_input)
