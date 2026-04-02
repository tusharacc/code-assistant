"""
4-persona pipeline:
  Architect → Implementer → Reviewer → Implementer (fix) → Tester [→ Implementer (fix) → Tester]*
  → Documentation writer.

Each persona runs with keep_alive=0 so Ollama unloads the model from RAM
immediately after every response. Only one model is in memory at a time.
Per-persona conversation histories are maintained across phases for coherent context.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field

from ..config import config
from ..logger import get_logger
from ..ui.console import console, print_rule, print_info, print_warning, print_error
from .base import Agent, Message
from .architect import make_architect
from .implementer import make_implementer
from .reviewer import make_reviewer
from .tester import make_tester
from .verifier import (
    verify_phase,
    print_verification,
    write_pipeline_artifact,
    verify_artifact,
    print_artifact_verification,
)

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

# Instruction appended to every phase prompt requiring a ## Handoff block.
# Each agent must call compute_file_sha256 for every file it writes, then
# list all (path, sha256) pairs in the ## Handoff section.
_HANDOFF_INSTRUCTION = (
    "\n\n---\n"
    "**MANDATORY — final step before finishing:**\n"
    "For every file you wrote or edited, call compute_file_sha256(path) to obtain its SHA-256.\n"
    "Then end your response with EXACTLY this block (no other text after it):\n\n"
    "## Handoff\n"
    "- /absolute/path/to/file1.ext sha256:<hash>\n"
    "- /absolute/path/to/file2.ext sha256:<hash>\n\n"
    "If you wrote no files, write:\n"
    "## Handoff\n"
    "- (no files written)\n"
)

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

# Symlink pointing at the most-recent pipeline artifacts dir
_LATEST_LINK = ".ca_pipeline/latest"

# Max rounds of reviewer → phantom-fix → re-review
_MAX_PHANTOM_FIX_ROUNDS = 2

# Patterns the reviewer uses when it can't find a file
_PHANTOM_PATTERNS = re.compile(
    r"(?:"
    r"not found|does not exist|missing file|file missing|"
    r"no such file|cannot find|couldn't find|could not find|"
    r"phantom|doesn't exist|file not found"
    r")",
    re.IGNORECASE,
)


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
        resume: bool = False,
    ) -> None:
        self.rag_context = rag_context
        self.initial_history: list[Message] = initial_history or []
        self.resume = resume
        # Populated by run() — keyed by phase name, plus "elapsed_total"
        self.metrics: dict = {}
        # Preserved after run() for feedback collection; None before first run
        self.last_state: PipelineState | None = None

    def _find_latest_artifacts(self):
        """Return the most recent .ca_pipeline/<ts>/ dir, or None."""
        from pathlib import Path as _Path
        base = _Path.cwd() / ".ca_pipeline"
        latest = base / "latest"
        if latest.is_symlink() and latest.resolve().is_dir():
            return latest.resolve()
        # Fallback: scan for newest timestamped dir
        if base.is_dir():
            dirs = sorted(
                [d for d in base.iterdir() if d.is_dir() and d.name != "latest"],
                key=lambda d: d.name,
                reverse=True,
            )
            return dirs[0] if dirs else None
        return None

    def _detect_resume_point(self, prev_dir) -> tuple[int, "PipelineState"]:
        """Inspect prev_dir artifacts and return (resume_from_phase, reconstructed_state).

        resume_from_phase is the first phase that is NOT done (1-7), or 8 if all done.
        """
        from pathlib import Path as _Path

        state = PipelineState()

        def _read(fname: str) -> str:
            try:
                p = _Path(prev_dir) / fname
                if p.exists():
                    return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
            return ""

        def _latest_test_results() -> str:
            """Return content of the highest-numbered 07_test_results_rN.md."""
            try:
                candidates = sorted(
                    [f for f in _Path(prev_dir).iterdir()
                     if f.name.startswith("07_test_results_r") and f.suffix == ".md"],
                    key=lambda f: f.name,
                    reverse=True,
                )
                if candidates:
                    return candidates[0].read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
            return ""

        def _manifest_files_exist(manifest_content: str) -> bool:
            """Check that at least one path listed in an impl manifest exists on disk."""
            if not manifest_content.strip():
                return False
            for line in manifest_content.splitlines():
                line = line.strip()
                if line.startswith("- ") and " sha256:" in line:
                    # Format: "- /path/to/file.py sha256:abc123"
                    path_part = line[2:].split(" sha256:")[0].strip()
                    if path_part and _Path(path_part).exists():
                        return True
            return False

        # --- Phase 1 done? ---
        arch_plan = _read("01_arch_plan.md")
        if not arch_plan.strip():
            return 1, state

        state.arch_plan = arch_plan
        state.arch_history = [
            Message(role="user", content="[Resumed] Architect plan loaded from previous run."),
            Message(role="assistant", content=arch_plan),
        ]

        # --- Phase 2 done? ---
        impl_manifest = _read("02_impl_manifest.md")
        if not impl_manifest.strip() or not _manifest_files_exist(impl_manifest):
            return 2, state

        # Build a synthetic impl_history listing what was written
        manifest_lines = [
            line.strip()[2:].split(" sha256:")[0].strip()
            for line in impl_manifest.splitlines()
            if line.strip().startswith("- ") and " sha256:" in line
        ]
        files_summary = "\n".join(f"  {p}" for p in manifest_lines) if manifest_lines else "  (none listed)"
        state.impl_history = [
            Message(
                role="user",
                content=f"[Resumed] Previous implementer run. Files written:\n{files_summary}",
            ),
            Message(role="assistant", content="Understood, implementation from previous run loaded."),
        ]

        # --- Phase 3 done? ---
        review_findings = _read("03_review_findings.md")
        if not review_findings.strip():
            return 3, state
        # Check for phantom-file mentions — if reviewer couldn't find files, phase 3 is not clean
        if _PHANTOM_PATTERNS.search(review_findings):
            return 3, state

        state.review_findings = review_findings

        # --- Phase 4 done? ---
        fix_manifest = _read("04_fix_manifest.md")
        if not fix_manifest.strip():
            return 4, state

        # --- Phase 5 done? ---
        run_instructions = _read("05_run_instructions.md")
        acceptance_criteria = _read("06_acceptance_criteria.md")
        if not run_instructions.strip() or not acceptance_criteria.strip():
            return 5, state

        state.run_instructions = run_instructions
        state.acceptance_criteria = acceptance_criteria

        # --- Phase 6 done? ---
        test_results = _latest_test_results()
        if not test_results.strip():
            return 6, state
        # Only consider phase 6 done if the latest test run PASSED
        if _VERDICT_FAIL.search(test_results):
            state.test_results = test_results
            return 6, state

        state.test_results = test_results

        # --- Phase 7 done? ---
        doc_output = _read("08_doc_output.md")
        if not doc_output.strip():
            return 7, state

        state.doc_output = doc_output

        # All phases done
        return 8, state

    def run(self, user_task: str) -> list[Message]:
        """Execute all pipeline phases. Returns all produced messages."""
        state = PipelineState()
        all_messages: list[Message] = [Message(role="user", content=user_task)]

        log.info("Pipeline start | task_chars=%d", len(user_task))

        # Artifact directory — one per pipeline run, named by timestamp
        from pathlib import Path as _Path
        import time as _time
        _run_ts = _time.strftime("%Y%m%d_%H%M%S")
        artifacts_dir = _Path.cwd() / ".ca_pipeline" / _run_ts
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        log.info("Pipeline artifacts dir: %s", artifacts_dir)

        # ── Resume logic ─────────────────────────────────────────────────
        resume_from = 0
        if self.resume:
            prev_dir = self._find_latest_artifacts()
            if prev_dir and prev_dir != artifacts_dir:
                resume_from, state = self._detect_resume_point(prev_dir)
                if resume_from > 1:
                    print_info(
                        f"Resuming from phase {resume_from} "
                        f"(previous run: {prev_dir.name})"
                    )
                    log.info(
                        "Pipeline resume | from_phase=%d prev_dir=%s",
                        resume_from, prev_dir,
                    )
                else:
                    print_info("No completed phases found in previous run — starting fresh.")
            else:
                print_info("No previous run found — starting fresh.")

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

        # ── Pre-pipeline git checkpoint ──────────────────────────────────
        # Snapshot all uncommitted changes before the pipeline modifies any files.
        # If the pipeline makes bad changes, `git reset --hard HEAD~1` restores
        # the exact state the workspace was in before this run.
        try:
            import subprocess as _sp
            _git_status = _sp.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=str(_Path.cwd()),
            )
            if _git_status.returncode == 0 and _git_status.stdout.strip():
                # There are uncommitted changes — snapshot them
                _sp.run(["git", "add", "-A"], cwd=str(_Path.cwd()), capture_output=True)
                _commit_msg = (
                    f"ca: pre-pipeline checkpoint\n\n"
                    f"Task: {user_task[:120]}\n"
                    f"Run: {_run_ts}\n\n"
                    f"Auto-committed by ca pipeline before phase 1.\n"
                    f"To revert: git reset --hard HEAD~1"
                )
                _commit_result = _sp.run(
                    ["git", "commit", "-m", _commit_msg],
                    capture_output=True, text=True, cwd=str(_Path.cwd()),
                )
                if _commit_result.returncode == 0:
                    log.info("Pre-pipeline git checkpoint created — revert: git reset --hard HEAD~1")
                    print_info("Git checkpoint created (revert: git reset --hard HEAD~1)")
                else:
                    log.warning(
                        "Pre-pipeline git commit failed (non-fatal): %s",
                        _commit_result.stderr.strip()
                    )
            elif _git_status.returncode == 0:
                log.info("Pre-pipeline git checkpoint: nothing to commit (clean working tree)")
            else:
                log.warning("Pre-pipeline git checkpoint: git not available or not a repo (non-fatal)")
        except Exception as _git_err:
            log.warning("Pre-pipeline git checkpoint failed (non-fatal): %s", _git_err)

        t0 = time.perf_counter()

        # ── Phase 1: Architect ───────────────────────────────────────────
        if resume_from <= 1:
            print_rule("pipeline · phase 1 · architect", style="dim cyan")
            t = time.perf_counter()
            before = _snap(arch)
            all_messages.extend(self._phase_architect(user_task, state, arch))
            self.metrics["architect"] = _phase_metrics(arch, before, time.perf_counter() - t)

            # Artifact: arch plan (pipeline writes it — architect has no tools)
            _art_path, _art_sha = write_pipeline_artifact(
                artifacts_dir, "01_arch_plan.md", state.arch_plan
            )
            _art_ok = verify_artifact("architect", _art_path, _art_sha, "arch_plan")
            print_artifact_verification("architect", _art_path, _art_ok, "01_arch_plan.md")
            if not _art_ok or not state.arch_plan.strip():
                print_error("Pipeline halted: architect produced no plan.")
                return all_messages
        else:
            print_info("↷ Phase 1 (architect) — skipped (loaded from previous run)")
            log.info("Resume: skipping phase 1")

        # ── Phase 2: Implementer (with optional Q&A with architect) ──────
        if resume_from <= 2:
            print_rule("pipeline · phase 2 · implementer", style="dim green")
            t = time.perf_counter()
            before_impl = _snap(impl)
            before_arch = _snap(arch)  # arch may be called during Q&A
            impl_msgs = self._phase_implementer(user_task, state, arch, impl)
            all_messages.extend(impl_msgs)
            self.metrics["implementer"] = _phase_metrics(impl, before_impl, time.perf_counter() - t)
            self.metrics["implementer_qa_arch"] = _phase_metrics(arch, before_arch, 0)

            # ── Gate: SHA-256 verify implementer handoff ─────────────────────
            _impl_verify = verify_phase("implementer", state.impl_history)
            print_verification(_impl_verify)

            # Artifact: impl manifest (paths + SHAs of every file claimed)
            _impl_manifest = _build_manifest("implementer", _impl_verify)
            _art_path, _art_sha = write_pipeline_artifact(
                artifacts_dir, "02_impl_manifest.md", _impl_manifest
            )
            _art_ok = verify_artifact("implementer", _art_path, _art_sha, "impl_manifest")
            print_artifact_verification("implementer", _art_path, _art_ok, "02_impl_manifest.md")

            if not _impl_verify.records:
                print_error(
                    "Pipeline halted: implementer made no write_file/edit_file calls. "
                    "Review and test phases require actual code on disk."
                )
                log.error("Pipeline halted: no file writes detected in impl_history")
                return all_messages
            if not _impl_verify.passed:
                print_error(
                    f"Pipeline halted: {len(_impl_verify.missing)} file(s) missing, "
                    f"{len(_impl_verify.mismatched)} SHA mismatch(es). "
                    "Reviewer cannot review phantom files."
                )
                log.error(
                    "Pipeline halted: verification failed | missing=%s mismatched=%s",
                    _impl_verify.missing, _impl_verify.mismatched,
                )
                return all_messages

            log.info("Implementer gate passed | files=%d", _impl_verify.file_count)

            # Post-implementer git snapshot — captures what was written before the
            # reviewer/fix phases run. Enables: git diff HEAD~1 to see what impl wrote.
            try:
                import subprocess as _sp2
                _sp2.run(["git", "add", "-A"], cwd=str(_Path.cwd()), capture_output=True)
                _sp2.run(
                    ["git", "commit", "-m",
                     f"ca: phase 2 implementer output\n\nTask: {user_task[:100]}\nRun: {_run_ts}"],
                    capture_output=True, cwd=str(_Path.cwd()),
                )
                log.info("Post-implementer git snapshot committed")
            except Exception as _gce:
                log.debug("Post-implementer git snapshot skipped: %s", _gce)

        else:
            print_info("↷ Phase 2 (implementer) — skipped (files already on disk)")
            log.info("Resume: skipping phase 2")

        # Re-index AST + refresh RAG so reviewer sees the files just written.
        self._reindex_and_refresh(user_task, state)

        # ── Phase 3: Reviewer ────────────────────────────────────────────
        if resume_from <= 3:
            print_rule("pipeline · phase 3 · reviewer", style="dim yellow")
            t = time.perf_counter()
            before = _snap(reviewer)
            all_messages.extend(self._phase_reviewer(user_task, state, reviewer))
            self.metrics["reviewer"] = _phase_metrics(reviewer, before, time.perf_counter() - t)

            # Artifact: review findings
            _art_path, _art_sha = write_pipeline_artifact(
                artifacts_dir, "03_review_findings.md", state.review_findings
            )
            _art_ok = verify_artifact("reviewer", _art_path, _art_sha, "review_findings")
            print_artifact_verification("reviewer", _art_path, _art_ok, "03_review_findings.md")
            if not _art_ok or not state.review_findings.strip():
                print_warning("Reviewer produced no findings — continuing with empty review.")

            # ── Phase 3b: Phantom-file fix loop ──────────────────────────────
            # If the reviewer found files that don't exist on disk, route back to
            # the implementer before proceeding to the quality-fix phase.
            # Max _MAX_PHANTOM_FIX_ROUNDS rounds to prevent infinite loops.
            for phantom_round in range(_MAX_PHANTOM_FIX_ROUNDS):
                phantoms = _parse_phantom_files(state.review_findings)
                if not phantoms:
                    break

                print_rule(
                    f"pipeline · phase 3b · phantom fix · round {phantom_round + 1}",
                    style="dim red",
                )
                console.print(
                    f"[yellow]Reviewer found {len(phantoms)} missing file(s) — "
                    f"routing back to implementer:[/yellow]"
                )
                for pf in phantoms:
                    console.print(f"  [dim red]✗ {pf}[/dim red]")

                t = time.perf_counter()
                before = _snap(impl)
                pfix_msgs = self._phase_phantom_fix(state, phantoms, impl)
                all_messages.extend(pfix_msgs)
                key = f"phantom_fix_round{phantom_round + 1}"
                self.metrics[key] = _phase_metrics(impl, before, time.perf_counter() - t)

                # Verify the phantom fix wrote the files
                _pfix_verify = verify_phase(key, pfix_msgs)
                print_verification(_pfix_verify)
                _pfix_manifest = _build_manifest(key, _pfix_verify)
                _art_path, _art_sha = write_pipeline_artifact(
                    artifacts_dir,
                    f"03b_phantom_fix_manifest_r{phantom_round + 1}.md",
                    _pfix_manifest,
                )
                verify_artifact(key, _art_path, _art_sha, f"phantom_fix_manifest_r{phantom_round + 1}")

                if not _pfix_verify.records:
                    print_warning(
                        f"Phantom fix round {phantom_round + 1}: implementer still wrote "
                        "no files — stopping phantom-fix loop."
                    )
                    log.warning("Phantom fix produced no writes in round %d", phantom_round + 1)
                    break

                # Re-run reviewer on the now-complete set of files
                print_rule(
                    f"pipeline · phase 3b · re-review · round {phantom_round + 1}",
                    style="dim yellow",
                )
                t = time.perf_counter()
                before = _snap(reviewer)
                all_messages.extend(self._phase_reviewer(user_task, state, reviewer))
                self.metrics[f"reviewer_recheck_round{phantom_round + 1}"] = _phase_metrics(
                    reviewer, before, time.perf_counter() - t
                )
                _art_path, _art_sha = write_pipeline_artifact(
                    artifacts_dir,
                    f"03b_review_recheck_r{phantom_round + 1}.md",
                    state.review_findings,
                )
                verify_artifact(
                    f"reviewer_recheck_round{phantom_round + 1}",
                    _art_path, _art_sha, "review_findings_recheck",
                )
            else:
                # Exhausted phantom-fix rounds — warn and continue anyway
                remaining = _parse_phantom_files(state.review_findings)
                if remaining:
                    print_warning(
                        f"Phantom-fix limit ({_MAX_PHANTOM_FIX_ROUNDS} rounds) reached. "
                        f"{len(remaining)} file(s) still missing — continuing to quality-fix phase."
                    )
                    log.warning(
                        "Phantom-fix limit reached | still_missing=%s", remaining
                    )
        else:
            print_info("↷ Phase 3 (reviewer) — skipped (loaded from previous run)")
            log.info("Resume: skipping phase 3")

        # ── Phase 4: Implementer fix (HIGH + MEDIUM issues only) ─────────
        if resume_from <= 4:
            high, medium = _parse_findings(state.review_findings)
            if high or medium:
                print_rule("pipeline · phase 4 · implementer fix", style="dim green")
                t = time.perf_counter()
                before = _snap(impl)
                fix_msgs = self._phase_fix(state, high, medium, impl)
                all_messages.extend(fix_msgs)
                self.metrics["implementer_fix"] = _phase_metrics(impl, before, time.perf_counter() - t)

                # Verify the fix + artifact
                _fix_verify = verify_phase("implementer_fix", fix_msgs)
                print_verification(_fix_verify)
                _fix_manifest = _build_manifest("implementer_fix", _fix_verify)
                _art_path, _art_sha = write_pipeline_artifact(
                    artifacts_dir, "04_fix_manifest.md", _fix_manifest
                )
                _art_ok = verify_artifact("implementer_fix", _art_path, _art_sha, "fix_manifest")
                print_artifact_verification("implementer_fix", _art_path, _art_ok, "04_fix_manifest.md")
                if _fix_verify.records and not _fix_verify.passed:
                    print_warning(
                        f"Fix verification: {len(_fix_verify.missing)} file(s) missing, "
                        f"{len(_fix_verify.mismatched)} SHA mismatch(es) — "
                        "continuing to test phase with possibly incomplete fixes."
                    )
                    log.warning(
                        "Fix phase verification failed (non-halting) | missing=%s mismatched=%s",
                        _fix_verify.missing, _fix_verify.mismatched,
                    )
                # Post-fix git snapshot — captures what the fix phase changed
                try:
                    import subprocess as _sp3
                    _sp3.run(["git", "add", "-A"], cwd=str(_Path.cwd()), capture_output=True)
                    _sp3.run(
                        ["git", "commit", "-m",
                         f"ca: phase 4 fix output\n\nTask: {user_task[:100]}\nRun: {_run_ts}"],
                        capture_output=True, cwd=str(_Path.cwd()),
                    )
                    log.info("Post-fix git snapshot committed")
                except Exception as _gce2:
                    log.debug("Post-fix git snapshot skipped: %s", _gce2)
            else:
                print_info("No HIGH/MEDIUM issues — skipping fix phase.")
                log.info("Pipeline: no HIGH/MEDIUM findings, skipping fix phase")
        else:
            print_info("↷ Phase 4 (implementer fix) — skipped (loaded from previous run)")
            log.info("Resume: skipping phase 4")

        # Re-index AST + refresh RAG so tester sees any files fixed in phase 4.
        # Reset test_history so the fresh AST outline is injected cleanly.
        state.test_history = []
        self._reindex_and_refresh(user_task, state)

        # ── Phase 5: Gather run info from implementer & architect ────────
        if resume_from <= 5:
            self._gather_run_info(state, arch, impl)

            # Artifacts: run instructions + acceptance criteria
            for _fname, _content, _label in [
                ("05_run_instructions.md", state.run_instructions, "run_instructions"),
                ("06_acceptance_criteria.md", state.acceptance_criteria, "acceptance_criteria"),
            ]:
                _art_path, _art_sha = write_pipeline_artifact(artifacts_dir, _fname, _content)
                _art_ok = verify_artifact("run_info", _art_path, _art_sha, _label)
                print_artifact_verification("run_info", _art_path, _art_ok, _fname)
        else:
            print_info("↷ Phase 5 (run info) — skipped (loaded from previous run)")
            log.info("Resume: skipping phase 5")

        # ── Phase 6+: Tester with fix-loop ───────────────────────────────
        if resume_from > 6:
            print_info("↷ Phase 6 (tester) — skipped (previous run passed all criteria)")
            log.info("Resume: skipping phase 6")
        for test_round in range(_MAX_TEST_FIX_ROUNDS if resume_from <= 6 else 0):
            round_label = f"pipeline · tester · round {test_round + 1}/{_MAX_TEST_FIX_ROUNDS}"
            print_rule(round_label, style="dim magenta")
            t = time.perf_counter()
            before = _snap(tester)
            all_messages.extend(self._phase_tester(state, tester, round_num=test_round))
            key = f"tester_round{test_round + 1}"
            self.metrics[key] = _phase_metrics(tester, before, time.perf_counter() - t)

            # Artifact: tester results for this round
            _art_path, _art_sha = write_pipeline_artifact(
                artifacts_dir,
                f"07_test_results_r{test_round + 1}.md",
                state.test_results,
            )
            _art_ok = verify_artifact(
                f"tester_round{test_round + 1}", _art_path, _art_sha,
                f"test_results_r{test_round + 1}",
            )
            print_artifact_verification(
                f"tester_round{test_round + 1}", _art_path, _art_ok,
                f"07_test_results_r{test_round + 1}.md",
            )

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
                tfix_msgs = self._phase_test_fix(state, failures, impl)
                all_messages.extend(tfix_msgs)
                self.metrics[f"test_fix_round{test_round + 1}"] = _phase_metrics(
                    impl, before, time.perf_counter() - t
                )

                # Verify test-fix writes + artifact
                _tfix_verify = verify_phase(f"test_fix_round{test_round + 1}", tfix_msgs)
                print_verification(_tfix_verify)
                _tfix_manifest = _build_manifest(f"test_fix_r{test_round + 1}", _tfix_verify)
                _art_path, _art_sha = write_pipeline_artifact(
                    artifacts_dir,
                    f"04b_test_fix_manifest_r{test_round + 1}.md",
                    _tfix_manifest,
                )
                _art_ok = verify_artifact(
                    f"test_fix_round{test_round + 1}", _art_path, _art_sha,
                    f"test_fix_manifest_r{test_round + 1}",
                )
                print_artifact_verification(
                    f"test_fix_round{test_round + 1}", _art_path, _art_ok,
                    f"04b_test_fix_manifest_r{test_round + 1}.md",
                )
                if _tfix_verify.records and not _tfix_verify.passed:
                    print_warning(
                        f"Test-fix verification: {len(_tfix_verify.missing)} file(s) missing, "
                        f"{len(_tfix_verify.mismatched)} SHA mismatch(es)."
                    )
                    log.warning(
                        "Test-fix round %d verification failed | missing=%s mismatched=%s",
                        test_round + 1, _tfix_verify.missing, _tfix_verify.mismatched,
                    )
            else:
                print_warning(
                    f"Max test-fix rounds ({_MAX_TEST_FIX_ROUNDS}) reached — "
                    f"{len(failures)} criterion/criteria still failing."
                )

        # ── Phase 7: Documentation writer ────────────────────────────────
        if resume_from <= 7:
            print_rule("pipeline · phase 7 · documentation", style="dim blue")
            t = time.perf_counter()
            before = _snap(impl)
            doc_msgs = self._phase_docs(user_task, state, impl)
            all_messages.extend(doc_msgs)
            self.metrics["docs"] = _phase_metrics(impl, before, time.perf_counter() - t)
        else:
            print_info("↷ Phase 7 (documentation) — skipped (loaded from previous run)")
            log.info("Resume: skipping phase 7")

        # Artifact: doc output + verify README written to disk
        if resume_from <= 7:
            _art_path, _art_sha = write_pipeline_artifact(
                artifacts_dir, "08_doc_output.md", state.doc_output
            )
            _art_ok = verify_artifact("docs", _art_path, _art_sha, "doc_output")
            print_artifact_verification("docs", _art_path, _art_ok, "08_doc_output.md")
            _docs_verify = verify_phase("docs", doc_msgs)
            print_verification(_docs_verify)

        self.metrics["elapsed_total"] = time.perf_counter() - t0

        # Print summary of all artifacts produced
        console.print(f"\n[dim]── pipeline artifacts → {artifacts_dir} ──[/dim]")

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

        # ── Update .ca_pipeline/latest symlink to this run's artifacts ───
        try:
            from pathlib import Path as _Path
            latest_link = _Path.cwd() / ".ca_pipeline" / "latest"
            if latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(artifacts_dir)
        except Exception as _e:
            log.warning("Could not update .ca_pipeline/latest symlink: %s", _e)

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

    def _reindex_and_refresh(self, user_task: str, state: PipelineState) -> None:
        """Re-index AST and RAG after a write phase so reviewers/testers see current files.

        Called after Phase 2 (implementer) and Phase 4 (fix) to ensure the reviewer
        and tester work with an up-to-date view of the codebase, not the pre-write snapshot.

        Updates:
          - self.rag_context    — re-queried with user_task; all subsequent agent.run() calls
                                  automatically get fresh retrieval context.
          - state.review_history — prepends AST outline if history is still empty.
          - state.test_history   — same.
        All errors are non-fatal — a failed re-index just means stale context, not a crash.
        """
        from pathlib import Path as _Path
        cwd = _Path.cwd()
        ast_outline = ""

        # 1. Re-index AST (SQLite — typically <1 s even for large codebases)
        try:
            from ..rag.ast_indexer import ASTIndexer
            from ..rag.ast_retriever import ASTRetriever
            _count = ASTIndexer().index_directory(cwd)
            _ast_r = ASTRetriever()
            if _ast_r.is_ready():
                ast_outline = _ast_r.get_outline() or ""
            log.info("Post-write AST re-index | symbols=%d outline_chars=%d", _count, len(ast_outline))
        except Exception as _e:
            log.warning("Post-write AST re-index failed (non-fatal): %s", _e)

        # 2. Re-query RAG (only if already indexed — no indexing is done here)
        try:
            from ..rag.retriever import CodebaseRetriever
            _r = CodebaseRetriever()
            if _r.is_ready():
                fresh = _r.query(user_task)
                if fresh:
                    self.rag_context = fresh
                    log.info("Post-write RAG context refreshed | chars=%d", len(fresh))
        except Exception as _e:
            log.warning("Post-write RAG refresh failed (non-fatal): %s", _e)

        # 3. Inject AST outline into upcoming phase histories (only if still empty)
        if ast_outline:
            _ast_ctx = [
                Message(
                    role="user",
                    content=(
                        "[Updated symbol map — reflects files written by implementer]\n\n"
                        + ast_outline
                    ),
                ),
                Message(
                    role="assistant",
                    content="Understood. I have the updated symbol map.",
                ),
            ]
            if not state.review_history:
                state.review_history = list(_ast_ctx)
            if not state.test_history:
                state.test_history = list(_ast_ctx)

        rag_status = f"{len(self.rag_context):,} chars" if self.rag_context else "not indexed"
        ast_status = f"{len(ast_outline):,} chars" if ast_outline else "not indexed"
        print_info(f"Context refreshed — AST: {ast_status} · RAG: {rag_status}")

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
            "Now implement it completely. "
            "Your FIRST action MUST be a write_file or edit_file tool call — "
            "do not write any explanatory text before your first tool call. "
            "Write production-ready code — no stubs or placeholders."
            + _HANDOFF_INSTRUCTION
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
            "Please review the implementation.\n\n"
            "IMPORTANT — start here:\n"
            "1. Call list_dir('.') to see all project files.\n"
            "2. Read every SOURCE file (*.py, *.js, *.html, *.css, etc.) using read_file.\n"
            "3. Do NOT read or report issues about anything inside `.ca_pipeline/` — "
            "those are internal pipeline artifacts, not project files.\n"
            "4. Do NOT call write_file or run_shell.\n\n"
            "Produce structured findings with ## HIGH Priority, ## MEDIUM Priority, "
            "## LOW Priority, and ## Summary sections.\n\n"
            "After your review, write your complete findings to "
            "`.ca_pipeline/review_findings.md` using write_file, "
            "then call compute_file_sha256 on it and end with:\n\n"
            "## Handoff\n"
            "- .ca_pipeline/review_findings.md sha256:<hash>"
        )
        state.review_history.append(Message(role="user", content=prompt))

        review_text, new_msgs = reviewer.run(
            state.review_history, rag_context=self.rag_context
        )
        state.review_history.extend(new_msgs)

        # `review_text` is the last model turn's text content.  If the reviewer's
        # final turn was a pure tool call (e.g. write_file with no accompanying text),
        # review_text will be "".  Fall back to the last substantial assistant message
        # in new_msgs so findings are never silently discarded.
        if not review_text.strip():
            for msg in reversed(new_msgs):
                if msg.role == "assistant" and len(msg.content.strip()) > 80:
                    review_text = msg.content
                    log.info(
                        "Phase 3 (reviewer): empty final_text — recovered %d chars "
                        "from last substantial assistant message",
                        len(review_text),
                    )
                    break

        state.review_findings = review_text
        log.info("Phase 3 (reviewer) complete | findings_chars=%d", len(review_text))
        return new_msgs

    def _phase_phantom_fix(
        self, state: PipelineState, phantoms: list[str], impl: Agent
    ) -> list[Message]:
        """
        Implementer writes files the reviewer reported as missing.

        Called when the reviewer finds phantom references — files it tried to
        read that don't exist on disk.  We give the implementer a targeted
        prompt listing only the missing files so it doesn't re-do work already
        done correctly.
        """
        file_list = "\n".join(f"  - {p}" for p in phantoms)
        prompt = (
            "The code reviewer tried to read the following files but they do not "
            "exist on disk:\n\n"
            f"{file_list}\n\n"
            "These files are required by the implementation plan. "
            "Write each missing file now using write_file. "
            "Do NOT re-write files that already exist — only create the missing ones.\n"
            + _HANDOFF_INSTRUCTION
        )
        state.impl_history.append(Message(role="user", content=prompt))

        fix_text, new_msgs = impl.run(
            state.impl_history, rag_context=self.rag_context
        )
        state.impl_history.extend(new_msgs)

        log.info(
            "Phantom fix complete | missing_files=%d response_chars=%d",
            len(phantoms), len(fix_text),
        )
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

        # Use a focused 2-message history instead of the full accumulated impl_history.
        # The full history can exceed context limits after a large Phase 2 run.
        # The original task message (impl_history[0]) gives the implementer enough
        # context; it can read any file it needs via read_file.
        focused = [state.impl_history[0], state.impl_history[-1]]
        fix_text, new_msgs = impl.run(focused, rag_context=self.rag_context)
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
        # Focused history: task context + the run question. Full history risks
        # overflowing context; the implementer can list_dir/read_file as needed.
        focused = [state.impl_history[0], state.impl_history[-1]]
        run_text, run_msgs = impl.run(focused, rag_context=self.rag_context)
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
                "Report PASS, FAIL, or MANUAL for each criterion with evidence from the command output.\n\n"
                f"After testing, write your results to `.ca_pipeline/test_results_r{round_num + 1}.md` "
                "using write_file, then call compute_file_sha256 on it and end with:\n\n"
                "## Handoff\n"
                f"- .ca_pipeline/test_results_r{round_num + 1}.md sha256:<hash>"
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

        # Verify the tester actually ran shell commands — not just produced a table
        # from imagination. If no run_shell calls appear in the new messages,
        # the tester hallucinated its results. Inject a recovery and retry once.
        _shell_calls = [
            tc for msg in new_msgs
            for tc in msg.tool_calls
            if tc.get("function", {}).get("name") == "run_shell"
        ]
        if not _shell_calls:
            log.warning(
                "Tester produced no run_shell calls — results are hallucinated. "
                "Injecting recovery | round=%d", round_num
            )
            recovery = Message(
                role="user",
                content=(
                    "You reported test results without running any shell commands. "
                    "Those results are not valid — you must call run_shell to actually "
                    "execute the code before reporting any outcome.\n\n"
                    "Start NOW: call run_shell to discover the project layout (ls -R), "
                    "then run the import check, then verify each acceptance criterion. "
                    "Do not report PASS or FAIL without command output as evidence."
                ),
            )
            state.test_history.append(recovery)
            test_text, retry_msgs = tester.run(state.test_history, rag_context=self.rag_context)
            state.test_history.extend(retry_msgs)
            new_msgs = new_msgs + [recovery] + retry_msgs

        # `test_text` is the last model turn's text content.  If the final turn
        # was a pure tool call (run_shell), test_text will be "".  Fall back to
        # the last substantial assistant message so results are never discarded.
        if not test_text.strip():
            all_msgs_for_fallback = new_msgs
            for msg in reversed(all_msgs_for_fallback):
                if msg.role == "assistant" and len(msg.content.strip()) > 80:
                    test_text = msg.content
                    log.info(
                        "Tester round %d: empty final_text — recovered %d chars "
                        "from last substantial assistant message",
                        round_num + 1, len(test_text),
                    )
                    break

        state.test_results = test_text
        log.info(
            "Tester round %d complete | shell_calls=%d results_chars=%d",
            round_num + 1, len(_shell_calls), len(test_text),
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

        # Focused history — same rationale as _phase_fix: avoids context overflow
        # after a large impl + test run while still giving the implementer the task.
        focused = [state.impl_history[0], state.impl_history[-1]]
        fix_text, new_msgs = impl.run(focused, rag_context=self.rag_context)
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
            "and its PASS/FAIL result from the tester\n\n"
            "## IMPORTANT — you MUST use write_file to create README.md\n"
            "Do NOT output the README as markdown text. Call write_file with the complete content.\n"
            "Required format:\n"
            '  {"name": "write_file", "arguments": {"path": "' + (readme_path) + '", "content": "..."}}\n'
            "Writing the file as plain text or in a code fence does nothing — "
            "only a write_file tool call creates the file on disk."
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

def _build_manifest(phase: str, verify_result: "VerificationResult") -> str:
    """Build a Markdown manifest from a VerificationResult for artifact storage."""
    from .verifier import VerificationResult  # noqa: PLC0415
    lines = [f"# File Manifest — {phase}\n"]
    if not verify_result.records:
        lines.append("(no files written)\n")
        return "\n".join(lines)
    for r in verify_result.records:
        status = "✓" if r.ok else "✗ FAIL"
        sha_disp = r.sha_actual or r.sha_claimed or "unknown"
        lines.append(f"- {status} `{r.path}`  sha256:{sha_disp}")
    lines.append(f"\nVerified: {sum(1 for r in verify_result.records if r.ok)}"
                 f"/{verify_result.file_count} files")
    return "\n".join(lines)


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


def _parse_phantom_files(review_text: str) -> list[str]:
    """
    Scan reviewer output for file-not-found references.

    Returns a deduplicated list of file paths the reviewer could not read.
    Looks for lines that contain phantom-file language adjacent to a path-like
    token (contains '/' or ends with a known extension).
    """
    phantoms: list[str] = []
    seen: set[str] = set()
    ext_re = re.compile(r"\.(py|rs|js|ts|toml|json|md|txt|yaml|yml|cfg|ini|sh)$", re.I)

    for line in review_text.splitlines():
        if not _PHANTOM_PATTERNS.search(line):
            continue
        # Extract path-like tokens from the line
        for token in re.findall(r"[\w./\\-]+", line):
            if ("/" in token or ext_re.search(token)) and token not in seen:
                # Exclude obviously non-path tokens
                if len(token) > 3 and not token.startswith("http"):
                    seen.add(token)
                    phantoms.append(token)

    return phantoms


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
