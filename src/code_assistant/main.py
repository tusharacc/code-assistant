"""
Code Assistant — main REPL entry point.

Usage:
    ca                                  start interactive session
    ca "explain main.py"                one-shot query
    ca --resume <name>                  resume a saved session
    ca --quick --req "..."              direct answer, no tools/files (alias: -q)
    ca --req-file requirements.md       load requirement doc into context
    ca --spec                           requirements spec mode (discuss → write spec)
    ca --spec --spec-out my_spec.txt    spec mode, save output to my_spec.txt
    ca --pipeline --req-file spec.txt   implement a saved spec with the full pipeline
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style as PtStyle
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

from .config import config
from .logger import get_logger, setup_logging
from .agents.base import Message
from .agents.orchestrator import Orchestrator
from .rag.indexer import CodebaseIndexer
from .rag.retriever import CodebaseRetriever
from .rag.provider import ContextProvider
from .rag.ast_retriever import ASTRetriever
from .session.history import History
from .session.persistence import save_session, load_session, list_sessions
from .ui.console import (
    console,
    print_rule,
    print_error,
    print_info,
    print_success,
    print_warning,
    confirm,
)

app = typer.Typer(add_completion=False)

# ── ca.config template ────────────────────────────────────────────────────────

_CA_CONFIG_TEMPLATE = """\
# code-assistant project configuration
# Auto-generated on first launch — values shown are your current effective defaults.
# Uncomment and edit any line to override it for this project only.
# Restart `ca` after saving.  Run `/config` to see the active source of each setting.
#
# Machine-level settings live in: ~/.code-assistant/config.toml
# Settings NOT listed here (feedback, sessions, log dirs) can only go there.

# ── Device / GPU ─────────────────────────────────────────────────────────────
# device = "{device}"   # auto | cpu | metal | cuda

# ── Models ───────────────────────────────────────────────────────────────────
# architect_model   = "{architect_model}"
# implementer_model = "{implementer_model}"
# reviewer_model    = "{reviewer_model}"
# tester_model      = "{tester_model}"
#
# GPU model presets — used when device = "metal" or "cuda"
# gpu_architect_model   = "{gpu_architect_model}"
# gpu_implementer_model = "{gpu_implementer_model}"
#
# Classification model (intent routing — keep this small and fast)
# classification_model = "{classification_model}"

# ── Inference ────────────────────────────────────────────────────────────────
# num_ctx     = {num_ctx}     # context window tokens (don't exceed available RAM)
# num_threads = {num_threads}      # CPU inference threads
# temperature = {temperature}    # 0.0 = deterministic, 1.0 = creative

# ── Pipeline & debate ────────────────────────────────────────────────────────
# use_pipeline  = {use_pipeline}   # arch → impl → review → test → docs
# auto_approve  = {auto_approve}    # auto-approve write_file / edit_file / run_shell
# debate_enabled = {debate_enabled}
# debate_rounds  = {debate_rounds}

# ── RAG ──────────────────────────────────────────────────────────────────────
# rag_top_k        = {rag_top_k}
# rag_always_query = {rag_always_query}

# ── Web tools ────────────────────────────────────────────────────────────────
# web_search_enabled = {web_search_enabled}   # requires duckduckgo-search or serper_api_key

# ── Feature flags ────────────────────────────────────────────────────────────
# project_context_enabled = {project_context_enabled}
"""

# ── RAG relevance heuristic ───────────────────────────────────────────────────

_RAG_MIN_CHARS = 30

def _needs_rag(text: str) -> bool:
    """Return True when input is long enough to benefit from codebase context.
    Replaces the old keyword heuristic — every substantive query gets context."""
    return len(text.strip()) >= _RAG_MIN_CHARS


# ── Prompt toolkit style ──────────────────────────────────────────────────────
_PT_STYLE = PtStyle.from_dict({
    "prompt": "bold",
})

_PROMPT_HISTORY_FILE = Path.home() / ".code-assistant" / ".repl_history"
_PROMPT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Slash command definitions ─────────────────────────────────────────────────

SLASH_HELP = """
[bold]Slash commands[/bold]

  [cyan]/add <file|dir>[/cyan]      Attach file(s) to conversation context
  [cyan]/summarize <file>[/cyan]    Read a file and stream a summary
  [cyan]/index <dir>[/cyan]         Embed a codebase directory into the RAG index
  [cyan]/ast <dir>[/cyan]           Build AST symbol index (Python/JS/TS/Rust)
  [cyan]/clear[/cyan]               Wipe conversation history
  [cyan]/save [name][/cyan]         Save session to disk
  [cyan]/resume <name>[/cyan]       Resume a saved session
  [cyan]/sessions[/cyan]            List saved sessions
  [cyan]/debate [on|off][/cyan]     Toggle dual-agent debate mode
  [cyan]/pipeline [on|off][/cyan]   Toggle 4-persona pipeline (arch→impl→review→test)
  [cyan]/compact[/cyan]             Manually compact history now
  [cyan]/config[/cyan]              Show current configuration (with source labels)
  [cyan]/config init[/cyan]         Create ca.config template in current directory
  [cyan]/rag[/cyan]                 Show RAG index status
  [cyan]/model arch <name>[/cyan]   Switch architect model
  [cyan]/model impl <name>[/cyan]   Switch implementer model
  [cyan]/help[/cyan]                Show this help
  [cyan]/exit[/cyan]  or  Ctrl-D    Quit

[dim]Tip: use [bold]ca --spec[/bold] to open requirements-gathering mode (architect interview → spec file)[/dim]
"""

SPEC_SLASH_HELP = """
[bold]Spec mode slash commands[/bold]

  [cyan]/finalize[/cyan]        Write the requirement document and exit
  [cyan]/clear[/cyan]           Restart the discussion from scratch
  [cyan]/help[/cyan]            Show this help
  [cyan]/exit[/cyan]  or Ctrl-D  Quit without writing the spec
"""


class REPL:
    def __init__(
        self,
        resume: str | None = None,
        resume_pipeline: bool = False,
        force_pipeline: bool = False,
    ) -> None:
        self.history = History()
        self.retriever: ContextProvider = CodebaseRetriever()
        self.ast_retriever = ASTRetriever()
        self.debate_enabled = config.debate_enabled
        self.pipeline_enabled = config.use_pipeline
        self.resume_pipeline = resume_pipeline
        # True when --pipeline was explicitly passed on the CLI (not just config default).
        # Forwarded to Orchestrator so it bypasses intent classification.
        self.force_pipeline = force_pipeline
        self._log = get_logger(__name__)
        self._prompt_session: PromptSession = PromptSession(
            history=FileHistory(str(_PROMPT_HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            style=_PT_STYLE,
        )

        if resume and isinstance(resume, str):
            loaded = load_session(resume)
            if loaded:
                self.history.append(loaded)

        # Auto-create ca.config on first launch in this directory.
        # Shows the effective defaults (from machine config) as commented-out lines.
        if not (Path.cwd() / "ca.config").exists():
            try:
                self._config_init(silent=True)
            except Exception as _e:
                self._log.warning("ca.config auto-create failed (non-fatal): %s", _e)

        # Silently generate code_assistant.md on first launch in this directory.
        # Purely programmatic — no LLM call, no console output.
        if config.project_context_enabled:
            try:
                from .project_context import ProjectContext
                if ProjectContext().ensure():
                    self._log.info(
                        "project_context: generated code_assistant.md for %s",
                        Path.cwd(),
                    )
            except Exception as _e:
                self._log.warning("project_context discovery failed (non-fatal): %s", _e)

        # If AST index exists, inject the compact symbol outline into session history.
        # Must happen BEFORE code_assistant.md injection (so ca.md stays at history[0-1]).
        if self.ast_retriever.is_ready():
            try:
                outline = self.ast_retriever.get_outline()
                if outline:
                    self.history.add_context_file("ast_symbol_map", outline)
                    self._log.info(
                        "AST outline injected | symbols=%d", self.ast_retriever.symbol_count()
                    )
            except Exception as _e:
                self._log.warning("AST outline injection failed (non-fatal): %s", _e)

        self._log.info(
            "REPL init | debate=%s pipeline=%s rag_ready=%s ast_ready=%s resumed=%s",
            self.debate_enabled, self.pipeline_enabled,
            self.retriever.is_ready(), self.ast_retriever.is_ready(), bool(resume),
        )

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self, one_shot: str | None = None, initial_input: str | None = None) -> None:
        if one_shot:
            self._log.info("One-shot mode | input_chars=%d", len(one_shot))
            self._handle_input(one_shot)
            return

        self._print_banner()

        # Auto-trigger first turn (e.g. from --req-file) before entering the loop
        if initial_input:
            self._log.info("Auto-triggering initial input | input_chars=%d", len(initial_input))
            self._handle_input(initial_input)
            # Pipeline mode is non-interactive: exit after the run completes
            # (whether it succeeded, halted on verifier failure, or hit an error).
            # The user gets the terminal back with a clear final status line.
            if self.pipeline_enabled:
                console.print("\n[dim]Pipeline run finished — returning control to terminal.[/dim]")
                return

        while True:
            try:
                raw = self._prompt_session.prompt(
                    "\n❯ ",
                    multiline=False,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self._log.info("REPL exited by user (EOF/KeyboardInterrupt)")
                console.print("\n[dim]Bye.[/dim]")
                break

            if not raw:
                continue

            if raw.startswith("/"):
                self._log.info("Slash command | %s", raw.split()[0])
                should_exit = self._handle_slash(raw)
                if should_exit:
                    self._log.info("REPL exited via /exit command")
                    break
            else:
                self._log.debug("User input | %s", raw[:200])
                self._handle_input(raw)

    # ── Input handler ─────────────────────────────────────────────────

    def _handle_input(self, text: str) -> None:
        rag_context: str | None = None
        if self.retriever.is_ready() and _needs_rag(text):
            rag_context = self.retriever.query(text) or None

        orchestrator = Orchestrator(
            history=self.history.all(),
            debate_enabled=self.debate_enabled,
            pipeline_enabled=self.pipeline_enabled,
            rag_context=rag_context,
            resume_pipeline=self.resume_pipeline,
            force_pipeline=self.force_pipeline,
        )
        try:
            new_messages = orchestrator.run(text)
            self.history.append(new_messages)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")

    # ── Slash commands ────────────────────────────────────────────────

    def _handle_slash(self, raw: str) -> bool:
        """Handle a slash command. Returns True if the REPL should exit."""
        parts = raw.split(maxsplit=2)
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/exit", "/quit", "/q"):
            console.print("[dim]Bye.[/dim]")
            return True

        elif cmd == "/help":
            console.print(SLASH_HELP)

        elif cmd == "/clear":
            self.history.clear()
            print_success("History cleared.")

        elif cmd == "/compact":
            before = self.history.total_chars()
            self.history.compact()
            after = self.history.total_chars()
            print_info(f"Compacted: {before} → {after} chars")

        elif cmd == "/save":
            name = args[0] if args else "session"
            save_session(self.history.all(), name)

        elif cmd == "/resume":
            if not args:
                print_error("Usage: /resume <session-name>")
            else:
                loaded = load_session(args[0])
                if loaded:
                    self.history.clear()
                    self.history.append(loaded)

        elif cmd == "/sessions":
            sessions = list_sessions()
            if sessions:
                console.print("\n[bold]Saved sessions:[/bold]")
                for s in sessions:
                    console.print(f"  [cyan]{s}[/cyan]")
            else:
                print_info("No saved sessions yet.")

        elif cmd == "/add":
            if not args:
                print_error("Usage: /add <file|dir>")
            else:
                self._add_context(args[0])

        elif cmd == "/summarize":
            if not args:
                print_error("Usage: /summarize <file>")
            else:
                self._summarize_file(args[0])

        elif cmd == "/index":
            if not args:
                print_error("Usage: /index <directory>")
            else:
                indexer = CodebaseIndexer()
                indexer.index_directory(args[0])
                self.retriever.refresh()
                print_success(f"RAG index updated. {self.retriever.collection_size()} chunks total.")

        elif cmd == "/ast":
            if not args:
                print_error("Usage: /ast <directory>")
            else:
                try:
                    from .rag.ast_indexer import ASTIndexer
                    indexer = ASTIndexer()
                    n = indexer.index_directory(Path(args[0]))
                    self.ast_retriever.refresh()
                    print_success(
                        f"AST index built — {n} symbols indexed "
                        f"({self.ast_retriever.symbol_count()} total)."
                    )
                except RuntimeError as e:
                    print_error(str(e))
                except Exception as e:
                    print_error(f"AST indexing failed: {e}")

        elif cmd == "/rag":
            if self.retriever.is_ready():
                print_info(f"RAG index ready — {self.retriever.collection_size()} chunks.")
            else:
                print_warning("RAG index not initialised. Use /index <dir> to build it.")

        elif cmd == "/debate":
            if not args:
                status = "on" if self.debate_enabled else "off"
                print_info(f"Debate mode is currently [bold]{status}[/bold]. Use /debate on|off")
            elif args[0].lower() == "on":
                self.debate_enabled = True
                print_success("Debate mode enabled.")
            elif args[0].lower() == "off":
                self.debate_enabled = False
                print_success("Debate mode disabled (single-agent mode).")
            else:
                print_error("Usage: /debate [on|off]")

        elif cmd == "/pipeline":
            if not args:
                status = "on" if self.pipeline_enabled else "off"
                print_info(
                    f"Pipeline mode is currently [bold]{status}[/bold]. Use /pipeline on|off\n"
                    "[dim]Pipeline: Architect → Implementer → Reviewer → Implementer (fix) → Tester[/dim]"
                )
            elif args[0].lower() == "on":
                self.pipeline_enabled = True
                print_success(
                    "Pipeline mode enabled. Complex tasks will run the full 4-persona flow."
                )
            elif args[0].lower() == "off":
                self.pipeline_enabled = False
                print_success("Pipeline mode disabled.")
            else:
                print_error("Usage: /pipeline [on|off]")

        elif cmd == "/config":
            if args and args[0].lower() == "init":
                self._config_init()
            else:
                self._print_config()

        elif cmd == "/model":
            self._switch_model(args)

        else:
            print_error(f"Unknown command: {cmd}. Type /help for a list.")

        return False

    # ── Context helpers ───────────────────────────────────────────────

    def _add_context(self, path_str: str) -> None:
        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            print_error(f"Not found: {path_str}")
            return

        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                self.history.add_context_file(str(p), content)
                print_success(f"Added {p} to context ({len(content)} chars).")
            except Exception as e:
                print_error(f"Could not read {p}: {e}")

        elif p.is_dir():
            files_added = 0
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.stat().st_size < 128_000:
                    try:
                        content = f.read_text(encoding="utf-8", errors="replace")
                        self.history.add_context_file(str(f), content)
                        files_added += 1
                    except Exception:
                        pass
            print_success(f"Added {files_added} files from {p} to context.")

    def _summarize_file(self, path_str: str) -> None:
        """Read a file and stream a summary using the architect model."""
        from .agents.base import Agent, Message

        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            print_error(f"Not found: {path_str}")
            return
        if not p.is_file():
            print_error(f"Not a file: {path_str}")
            return
        try:
            if p.stat().st_size > 200_000:
                print_error(f"File too large to summarize (>200 KB): {p.name}")
                return
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print_error(f"Could not read {p.name}: {e}")
            return

        agent = Agent(
            model=config.effective_architect_model(),
            system_prompt=(
                "You are a concise technical assistant. "
                "When given a file, summarize its purpose, structure, and key points clearly."
            ),
            role_label="summary",
            use_tools=False,
            keep_alive=0,
        )
        messages = [Message(
            role="user",
            content=f"Summarize this file (`{p.name}`):\n\n```\n{content}\n```",
        )]
        try:
            agent.run(messages)
        except Exception as e:
            print_error(f"Summarization failed: {e}")

    def _switch_model(self, args: list[str]) -> None:
        if len(args) < 2:
            print_error("Usage: /model arch <name>  or  /model impl <name>")
            return
        which, name = args[0].lower(), args[1]
        if which in ("arch", "architect"):
            config.architect_model = name
            print_success(f"Architect model → {name}")
        elif which in ("impl", "implementer"):
            config.implementer_model = name
            print_success(f"Implementer model → {name}")
        else:
            print_error("Usage: /model arch <name>  or  /model impl <name>")

    # ── Display ───────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        debate_status = "[green]on[/green]" if self.debate_enabled else "[dim]off[/dim]"
        pipeline_status = "[green]on[/green]" if self.pipeline_enabled else "[dim]off[/dim]"
        rag_status = (
            f"[green]{self.retriever.collection_size()} chunks[/green]"
            if self.retriever.is_ready()
            else "[dim]not indexed[/dim]"
        )
        ast_status = (
            f"[green]{self.ast_retriever.symbol_count()} symbols[/green]"
            if self.ast_retriever.is_ready()
            else "[dim]not indexed[/dim]"
        )
        approve_status = "[green]auto[/green]" if config.auto_approve else "[yellow]manual[/yellow]"
        device_label = f"[cyan]{config.device}[/cyan]" if config.device != "auto" else "[dim]auto[/dim]"
        console.print(Panel(
            f"[bold white]Code Assistant[/bold white]\n"
            f"[dim]Architect:[/dim] [cyan]{config.effective_architect_model()}[/cyan]   "
            f"[dim]Implementer:[/dim] [cyan]{config.effective_implementer_model()}[/cyan]\n"
            f"[dim]Reviewer/Tester:[/dim] [cyan]{config.reviewer_model}[/cyan]   "
            f"[dim]Device:[/dim] {device_label}\n"
            f"[dim]Debate:[/dim] {debate_status}   "
            f"[dim]Pipeline:[/dim] {pipeline_status}   "
            f"[dim]RAG:[/dim] {rag_status}   "
            f"[dim]AST:[/dim] {ast_status}   "
            f"[dim]Approve:[/dim] {approve_status}\n"
            f"[dim]Type [bold]/help[/bold] for commands · Ctrl-D to exit[/dim]",
            border_style="cyan",
            padding=(0, 1),
        ))

    def _print_config(self) -> None:
        sources = config.config_sources()
        source_styles = {
            "project": "[bold green]project[/bold green]",
            "machine": "[yellow]machine[/yellow]",
            "env":     "[cyan]env[/cyan]",
            "default": "[dim]default[/dim]",
        }
        rows = [
            ("Architect model",    config.architect_model,    "architect_model"),
            ("Implementer model",  config.implementer_model,  "implementer_model"),
            ("Reviewer model",     config.reviewer_model,     "reviewer_model"),
            ("Tester model",       config.tester_model,       "tester_model"),
            ("Embed model",        config.embed_model,        "embed_model"),
            ("Ollama host",        config.ollama_host,        "ollama_host"),
            ("num_ctx",            str(config.num_ctx),       "num_ctx"),
            ("num_threads",        str(config.num_threads),   "num_threads"),
            ("temperature",        str(config.temperature),   "temperature"),
            ("Debate enabled",     str(self.debate_enabled),  "debate_enabled"),
            ("Debate rounds",      str(config.debate_rounds), "debate_rounds"),
            ("Pipeline enabled",   str(self.pipeline_enabled),"use_pipeline"),
            ("Auto-approve tools", str(config.auto_approve),  "auto_approve"),
            ("RAG top_k",          str(config.rag_top_k),     "rag_top_k"),
            ("Sessions dir",       config.sessions_dir,       "sessions_dir"),
            ("Log level",          config.log_level,          "log_level"),
            ("Log dir",            config.log_dir,            "log_dir"),
        ]
        console.print("\n[bold]Configuration[/bold]")

        # Show which config files are active
        project_cfg = Path.cwd() / "ca.config"
        machine_cfg = Path.home() / ".code-assistant" / "config.toml"
        console.print(
            f"  [dim]{'Project config':<20}[/dim] "
            + ("[green]ca.config[/green]" if project_cfg.exists()
               else f"[dim]not found ({project_cfg})[/dim]")
        )
        console.print(
            f"  [dim]{'Machine config':<20}[/dim] "
            + ("[yellow]~/.code-assistant/config.toml[/yellow]" if machine_cfg.exists()
               else "[dim]not found (~/.code-assistant/config.toml)[/dim]")
        )
        console.print()

        for key, val, field_name in rows:
            src_label = source_styles.get(
                sources.get(field_name, "default"), "[dim]default[/dim]"
            )
            console.print(
                f"  [dim]{key:<20}[/dim] [white]{val:<32}[/white] {src_label}"
            )

        console.print()
        console.print(
            "  [dim]Run [bold]/config init[/bold] to create a ca.config template "
            "in this directory.[/dim]"
        )

    def _config_init(self, silent: bool = False) -> None:
        """Write a commented ca.config template into the current directory.

        silent=True — auto-called on first launch; prints a one-line hint instead
                      of the full usage note (does NOT overwrite an existing file).
        """
        dest = Path.cwd() / "ca.config"
        if dest.exists():
            if not silent:
                print_warning(f"ca.config already exists at {dest} — not overwriting.")
            return
        template = _CA_CONFIG_TEMPLATE.format(
            device=config.device,
            architect_model=config.architect_model,
            implementer_model=config.implementer_model,
            reviewer_model=config.reviewer_model,
            tester_model=config.tester_model,
            gpu_architect_model=config.gpu_architect_model,
            gpu_implementer_model=config.gpu_implementer_model,
            classification_model=config.classification_model or "",
            num_ctx=config.num_ctx,
            num_threads=config.num_threads,
            temperature=config.temperature,
            use_pipeline=str(config.use_pipeline).lower(),
            auto_approve=str(config.auto_approve).lower(),
            debate_enabled=str(config.debate_enabled).lower(),
            debate_rounds=config.debate_rounds,
            rag_top_k=config.rag_top_k,
            rag_always_query=str(config.rag_always_query).lower(),
            web_search_enabled=str(config.web_search_enabled).lower(),
            project_context_enabled=str(config.project_context_enabled).lower(),
        )
        dest.write_text(template, encoding="utf-8")
        if silent:
            print_info(
                f"Created ca.config with defaults — uncomment any line to override for this project."
            )
            self._log.info("ca.config auto-created at %s", dest)
        else:
            print_success(f"Created: {dest}")
            console.print(
                "[dim]Uncomment and edit any settings you want to override for this project.\n"
                "Restart [bold]ca[/bold] after saving to apply changes.[/dim]"
            )


# ── Spec mode REPL ───────────────────────────────────────────────────────────

class SpecREPL:
    """
    Architect-driven requirements-gathering REPL.

    The architect interviews the user to clarify goals, constraints, and testable
    acceptance criteria.  When the user is satisfied they type /finalize and the
    architect writes a structured spec document to disk.  That file can then be
    fed straight into pipeline mode::

        ca --pipeline --req-file <output_path>
    """

    def __init__(self, output_path: Path | None = None) -> None:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path: Path = output_path or (Path.cwd() / f"spec_{stamp}.txt")
        self.history: list[Message] = []
        self._log = get_logger(__name__)
        self._prompt_session: PromptSession = PromptSession(
            history=FileHistory(str(_PROMPT_HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            style=_PT_STYLE,
        )

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self) -> None:
        from .agents.architect import make_spec_architect
        architect = make_spec_architect()

        self._print_banner()

        # Architect opens the interview with a greeting
        self._call_architect(
            architect,
            Message(
                role="user",
                content=(
                    "Start the requirements gathering session. Briefly greet the user "
                    "and ask what they want to build. Be concise (2-3 sentences)."
                ),
            ),
        )

        while True:
            try:
                raw = self._prompt_session.prompt("\n❯ ", multiline=False).strip()
            except (EOFError, KeyboardInterrupt):
                self._log.info("Spec REPL exited (EOF/KeyboardInterrupt)")
                console.print("\n[dim]Spec session ended — no file written.[/dim]")
                break

            if not raw:
                continue

            if raw.startswith("/"):
                should_exit = self._handle_slash(raw, architect)
                if should_exit:
                    break
            else:
                self._call_architect(architect, Message(role="user", content=raw))

    # ── Helpers ────────────────────────────────────────────────────────

    def _call_architect(self, architect, user_msg: Message) -> str:
        """Append user_msg to history, run architect, extend history, return text."""
        self.history.append(user_msg)
        text, new_msgs = architect.run(self.history)
        self.history.extend(new_msgs)
        return text

    def _handle_slash(self, raw: str, architect) -> bool:
        """Handle a slash command.  Returns True if the REPL should exit."""
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd in ("/exit", "/quit", "/q"):
            console.print("[dim]Spec session ended — no file written.[/dim]")
            return True

        elif cmd == "/finalize":
            self._finalize(architect)
            return True

        elif cmd == "/clear":
            self.history.clear()
            print_success("Discussion cleared — starting over.")
            self._call_architect(
                architect,
                Message(
                    role="user",
                    content=(
                        "The discussion has been reset. Briefly greet the user again "
                        "and ask what they want to build."
                    ),
                ),
            )

        elif cmd == "/help":
            console.print(SPEC_SLASH_HELP)

        else:
            print_error(f"Unknown command: {cmd}. Type /help")

        return False

    def _finalize(self, architect) -> None:
        """Ask the architect to write the spec file, then confirm and print next steps."""
        print_info(f"Generating spec document → {self.output_path}")
        finalize_msg = Message(
            role="user",
            content=(
                f"The user is satisfied with the discussion. Please now write the full "
                f"requirement document using write_file to this exact path:\n\n"
                f"  {self.output_path}\n\n"
                f"Follow the spec format from your instructions (Overview, Goals, "
                f"Acceptance Criteria, Technical Constraints, Out of Scope, "
                f"Implementation Notes). Make every acceptance criterion a runnable "
                f"command with expected output."
            ),
        )
        self._call_architect(architect, finalize_msg)

        if self.output_path.exists():
            print_success(f"\nSpec written: {self.output_path}")
            console.print(
                Panel(
                    f"[bold]To implement this spec, run:[/bold]\n\n"
                    f"  [bold cyan]ca --pipeline --req-file {self.output_path}[/bold cyan]",
                    border_style="green",
                    padding=(0, 2),
                )
            )
            # Update Open Requirements section in code_assistant.md
            if config.project_context_enabled:
                try:
                    spec_content = self.output_path.read_text(encoding="utf-8")
                    from .project_context import ProjectContext
                    ProjectContext().update_from_spec(self.output_path, spec_content)
                    self._log.info(
                        "project_context: updated Open Requirements from %s",
                        self.output_path.name,
                    )
                except Exception as _e:
                    self._log.warning(
                        "project_context spec update failed (non-fatal): %s", _e
                    )
        else:
            print_warning(
                f"Spec file not detected at {self.output_path}.\n"
                "Check the architect output above — it may have saved to a different path."
            )

    # ── Display ────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        console.print(Panel(
            f"[bold white]Code Assistant — Requirements Spec Mode[/bold white]\n\n"
            f"[dim]The architect will interview you to define your project requirements.[/dim]\n"
            f"[dim]When satisfied, type [bold cyan]/finalize[/bold cyan] to generate the spec document.[/dim]\n\n"
            f"[dim]Spec output:[/dim] [cyan]{self.output_path}[/cyan]\n"
            f"[dim]Commands: [bold]/finalize[/bold] · [bold]/clear[/bold] · "
            f"[bold]/help[/bold] · [bold]/exit[/bold][/dim]",
            border_style="yellow",
            padding=(0, 1),
        ))


# ── Quick mode helper ─────────────────────────────────────────────────────────

def _run_quick(query: str, log_level: str) -> None:
    """
    Direct-answer mode: one model call, no tools, no REPL overhead.
    Output streams straight to the terminal without agent headers or panels.
    """
    from .agents.quick import make_quick_agent
    from .agents.base import Message
    from .ui.console import stream_token

    log = get_logger(__name__)

    effective_level = log_level.upper()
    log_path = setup_logging(level=effective_level, log_dir=config.log_dir)
    log.info("Quick mode | query_chars=%d", len(query))

    agent = make_quick_agent()
    messages = [Message(role="user", content=query)]

    # Stream the raw text ourselves so there are no headers or panels
    import ollama as _ollama

    kwargs = {
        "model": agent.model,
        "messages": [{"role": "system", "content": agent.system_prompt}]
                   + [m.to_dict() for m in messages],
        "stream": True,
        "options": agent._ollama_opts,
        "keep_alive": 0,
    }
    try:
        accumulated = ""
        for chunk in _ollama.chat(**kwargs):
            delta = chunk.message.content or ""
            if delta:
                accumulated += delta
                stream_token(delta)
        console.print()   # trailing newline
        log.info("Quick mode complete | response_chars=%d", len(accumulated))
    except _ollama.ResponseError as e:
        print_error(f"Ollama error: {e.error}")
    except Exception as e:
        print_error(f"Error: {e}")


# ── CLI entry points ──────────────────────────────────────────────────────────

@app.command()
def main(
    prompt: Optional[str] = typer.Argument(None, help="One-shot prompt (skip REPL)"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume a saved session"),
    resume_pipeline: bool = typer.Option(
        False, "--resume-pipeline",
        help="Resume pipeline from last failure point instead of starting fresh",
    ),
    no_debate: bool = typer.Option(False, "--no-debate", help="Disable dual-agent debate mode"),
    pipeline: bool = typer.Option(False, "--pipeline", "-p", help="Enable 4-persona pipeline mode"),
    quick: bool = typer.Option(
        False, "--quick", "-q",
        help="Quick mode: direct answer, no tools/files, no REPL",
    ),
    req: Optional[str] = typer.Option(
        None, "--req",
        help="Query string for --quick mode (alternative to the positional argument)",
    ),
    req_file: Optional[Path] = typer.Option(
        None, "--req-file", "-f",
        help="Path to a requirement document file to load into context at startup",
        exists=True, file_okay=True, dir_okay=False, readable=True,
    ),
    context: list[Path] = typer.Option(
        [], "--context", "-c",
        help="File to load into context at startup (repeatable: -c file1.md -c file2.md)",
        exists=True, file_okay=True, dir_okay=False, readable=True,
    ),
    spec: bool = typer.Option(
        False, "--spec", "-s",
        help="Requirements spec mode: discuss requirements with the architect, then write a spec document",
    ),
    spec_out: Optional[Path] = typer.Option(
        None, "--spec-out",
        help="Output path for the spec document (default: spec_YYYYMMDD_HHMMSS.txt in current dir)",
        file_okay=True, dir_okay=False,
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override implementer model"),
    no_auto_approve: bool = typer.Option(
        False, "--no-auto-approve",
        help="Prompt for confirmation before every write_file / edit_file / run_shell call",
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", "-l",
        help="Log verbosity: DEBUG | INFO | ERROR  (default: from config, currently DEBUG)",
    ),
) -> None:
    """Local code assistant powered by Ollama — with dual-agent debate and 4-persona pipeline."""
    effective_level = (log_level or config.log_level).upper()

    # ── Quick mode — bypass everything, just answer ──────────────────
    if quick:
        query = req or prompt
        if not query:
            print_error("Provide a query: ca --quick --req \"...\"  or  ca --quick \"...\"")
            raise typer.Exit(1)
        _run_quick(query, effective_level)
        return

    # ── Logging must be configured first, before any other module logs ──
    log_path = setup_logging(level=effective_level, log_dir=config.log_dir)
    log = get_logger(__name__)
    console.print(
        f"[dim]Logging → [cyan]{log_path / 'code_assistant.log'}[/cyan]"
        f"  level=[bold]{effective_level}[/bold][/dim]"
    )

    # ── Apply remaining CLI overrides to config ──────────────────────
    if no_debate:
        config.debate_enabled = False
    if pipeline:
        config.use_pipeline = True
    if model:
        config.implementer_model = model
    if no_auto_approve:
        config.auto_approve = False

    # ── Spec mode — architect-driven requirements gathering ──────────
    if spec:
        spec_repl = SpecREPL(output_path=spec_out)
        spec_repl.run()
        return

    repl = REPL(resume=resume, resume_pipeline=resume_pipeline, force_pipeline=bool(pipeline))

    # ── Load requirement document into context ───────────────────────
    initial_input: str | None = None
    if req_file:
        try:
            content = req_file.read_text(encoding="utf-8", errors="replace")
            repl.history.add_context_file(str(req_file.resolve()), content)
            print_success(f"Loaded requirement document: {req_file} ({len(content)} chars)")
            # If no explicit prompt was given, auto-trigger implementation of the requirements
            if not prompt:
                initial_input = (
                    f"Please implement everything described in the requirement document "
                    f"'{req_file.name}' that was just loaded into context."
                )
        except Exception as e:
            print_error(f"Could not read requirement file {req_file}: {e}")
            raise typer.Exit(1)

    # Load explicit context files (--context flags + config.context_files).
    # Must load AFTER req_file and BEFORE code_assistant.md so that the
    # prepend order keeps ca.md at history[0] (most prominent position).
    _ctx_paths: list[Path] = list(context)   # from --context CLI flags
    for _cf in config.context_files:         # from ca.config or machine config
        _p = (Path(_cf) if Path(_cf).is_absolute() else Path.cwd() / _cf).expanduser()
        if _p not in _ctx_paths:
            _ctx_paths.append(_p)

    for _ctx_path in _ctx_paths:
        if not _ctx_path.exists():
            print_warning(f"Context file not found: {_ctx_path}")
            continue
        try:
            if _ctx_path.stat().st_size > 200_000:
                print_warning(f"Skipping [cyan]{_ctx_path.name}[/cyan] — too large (>200 KB).")
                continue
            _ctx_content = _ctx_path.read_text(encoding="utf-8", errors="replace")
            repl.history.add_context_file(str(_ctx_path.resolve()), _ctx_content)
            print_info(
                f"Loaded [cyan]{_ctx_path.name}[/cyan] into context ({len(_ctx_content):,} chars)."
            )
            log.debug("context file: loaded %s (%d chars)", _ctx_path, len(_ctx_content))
        except Exception as _e:
            print_warning(f"Could not load context file {_ctx_path.name}: {_e}")
            log.warning("context file load failed (non-fatal): %s", _e)

    # Load project context file (code_assistant.md) into context if it exists.
    # This must happen AFTER req_file so that add_context_file()'s prepend
    # places ca.md at history[0] (most prominent position).
    if config.project_context_enabled:
        _ca_md = Path.cwd() / config.project_context_file
        if _ca_md.exists():
            try:
                _ca_content = _ca_md.read_text(encoding="utf-8", errors="replace")
                repl.history.add_context_file(str(_ca_md), _ca_content)
                log.debug(
                    "project_context: loaded %s (%d chars)",
                    _ca_md, len(_ca_content),
                )
            except Exception as _e:
                log.warning("project_context load failed (non-fatal): %s", _e)

    repl.run(one_shot=prompt, initial_input=initial_input)


def run() -> None:
    """Entry point — delegates to typer app so sys.argv is parsed correctly."""
    app()


if __name__ == "__main__":
    run()
