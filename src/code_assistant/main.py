"""
Code Assistant — main REPL entry point.

Usage:
    ca                          start interactive session
    ca "explain main.py"        one-shot query
    ca --resume <name>          resume a saved session
    ca --quick --req "..."      direct answer, no tools/files (alias: -q)
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

# ── RAG relevance heuristic ───────────────────────────────────────────────────

_TASK_KEYWORDS = frozenset({
    "implement", "build", "create", "write", "fix", "refactor", "add", "make",
    "design", "update", "delete", "remove", "debug", "test", "generate", "migrate",
    "integrate", "deploy", "optimize", "review", "analyse", "analyze",
})

def _needs_rag(text: str) -> bool:
    """Return True only for inputs that likely benefit from codebase context."""
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    words = set(stripped.lower().split())
    return bool(words & _TASK_KEYWORDS)


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
  [cyan]/index <dir>[/cyan]         Embed a codebase directory into the RAG index
  [cyan]/clear[/cyan]               Wipe conversation history
  [cyan]/save [name][/cyan]         Save session to disk
  [cyan]/resume <name>[/cyan]       Resume a saved session
  [cyan]/sessions[/cyan]            List saved sessions
  [cyan]/debate [on|off][/cyan]     Toggle dual-agent debate mode
  [cyan]/pipeline [on|off][/cyan]   Toggle 4-persona pipeline (arch→impl→review→test)
  [cyan]/compact[/cyan]             Manually compact history now
  [cyan]/config[/cyan]              Show current configuration
  [cyan]/rag[/cyan]                 Show RAG index status
  [cyan]/model arch <name>[/cyan]   Switch architect model
  [cyan]/model impl <name>[/cyan]   Switch implementer model
  [cyan]/help[/cyan]                Show this help
  [cyan]/exit[/cyan]  or  Ctrl-D    Quit
"""


class REPL:
    def __init__(self, resume: str | None = None) -> None:
        self.history = History()
        self.retriever = CodebaseRetriever()
        self.debate_enabled = config.debate_enabled
        self.pipeline_enabled = config.use_pipeline
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

        self._log.info(
            "REPL init | debate=%s pipeline=%s rag_ready=%s resumed=%s",
            self.debate_enabled, self.pipeline_enabled, self.retriever.is_ready, bool(resume),
        )

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self, one_shot: str | None = None) -> None:
        if one_shot:
            self._log.info("One-shot mode | input_chars=%d", len(one_shot))
            self._handle_input(one_shot)
            return

        self._print_banner()

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
        if self.retriever.is_ready and _needs_rag(text):
            rag_context = self.retriever.query(text) or None

        orchestrator = Orchestrator(
            history=self.history.all(),
            debate_enabled=self.debate_enabled,
            pipeline_enabled=self.pipeline_enabled,
            rag_context=rag_context,
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

        elif cmd == "/index":
            if not args:
                print_error("Usage: /index <directory>")
            else:
                indexer = CodebaseIndexer()
                indexer.index_directory(args[0])
                # Refresh retriever after indexing
                self.retriever = CodebaseRetriever()
                print_success(f"RAG index updated. {self.retriever.collection_size()} chunks total.")

        elif cmd == "/rag":
            if self.retriever.is_ready:
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
            if self.retriever.is_ready
            else "[dim]not indexed[/dim]"
        )
        console.print(Panel(
            f"[bold white]Code Assistant[/bold white]\n"
            f"[dim]Architect:[/dim] [cyan]{config.architect_model}[/cyan]   "
            f"[dim]Implementer:[/dim] [cyan]{config.implementer_model}[/cyan]\n"
            f"[dim]Reviewer/Tester:[/dim] [cyan]{config.reviewer_model}[/cyan]\n"
            f"[dim]Debate:[/dim] {debate_status}   "
            f"[dim]Pipeline:[/dim] {pipeline_status}   "
            f"[dim]RAG:[/dim] {rag_status}\n"
            f"[dim]Type [bold]/help[/bold] for commands · Ctrl-D to exit[/dim]",
            border_style="cyan",
            padding=(0, 1),
        ))

    def _print_config(self) -> None:
        rows = [
            ("Architect model", config.architect_model),
            ("Implementer model", config.implementer_model),
            ("Reviewer model", config.reviewer_model),
            ("Tester model", config.tester_model),
            ("Embed model", config.embed_model),
            ("Ollama host", config.ollama_host),
            ("num_ctx", str(config.num_ctx)),
            ("num_threads", str(config.num_threads)),
            ("temperature", str(config.temperature)),
            ("Debate enabled", str(self.debate_enabled)),
            ("Debate rounds", str(config.debate_rounds)),
            ("Pipeline enabled", str(self.pipeline_enabled)),
            ("RAG top_k", str(config.rag_top_k)),
            ("Sessions dir", config.sessions_dir),
            ("Log level", config.log_level),
            ("Log dir", config.log_dir),
        ]
        console.print("\n[bold]Configuration[/bold]")
        for key, val in rows:
            console.print(f"  [dim]{key:<20}[/dim] [white]{val}[/white]")


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
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override implementer model"),
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

    repl = REPL(resume=resume)
    repl.run(one_shot=prompt)


def run() -> None:
    """Entry point — delegates to typer app so sys.argv is parsed correctly."""
    app()


if __name__ == "__main__":
    run()
