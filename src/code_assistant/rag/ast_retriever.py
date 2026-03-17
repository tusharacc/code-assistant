"""
AST retriever — query the SQLite symbol table and generate compact outlines.

Two modes of use:
  1. get_outline()       — compact per-file summary, injected at session start
  2. search_symbols()    — substring search, exposed as a tool the LLM can call

Output is designed to be minimal (~50 lines for a medium project) while giving
the LLM enough structural context to navigate multi-language repos without
needing to read every file first.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from ..config import config
from ..logger import get_logger

log = get_logger(__name__)

_DB_FILENAME = "index.db"

_LANG_DISPLAY: dict[str, str] = {
    "python":     "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "tsx":        "TSX",
    "rust":       "Rust",
}


class ASTRetriever:
    """
    Read-only access to the AST symbol table built by ASTIndexer.

    Instantiated in REPL.__init__ and in the ast_tool.py singleton.
    Cheap to create — just opens a SQLite connection.
    """

    def __init__(self, ast_path: str | None = None) -> None:
        self._db_path = Path(ast_path or config.ast_path) / _DB_FILENAME
        self._conn: sqlite3.Connection | None = self._connect()

    def _connect(self) -> sqlite3.Connection | None:
        if not self._db_path.exists():
            return None
        try:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            if count == 0:
                conn.close()
                return None
            log.info("AST retriever ready | symbols=%d", count)
            return conn
        except Exception as e:
            log.debug("AST retriever: could not open DB: %s", e)
            return None

    def refresh(self) -> None:
        """Re-open the DB after a re-index."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = self._connect()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._conn is not None

    def symbol_count(self) -> int:
        if not self.is_ready():
            return 0
        try:
            return self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]  # type: ignore[union-attr]
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Session-start outline
    # ------------------------------------------------------------------

    def get_outline(self, max_files: int = 30) -> str:
        """
        Return a compact markdown outline of the project's symbol structure.

        Format (one content line per file):
            # Symbol Map [Python: 23 · Rust: 12 · TypeScript: 45]

            ## src/main.py [Python]
            class App :5 · fn setup() :12 · fn run() :45

            ## src-tauri/src/commands.rs [Rust]
            fn greet :1 · struct Config :15 · impl Config → [new, load] :30

        Methods and nested symbols are suppressed from top-level listing when
        their parent is already shown (e.g. impl bodies shown inline).
        """
        if not self.is_ready():
            return ""

        try:
            rows = self._conn.execute(  # type: ignore[union-attr]
                """
                SELECT file, language, kind, name, signature, line_start, parent
                FROM symbols
                ORDER BY file, line_start
                """
            ).fetchall()
        except Exception as e:
            log.error("AST outline query failed: %s", e)
            return ""

        if not rows:
            return ""

        # Count per language
        lang_counts: dict[str, int] = defaultdict(int)
        # Group by file: {file: [(kind, name, sig, line, parent), ...]}
        by_file: dict[str, list] = defaultdict(list)
        for r in rows:
            lang_counts[r["language"]] += 1
            by_file[r["file"]].append(r)

        total_files = len(by_file)
        # Sort by top-level symbol count descending so structurally rich files
        # (e.g. state.rs, lib.rs) appear before repetitive single-function files.
        files_shown = sorted(
            by_file.keys(),
            key=lambda f: sum(1 for r in by_file[f] if not r["parent"]),
            reverse=True,
        )[:max_files]

        # Header
        lang_summary = " · ".join(
            f"{_LANG_DISPLAY.get(lang, lang.capitalize())}: {cnt}"
            for lang, cnt in sorted(lang_counts.items())
        )
        lines = [f"# Symbol Map [{lang_summary}]\n"]

        for filepath in files_shown:
            file_rows = by_file[filepath]
            # Only show top-level symbols (parent == "") in the compact line.
            # Methods/nested items are already embedded in impl/class signatures.
            top_level = [r for r in file_rows if not r["parent"]]
            if not top_level:
                continue

            lang = file_rows[0]["language"]
            lines.append(f"## {filepath} [{_LANG_DISPLAY.get(lang, lang.capitalize())}]")

            parts = []
            for r in top_level:
                # Use the stored signature directly (impl blocks already have method list)
                sig = r["signature"]
                # Shorten: strip the language keyword prefix if it duplicates the kind
                # e.g. "def setup(self)" → "setup(self)", "fn greet" → "greet"
                compact = _compact_sig(sig, r["name"], r["kind"])
                parts.append(f"{compact} :{r['line_start']}")

            # Join on dot-separator; wrap at ~100 chars
            content_line = " · ".join(parts)
            if len(content_line) > 120:
                # Hard-limit: show first N symbols + count of rest
                shown_parts: list[str] = []
                running = 0
                for part in parts:
                    if running + len(part) + 3 > 110:
                        break
                    shown_parts.append(part)
                    running += len(part) + 3
                hidden = len(parts) - len(shown_parts)
                content_line = " · ".join(shown_parts)
                if hidden:
                    content_line += f" · [+{hidden} more]"

            lines.append(content_line)
            lines.append("")  # blank between files

        if total_files > max_files:
            lines.append(
                f"_... {total_files - max_files} more files not shown. "
                f"Use find_symbols() to search._"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool-level symbol search
    # ------------------------------------------------------------------

    def search_symbols(self, query: str, top_k: int = 20, kind: str = "") -> str:
        """
        Substring search over symbol names (case-insensitive).

        Returns a formatted list of matching symbols with file:line context.
        """
        if not self.is_ready():
            return ""

        params: list = [f"%{query}%"]
        sql = """
            SELECT file, language, kind, name, signature, line_start, parent
            FROM symbols
            WHERE name LIKE ? COLLATE NOCASE
        """
        if kind:
            sql += " AND kind = ?"
            params.append(kind.lower())

        sql += " ORDER BY file, line_start LIMIT ?"
        params.append(top_k)

        try:
            rows = self._conn.execute(sql, params).fetchall()  # type: ignore[union-attr]
        except Exception as e:
            log.error("AST search failed: %s", e)
            return ""

        if not rows:
            return f"No symbols matching '{query}'" + (f" (kind={kind})" if kind else "") + "."

        parts = [f"Symbols matching '{query}':\n"]
        for r in rows:
            parent_note = f" [in {r['parent']}]" if r["parent"] else ""
            parts.append(
                f"  {r['kind']:12s}  {r['signature']}{parent_note}\n"
                f"              → {r['file']}:{r['line_start']}"
            )

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _compact_sig(signature: str, name: str, kind: str) -> str:
    """
    Shorten a signature for the one-line outline.

    Strips common language keywords so the outline reads like pseudo-code:
      "def setup(self) -> None"  → "setup(self) → None"
      "fn greet(name: String)"   → "greet(name: String)"
      "async function fetch()"   → "fetch()"
      "interface Config"         → "Config (iface)"
      "impl AppState → [new]"    → kept as-is (already compact)
    """
    # Strip language keywords at the start
    for prefix in ("async def ", "def ", "async fn ", "fn ", "pub fn ",
                   "pub async fn ", "async function ", "function ",
                   "class ", "struct ", "enum ", "trait ", "type ",
                   "interface "):
        if signature.startswith(prefix):
            signature = signature[len(prefix):]
            break

    # For interface / type, append a kind hint to avoid ambiguity
    if kind == "interface":
        signature = f"{signature} (iface)"
    elif kind == "type":
        signature = f"{signature} (type)"

    # Hard limit for the inline display
    if len(signature) > 60:
        signature = signature[:57] + "..."
    return signature
