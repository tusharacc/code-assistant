"""
AST-based symbol extractor for Python, JavaScript, TypeScript, and Rust.

Extracts a lightweight symbol table (function/class/struct names + compact
signatures + file:line) and stores it in a SQLite database.

We store the *symbol table*, not the raw AST — this keeps the DB tiny
(~1 MB for large codebases) and the LLM outline human-readable.

Optional dependency group:
    pip install "code-assistant[ast]"
    # installs: tree-sitter, tree-sitter-python, tree-sitter-javascript,
    #           tree-sitter-typescript, tree-sitter-rust
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..config import config
from ..logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SymbolRecord:
    file: str          # relative path from project root
    language: str      # "python" | "javascript" | "typescript" | "rust"
    kind: str          # "function" | "class" | "struct" | "impl" | "trait"
                       # | "interface" | "type" | "enum" | "method"
    name: str
    signature: str     # compact one-liner, no body, ≤120 chars
    line_start: int    # 1-based
    line_end: int
    parent: str = ""   # containing class/impl name for methods

# ---------------------------------------------------------------------------
# Language / extension maps
# ---------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, str] = {
    ".py":  "python",
    ".js":  "javascript",
    ".jsx": "javascript",
    ".ts":  "typescript",
    ".tsx": "tsx",
    ".rs":  "rust",
}

_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".chroma", ".ca_ast", "node_modules", "__pycache__",
    "target", ".venv", "venv", "env", "dist", "build",
    ".next", ".nuxt", ".svelte-kit",
})

# ---------------------------------------------------------------------------
# Node-type → kind mapping
# ---------------------------------------------------------------------------

_SYMBOL_KINDS: dict[str, str] = {
    # Python
    "function_definition":           "function",
    "async_function_definition":     "function",
    "class_definition":              "class",
    # JavaScript / TypeScript
    "function_declaration":          "function",
    "generator_function_declaration":"function",
    "class_declaration":             "class",
    "method_definition":             "method",
    "interface_declaration":         "interface",
    "type_alias_declaration":        "type",
    # Rust
    "function_item":                 "function",
    "struct_item":                   "struct",
    "enum_item":                     "enum",
    "trait_item":                    "trait",
    "impl_item":                     "impl",
    "type_item":                     "type",
}

# Container nodes whose bodies we recurse into to collect methods
_CONTAINER_TYPES: frozenset[str] = frozenset({
    "class_definition",   # Python
    "class_declaration",  # JS/TS
    "impl_item",          # Rust
    "trait_item",         # Rust
})

# Node types that represent a function body — we never recurse into these
_BODY_TYPES: frozenset[str] = frozenset({
    "block",            # Python function/if/for body
    "statement_block",  # JS/TS function body
    # Rust function body is "block" too — same token
})

# ---------------------------------------------------------------------------
# Tree-sitter lazy loader
# ---------------------------------------------------------------------------

def _load_languages() -> tuple[type | None, dict[str, object], list[str]]:
    """
    Import tree-sitter core and all language packages.

    Returns (Parser class, {lang: Language}, [missing_package_names]).
    Parser is None if tree-sitter core is not installed.
    """
    try:
        from tree_sitter import Language, Parser  # type: ignore[import]
    except ImportError:
        return None, {}, ["tree-sitter"]

    languages: dict[str, object] = {}
    missing: list[str] = []

    _lang_imports = [
        ("python",     "tree_sitter_python",     lambda m: m.language()),
        ("javascript", "tree_sitter_javascript",  lambda m: m.language()),
        ("typescript", "tree_sitter_typescript",  lambda m: m.language_typescript()),
        ("tsx",        "tree_sitter_typescript",  lambda m: m.language_tsx()),
        ("rust",       "tree_sitter_rust",        lambda m: m.language()),
    ]
    for lang_name, pkg_name, getter in _lang_imports:
        if lang_name in languages:   # tsx reuses tree_sitter_typescript
            continue
        try:
            import importlib
            mod = importlib.import_module(pkg_name)
            languages[lang_name] = Language(getter(mod))
            if lang_name == "typescript":
                # Also register tsx using same module
                languages["tsx"] = Language(mod.language_tsx())
        except (ImportError, AttributeError):
            if pkg_name not in [p for p, _, _ in _lang_imports[:_lang_imports.index((lang_name, pkg_name, getter))]]:
                missing.append(pkg_name.replace("_", "-"))

    return Parser, languages, missing

# ---------------------------------------------------------------------------
# Signature extraction helper
# ---------------------------------------------------------------------------

def _build_signature(node, content: bytes, kind: str, name: str) -> str:
    """
    Extract a compact one-line signature from a node's text.

    Cuts at the first `{` (body delimiter). Whitespace is collapsed so
    multi-line Rust function signatures (params on separate lines) are
    joined into a single readable line. Falls back to the node name if
    nothing useful is found.
    """
    raw = content[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    cut = raw.find("{")
    if cut == -1:
        cut = len(raw)
    # Collapse newlines and extra spaces — handles multi-line Rust signatures
    sig = " ".join(raw[:cut].split())
    if len(sig) > 120:
        sig = sig[:117] + "..."
    return sig or name

# ---------------------------------------------------------------------------
# Recursive tree walker
# ---------------------------------------------------------------------------

def _walk_symbols(
    node,
    content: bytes,
    language: str,
    rel_path: str,
    parent: str = "",
    depth: int = 0,
) -> Iterator[SymbolRecord]:
    """
    Yield SymbolRecords by walking a tree-sitter node tree.

    Only extracts top-level symbols and direct members of class/impl/trait
    bodies. Does NOT recurse into function bodies.
    """
    if depth > 8:
        return  # safety limit

    for child in node.children:
        kind = _SYMBOL_KINDS.get(child.type)

        if kind is not None:
            # Rust impl_item uses the "type" field to identify the implemented type
            # (e.g. `impl AppConfig` → type_identifier "AppConfig"), not "name".
            # All other node types use the standard "name" field.
            if child.type == "impl_item":
                name_node = child.child_by_field_name("type")
            else:
                name_node = child.child_by_field_name("name")
            if name_node is None:
                # Some JS arrow-function consts don't have a "name" field at this level
                continue

            name = content[name_node.start_byte:name_node.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            if not name:
                continue

            # For Rust impl blocks, gather method names to embed in signature
            if child.type == "impl_item":
                methods = _collect_impl_methods(child, content)
                if methods:
                    sig = f"impl {name} → [{', '.join(methods)}]"
                else:
                    sig = f"impl {name}"
            else:
                sig = _build_signature(child, content, kind, name)

            yield SymbolRecord(
                file=rel_path,
                language=language,
                kind=kind,
                name=name,
                signature=sig,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                parent=parent,
            )

            # Recurse into container bodies (class, impl, trait) to get methods
            if child.type in _CONTAINER_TYPES:
                body = (
                    child.child_by_field_name("body")
                    or child.child_by_field_name("declaration_list")
                    or child.child_by_field_name("class_body")
                )
                if body is not None:
                    yield from _walk_symbols(
                        body, content, language, rel_path,
                        parent=name, depth=depth + 1,
                    )

        elif child.type not in _BODY_TYPES and child.child_count > 0:
            # Recurse into non-body structural nodes (modules, export statements, etc.)
            yield from _walk_symbols(
                child, content, language, rel_path,
                parent=parent, depth=depth + 1,
            )


def _collect_impl_methods(impl_node, content: bytes) -> list[str]:
    """Return function names directly inside a Rust impl block body."""
    body = (
        impl_node.child_by_field_name("body")
        or impl_node.child_by_field_name("declaration_list")
    )
    if body is None:
        return []
    names = []
    for child in body.children:
        if child.type == "function_item":
            name_node = child.child_by_field_name("name")
            if name_node:
                names.append(
                    content[name_node.start_byte:name_node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                )
    return names

# ---------------------------------------------------------------------------
# ASTIndexer
# ---------------------------------------------------------------------------

class ASTIndexer:
    """
    Parse source files and store a compact symbol table in SQLite.

    Trigger via: ASTIndexer().index_directory(Path("./myproject"))
    """

    _DB_FILENAME = "index.db"

    def __init__(self, ast_path: str | None = None) -> None:
        self._ast_dir = Path(ast_path or config.ast_path)
        self._ast_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._ast_dir / self._DB_FILENAME
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                file       TEXT    NOT NULL,
                language   TEXT    NOT NULL,
                kind       TEXT    NOT NULL,
                name       TEXT    NOT NULL,
                signature  TEXT    NOT NULL,
                line_start INTEGER NOT NULL,
                line_end   INTEGER NOT NULL,
                parent     TEXT    DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_name
                ON symbols(name COLLATE NOCASE);
            CREATE INDEX IF NOT EXISTS idx_symbols_file
                ON symbols(file);
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_directory(self, root: Path) -> int:
        """
        Walk root, parse all supported files, store symbols in SQLite.
        Returns total number of symbols indexed.

        Clears the previous index on each run (full rebuild).
        Raises RuntimeError if tree-sitter is not installed.
        """
        Parser, languages, missing = _load_languages()

        if Parser is None:
            raise RuntimeError(
                "tree-sitter is not installed.\n"
                "Run: pip install 'code-assistant[ast]'"
            )
        if missing:
            log.warning(
                "AST: some language packages missing (%s); those languages will be skipped",
                ", ".join(missing),
            )
        if not languages:
            raise RuntimeError(
                "No tree-sitter language packages found.\n"
                "Run: pip install 'code-assistant[ast]'"
            )

        root = Path(root).resolve()
        if not root.exists():
            raise RuntimeError(f"Directory not found: {root}")
        if not root.is_dir():
            raise RuntimeError(f"Not a directory: {root}")
        log.info("AST index start | root=%s languages=%s", root, list(languages))

        # Clear previous index
        self._conn.execute("DELETE FROM symbols")
        self._conn.commit()

        parsers: dict[str, object] = {}
        total = 0

        for filepath in self._walk(root):
            ext = filepath.suffix.lower()
            lang = _EXT_TO_LANG.get(ext)
            ts_lang = languages.get(lang) if lang else None
            if ts_lang is None:
                continue

            if lang not in parsers:
                parsers[lang] = Parser(ts_lang)

            try:
                content = filepath.read_bytes()
                tree = parsers[lang].parse(content)
                rel_path = str(filepath.relative_to(root))
                symbols = list(_walk_symbols(tree.root_node, content, lang, rel_path))
                self._insert_symbols(symbols)
                total += len(symbols)
                log.debug("AST | %s → %d symbols", rel_path, len(symbols))
            except Exception as e:
                log.warning("AST: failed to parse %s: %s", filepath, e)

        self._conn.execute(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)", ("symbol_count", str(total))
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)", ("root", str(root))
        )
        self._conn.commit()
        log.info("AST index complete | root=%s symbols=%d", root, total)
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _walk(self, root: Path) -> Iterator[Path]:
        """Yield all supported source files, skipping common build/vendor dirs."""
        for p in sorted(root.rglob("*")):
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if p.is_file() and p.suffix.lower() in _EXT_TO_LANG:
                yield p

    def _insert_symbols(self, symbols: list[SymbolRecord]) -> None:
        if not symbols:
            return
        self._conn.executemany(
            """
            INSERT INTO symbols
                (file, language, kind, name, signature, line_start, line_end, parent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (s.file, s.language, s.kind, s.name, s.signature,
                 s.line_start, s.line_end, s.parent)
                for s in symbols
            ],
        )
