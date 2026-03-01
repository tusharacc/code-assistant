"""
RAG indexer — chunks source files and stores embeddings in ChromaDB.

Chunking is language-aware (splits at function/class boundaries) with
a fallback to fixed-size character chunks.

Performance notes:
- OllamaEmbedder is a module-level singleton (stateless, safe to share).
- Embeddings use the batch ollama.embed() API — one HTTP call per chunk batch.
- Regex patterns are pre-compiled at module load (not per file).
- Re-indexing skips files whose SHA-256 hash has not changed since the last run.
- index_directory() shows a Rich Progress bar instead of per-file prints.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Generator

import chromadb
import ollama

from ..config import config
from ..logger import get_logger
from ..ui.console import console, print_success, print_error

log = get_logger(__name__)

# ── Language patterns for semantic chunking ──────────────────────────────────

_LANG_PATTERNS: dict[str, list[str]] = {
    ".py":   [r"^(class |def |\basync def )"],
    ".js":   [r"^(function |class |const \w+ = |export (default )?(function|class))"],
    ".ts":   [r"^(function |class |const \w+ = |export (default )?(function|class)|interface |type )"],
    ".tsx":  [r"^(function |class |const \w+ = |export (default )?(function|class)|interface |type )"],
    ".jsx":  [r"^(function |class |const \w+ = |export (default )?(function|class))"],
    ".go":   [r"^func "],
    ".rs":   [r"^(pub )?(fn |struct |enum |impl |trait )"],
    ".java": [r"^(public |private |protected )?(class |interface |enum |.* \w+\()"],
    ".cpp":  [r"^(\w[\w:*&<> ]+ \w+\()"],
    ".c":    [r"^(\w[\w*& ]+ \w+\()"],
    ".rb":   [r"^(def |class |module )"],
    ".php":  [r"^(function |class )"],
}

# Pre-compiled patterns — built once at module load, keyed by file extension
_COMPILED_PATTERNS: dict[str, re.Pattern] = {
    ext: re.compile("|".join(pats), re.MULTILINE)
    for ext, pats in _LANG_PATTERNS.items()
}

# File extensions we'll index
_INDEXABLE_EXTENSIONS = set(_LANG_PATTERNS.keys()) | {
    ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".sh", ".bash",
    ".sql", ".html", ".css", ".scss",
}

# Dirs to always skip
_SKIP_DIRS = {
    ".git", "__pycache__", ".chroma", "node_modules", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", "coverage",
}


# ── Shared embedder singleton ─────────────────────────────────────────────────

class OllamaEmbedder(chromadb.EmbeddingFunction):
    """ChromaDB embedding function backed by Ollama.

    Uses the batch ollama.embed() API — a full list of texts is embedded
    in a single HTTP round-trip instead of one call per text.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:  # type: ignore[override]
        if not input:
            return []
        try:
            resp = ollama.embed(model=config.embed_model, input=input)
            return resp["embeddings"]
        except Exception as e:
            log.error(
                "Batch embedding error | model=%s texts=%d | %s",
                config.embed_model, len(input), e, exc_info=True,
            )
            print_error(f"Embedding error: {e}")
            # Return zero-vectors so ChromaDB does not crash; these chunks score low
            return [[0.0] * 768] * len(input)   # nomic-embed-text dimension


# Module-level singleton — stateless, safe to share with the retriever
_SHARED_EMBEDDER = OllamaEmbedder()


class CodebaseIndexer:
    def __init__(self, chroma_path: str | None = None) -> None:
        path = chroma_path or config.chroma_path
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name="codebase",
            embedding_function=_SHARED_EMBEDDER,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ────────────────────────────────────────────────────

    def index_directory(self, root: str) -> int:
        """Index all source files under root. Returns number of chunks stored.

        Files whose SHA-256 hash has not changed since the last run are skipped,
        making subsequent calls near-instant for large, mostly-unchanged codebases.
        """
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn,
        )

        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            log.error("index_directory | not a directory: %s", root)
            print_error(f"Not a directory: {root}")
            return 0

        files = list(_iter_source_files(root_path))
        log.info("index_directory | start | root=%s files=%d", root_path, len(files))

        total = 0
        skipped = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Indexing {len(files)} files…", total=len(files))

            for file_path in files:
                rel = str(file_path.relative_to(root_path))
                progress.update(task, description=f"[dim]{file_path.name}[/dim]")
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    if not text.strip():
                        continue

                    file_hash = hashlib.sha256(text.encode()).hexdigest()

                    # Skip file if already indexed with the same content hash
                    if self._is_file_unchanged(rel, file_hash):
                        log.debug("index_directory | unchanged, skipping: %s", rel)
                        skipped += 1
                        continue

                    chunks = _chunk_text(text, file_path)
                    if not chunks:
                        continue

                    self._upsert_chunks(file_path, root_path, chunks, file_hash)
                    total += len(chunks)
                    log.debug(
                        "index_directory | indexed %s — %d chunk(s)", rel, len(chunks)
                    )
                except Exception as e:
                    log.error("index_directory | skipping %s: %s", file_path, e, exc_info=True)
                    print_error(f"Skipping {file_path.name}: {e}")
                finally:
                    progress.advance(task)

        log.info(
            "index_directory | complete | new_chunks=%d files=%d skipped=%d",
            total, len(files), skipped,
        )
        print_success(
            f"Indexed {total} chunks from {len(files) - skipped} files "
            f"({skipped} unchanged, skipped)."
        )
        return total

    def index_file(self, path: str) -> int:
        """Index a single file."""
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            log.error("index_file | not a file: %s", path)
            print_error(f"Not a file: {path}")
            return 0
        text = file_path.read_text(encoding="utf-8", errors="replace")
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        chunks = _chunk_text(text, file_path)
        log.info("index_file | %s — %d chunks", path, len(chunks))
        self._upsert_chunks(file_path, file_path.parent, chunks, file_hash)
        return len(chunks)

    def collection_size(self) -> int:
        return self._collection.count()

    # ── Internal ──────────────────────────────────────────────────────

    def _is_file_unchanged(self, rel: str, file_hash: str) -> bool:
        """Return True if any chunk for this file already carries the same hash."""
        try:
            result = self._collection.get(
                where={"$and": [{"file": {"$eq": rel}}, {"file_hash": {"$eq": file_hash}}]},
                limit=1,
                include=[],   # only IDs needed — skip documents and embeddings
            )
            return len(result["ids"]) > 0
        except Exception:
            return False  # on any error, re-index to be safe

    def _upsert_chunks(
        self, file_path: Path, root: Path, chunks: list[str], file_hash: str
    ) -> None:
        rel = str(file_path.relative_to(root))
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = _chunk_id(rel, i, chunk)
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"file": rel, "chunk_index": i, "file_hash": file_hash})

        # Upsert in batches of 64 to avoid memory spikes
        batch = 64
        for start in range(0, len(ids), batch):
            self._collection.upsert(
                ids=ids[start : start + batch],
                documents=docs[start : start + batch],
                metadatas=metas[start : start + batch],
            )


# ── Chunking helpers ──────────────────────────────────────────────────────────

def _iter_source_files(root: Path) -> Generator[Path, None, None]:
    for entry in root.rglob("*"):
        if any(skip in entry.parts for skip in _SKIP_DIRS):
            continue
        if entry.is_file() and entry.suffix in _INDEXABLE_EXTENSIONS:
            if entry.stat().st_size < 512_000:  # skip files >500 KB
                yield entry


def _chunk_text(text: str, path: Path) -> list[str]:
    """Chunk file content using pre-compiled language-aware patterns."""
    if not text.strip():
        return []
    pattern = _COMPILED_PATTERNS.get(path.suffix.lower())
    if pattern:
        return _semantic_chunks(text, pattern, path.name)
    return _fixed_chunks(text)


def _semantic_chunks(text: str, pattern: re.Pattern, filename: str) -> list[str]:
    """Split at function/class boundaries, then cap at chunk_size."""
    lines = text.splitlines(keepends=True)

    split_points = [0]
    for i, line in enumerate(lines):
        if pattern.match(line):
            split_points.append(i)
    split_points.append(len(lines))

    raw_chunks: list[str] = []
    for start, end in zip(split_points, split_points[1:]):
        chunk = "".join(lines[start:end])
        if chunk.strip():
            raw_chunks.append(f"# file: {filename}\n" + chunk)

    result: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) > config.chunk_size:
            result.extend(_fixed_chunks(chunk))
        else:
            result.append(chunk)
    return result


def _fixed_chunks(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Character-level chunking with overlap."""
    sz = size or config.chunk_size
    ov = overlap or config.chunk_overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + sz, len(text))
        chunks.append(text[start:end])
        start += sz - ov
    return chunks


def _chunk_id(rel_path: str, index: int, content: str) -> str:
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{rel_path}::chunk_{index}::{content_hash}"
