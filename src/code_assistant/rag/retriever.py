"""
RAG retriever — query ChromaDB for relevant code chunks.

Implements the ContextProvider ABC so future backends (e.g. AST graph) can
be swapped in transparently.

Improvements over v1:
- Jaccard-based deduplication: fetches k*3 candidates, drops near-duplicates
- Structured output format: ### path:line [lang · type · symbol]
- Language-aware code fences (```python instead of plain ```)
- is_ready() is a regular method (not a property) to satisfy the ABC
- Optional include_paths / exclude_paths filters (pass-through for now)
"""
from __future__ import annotations

import re
from typing import Sequence

import chromadb

from ..config import config
from ..logger import get_logger
from .indexer import _SHARED_EMBEDDER
from .provider import ContextProvider

log = get_logger(__name__)

# Jaccard overlap threshold: drop a chunk if its word overlap with any already-
# accepted chunk from the same file exceeds this value.
_DEDUP_THRESHOLD = 0.5

# Pre-compiled regex to strip the '# file: path | L42-L67' prefix line
_PREFIX_LINE_RE = re.compile(r"^#\s+file:.*\n", re.MULTILINE)


class CodebaseRetriever(ContextProvider):
    def __init__(
        self,
        chroma_path: str | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> None:
        path = chroma_path or config.chroma_path
        self._client = chromadb.PersistentClient(path=path)
        self._collection = None
        self._count: int = 0
        self.include_paths = include_paths
        self.exclude_paths = exclude_paths
        self._connect()

    def _connect(self) -> None:
        """Acquire the ChromaDB collection (called at init and after re-index)."""
        try:
            self._collection = self._client.get_collection(
                name="codebase",
                embedding_function=_SHARED_EMBEDDER,
            )
            self._count = self._collection.count()
            log.info("RAG retriever ready | collection_size=%d", self._count)
        except Exception:
            self._collection = None
            self._count = 0
            log.debug("RAG retriever | no index found — RAG disabled until /index is run")

    def refresh(self) -> None:
        """Re-acquire the collection after a re-index without rebuilding the client."""
        self._connect()

    def is_ready(self) -> bool:
        return self._collection is not None and self._count > 0

    def collection_size(self) -> int:
        return self._count if self.is_ready() else 0

    def query(self, text: str, top_k: int | None = None) -> str:
        """
        Return the top-k most relevant code chunks as a formatted string.

        Fetches k*3 candidates from ChromaDB, deduplicates by Jaccard
        similarity, then formats the survivors into structured blocks ready
        to inject into a system prompt or message pair.
        """
        if not self.is_ready():
            return ""

        k = top_k or config.rag_top_k
        # Fetch more candidates so dedup has room to work
        fetch_n = min(k * 3, self._count)
        if fetch_n == 0:
            return ""

        kwargs: dict = {
            "query_texts": [text],
            "n_results": fetch_n,
        }
        where = self._build_where_filter()
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)  # type: ignore[union-attr]
        except Exception as e:
            log.error("RAG query failed: %s", e, exc_info=True)
            return ""

        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            log.debug("RAG query | no results for: %s", text[:100])
            return ""

        deduped_docs, deduped_metas = _deduplicate(docs, metas, k)

        files_hit = [
            (m.get("file", "?") if m else "?") for m in deduped_metas
        ]
        log.info(
            "RAG query | fetch=%d deduped=%d top_k=%d files=%s",
            len(docs), len(deduped_docs), k, files_hit,
        )
        log.debug("RAG query text | %s", text[:200])

        return _format_results(deduped_docs, deduped_metas)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_where_filter(self) -> dict | None:
        """Build a ChromaDB where-filter for path inclusion/exclusion (future use)."""
        # Pass-through for now — include_paths / exclude_paths are stored but
        # not yet wired to ChromaDB filters (requires metadata-aware queries).
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set[str]:
    """Return a set of lowercase words from text (for Jaccard computation)."""
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _deduplicate(
    docs: Sequence[str],
    metas: Sequence[dict | None],
    max_results: int,
) -> tuple[list[str], list[dict | None]]:
    """
    Return at most max_results chunks with no near-duplicates.

    Two chunks are considered near-duplicates if they come from the same file
    AND their word-level Jaccard similarity exceeds _DEDUP_THRESHOLD.
    """
    accepted_docs:  list[str]           = []
    accepted_metas: list[dict | None]   = []
    # Per-file list of word-sets for already-accepted chunks
    file_word_sets: dict[str, list[set[str]]] = {}

    for doc, meta in zip(docs, metas):
        if len(accepted_docs) >= max_results:
            break

        file_key = (meta.get("file", "") if meta else "")
        doc_words = _word_set(doc)

        # Check overlap against accepted chunks from the same file
        duplicate = False
        for prev_words in file_word_sets.get(file_key, []):
            if _jaccard(doc_words, prev_words) > _DEDUP_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            accepted_docs.append(doc)
            accepted_metas.append(meta)
            file_word_sets.setdefault(file_key, []).append(doc_words)

    return accepted_docs, accepted_metas


def _format_results(
    docs: Sequence[str],
    metas: Sequence[dict | None],
) -> str:
    """
    Format retrieved chunks into structured blocks.

    Output example:
        ### src/foo.py:42 [python · function · bar]
        ```python
        def bar(x):
            return x + 1
        ```
    """
    parts: list[str] = []

    for doc, meta in zip(docs, metas):
        if meta:
            file_path   = meta.get("file", "unknown")
            line_start  = meta.get("line_start", 0)
            language    = meta.get("language", "")
            chunk_type  = meta.get("chunk_type", "")
            symbol_name = meta.get("symbol_name", "")
        else:
            file_path = language = chunk_type = symbol_name = ""
            line_start = 0

        # Build header annotation
        annotation_parts = [p for p in (language, chunk_type, symbol_name) if p]
        annotation = " · ".join(annotation_parts)
        loc = f":{line_start}" if line_start else ""
        header = f"### {file_path}{loc}"
        if annotation:
            header += f" [{annotation}]"

        # Strip the '# file: path | L...' prefix line that was added during indexing
        body = _PREFIX_LINE_RE.sub("", doc, count=1).strip()

        fence_lang = language or ""
        parts.append(f"{header}\n```{fence_lang}\n{body}\n```")

    return "\n\n".join(parts)
