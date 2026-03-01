"""
RAG retriever — query ChromaDB for relevant code chunks.
"""
from __future__ import annotations

import chromadb

from ..config import config
from ..logger import get_logger
from .indexer import _SHARED_EMBEDDER

log = get_logger(__name__)


class CodebaseRetriever:
    def __init__(self, chroma_path: str | None = None) -> None:
        path = chroma_path or config.chroma_path
        self._client = chromadb.PersistentClient(path=path)
        self._collection = None
        self._ready = False
        self._count: int = 0   # cached — avoids an extra ChromaDB round-trip on every query
        self._connect()

    def _connect(self) -> None:
        """Acquire the ChromaDB collection (called at init and after re-index)."""
        try:
            self._collection = self._client.get_collection(
                name="codebase",
                embedding_function=_SHARED_EMBEDDER,
            )
            self._ready = True
            self._count = self._collection.count()
            log.info("RAG retriever ready | collection_size=%d", self._count)
        except Exception:
            self._collection = None
            self._ready = False
            self._count = 0
            log.debug("RAG retriever | no index found — RAG disabled until /index is run")

    def refresh(self) -> None:
        """Re-acquire the collection after a re-index without rebuilding the client."""
        self._connect()

    @property
    def is_ready(self) -> bool:
        return self._ready and self._collection is not None

    def query(self, text: str, top_k: int | None = None) -> str:
        """
        Return the top-k most relevant code chunks as a single formatted string,
        ready to inject into a system prompt.
        """
        if not self.is_ready:
            return ""

        k = top_k or config.rag_top_k
        # Use cached count to avoid an extra ChromaDB call; cap to avoid errors on tiny indexes
        n = min(k, self._count) if self._count > 0 else k
        try:
            results = self._collection.query(  # type: ignore[union-attr]
                query_texts=[text],
                n_results=n,
            )
        except Exception as e:
            log.error("RAG query failed: %s", e, exc_info=True)
            return ""

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            log.debug("RAG query | no results for query: %s", text[:100])
            return ""

        files_hit = [m.get("file", "?") if m else "?" for m in metas]
        log.info(
            "RAG query | top_k=%d hits=%d files=%s",
            k, len(docs), files_hit,
        )
        log.debug("RAG query text | %s", text[:200])

        parts = []
        for doc, meta in zip(docs, metas):
            file_label = meta.get("file", "unknown") if meta else "unknown"
            parts.append(f"### {file_label}\n```\n{doc.strip()}\n```")

        return "\n\n".join(parts)

    def collection_size(self) -> int:
        if not self.is_ready:
            return 0
        return self._count
