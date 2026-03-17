"""
ContextProvider — abstract base class for retrieval backends.

Concrete implementations:
  - CodebaseRetriever (rag/retriever.py)  — ChromaDB-backed RAG
  - Future: ASTProvider — graph-based symbol lookup (plugin, not built yet)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ContextProvider(ABC):
    """Plugin interface for codebase context retrieval."""

    @abstractmethod
    def query(self, text: str, top_k: int | None = None) -> str:
        """Return relevant context as a formatted string for injection into a prompt."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the backend has an index and can serve queries."""
        ...

    @abstractmethod
    def refresh(self) -> None:
        """Re-acquire the underlying index after a re-index without rebuilding the client."""
        ...

    @abstractmethod
    def collection_size(self) -> int:
        """Return the number of chunks/nodes in the index (0 when not ready)."""
        ...

    def index_directory(self, path: Path | str) -> int:
        """
        Build or update the index for a directory.

        Default implementation raises NotImplementedError — backends that don't
        own their own indexing (e.g. a read-only AST provider) don't need to
        implement this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement index_directory()"
        )
