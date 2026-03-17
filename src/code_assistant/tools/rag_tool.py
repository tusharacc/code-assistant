"""
RAG tool — exposes search_codebase() as a callable tool for the LLM.

This is the pull side of RAG: the model calls this when it determines it needs
more codebase context than was automatically injected before the turn.
"""
from __future__ import annotations

from ..rag.retriever import CodebaseRetriever
from ..logger import get_logger

log = get_logger(__name__)

# Lazy singleton — opens the same .chroma/ directory as REPL.retriever,
# so it always reads whatever the latest index contains.
_retriever: CodebaseRetriever | None = None


def _get_retriever() -> CodebaseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CodebaseRetriever()
    return _retriever


def search_codebase(query: str, top_k: int | None = None) -> str:
    """
    Query the codebase RAG index for code relevant to the query string.

    Returns formatted code chunks with file paths and line numbers,
    or an error string if the index has not been built yet.
    """
    r = _get_retriever()
    # If imported before /index ran, refresh once — cheap (one ChromaDB call)
    if not r.is_ready():
        r.refresh()
    if not r.is_ready():
        return (
            "Error: RAG index not built. "
            "Ask the user to run /index <directory> first."
        )
    log.info("search_codebase tool | query=%s top_k=%s", query[:80], top_k)
    result = r.query(query, top_k=top_k)
    return result if result else "No relevant code found for that query."


TOOL_HANDLERS: dict = {
    "search_codebase": search_codebase,
}
