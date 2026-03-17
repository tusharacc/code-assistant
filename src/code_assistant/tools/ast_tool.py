"""
AST tool — exposes find_symbols() as a callable tool for the LLM.

Complements search_codebase (semantic/RAG) with exact structural lookup:
  - search_codebase: "how does authentication work?" (semantic)
  - find_symbols:    "show me all structs", "where is AppConfig defined?" (structural)
"""
from __future__ import annotations

from ..rag.ast_retriever import ASTRetriever
from ..logger import get_logger

log = get_logger(__name__)

# Lazy singleton — same DB path as REPL.ast_retriever
_retriever: ASTRetriever | None = None


def _get_retriever() -> ASTRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ASTRetriever()
    return _retriever


def find_symbols(query: str, kind: str = "") -> str:
    """
    Search the AST symbol index for functions, classes, structs, interfaces,
    etc. by name (case-insensitive substring match).

    Returns file paths, line numbers, and compact signatures for matches.

    Parameters
    ----------
    query : str
        Name or partial name to search for.
        Examples: "AppConfig", "handle", "route"
    kind : str, optional
        Filter by symbol kind: function | class | struct | interface |
        trait | impl | enum | type | method
        Leave empty to search all kinds.
    """
    r = _get_retriever()
    if not r.is_ready():
        r.refresh()
    if not r.is_ready():
        return (
            "Error: AST index not built. "
            "Ask the user to run /ast <directory> first."
        )
    log.info("find_symbols tool | query=%s kind=%s", query[:80], kind or "any")
    result = r.search_symbols(query, kind=kind)
    return result or f"No symbols found matching '{query}'."


TOOL_HANDLERS: dict = {
    "find_symbols": find_symbols,
}
