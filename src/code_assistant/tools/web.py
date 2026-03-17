"""
Web tools: fetch_url and web_search.

fetch_url  — fetch any URL and return readable plain text. Always available;
             uses only stdlib (urllib + html.parser). No API key required.

web_search — search the web and return top results with titles, URLs, snippets.
             Requires web_search_enabled = true in config.
             Uses Serper (https://serper.dev) if serper_api_key is set,
             otherwise falls back to DuckDuckGo (requires duckduckgo-search package).

Toggle web search on/off:
  ca.config or ~/.code-assistant/config.toml:
      web_search_enabled = true
      serper_api_key     = "your-key-here"   # optional; omit for DuckDuckGo
"""
import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any

from ..logger import get_logger

log = get_logger(__name__)

_MAX_FETCH_CHARS = 8_000   # truncate pages; keeps context manageable
_SEARCH_RESULTS = 5        # number of results to return


# ---------------------------------------------------------------------------
# HTML → plain text (stdlib only)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimal, robust HTML → plain text using stdlib html.parser."""

    _SKIP_TAGS = frozenset({
        "script", "style", "noscript", "head", "meta", "link",
        "nav", "footer", "aside",
    })
    _BLOCK_TAGS = frozenset({
        "p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "blockquote", "pre", "article", "section", "header",
    })

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        return extractor.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html).strip()


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

def fetch_url(url: str) -> str:
    """Fetch a web page or URL and return its plain-text content."""
    log.info("fetch_url | url=%s", url)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; code-assistant/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get_content_type() or ""
            charset = resp.headers.get_content_charset() or "utf-8"
            raw_bytes = resp.read(512_000)  # cap at 500 KB

        if "text" not in content_type and "json" not in content_type:
            return (
                f"Error: non-text content type '{content_type}' — "
                "cannot read binary content."
            )

        raw = raw_bytes.decode(charset, errors="replace")

        if "html" in content_type:
            text = _html_to_text(raw)
        else:
            text = raw  # JSON, plain text, markdown, etc.

        if len(text) > _MAX_FETCH_CHARS:
            text = (
                text[:_MAX_FETCH_CHARS]
                + f"\n\n[… truncated — {len(text):,} chars total. "
                "Use fetch_url on a more specific sub-page for full content.]"
            )

        log.debug("fetch_url | %s — returned %d chars", url, len(text))
        return text

    except urllib.error.HTTPError as e:
        log.error("fetch_url | HTTP %s: %s", e.code, url)
        return f"Error: HTTP {e.code} fetching {url}"
    except urllib.error.URLError as e:
        log.error("fetch_url | URL error: %s", e.reason)
        return f"Error: could not reach {url}: {e.reason}"
    except Exception as e:
        log.error("fetch_url | unexpected: %s", e, exc_info=True)
        return f"Error fetching {url}: {e}"


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """Search the web and return top results."""
    # Import here to avoid circular import (config imports nothing from tools)
    from ..config import config  # noqa: PLC0415

    if not config.web_search_enabled:
        return (
            "Error: web search is disabled. "
            "Enable it with  web_search_enabled = true  in ca.config or "
            "~/.code-assistant/config.toml. "
            "Optionally set  serper_api_key = \"your-key\"  for Serper, "
            "or install duckduckgo-search for a free fallback."
        )

    log.info("web_search | query=%s", query)

    if config.serper_api_key:
        return _search_serper(query, config.serper_api_key)
    return _search_duckduckgo(query)


def _search_serper(query: str, api_key: str) -> str:
    """Search via Serper.dev — Google results, 2500 free queries/month."""
    try:
        payload = json.dumps({"q": query, "num": _SEARCH_RESULTS}).encode()
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=payload,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        organic = data.get("organic", [])[:_SEARCH_RESULTS]
        if not organic:
            return f"No results found for: {query}"

        lines = [f"Search results for: **{query}**\n"]
        for i, item in enumerate(organic, 1):
            title   = item.get("title", "")
            link    = item.get("link", "")
            snippet = item.get("snippet", "")
            lines.append(f"{i}. {title}\n   {link}\n   {snippet}\n")

        return "\n".join(lines)

    except Exception as e:
        log.error("web_search serper | error: %s", e, exc_info=True)
        return f"Error searching via Serper: {e}"


def _search_duckduckgo(query: str) -> str:
    """Search via DuckDuckGo — free, no API key. Requires duckduckgo-search package."""
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        return (
            "Error: no search backend available. "
            "Either set  serper_api_key  in config, "
            "or install the free DuckDuckGo backend:  pip install duckduckgo-search"
        )

    try:
        results = list(DDGS().text(query, max_results=_SEARCH_RESULTS))
        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: **{query}**\n"]
        for i, r in enumerate(results, 1):
            title   = r.get("title", "")
            href    = r.get("href", "")
            body    = r.get("body", "")
            lines.append(f"{i}. {title}\n   {href}\n   {body}\n")

        return "\n".join(lines)

    except Exception as e:
        log.error("web_search ddg | error: %s", e, exc_info=True)
        return f"Error searching via DuckDuckGo: {e}"


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

TOOL_HANDLERS: dict[str, Any] = {
    "fetch_url":  fetch_url,
    "web_search": web_search,
}
