"""
Logging setup for Code Assistant.

Creates a `ca_logs/` directory in the current working directory and writes
structured, rotating log files there.

Log levels
----------
DEBUG  — everything: full prompts, responses, tool args/results, RAG chunks
INFO   — key lifecycle events: session start, tool calls (name only), model
         swaps, indexing progress, session save/load
ERROR  — all exceptions, Ollama errors, file/shell failures

Usage
-----
In any module:

    from .logger import get_logger
    log = get_logger(__name__)
    log.debug("full prompt: %s", messages)
    log.info("tool called: %s", tool_name)
    log.error("failed to read file: %s", exc)

The root level is configured once when setup_logging() is called from main.py.
Until then get_logger() returns a no-op (NullHandler) logger so imports are safe.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

PACKAGE = "code_assistant"

# Rotation: 5 MB per file, keep 5 backups → max ~25 MB on disk
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5

_FILE_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)-30s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Shorter format for the optional console handler (not used in normal operation)
_CONSOLE_FORMAT = "[%(levelname)s] %(name)s | %(message)s"

# Level name → int  (mirrors logging module constants)
LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO":  logging.INFO,
    "ERROR": logging.ERROR,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger namespaced under the package.

    Safe to call before setup_logging() — the root package logger has a
    NullHandler attached at import time, so nothing is lost.
    """
    # Strip absolute module path down to relative: code_assistant.agents.base → agents.base
    short = name.replace(f"{PACKAGE}.", "") if name.startswith(PACKAGE) else name
    return logging.getLogger(f"{PACKAGE}.{short}")


def setup_logging(
    level: str = "DEBUG",
    log_dir: str | Path = "ca_logs",
) -> Path:
    """
    Configure the package-level logger.

    Must be called once at application startup (main.py) before any agent
    or tool code runs.

    Parameters
    ----------
    level   : "DEBUG" | "INFO" | "ERROR"
    log_dir : directory where log files are written (created if missing)

    Returns the resolved log directory path.
    """
    level_int = LEVELS.get(level.upper(), logging.DEBUG)

    log_path = Path(log_dir).resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "code_assistant.log"

    # ── Root package logger ────────────────────────────────────────────
    root_logger = logging.getLogger(PACKAGE)
    root_logger.setLevel(level_int)

    # Avoid adding duplicate handlers if setup_logging() is called twice
    if root_logger.handlers:
        root_logger.handlers.clear()

    # ── Rotating file handler ──────────────────────────────────────────
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level_int)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))

    root_logger.addHandler(file_handler)

    # ── Startup entry ─────────────────────────────────────────────────
    root_logger.info(
        "=== Code Assistant started | log_level=%s | log_file=%s ===",
        level.upper(),
        log_file,
    )

    return log_path


# ── Module-level safety net ────────────────────────────────────────────────────
# Attach a NullHandler so that get_logger() calls before setup_logging()
# don't raise "No handlers could be found for logger X".
logging.getLogger(PACKAGE).addHandler(logging.NullHandler())
