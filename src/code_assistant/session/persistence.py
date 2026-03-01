"""
Session save/load — serialise History to/from JSON files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..agents.base import Message
from ..config import config
from ..logger import get_logger
from ..ui.console import print_success, print_error, print_info

log = get_logger(__name__)


def _sessions_dir() -> Path:
    d = Path(config.sessions_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(messages: list[Message], name: str | None = None) -> str:
    """Save messages to a JSON file. Returns the file path."""
    timestamp = int(time.time())
    filename = f"{name or 'session'}_{timestamp}.json"
    path = _sessions_dir() / filename

    data = {
        "saved_at": timestamp,
        "messages": [
            {"role": m.role, "content": m.content}
            for m in messages
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Session saved | file=%s messages=%d", path, len(messages))
    print_success(f"Session saved: {path}")
    return str(path)


def load_session(name_or_path: str) -> list[Message]:
    """
    Load a session by filename (e.g. 'session_1234.json') or full path.
    Also accepts a session name prefix to find the latest matching file.
    """
    log.info("Loading session | query=%s", name_or_path)

    # Try as a direct path first
    direct = Path(name_or_path).expanduser()
    if direct.exists():
        return _read_session(direct)

    # Try in sessions directory
    sessions = _sessions_dir()
    candidate = sessions / name_or_path
    if candidate.exists():
        return _read_session(candidate)

    # Try as a prefix match (return the most recent)
    matches = sorted(sessions.glob(f"{name_or_path}*.json"), reverse=True)
    if matches:
        return _read_session(matches[0])

    log.error("Session not found | query=%s", name_or_path)
    print_error(f"Session not found: {name_or_path}")
    return []


def list_sessions() -> list[str]:
    """Return names of saved sessions, newest first."""
    sessions = _sessions_dir()
    files = sorted(sessions.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    log.debug("list_sessions | found %d sessions", len(files))
    return [f.name for f in files]


def _read_session(path: Path) -> list[Message]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = [
            Message(role=m["role"], content=m["content"])
            for m in data.get("messages", [])
        ]
        log.info("Session loaded | file=%s messages=%d", path.name, len(messages))
        print_info(f"Loaded {len(messages)} messages from {path.name}")
        return messages
    except Exception as e:
        log.error("Failed to load session | file=%s | %s", path, e, exc_info=True)
        print_error(f"Failed to load session {path}: {e}")
        return []
