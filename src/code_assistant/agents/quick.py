"""
Quick agent — fast 7B model, no tools, concise direct answers only.
Used by `ca --quick / -q` for one-off lookups without the full agentic loop.
"""
from ..config import config
from .base import Agent

QUICK_SYSTEM = """\
You are a concise technical assistant. Answer the user's question directly and briefly.

Rules:
- Commands or flags: output the command/flag only. No preamble, no explanation unless asked.
- Config file templates: output the file content only, as a code block. Do NOT create files.
- Short code snippets: output the code block only.
- Concepts or definitions: 1-3 sentences maximum.
- No greetings, no "certainly!", no "here you go", no closing remarks.
- If the full answer fits in one line, use one line.
- Prefer code blocks (```) for anything the user would copy-paste.
"""


def make_quick_agent() -> Agent:
    return Agent(
        model=config.architect_model,   # 7B — fastest available
        system_prompt=QUICK_SYSTEM,
        role_label="quick",
        use_tools=False,                # read-only answers, never touches the filesystem
        keep_alive=0,
    )
