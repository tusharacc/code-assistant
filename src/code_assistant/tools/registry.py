"""
Tool registry — JSON schemas for Ollama's tool-calling API, plus unified dispatch.
"""
from .file_ops import TOOL_HANDLERS as FILE_HANDLERS
from .shell import TOOL_HANDLERS as SHELL_HANDLERS
from .rag_tool import TOOL_HANDLERS as RAG_HANDLERS
from .ast_tool import TOOL_HANDLERS as AST_HANDLERS
from .web import TOOL_HANDLERS as WEB_HANDLERS

# All tool handlers in one dict
ALL_HANDLERS = {**FILE_HANDLERS, **SHELL_HANDLERS, **RAG_HANDLERS, **AST_HANDLERS, **WEB_HANDLERS}


_TOOL_SCHEMAS: list[dict] | None = None


def get_tool_schemas() -> list[dict]:
    """Return tool definitions in Ollama / OpenAI function-calling format.
    Built once and cached — schemas are constant at runtime.
    web_search is only included when config.web_search_enabled is True.
    """
    global _TOOL_SCHEMAS
    if _TOOL_SCHEMAS is not None:
        return _TOOL_SCHEMAS

    from ..config import config  # noqa: PLC0415 — avoid circular import at module level

    _TOOL_SCHEMAS = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read the full contents of a file from disk. "
                    "Returns file content with line numbers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write content to a file, creating it or overwriting it. "
                    "Shows a diff and asks the user for confirmation first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full content to write to the file.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": (
                    "Make a targeted edit: replace the first occurrence of old_string "
                    "with new_string in the file. Shows a diff and asks for confirmation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to edit.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": (
                                "The exact string to replace, including all surrounding "
                                "whitespace and indentation."
                            ),
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The replacement string.",
                        },
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List directory contents as an indented tree.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list (default: current dir).",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "How deep to traverse (default: 3).",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob_files",
                "description": "Find files matching a glob pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern, e.g. '**/*.py' or 'src/*.ts'.",
                        },
                        "root": {
                            "type": "string",
                            "description": "Root directory to search from (default: '.').",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": (
                    "Search the project's RAG index for code relevant to a query. "
                    "Use when you need to find how something is implemented, locate "
                    "a function or class definition, or understand patterns in the "
                    "codebase — especially when you don't know the exact file path."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural language or keyword search query. "
                                "Examples: 'authentication middleware', "
                                "'database connection pool', 'parse_config function'."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results to return (default: 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_symbols",
                "description": (
                    "Search the AST symbol index for functions, classes, structs, "
                    "interfaces, traits, enums, and types by name. Use for structural "
                    "lookup: 'where is AppConfig defined?', 'show me all structs', "
                    "'what methods does Handler have?'. "
                    "Complements search_codebase (semantic) — use this for exact "
                    "name-based structural navigation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Name or partial name to search (case-insensitive substring). "
                                "Examples: 'AppConfig', 'handle', 'route', 'store'."
                            ),
                        },
                        "kind": {
                            "type": "string",
                            "description": (
                                "Optional: filter by symbol kind. "
                                "One of: function | class | struct | interface | "
                                "trait | impl | enum | type | method. "
                                "Leave empty to search all kinds."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": (
                    "Fetch a web page or URL and return its plain-text content. "
                    "Use to read documentation, API references, GitHub files, or any "
                    "publicly accessible URL. Returns up to 8,000 characters. "
                    "Only use URLs returned by web_search or provided by the user — "
                    "do not construct or guess URLs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full URL to fetch (must include https:// or http://).",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
        *([
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web and return top results with titles, URLs, and snippets. "
                        "Use only when local codebase tools (read_file, search_codebase, "
                        "find_symbols) cannot answer the question — e.g. external library docs, "
                        "framework API behavior, compiler error explanations, package versions. "
                        "Follow up with fetch_url to read full pages from the results."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Search query. Be specific. "
                                    "Examples: 'rust tokio spawn timeout example', "
                                    "'tauri v2 invoke command typescript'."
                                ),
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ] if config.web_search_enabled else []),
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": (
                    "Execute a shell command. Always prompts user for confirmation first. "
                    "Use for running tests, installing packages, building projects, etc."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        },
                        "working_dir": {
                            "type": "string",
                            "description": "Working directory (default: current dir).",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
    ]
    return _TOOL_SCHEMAS


def execute_tool(name: str, args: dict) -> str:
    """Look up and call the named tool with the given arguments."""
    handler = ALL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    try:
        return handler(**args)
    except TypeError as e:
        return f"Error: bad arguments for tool '{name}': {e}"
    except Exception as e:
        return f"Error in tool '{name}': {e}"
