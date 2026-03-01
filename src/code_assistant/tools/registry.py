"""
Tool registry — JSON schemas for Ollama's tool-calling API, plus unified dispatch.
"""
from .file_ops import TOOL_HANDLERS as FILE_HANDLERS
from .shell import TOOL_HANDLERS as SHELL_HANDLERS

# All tool handlers in one dict
ALL_HANDLERS = {**FILE_HANDLERS, **SHELL_HANDLERS}


_TOOL_SCHEMAS: list[dict] | None = None


def get_tool_schemas() -> list[dict]:
    """Return tool definitions in Ollama / OpenAI function-calling format.
    Built once and cached — schemas are constant at runtime.
    """
    global _TOOL_SCHEMAS
    if _TOOL_SCHEMAS is not None:
        return _TOOL_SCHEMAS
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
