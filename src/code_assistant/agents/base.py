"""
Base Agent — wraps an Ollama model with a system prompt, tool-calling loop,
and streaming output. This is the core agentic loop.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

import ollama

from ..config import config
from ..logger import get_logger
from ..tools.registry import get_tool_schemas, execute_tool
from ..ui.console import (
    console,
    print_agent_header,
    stream_token,
    print_tool_call,
    print_tool_result,
    print_error,
)

log = get_logger(__name__)


@dataclass
class Message:
    role: str       # "user" | "assistant" | "tool" | "system"
    content: str
    tool_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


class Agent:
    """
    A single agent backed by one Ollama model.

    Parameters
    ----------
    model : str
        Ollama model name, e.g. "qwen2.5-coder:14b"
    system_prompt : str
        The agent's persona and instructions.
    role_label : str
        Display name used in the UI ("architect" | "implementer")
    use_tools : bool
        Whether this agent may call tools (file/shell ops).
        Architect gets False (planning only); Implementer gets True.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        role_label: str = "assistant",
        use_tools: bool = True,
        keep_alive: int | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.role_label = role_label
        self.use_tools = use_tools
        self._keep_alive = keep_alive  # None = Ollama default (5 min); 0 = unload immediately
        self._tools = get_tool_schemas() if use_tools else []
        self._ollama_opts: dict = {
            "num_threads": config.num_threads,
            "num_ctx": config.num_ctx,
            "num_batch": config.num_batch,
            "temperature": config.temperature,
        }
        num_gpu = config.ollama_num_gpu()
        if num_gpu is not None:
            self._ollama_opts["num_gpu"] = num_gpu
        # Accumulated token counts — read by benchmark harness
        self.token_in: int = 0
        self.token_out: int = 0
        self.api_calls: int = 0
        log.debug(
            "Agent created | role=%s model=%s use_tools=%s",
            role_label, model, use_tools,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        messages: list[Message],
        rag_context: str | None = None,
        silent: bool = False,
    ) -> tuple[str, list[Message]]:
        """
        Run the agentic loop on `messages`.

        Returns (final_text, new_messages_to_append_to_history).
        Streams output to the terminal unless silent=True.
        """
        if not silent:
            print_agent_header(self.role_label)

        # Build the raw message list for Ollama
        raw: list[dict] = [{"role": "system", "content": self.system_prompt}]
        raw.extend(m.to_dict() for m in messages)

        if rag_context:
            # Inject RAG context as a synthetic user/assistant pair immediately
            # before the last user message — recency bias ensures the model
            # attends to it rather than deprioritising it at the end of the
            # system prompt.
            insert_at = len(raw)
            for idx in range(len(raw) - 1, -1, -1):
                if raw[idx]["role"] == "user":
                    insert_at = idx
                    break
            raw.insert(insert_at, {
                "role": "assistant",
                "content": "Understood. I will use this codebase context when answering.",
            })
            raw.insert(insert_at, {
                "role": "user",
                "content": (
                    "Here is relevant codebase context retrieved from the RAG index. "
                    "Reference specific files and line numbers when using it:\n\n"
                    + rag_context
                ),
            })

        log.info(
            "Agent run start | role=%s model=%s messages=%d rag=%s",
            self.role_label, self.model, len(messages), "yes" if rag_context else "no",
        )
        # Log the full prompt at DEBUG — guard _truncate() so it only runs when needed
        if log.isEnabledFor(logging.DEBUG):
            for i, m in enumerate(messages):
                log.debug(
                    "PROMPT[%d] role=%s | %s",
                    i, m.role, _truncate(m.content, 2000),
                )

        new_messages: list[Message] = []
        final_text = ""
        max_tool_rounds = 10  # safety limit

        for round_num in range(max_tool_rounds):
            log.debug("Tool loop round %d | role=%s", round_num, self.role_label)
            text, tool_calls_raw = self._call_model(raw, silent)
            final_text = text

            # Append assistant turn to both raw history and new_messages
            assistant_msg: dict = {"role": "assistant", "content": text}
            if tool_calls_raw:
                assistant_msg["tool_calls"] = tool_calls_raw
            raw.append(assistant_msg)
            new_messages.append(Message(
                role="assistant",
                content=text,
                tool_calls=tool_calls_raw,
            ))

            if not tool_calls_raw:
                # Recovery: model dumped code as markdown instead of calling tools.
                # Detect by looking for code fences in a tools-enabled agent on round 0.
                if round_num == 0 and self.use_tools and "```" in text:
                    log.warning(
                        "Tool-use failure: model output markdown code with no tool calls "
                        "— injecting recovery prompt | role=%s chars=%d",
                        self.role_label, len(text),
                    )
                    raw.append({
                        "role": "user",
                        "content": (
                            "Your response contained code in markdown format. "
                            "That code does NOT exist on disk — the task has NOT been completed. "
                            "You MUST call write_file or edit_file for every file. "
                            "Start NOW by calling write_file for the first file. Do not explain."
                        ),
                    })
                    continue  # retry once with the correction
                log.debug("No tool calls — agentic loop complete | role=%s", self.role_label)
                break  # No tool calls — we're done

            # Execute each tool call
            for tc in tool_calls_raw:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {}

                log.info("TOOL CALL | %s(%s)", fn_name, _fmt_args(fn_args))

                if not silent:
                    print_tool_call(fn_name, fn_args)

                result = execute_tool(fn_name, fn_args)

                log.debug("TOOL RESULT | %s → %s", fn_name, _truncate(result, 1000))

                if not silent:
                    print_tool_result(fn_name, result)

                tool_msg: dict = {"role": "tool", "content": result}
                raw.append(tool_msg)
                new_messages.append(Message(role="tool", content=result))

        log.info(
            "Agent run complete | role=%s response_chars=%d new_messages=%d",
            self.role_label, len(final_text), len(new_messages),
        )
        return final_text, new_messages

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_system(self) -> str:
        return self.system_prompt

    def _call_model(self, raw_messages: list[dict], silent: bool) -> tuple[str, list[dict]]:
        """
        Stream one model call. Returns (full_text, tool_calls).
        tool_calls is a list of dicts in OpenAI format.
        """
        text_accumulator = ""
        tool_calls_raw: list[dict] = []

        kwargs: dict = {
            "model": self.model,
            "messages": raw_messages,
            "stream": True,
            "options": self._ollama_opts,
        }
        if self._tools:
            kwargs["tools"] = self._tools
        if self._keep_alive is not None:
            kwargs["keep_alive"] = self._keep_alive

        log.debug(
            "Calling Ollama | model=%s messages=%d tools=%d",
            self.model, len(raw_messages), len(self._tools),
        )

        try:
            stream = ollama.chat(**kwargs)
            last_chunk = None
            for chunk in stream:
                last_chunk = chunk
                msg = chunk.message

                # Accumulate text content
                delta = msg.content or ""
                if delta:
                    text_accumulator += delta
                    # Stream token-by-token only for agents without tools (quick/conversational).
                    # Tool-using agents buffer the full response so we can detect code dumps
                    # and suppress them from the console (showing a headline instead).
                    if not silent and not self.use_tools:
                        stream_token(delta)

                # Tool calls appear in the final chunk
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_raw.append({
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        })

            if not silent:
                if not self.use_tools and text_accumulator:
                    console.print()  # newline after streamed content
                elif text_accumulator:
                    # Tool-using agent: decide whether to print full text or a headline.
                    # Large code-fence responses (>= 2000 chars) are suppressed on console
                    # — the full content is logged at DEBUG for troubleshooting.
                    _CODE_DUMP_THRESHOLD = 2000
                    if len(text_accumulator) >= _CODE_DUMP_THRESHOLD and "```" in text_accumulator:
                        console.print(
                            f"[dim]⋯ Received code response "
                            f"({len(text_accumulator):,} chars) — executing...[/dim]"
                        )
                        log.debug(
                            "Full model response suppressed from console | role=%s chars=%d\n%s",
                            self.role_label, len(text_accumulator), text_accumulator,
                        )
                    else:
                        console.print(text_accumulator, markup=False, highlight=False)
                        console.print()

            # Capture token counts from the final chunk (Ollama only populates these there)
            if last_chunk is not None:
                self.token_in  += getattr(last_chunk, "prompt_eval_count", 0) or 0
                self.token_out += getattr(last_chunk, "eval_count", 0) or 0
            self.api_calls += 1

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Ollama response | model=%s chars=%d tool_calls=%d",
                    self.model, len(text_accumulator), len(tool_calls_raw),
                )
                if text_accumulator:
                    log.debug("RESPONSE | %s", _truncate(text_accumulator, 3000))

        except ollama.ResponseError as e:
            error_msg = f"[Ollama error: {e.error}]"
            log.error("Ollama ResponseError | model=%s error=%s", self.model, e.error)
            if not silent:
                print_error(error_msg)
            return error_msg, []
        except Exception as e:
            error_msg = f"[Error: {e}]"
            log.error("Unexpected error calling Ollama | model=%s", self.model, exc_info=True)
            if not silent:
                print_error(error_msg)
            return error_msg, []

        # Fallback: some model variants emit tool calls as JSON text in content
        # rather than through the tool_calls API field. Detect and parse those.
        if not tool_calls_raw and text_accumulator.strip():
            parsed = _try_parse_text_tool_calls(text_accumulator.strip())
            if parsed:
                log.debug(
                    "Fallback text tool-call parser matched %d call(s)", len(parsed)
                )
                tool_calls_raw = parsed
                text_accumulator = ""  # consumed — don't show as text

        return text_accumulator, tool_calls_raw


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_KNOWN_TOOLS = {
    "read_file", "write_file", "edit_file",
    "list_dir", "glob_files", "run_shell",
    "search_codebase",
}


def _try_parse_text_tool_calls(text: str) -> list[dict]:
    """
    Detect tool calls embedded as JSON anywhere in a model's text response.

    Some model variants emit tool calls as JSON inside the response text
    rather than through the API tool_calls field. The content may be:
      - A ```json ... ``` fenced block (potentially containing nested fences
        inside string values, e.g. README content with ```sh blocks)
      - A bare JSON object anywhere in the text

    Strategy
    --------
    1. Use balanced-brace scanning to extract all top-level JSON objects.
       This correctly handles { } inside JSON string values (including ones
       that contain triple-backtick sequences) because it tracks string
       boundaries and escape sequences character-by-character.
    2. Try json.loads on each candidate.
    3. Accept any object whose "name" is a known tool and "arguments" is a dict.

    Returns an empty list if nothing recognisable is found.
    """
    result: list[dict] = []
    candidates = list(_extract_json_objects(text))

    if not candidates:
        # Last resort: try the entire text as one blob
        candidates = [text.strip()]

    for block in candidates:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue

        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = (
                item.get("name")
                or (item.get("function") or {}).get("name")
            )
            args = item.get("arguments") or item.get("parameters") or {}
            if name in _KNOWN_TOOLS and isinstance(args, dict):
                result.append({"function": {"name": name, "arguments": args}})

    if result:
        log.debug(
            "Fallback parser extracted %d tool call(s) from %d candidate(s)",
            len(result), len(candidates),
        )
    return result


def _extract_json_objects(text: str):
    """
    Yield every top-level JSON object string found in `text` by tracking
    balanced braces, string boundaries, and escape sequences.

    This is robust against JSON strings that contain triple-backticks,
    nested braces, or any other characters that confuse regex approaches.
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] != '{':
            i += 1
            continue

        # Found the start of a potential JSON object — scan to matching '}'
        depth = 0
        in_string = False
        escape_next = False
        start = i
        j = i

        while j < n:
            ch = text[j]

            if escape_next:
                escape_next = False
                j += 1
                continue

            if ch == '\\' and in_string:
                escape_next = True
                j += 1
                continue

            if ch == '"':
                in_string = not in_string
                j += 1
                continue

            if in_string:
                j += 1
                continue

            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    yield text[start:j + 1]
                    i = j + 1
                    break

            j += 1
        else:
            # Ran off the end without closing — not a valid object
            i += 1


def _truncate(text: str, max_chars: int) -> str:
    """Truncate long strings for log readability."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"… [+{len(text) - max_chars} chars]"


def _fmt_args(args: dict) -> str:
    """Format tool arguments as a compact key=value string."""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 80:
            val = val[:80] + "…"
        parts.append(f"{k}={val!r}")
    return ", ".join(parts)
