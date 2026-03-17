# CLAUDE.md — Code Assistant

This file gives Claude Code the full context needed to work on this repository.
Read it before making any changes.

---

## What this project is

A local AI coding assistant backed by Ollama. The `ca` CLI starts an interactive
REPL where the user can ask questions, write code, edit files, and run shell commands
— all processed by local LLMs (default: `qwen2.5-coder:7b` / `14b`).

Three execution paths exist:
- **Single agent** — simple queries routed straight to the Implementer
- **Debate mode** — Architect proposes, Implementer reviews/critiques, then codes
- **Pipeline mode** — 7-phase sequential flow (Architect → Implementer → Reviewer →
  Implementer fix → Tester × 3 → Docs); only one model in RAM at a time

---

## Repository layout

```
src/code_assistant/
├── main.py              CLI + REPL (Typer app, PromptSession, slash commands)
├── config.py            Layered config — pydantic-settings + TOML sources, CA_ env prefix
├── project_context.py   ProjectScanner + ProjectContext — per-project AI memory file
├── logger.py            Rotating file logger, DEBUG/INFO/ERROR levels
├── agents/
│   ├── base.py          Agent class + Message dataclass + agentic loop
│   ├── architect.py     make_architect(), make_spec_architect()
│   ├── implementer.py   make_implementer()
│   ├── reviewer.py      make_reviewer()
│   ├── tester.py        make_tester()
│   ├── quick.py         make_quick_agent() — no tools, streaming only
│   ├── orchestrator.py  Mode selection + debate flow
│   └── pipeline.py      PipelineState dataclass + Pipeline.run() 7-phase driver
├── tools/
│   ├── registry.py      get_tool_schemas(), execute_tool() dispatch
│   ├── file_ops.py      read_file / write_file / edit_file / list_dir / glob_files
│   └── shell.py         run_shell() — always shows command, prompts for y/n
├── rag/
│   ├── indexer.py       OllamaEmbedder + CodebaseIndexer (ChromaDB, semantic chunking)
│   └── retriever.py     CodebaseRetriever.query() → formatted string for system prompt
├── session/
│   ├── history.py       History class — append, compact, add_context_file
│   └── persistence.py   save_session / load_session / list_sessions
├── feedback/
│   ├── collector.py     FeedbackRecord dataclass + collect() + save()
│   ├── enricher.py      load_examples() + enrich_impl_system() + enrich_tester_system()
│   └── export.py        export_chatml() + CLI entry for LoRA data export
└── ui/
    ├── console.py       All Rich output — panels, streaming, confirmations, diffs
    └── diff.py          print_diff() for coloured unified diffs
```

Entry point: `ca = "code_assistant.main:run"` (defined in `pyproject.toml`).

---

## Key classes and data flow

### `Message` (agents/base.py)

```python
@dataclass
class Message:
    role: str           # "user" | "assistant" | "tool" | "system"
    content: str
    tool_calls: list[dict] = field(default_factory=list)
```

Tool calls format (matches Ollama/OpenAI):
```python
[{"function": {"name": "edit_file", "arguments": {"path": "...", "old_string": "...", "new_string": "..."}}}]
```

### `Agent` (agents/base.py)

Wraps one Ollama model. Key attributes:
- `model: str` — Ollama model name
- `system_prompt: str` — Full system prompt (can be enriched by `enricher.py`)
- `use_tools: bool` — Whether tool schemas are included
- `token_in / token_out / api_calls` — Accumulated counters for benchmarks
- `_keep_alive: int | None` — 0 = unload after each call (pipeline mode)

`agent.run(messages, rag_context)` returns `(final_text, new_messages)`.
`new_messages` contains all assistant + tool turns produced this invocation.

### `Pipeline` (agents/pipeline.py)

```python
class Pipeline:
    rag_context: str | None
    initial_history: list[Message]
    metrics: dict               # populated after run(), keyed by phase name
    last_state: PipelineState | None  # set at end of run() for feedback collection

    def run(self, user_task: str) -> list[Message]: ...
```

`PipelineState` accumulates per-persona histories and string artifacts:
```python
@dataclass
class PipelineState:
    arch_history:    list[Message]
    impl_history:    list[Message]   # ← fed to feedback collector
    review_history:  list[Message]
    test_history:    list[Message]
    docs_history:    list[Message]

    arch_plan:        str
    acceptance_criteria: str
    review_findings:  str            # ← fed to feedback collector
    run_instructions: str
    test_results:     str            # ← fed to feedback collector
    doc_output:       str
```

### `Orchestrator` (agents/orchestrator.py)

Called once per user turn from `REPL._handle_input()`. Mode selection:

```python
is_complex = _is_complex_task(user_input)   # keyword heuristic
use_pipeline = pipeline_enabled and is_complex
use_debate   = not use_pipeline and debate_enabled and is_complex
```

Returns `list[Message]` — these are appended to `REPL.history`.

---

## Adding a new tool

1. Add the handler function in `tools/file_ops.py` or `tools/shell.py`.
2. Add the JSON schema and dispatch entry in `tools/registry.py`:
   ```python
   # Schema
   _TOOL_SCHEMAS.append({
       "type": "function",
       "function": {
           "name": "my_tool",
           "description": "What it does.",
           "parameters": {
               "type": "object",
               "properties": {
                   "path": {"type": "string", "description": "..."}
               },
               "required": ["path"],
           },
       },
   })

   # Dispatch
   _HANDLERS["my_tool"] = my_tool_function
   ```
3. No other changes needed — `Agent` calls `get_tool_schemas()` and `execute_tool()`
   automatically from the registry.

---

## Adding a new persona

1. Create `agents/my_persona.py` with a `make_my_persona(keep_alive=None)` factory.
2. Call `Agent(model=..., system_prompt=..., role_label="...", use_tools=...)`.
3. Add the phase to `pipeline.py` if it belongs in the pipeline.
4. Expose a phase metrics entry in `Pipeline.metrics`.

---

## Configuration system

`config.py` uses `pydantic_settings.BaseSettings` with a **layered priority chain**:

```
Priority (highest → lowest)
  1. init kwargs          — runtime mutations: CLI flags, /model slash command
  2. CA_* env vars        — e.g. CA_IMPLEMENTER_MODEL=starcoder2:15b
  3. .env file            — loaded from project root (good for secrets)
  4. ca.config            — project-level flat TOML (cwd/ca.config)
  5. ~/.code-assistant/config.toml  — machine-level flat TOML
  6. built-in defaults    — field defaults in the Config class
```

### Key components in `config.py`

**`_PROJECT_EXCLUDED_FIELDS`** — frozenset of fields that are silently dropped when
loaded from `ca.config`. Contains: `feedback_enabled`, `feedback_dir`, `few_shot_max`,
`sessions_dir`, `log_dir`, `log_level`. These are machine-scope concerns that must not
be fragmented per project.

**`TomlFileSource(settings_cls, path, exclude_fields=None)`** — custom
`PydanticBaseSettingsSource` that reads flat TOML files. Both TOML files share the
same class; the project source passes `exclude_fields=_PROJECT_EXCLUDED_FIELDS`.

**`Config.settings_customise_sources()`** — classmethod that inserts the two TOML
sources between `dotenv_settings` and `file_secret_settings` in the priority chain.

**`Config.config_sources() -> dict[str, str]`** — instance method that re-reads both
TOML files and env vars to return `{field_name: "project"|"machine"|"env"|"default"}`.
Used by `/config` slash command to show source annotations.

**`_resolve_feedback_dir` validator** — `@field_validator("feedback_dir", mode="before")`
that calls `Path(v).expanduser().resolve()`. Safety net ensuring `feedback_dir` is
always absolute even if set via env var to a relative string.

### Adding a new setting

```python
my_setting: int = 42   # add to Config class body
```

Available immediately as `config.my_setting` and overridable via `CA_MY_SETTING=99`,
`ca.config`, or `~/.code-assistant/config.toml`.

If the setting should be machine-level only (like feedback settings), add its field
name to `_PROJECT_EXCLUDED_FIELDS`.

---

## Per-project memory (`project_context.py`)

`project_context.py` manages a `code_assistant.md` file in each project directory.
This is AI-facing context — loaded into session history at startup so the AI knows
the project without being re-briefed each session.

### `DiscoveryResult` dataclass

Holds metadata extracted without any LLM call:
`name`, `project_type` ("python"|"node"|"other"), `description`, `language`,
`package_manager`, `test_runner`, `key_dependencies`, `entry_points`,
`module_layout`, `key_files`.

### `ProjectScanner.scan(root: Path) -> DiscoveryResult`

Purely programmatic. Priority:
1. `pyproject.toml` → `tomllib.loads()` for name, description, deps, scripts
2. `package.json` → `json.loads()` for name, description, deps, scripts
3. Fallback: uses directory name, "other" type

Detects test runner from deps ("pytest" → pytest, "jest" → jest).
Detects package manager from lock files (`uv.lock` → uv, `yarn.lock` → yarn, etc.).
Module layout: top-level non-hidden directories (`[:8]`).
Key files: checks for `README.md`, `Makefile`, `Dockerfile`, `.github/workflows`.

### `ProjectContext(directory: Path | None = None)`

`directory` defaults to `Path.cwd()`. Manages `directory / config.project_context_file`.

**`ensure() -> bool`**
If the file exists, returns `False` (idempotent).
Otherwise scans with `ProjectScanner`, writes initial file, returns `True`.
Silent — no console output, called from `REPL.__init__()`.

**`update_from_pipeline(state: PipelineState, task: str) -> None`**
Full regeneration after a pipeline run. Preserves work history (appends a row).
Preserves `## Open Requirements` from the previous file.
Called from `Pipeline.run()` after the feedback collection block.

**`update_from_spec(spec_path: Path, spec_content: str) -> None`**
Updates only the `## Open Requirements` section. All other sections preserved.
Called from `SpecREPL._finalize()` after the spec file is written.

**`_write(content: str) -> None`** — atomic write via `.tmp` → `Path.replace()`.

### Module-level helpers

| Function | Description |
|---|---|
| `_extract_files_written(impl_history)` | Scans tool calls for `write_file`/`edit_file`, returns deduped path list |
| `_parse_test_verdict(test_results)` | Returns "PASS", "FAIL", or "PARTIAL" |
| `_parse_acceptance_criteria(criteria, test_results)` | Returns `[(criterion, "PASS"\|"FAIL"\|"UNKNOWN")]` |
| `_summarise_arch_plan(arch_plan, max_chars=1200)` | Truncates at paragraph boundary |
| `_parse_spec_criteria(spec_content)` | Extracts acceptance criteria lines from spec |
| `_extract_work_history(existing_content)` | Parses existing `## Work History` rows |

### Integration points in `main.py`

```python
# REPL.__init__() — after session resume block
if config.project_context_enabled:
    ProjectContext().ensure()          # silent first-run generation

# main() — after req_file load (so ca.md prepends to history[0])
if config.project_context_enabled:
    repl.history.add_context_file(ca_md_path, content)

# SpecREPL._finalize() — after spec file written
if config.project_context_enabled:
    ProjectContext().update_from_spec(output_path, spec_content)
```

```python
# Pipeline.run() — after feedback collection block
if config.project_context_enabled:
    ProjectContext().update_from_pipeline(state, user_task)
```

---

## Feedback / few-shot learning

### Data model (`feedback/collector.py`)

```python
@dataclass
class FeedbackRecord:
    id: str            # uuid4 hex
    timestamp: str     # ISO-8601 UTC
    model: str         # e.g. "qwen2.5-coder:14b"
    phase: str         # "implementer" | "tester"
    mistake_type: str  # "tool_error" | "review_issue" | "test_failure"
    context: str       # assistant text that led to mistake (≤400 chars)
    error_signal: str  # error / reviewer / tester feedback (≤400 chars)
    correction: str    # next assistant message that fixed it (≤600 chars)
    tags: list[str]    # ["edit_file", "syntax", "lambda"] — for retrieval
```

Saved as JSON-lines in `~/.code-assistant/feedback/feedback.jsonl`.
This path is **always absolute** and **never per-project** — enforced by
`_PROJECT_EXCLUDED_FIELDS` and the `_resolve_feedback_dir` validator.

### Integration points in `pipeline.py`

**Before phase 1** (few-shot injection):
```python
impl.system_prompt = enrich_impl_system(impl.system_prompt, feedback_dir, max_n)
tester.system_prompt = enrich_tester_system(tester.system_prompt, feedback_dir, max_n)
```

**After phase 7** (feedback collection):
```python
self.last_state = state                      # expose state for collector
records = collect(self, user_task)           # extract pairs
save(records, feedback_dir)                  # append to JSONL
```

Both blocks are wrapped in `try/except` — a feedback failure must never abort a run.

### Three extractors (all in `collector.py`)

| Extractor | Trigger pattern in `impl_history` |
|---|---|
| `extract_tool_errors` | `assistant (tool_calls)` → `tool ("Error:...")` → `assistant (correction)` |
| `extract_review_cycles` | user message containing `"code reviewer found these issues"` |
| `extract_test_cycles` | user message containing `"failing acceptance criteria"` or `"the tester found"` |

All tool errors start with `"Error:"` — consistent across `file_ops.py` and `shell.py`.

---

## Running the application

```bash
# Install (editable)
pip install -e .

# Interactive REPL
ca

# One-shot
ca "implement a queue with push/pop/peek"

# Pipeline with spec file
ca --pipeline --req-file spec.txt

# Quick answer
ca -q "what does git reflog do"

# Override model for a single session
ca --model qwen2.5-coder:32b
```

Ensure Ollama is running (`ollama serve`) and required models are pulled.

---

## Running the benchmarks

```bash
pip install -e ".[benchmark]"
export ANTHROPIC_API_KEY=sk-ant-...

# Single requirement
python -m benchmarks.harness --req benchmarks/req_01_calculator.txt

# Compare runs
python benchmarks/compare.py \
    --dirs benchmarks/results/<ts1> benchmarks/results/<ts2> ...
```

Results are saved to `benchmarks/results/<timestamp>/`:
- `ca/` — code-assistant output
- `claude/` — Claude API output
- `report.json` — structured metrics
- `report.md` — human-readable comparison table

---

## Logging

Logs are written to `<CA_LOG_DIR>/code_assistant.log` (default: `ca_logs/`).
Rotates at 5 MB, keeps 5 backups.

```bash
tail -f ca_logs/code_assistant.log            # follow live
grep "TOOL CALL" ca_logs/code_assistant.log   # all tool calls
grep "ERROR" ca_logs/code_assistant.log       # errors only
```

Log levels:
- `DEBUG` (default) — full prompts, responses, tool args, RAG chunks
- `INFO` — lifecycle events only
- `ERROR` — failures only

---

## Common pitfalls

### 1. `self._system` vs `self.system_prompt`

The Agent attribute is `self.system_prompt` (not `_system`). The few-shot enricher
modifies it directly:
```python
impl.system_prompt = enrich_impl_system(impl.system_prompt, ...)
```

### 2. `PipelineState` is local to `Pipeline.run()`

Before the feedback loop was added, `state` was a local variable discarded on return.
Now `self.last_state = state` saves it. Callers of `pipeline.run()` can read
`pipeline.last_state` after the call. The orchestrator currently ignores it but
the feedback collector and project context updater use it.

### 3. Tool result format

`execute_tool()` always returns a plain string. Errors start with `"Error:"`:
```python
return f"Error: file not found: {path}"
return f"Error: permission denied: {path}"
return f"Error: the exact string was not found in {path}. ..."
```

The feedback collector checks `result.startswith("Error")` or `"Error:" in result[:80]`.

### 4. Multiple tool calls in one assistant turn

A single `Message(role="assistant", tool_calls=[tc1, tc2, ...])` can have multiple
tool calls. The tool result messages follow in order:
```
msg[i]   role="assistant"  tool_calls=[tc0, tc1]
msg[i+1] role="tool"       (result of tc0)
msg[i+2] role="tool"       (result of tc1)
msg[i+3] role="assistant"  ...
```

The feedback collector accounts for this with `tool_idx = j - (i + 1)`.

### 5. Pipeline uses `keep_alive=0`

In pipeline mode all four agents are created with `keep_alive=0`. This tells Ollama
to unload the model from RAM immediately after each response. Only one model is ever
in memory at a time — critical for 32 GB machines.

In debate mode and single-agent mode `keep_alive` is left as `None` (Ollama default:
5 minutes), so the model stays warm for the interactive REPL.

### 6. Pydantic + `.env` file + TOML sources

If the `.env` file contains non-`CA_`-prefixed keys (e.g. `ANTHROPIC_API_KEY`),
pydantic-settings will load them but not match any field. Without `"extra": "ignore"`
this raises `ValidationError`. The setting is already in `config.py` — do not remove it.

The config now also loads two TOML files via `TomlFileSource` in
`settings_customise_sources()`. TOML parse errors are silently swallowed (returns `{}`).
Invalid keys in TOML files are also silently ignored due to `"extra": "ignore"`.

### 7. Circular imports

`pipeline.py` imports `config` at the top level. The `feedback/collector.py`,
`feedback/enricher.py`, and `project_context.py` modules are imported lazily inside
`pipeline.run()` to avoid circular import chains. Keep these imports inside the
try/except blocks in `pipeline.py`, not at module level.

### 8. Fallback tool-call parser

Some Ollama model variants emit tool calls as JSON in the response text rather than
through the API field. `base.py` includes `_try_parse_text_tool_calls()` which scans
the raw text for JSON objects matching known tool names. This is the correct behaviour
— do not remove it.

### 9. `feedback_dir` must always be machine-level

`_PROJECT_EXCLUDED_FIELDS` in `config.py` prevents `ca.config` from supplying
`feedback_enabled`, `feedback_dir`, `few_shot_max`, `sessions_dir`, `log_dir`,
and `log_level`. This is enforced in `TomlFileSource.__call__()` and
`get_field_value()`.

Additionally, the `_resolve_feedback_dir` validator calls
`Path(v).expanduser().resolve()` on every value that reaches the field, ensuring
even an env-var or machine-config relative path is made absolute. A relative
`feedback_dir` would otherwise silently store examples in the project directory
and fragment the shared pool needed for LoRA fine-tuning.

To change feedback settings, edit `~/.code-assistant/config.toml` — never `ca.config`.

### 10. Config loads at import time

The singleton `config = Config()` is created when `config.py` is first imported.
`Path.cwd()` (for `ca.config`) and `Path.home()` (for the machine TOML) are
evaluated at that moment. Editing a TOML file mid-session has no effect until `ca`
is restarted. Do not attempt to reload or re-instantiate `Config` — all 16 modules
that import `config` hold a reference to the same singleton object.

### 11. `add_context_file()` prepend order

`History.add_context_file()` **prepends** two synthetic messages to history. The
last call produces the messages at `history[0]` and `history[1]`.

`code_assistant.md` must be the first thing the AI sees, so it must be loaded
**last** (after req_file). In `main()`:

```python
repl.history.add_context_file(req_file, ...)   # prepend 1st → goes to history[2+]
repl.history.add_context_file(ca_md, ...)      # prepend 2nd → becomes history[0]
```

---

## Testing

There is no dedicated test suite at the `src/` level. The `projects/calculator/tests/`
directory contains a sample test suite for the calculator project generated by the
pipeline. To run it:

```bash
cd projects/calculator
python -m pytest tests/ -v
```

For the feedback module, the functional tests are run inline:

```bash
python -c "
from src.code_assistant.feedback.collector import extract_tool_errors
from src.code_assistant.agents.base import Message
# ... (see bottom of this file for test script)
"
```

For `project_context.py`, run the inline smoke tests:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from code_assistant.project_context import ProjectScanner, ProjectContext

# Scan this repo
result = ProjectScanner().scan(Path('.'))
print(result)

# Test ensure() in a temp dir
import tempfile, shutil
d = Path(tempfile.mkdtemp())
(d / 'pyproject.toml').write_text('[project]\nname = \"x\"\ndescription = \"\"')
ctx = ProjectContext(d)
assert ctx.ensure() == True    # created
assert ctx.ensure() == False   # idempotent
shutil.rmtree(d)
print('project_context OK')
"
```

A full test harness across all three requirements:

```bash
for req in benchmarks/req_*.txt; do
    python -m benchmarks.harness --req \"$req\"
done
```

---

## Dependency notes

| Package | Why |
|---|---|
| `ollama>=0.4.0` | Streaming chat API with native tool-calling support |
| `rich>=13.0.0` | All terminal output — panels, colours, diffs, tables |
| `typer>=0.9.0` | CLI argument parsing with `--help` generation |
| `chromadb>=0.5.0` | Local vector DB for RAG; uses SQLite under the hood |
| `pydantic>=2.0.0` | Data validation for `Message`, `FeedbackRecord`, config |
| `pydantic-settings>=2.0.0` | Load config from env + `.env` + TOML files |
| `prompt_toolkit>=3.0.0` | REPL with history, auto-suggest, multi-line input |
| `anthropic>=0.40.0` | Optional — only for benchmark comparison (`pip install -e ".[benchmark]"`) |
| `httpx` | Used in `benchmarks/harness/claude_runner.py` for timeout control |
| `tomllib` | Python 3.11 stdlib — parses `ca.config` and `~/.code-assistant/config.toml` |

---

## Files that should never be committed

- `.env` — may contain `ANTHROPIC_API_KEY`
- `ca.config` — project-specific model/inference overrides (not a secret, but local)
- `code_assistant.md` — auto-generated AI memory file; regenerated each pipeline run
- `ca_logs/` — debug logs (already in `.gitignore`)
- `.chroma/` — RAG index (already in `.gitignore`)
- `~/.code-assistant/feedback/feedback.jsonl` — lives outside the repo by design
- `~/.code-assistant/config.toml` — machine-level config; lives outside the repo

---

## Quick reference

```bash
# Start REPL
ca

# Pipeline from spec
ca --pipeline --req-file spec.txt

# Create project config template
ca  # then type: /config init

# Show config with source labels (project / machine / env / default)
ca  # then type: /config

# Check machine-level config
cat ~/.code-assistant/config.toml

# Check accumulated feedback count (target: 200+ for LoRA)
wc -l ~/.code-assistant/feedback/feedback.jsonl

# Inspect a few feedback records
cat ~/.code-assistant/feedback/feedback.jsonl | python -m json.tool | head -80

# Export feedback for fine-tuning
python -m code_assistant.feedback.export --out training.jsonl

# Tail logs
tail -f ca_logs/code_assistant.log

# Run benchmark
python -m benchmarks.harness --req benchmarks/req_01_calculator.txt

# Syntax check all source files
python -m py_compile src/code_assistant/**/*.py
```
