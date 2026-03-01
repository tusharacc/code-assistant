# Code Assistant

A local, privacy-first code assistant that runs entirely on your machine using [Ollama](https://ollama.com). Interactive REPL, file editing with diffs, shell execution, codebase RAG, dual-agent debate mode, and a 4-persona pipeline where Architect → Implementer → Reviewer → Tester each run sequentially with only one model loaded in RAM at a time.

No internet required after setup. No API keys. No GPU needed.

---

## How it works

### Standard mode (debate)

Two models collaborate on complex tasks:

| Role | Default model | Job |
|---|---|---|
| **Architect** | `qwen2.5-coder:7b` | Analyses the task, proposes a design, calls out edge cases. No tool access — planning only. |
| **Implementer** | `qwen2.5-coder:14b` | Reviews the plan, pushes back if needed, then writes the code using tools. |
| **Embeddings** | `nomic-embed-text` | Powers the RAG index for codebase-aware context. |

For simple tasks (explain, read, describe) a single agent responds directly. For complex tasks (implement, build, refactor, design) the debate mode kicks in automatically.

```
User: "Implement a thread-safe LRU cache"

[Architect]   → Proposes: OrderedDict + RLock, maxsize=0 means unbounded
[Implementer] → Pushes back: use RLock not Lock (get() may call put() internally)
[Architect]   → Revises plan, both agree
[Implementer] → Writes the final code, creates the file
```

### Pipeline mode (`--pipeline` / `/pipeline on`)

A 4-persona sequential pipeline where **only one model is in RAM at a time** (`keep_alive=0`):

| Phase | Persona | Default model | Job |
|---|---|---|---|
| 1 | **Architect** | `qwen2.5-coder:7b` | Produces a detailed implementation plan |
| 2 | **Implementer** | `qwen2.5-coder:14b` | Codes the plan; can Q&A with Architect mid-flight |
| 3 | **Reviewer** | `qwen2.5-coder:7b` | Inspects files, produces HIGH/MEDIUM/LOW findings |
| 4 | **Implementer (fix)** | `qwen2.5-coder:14b` | Fixes HIGH and MEDIUM issues; skipped if none found |
| 5 | **Tester** | `qwen2.5-coder:7b` | Runs shell commands, verifies acceptance criteria, reports PASS/FAIL |

The Implementer can ask the Architect a clarifying question mid-implementation using a `@@QUESTION_FOR_ARCHITECT: ...@@` tag, and the pipeline routes the answer back automatically.

### Quick mode (`-q` / `--quick`)

Direct answer with no tools, no REPL, no panels — just a raw streamed response. Ideal for shell lookups, config snippets, or one-liner questions:

```bash
ca --quick --req "command to list hidden files"
ca -q "git stash pop vs apply"
ca -q --req "openvpn config file structure"
```

---

## Prerequisites

### 1. Hardware

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | Not required |
| Disk | ~15 GB free | 20 GB free |

The assistant is tuned for CPU inference. On a modern 8-core CPU at 32 GB RAM expect roughly:
- `qwen2.5-coder:7b` → 10–20 tokens/sec
- `qwen2.5-coder:14b` → 5–10 tokens/sec

### 2. OS

Linux (primary target) or macOS. Not tested on Windows.

### 3. Python

Python 3.11 or newer.

```bash
python3 --version
```

### 4. Ollama

Install Ollama from [ollama.com](https://ollama.com/download) or via the one-line installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify it is running:

```bash
ollama --version
ollama list
```

---

## Model setup

Pull the required models. This is a one-time download (~13 GB total).

```bash
# Architect / Reviewer / Tester — fast, planning and analysis (~4.5 GB)
ollama pull qwen2.5-coder:7b

# Implementer — sharper, writes the actual code (~8.5 GB)
ollama pull qwen2.5-coder:14b

# Embeddings — for RAG codebase indexing (~274 MB)
ollama pull nomic-embed-text
```

> **Why these models?**
> Both `qwen2.5-coder` variants are trained specifically on code, support native function/tool calling through Ollama's API, and fit comfortably in a 32 GB system at Q4_K_M quantisation. At 4-bit precision they use ~4.5 GB and ~8.5 GB respectively.
>
> With `keep_alive=0` in pipeline mode, Ollama unloads each model immediately after it responds — only one model occupies RAM at a time.

### Alternative models

You can swap models at any time via `/model` in the REPL or by setting environment variables.

| Model | Size (Q4) | Speed | Notes |
|---|---|---|---|
| `qwen2.5-coder:7b` | 4.5 GB | Fast | Default architect/reviewer/tester |
| `qwen2.5-coder:14b` | 8.5 GB | Medium | Default implementer |
| `qwen2.5-coder:32b` | 19 GB | Slow | Fits in 32 GB but ~1–2 t/s on CPU |
| `deepseek-coder-v2:16b` | 10 GB | Medium | Good alternative implementer |
| `codellama:13b` | 7.5 GB | Medium | Older but reliable |
| `phi4:14b` | 8.5 GB | Medium | Strong reasoning, less code-specific |

---

## Installation

```bash
git clone <repo-url>
cd code-assistant
pip install -e .
```

This installs the `ca` command globally (within your Python environment).

---

## Usage

### Start the interactive REPL

```bash
ca
```

### One-shot query (no REPL)

```bash
ca "explain what this project does"
ca "write a binary search in Python and save it to search.py"
```

### Quick mode — direct answers, no REPL

```bash
ca --quick --req "ls flag for hidden files"
ca -q --req "dockerfile for a python fastapi app"
ca -q "git cherry-pick syntax"
```

### Options

```
ca [OPTIONS] [PROMPT]

Options:
  --resume,    -r TEXT   Resume a saved session by name
  --no-debate            Disable dual-agent debate (single-agent mode)
  --pipeline,  -p        Enable 4-persona pipeline mode
  --quick,     -q        Quick mode: direct answer, no tools/files, no REPL
  --req        TEXT      Query for --quick mode (alternative to positional arg)
  --model,     -m TEXT   Override the implementer model
  --log-level, -l TEXT   Log verbosity: DEBUG | INFO | ERROR
  --help                 Show help and exit
```

---

## Slash commands

Type these inside the REPL:

| Command | Description |
|---|---|
| `/add <file\|dir>` | Attach a file or directory into the conversation context |
| `/index <dir>` | Embed a codebase into the RAG index (enables semantic retrieval) |
| `/rag` | Show RAG index status and chunk count |
| `/debate [on\|off]` | Toggle dual-agent debate mode |
| `/pipeline [on\|off]` | Toggle 4-persona pipeline mode |
| `/save [name]` | Save the current session to disk |
| `/resume <name>` | Load a previously saved session |
| `/sessions` | List all saved sessions |
| `/clear` | Wipe conversation history |
| `/compact` | Manually summarise old history to free context space |
| `/model arch <name>` | Switch the architect model |
| `/model impl <name>` | Switch the implementer model |
| `/config` | Show current configuration |
| `/help` | Show command reference |
| `/exit` or Ctrl-D | Quit |

---

## Tools available to the model

The implementer and tester agents can call these tools autonomously during a response:

| Tool | What it does |
|---|---|
| `read_file` | Read any file with line numbers |
| `write_file` | Create or overwrite a file — shows diff, asks confirmation |
| `edit_file` | Replace a specific string in a file — shows diff, asks confirmation |
| `list_dir` | List a directory as an indented tree |
| `glob_files` | Find files by glob pattern |
| `run_shell` | Execute a shell command — **always asks confirmation** |

File edits show a coloured diff before applying. Shell commands always require an explicit `y` from you — the model can never run commands silently.

---

## RAG — indexing your codebase

For large projects, index the codebase so the assistant can retrieve relevant code automatically:

```bash
# Inside the REPL:
/index ./my-project

# Or from the command line:
ca "/index ./my-project"
```

The index is stored in `.chroma/` in your current directory and persists across sessions. Re-indexing skips files whose SHA-256 hash has not changed since the last run, making subsequent calls near-instant for large codebases.

Supported languages for semantic chunking (splits at function/class boundaries): Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP. All other text-based files use fixed-size chunks.

RAG queries are automatically skipped for short or conversational inputs — context is only retrieved when the input looks like an implementation or analysis task.

---

## Configuration

All settings can be overridden via environment variables (prefix `CA_`) or a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `CA_ARCHITECT_MODEL` | `qwen2.5-coder:7b` | Architect model name |
| `CA_IMPLEMENTER_MODEL` | `qwen2.5-coder:14b` | Implementer model name |
| `CA_REVIEWER_MODEL` | `qwen2.5-coder:7b` | Reviewer model (pipeline mode) |
| `CA_TESTER_MODEL` | `qwen2.5-coder:7b` | Tester model (pipeline mode) |
| `CA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for RAG |
| `CA_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `CA_NUM_THREADS` | (all CPU cores) | Threads used for inference |
| `CA_NUM_CTX` | `8192` | Context window size |
| `CA_TEMPERATURE` | `0.2` | Sampling temperature |
| `CA_DEBATE_ENABLED` | `true` | Enable dual-agent debate |
| `CA_DEBATE_ROUNDS` | `2` | Max architect↔implementer rounds |
| `CA_USE_PIPELINE` | `false` | Enable 4-persona pipeline by default |
| `CA_RAG_TOP_K` | `5` | Chunks retrieved per query |
| `CA_SESSIONS_DIR` | `~/.code-assistant/sessions` | Where sessions are saved |

Example `.env`:

```env
CA_IMPLEMENTER_MODEL=qwen2.5-coder:32b-instruct-q4_K_M
CA_NUM_CTX=4096
CA_DEBATE_ENABLED=false
CA_USE_PIPELINE=true
```

---

## Project layout

```
src/code_assistant/
├── config.py               Typed configuration (pydantic-settings)
├── main.py                 REPL, CLI entry point, slash commands
├── agents/
│   ├── base.py             Core agentic loop: streaming + tool calling
│   ├── architect.py        Planner persona and system prompt
│   ├── implementer.py      Coder/critic persona and system prompt
│   ├── reviewer.py         Code reviewer persona (read-only tools)
│   ├── tester.py           QA tester persona (run_shell, PASS/FAIL report)
│   ├── quick.py            Minimal agent for --quick mode
│   ├── pipeline.py         4-persona pipeline orchestration
│   └── orchestrator.py     Mode selection: debate / pipeline / single
├── tools/
│   ├── registry.py         Tool schemas (JSON) and dispatch table
│   ├── file_ops.py         read / write / edit / list_dir / glob
│   └── shell.py            run_shell with confirmation gate
├── rag/
│   ├── indexer.py          File chunking and ChromaDB ingestion
│   └── retriever.py        Semantic query and context formatting
├── session/
│   ├── history.py          Conversation history with auto-compaction
│   └── persistence.py      Save / load sessions as JSON
└── ui/
    ├── console.py          Rich streaming, panels, agent headers
    └── diff.py             Coloured unified diffs
```
