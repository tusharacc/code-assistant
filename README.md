# Code Assistant

A local, privacy-first AI coding assistant that runs **entirely on your machine** using [Ollama](https://ollama.com). No internet required after setup. No API keys. No GPU needed.

The assistant can hold an interactive REPL session, edit files with coloured diffs, run shell commands with your approval, retrieve context from your codebase (RAG), and — for complex implementation tasks — run a full **4-persona pipeline** where Architect, Implementer, Reviewer, and Tester collaborate sequentially with only one model in RAM at a time.

After each pipeline run the assistant automatically learns from its own mistakes, saving mistake→correction pairs to a machine-level shared pool and injecting the most relevant ones into future prompts as few-shot examples — no GPU or weight updates required.

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [Model setup](#model-setup)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Modes](#modes)
   - [Standard (debate)](#standard-mode-debate)
   - [Pipeline](#pipeline-mode)
   - [Quick](#quick-mode)
   - [Spec (requirements gathering)](#spec-mode)
7. [Slash commands](#slash-commands)
8. [Agent tools](#agent-tools)
9. [RAG — indexing your codebase](#rag--indexing-your-codebase)
10. [Feedback loop & few-shot learning](#feedback-loop--few-shot-learning)
11. [Per-project memory](#per-project-memory-code_assistantmd)
12. [Session management](#session-management)
13. [Configuration](#configuration)
14. [Benchmarks](#benchmarks)
15. [Project layout](#project-layout)
16. [Troubleshooting](#troubleshooting)

---

## How it works

### Three execution paths

```
User input
    │
    ├─ Short / conversational ──▶  Single agent (Implementer)
    │
    ├─ Complex task + debate on ──▶  Debate mode
    │       Architect proposes → Implementer reviews → both agree → Implementer codes
    │
    └─ Complex task + pipeline on ──▶  Pipeline mode (recommended for large tasks)
            Phase 1  Architect      — plan
            Phase 2  Implementer    — code  (can Q&A the Architect mid-flight)
            Phase 3  Reviewer       — inspect files, HIGH/MEDIUM/LOW findings
            Phase 4  Implementer    — fix HIGH + MEDIUM issues
            Phase 5  Tester         — run acceptance criteria, PASS/FAIL table
            Phase 6  Tester-fix ×3  — implementer fixes failures, tester re-verifies
            Phase 7  Implementer    — write README.md
```

### Persona summary

| Persona | Default model | Tool access | Job |
|---|---|---|---|
| **Architect** | `qwen2.5-coder:7b` | None (read-only in spec mode) | Plan, design, acceptance criteria |
| **Implementer** | `qwen2.5-coder:14b` | Full (read, write, edit, shell) | Code, fix, document |
| **Reviewer** | `qwen2.5-coder:7b` | Read-only | Inspect, produce findings |
| **Tester** | `qwen2.5-coder:7b` | Read + shell | Run tests, verify criteria |

### The feedback loop

Every time the pipeline completes, the assistant automatically extracts **mistake → correction** pairs from the conversation history:

| Type | What is captured |
|---|---|
| `tool_error` | Assistant made a tool call → got `Error:...` → then fixed it |
| `review_issue` | Reviewer found HIGH/MEDIUM issues → Implementer fixed them |
| `test_failure` | Tester reported FAIL → Implementer applied a fix |

These examples are saved to `~/.code-assistant/feedback/feedback.jsonl` (shared across all projects on the machine) and injected as a `## Common mistakes to avoid` section into the Implementer and Tester system prompts on the **next run** — so the model learns from its own past errors without needing any GPU or weight updates.

After each pipeline run, the per-project `code_assistant.md` memory file is also updated with the latest architecture summary, files written, and test outcomes — see [Per-project memory](#per-project-memory-code_assistantmd).

---

## Prerequisites

### Hardware

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | Not required |
| Disk | ~15 GB | ~20 GB |

Expected throughput on CPU (no GPU):

| Model | RAM (Q4) | Speed |
|---|---|---|
| `qwen2.5-coder:7b` | ~4.5 GB | 10–20 tok/s |
| `qwen2.5-coder:14b` | ~8.5 GB | 5–10 tok/s |

In pipeline mode `keep_alive=0` ensures only one model is in RAM at a time.

### Software

- **OS:** Linux (primary) or macOS. Not tested on Windows.
- **Python:** 3.11 or newer (`python3 --version`)
- **Ollama:** Install from [ollama.com](https://ollama.com/download) or:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version          # verify
```

---

## Model setup

One-time download (~13 GB total):

```bash
# Architect / Reviewer / Tester — fast planner (~4.5 GB)
ollama pull qwen2.5-coder:7b

# Implementer — sharper coder (~8.5 GB)
ollama pull qwen2.5-coder:14b

# Embeddings — for RAG indexing (~274 MB)
ollama pull nomic-embed-text
```

### Why these models?

Both `qwen2.5-coder` variants are code-specific, support Ollama's native function-calling API, and fit comfortably in a 32 GB machine at Q4_K_M quantisation. Using `keep_alive=0` in pipeline mode unloads each model from RAM immediately after it responds.

### Alternative implementer models

| Model | Size (Q4) | Speed | Notes |
|---|---|---|---|
| `qwen2.5-coder:7b` | 4.5 GB | Fast | Default arch/reviewer/tester |
| `qwen2.5-coder:14b` | 8.5 GB | Medium | **Default implementer** |
| `qwen2.5-coder:32b` | 19 GB | Slow | Fits in 32 GB; ~1–2 tok/s |
| `deepseek-coder-v2:16b` | 10 GB | Medium | Good implementer alternative |
| `starcoder2:15b` | 9 GB | Medium | Good alternative |
| `codellama:13b` | 7.5 GB | Medium | Older but reliable |
| `phi4:14b` | 8.5 GB | Medium | Strong reasoning |

Switch models per-project using `ca.config` — see [Configuration](#configuration).

---

## Installation

```bash
git clone <repo-url>
cd code-assistant
pip install -e .
```

This installs the `ca` command into your Python environment.

For benchmark support (optional — requires Anthropic API key):

```bash
pip install -e ".[benchmark]"
```

---

## Quick start

```bash
# Interactive REPL
ca

# One-shot implementation
ca "write a binary search in Python and save it to search.py"

# Direct answer — no REPL, no tools
ca -q "git stash pop vs apply"

# Full pipeline run from a spec file
ca --pipeline --req-file my_spec.txt

# Gather requirements and write a spec
ca --spec
```

---

## Modes

### Standard mode (debate)

Default for interactive sessions. For complex tasks the two agents collaborate:

```
User: "Implement a thread-safe LRU cache"

[Architect]   → Proposes: OrderedDict + RLock, maxsize=0 means unbounded
[Implementer] → Pushes back: use RLock not Lock (get() may call put() internally)
[Architect]   → Revises plan, both agree
[Implementer] → Writes code, creates files, runs verification
```

For simple tasks (explain, describe, read) a single agent responds directly. The mode selection is automatic based on the presence of keywords like _implement_, _build_, _create_, _refactor_.

```bash
ca                           # interactive REPL, debate on
ca "implement a CSV parser"  # one-shot, debate kicks in
ca --no-debate "explain X"   # force single-agent
```

Toggle inside the REPL:

```
/debate on
/debate off
```

### Pipeline mode

A 7-phase sequential workflow where **only one model occupies RAM at a time**.
Best for large, well-defined requirements documents.

```bash
ca --pipeline --req-file spec.txt   # from a spec file
ca -p "build a REST API for todos"  # inline
```

Toggle inside the REPL:

```
/pipeline on
/pipeline off
```

**Phase breakdown:**

| # | Persona | Description |
|---|---|---|
| 1 | Architect | Reads spec, produces detailed implementation plan |
| 2 | Implementer | Codes the plan; uses `@@QUESTION_FOR_ARCHITECT: ...@@` for Q&A |
| 3 | Reviewer | Reads all created files; outputs HIGH / MEDIUM / LOW findings |
| 4 | Implementer | Fixes HIGH and MEDIUM issues; skipped if none found |
| 5 | Gather info | Implementer describes run commands; Architect lists acceptance criteria |
| 6 | Tester (×3) | Runs shell commands, verifies each criterion; loop until PASS or 3 rounds |
| 7 | Implementer | Writes `README.md` with usage, criteria status, and run instructions |

The Implementer can ask the Architect a question mid-implementation by embedding `@@QUESTION_FOR_ARCHITECT: your question here@@` in its response. The pipeline routes the question, feeds the answer back, and the Implementer continues.

### Quick mode

Streams a direct answer with no tools, no panels, no REPL. Ideal for shell lookups, config snippets, or one-liner questions:

```bash
ca --quick --req "ls flag to show hidden files"
ca -q "dockerfile for a Python FastAPI app"
ca -q "difference between INNER JOIN and LEFT JOIN"
```

### Spec mode

The Architect interviews you about your project requirements and produces a structured specification document that can then feed directly into pipeline mode:

```bash
# Interactive interview → writes spec_YYYYMMDD_HHMMSS.txt
ca --spec

# Save spec to a custom path
ca --spec --spec-out requirements/my_feature.txt
```

Inside the spec REPL:

| Command | Description |
|---|---|
| `/finalize` | Write the spec document and exit |
| `/clear` | Restart the discussion from scratch |
| `/help` | Show commands |
| `/exit` | Quit without writing |

The output spec follows this format:

```markdown
# <Project Name>
## Overview
## Goals
## Acceptance Criteria
## Technical Constraints
## Out of Scope
## Implementation Notes
```

Feed it straight into the pipeline:

```bash
ca --pipeline --req-file my_feature.txt
```

After `/finalize`, the project's `code_assistant.md` is automatically updated with the new acceptance criteria in an `## Open Requirements` section.

---

## Slash commands

Type these inside the REPL at any time:

| Command | Description |
|---|---|
| `/add <file\|dir>` | Attach a file or entire directory to the conversation context |
| `/index <dir>` | Index a codebase directory into the RAG vector store |
| `/rag` | Show RAG index status and total chunk count |
| `/debate [on\|off]` | Toggle dual-agent debate mode |
| `/pipeline [on\|off]` | Toggle 4-persona pipeline mode |
| `/save [name]` | Save the current session to disk |
| `/resume <name>` | Load a previously saved session |
| `/sessions` | List all saved sessions |
| `/clear` | Wipe conversation history |
| `/compact` | Manually summarise older history to free context space |
| `/model arch <name>` | Hot-swap the architect model |
| `/model impl <name>` | Hot-swap the implementer model |
| `/config` | Print configuration with source labels (project / machine / env / default) |
| `/config init` | Create a `ca.config` project template in the current directory |
| `/help` | Show command reference |
| `/exit` or Ctrl-D | Quit |

---

## Agent tools

The Implementer and Tester can call these tools autonomously during a response. Every file write and every shell command shows what it is about to do and waits for your `y` before proceeding (unless `auto_approve=true`).

| Tool | Description |
|---|---|
| `read_file(path)` | Read a file; returns content with line numbers |
| `write_file(path, content)` | Create or overwrite — shows a coloured diff, asks confirmation |
| `edit_file(path, old_string, new_string)` | Targeted string replacement — shows diff, asks confirmation |
| `list_dir(path)` | Print directory as an indented tree |
| `glob_files(pattern)` | Find files matching a glob pattern |
| `run_shell(command)` | Execute a shell command — **always prompts, never runs silently** |

The Reviewer and Architect have read-only access (`read_file`, `list_dir`, `glob_files`) to inspect the implementation without modifying anything.

Set `CA_AUTO_APPROVE=true` (default) to skip confirmation prompts — useful in batch / CI scenarios. Override with `--no-auto-approve` to always be prompted.

---

## RAG — indexing your codebase

For large projects, build a semantic index so the assistant retrieves relevant code chunks automatically:

```bash
# Inside the REPL:
/index ./my-project

# From the command line:
ca "/index ./my-project"
```

The index is stored in `.chroma/` in your current directory and persists across sessions. Re-indexing skips unchanged files (SHA-256 hash tracking), so subsequent runs are near-instant.

**Supported languages for semantic chunking** (splits at function / class boundaries):
Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP

All other text-based files use fixed-size chunks (1 500 chars, 200 char overlap).

**Retrieval is automatic and conditional** — context is only fetched when the input looks like an implementation or analysis task (contains action keywords). Short conversational questions skip retrieval entirely.

```bash
# Tune how many chunks are injected
CA_RAG_TOP_K=8        # default: 5
CA_CHUNK_SIZE=2000    # default: 1500 chars
```

---

## Feedback loop & few-shot learning

The assistant learns from its own mistakes **without requiring GPU or fine-tuning** — purely through prompt engineering.

### How it works

```
Pipeline.run() completes
        │
        ▼
FeedbackCollector.collect()
   ├─ extract_tool_errors(impl_history)
   │       assistant used a tool → got "Error:..." → then corrected itself
   ├─ extract_review_cycles(impl_history, review_findings)
   │       reviewer flagged HIGH/MEDIUM → implementer fixed them
   └─ extract_test_cycles(impl_history, test_results)
           tester found FAIL → implementer applied a fix
                │
                ▼
        ~/.code-assistant/feedback/feedback.jsonl  ← shared across ALL projects
                │
        (next run)
                ▼
FewShotEnricher.enrich_impl_system()   ← injects tool_error + review_issue examples
FewShotEnricher.enrich_tester_system() ← injects test_failure examples
```

### What the injected section looks like

```
## Common mistakes to avoid (learned from past runs)

### Mistake 1 — tool_error [edit_file, syntax, lambda]
WRONG — caused this error:
  Error: the exact string was not found in app.py. SyntaxError: invalid syntax

The assistant tried:
  'factorial': lambda x: math.factorial(x) if x >= 0 else raise ValueError(...)

CORRECT fix applied afterwards:
  def _safe_factorial(n):
      if n < 0:
          raise ValueError("factorial of negative number")
      return math.factorial(n)
  ...
  'factorial': _safe_factorial,

---
```

### Machine-level accumulation

Feedback examples accumulate from **all projects** into a single shared pool at
`~/.code-assistant/feedback/feedback.jsonl`. This is intentional — 200+ examples are
needed before LoRA fine-tuning becomes worthwhile, and fragmenting examples per-project
would make that threshold unreachable.

The `feedback_dir`, `feedback_enabled`, and `few_shot_max` settings are **machine-level
only** and cannot be overridden in a project's `ca.config`. Set them in
`~/.code-assistant/config.toml` if you need to change them.

### Configuration

```toml
# ~/.code-assistant/config.toml  (machine-level only)
feedback_enabled = true
feedback_dir     = "~/.code-assistant/feedback"   # default; absolute path required
few_shot_max     = 3
```

### Inspecting collected feedback

```bash
# Show all saved records
cat ~/.code-assistant/feedback/feedback.jsonl | python -m json.tool | head -80

# Count total examples
wc -l ~/.code-assistant/feedback/feedback.jsonl

# Count records by type
grep -o '"mistake_type": "[^"]*"' ~/.code-assistant/feedback/feedback.jsonl | sort | uniq -c
```

### Tier-2: LoRA fine-tuning (GPU required)

Once you have accumulated ~200+ quality examples, export them as ChatML JSONL for actual weight-level fine-tuning:

```bash
# Export to ChatML format
python -m code_assistant.feedback.export --out training.jsonl

# Fine-tune with unsloth (separate GPU machine)
pip install unsloth
python unsloth_train.py \
    --model qwen2.5-coder:14b \
    --data training.jsonl \
    --out ./lora_adapter

# Load adapter into Ollama
echo "FROM qwen2.5-coder:14b
ADAPTER ./lora_adapter" > Modelfile
ollama create qwen-ca-impl -f Modelfile

# Use the fine-tuned model (project-level override)
# In ca.config:
# implementer_model = "qwen-ca-impl"
```

---

## Per-project memory (`code_assistant.md`)

On the first `ca` launch in any directory, a `code_assistant.md` file is silently
created by scanning the project's metadata — no LLM call, no console output.

This file is the AI's **persistent memory for the project**. It is loaded into context
at the start of every session so the assistant already knows the project structure,
recent build outcomes, and any pending requirements — without you having to re-explain
them each time.

It is separate from `README.md`. `README.md` is for human users; `code_assistant.md`
is for the AI.

### What it contains

| Section | Populated by | Content |
|---|---|---|
| `## Project` | First launch | Type, language, package manager, test runner, entry points, key deps |
| `## Module Layout` | First launch | Top-level directories |
| `## Architecture` | Pipeline run | Architect's plan (truncated to ~1200 chars) |
| `## Key Files` | Pipeline run | Files created or edited by the Implementer |
| `## Current Status` | Pipeline run | PASS / PARTIAL / FAIL with date |
| `## Acceptance Criteria` | Pipeline run | Per-criterion PASS/FAIL table |
| `## Work History` | Pipeline run | Cumulative table: date, task, outcome |
| `## Open Requirements` | `ca --spec /finalize` | Checklist from the latest spec |

### Update lifecycle

```
ca launch (any directory)
    └─ ProjectContext().ensure()      ← creates code_assistant.md if missing (silent)
    └─ history.add_context_file()    ← loads it into session context

ca --spec → /finalize
    └─ update_from_spec()            ← updates ## Open Requirements

ca --pipeline
    └─ Pipeline.run() completes
            └─ update_from_pipeline() ← full regeneration (preserves Work History)
```

### Sample file (after first pipeline run)

```markdown
# Project: my-app

> Auto-generated by code-assistant. Do not edit manually — it will be overwritten after each pipeline run.

## Project

- **Type**: Python (pyproject.toml)
- **Description**: A CLI that parses JSON logs
- **Test runner**: pytest

## Architecture

Build a CLI using argparse...

## Key Files

- src/my_app/main.py
- src/my_app/parser.py
- tests/test_parser.py

## Current Status

**✓ PASS** — all acceptance criteria passed (2026-03-15)

## Work History

| Date | Task | Outcome |
|------|------|---------|
| 2026-03-15 | Build a Python script that parses JSON logs | PASS |

## Open Requirements

*(none)*
```

### Configuration

```bash
CA_PROJECT_CONTEXT_ENABLED=false   # disable entirely
CA_PROJECT_CONTEXT_FILE=ai.md      # change the filename (default: code_assistant.md)
```

---

## Session management

Sessions save the full conversation history to `~/.code-assistant/sessions/`.

```bash
# Save inside REPL
/save my-feature

# Load on next launch
ca --resume my-feature

# From inside REPL
/resume my-feature

# List all sessions
/sessions
```

Sessions are stored as plain JSON and can be inspected or edited manually.

**History compaction:** When the total conversation exceeds `CA_MAX_HISTORY_CHARS` (default 24 000), the assistant automatically summarises the oldest half of the conversation to keep the context window manageable. Trigger manually with `/compact`.

---

## Configuration

Settings are resolved in this priority order (highest wins):

| Priority | Source | Notes |
|---|---|---|
| 1 (highest) | CLI flags / `/model` command | Runtime-only; not persisted |
| 2 | `CA_*` environment variables | e.g. `CA_IMPLEMENTER_MODEL=...` |
| 3 | `.env` file | In the project root; good for secrets |
| 4 | `ca.config` | **Project-level TOML** — in the project root |
| 5 | `~/.code-assistant/config.toml` | **Machine-level TOML** — applies to all projects |
| 6 (lowest) | Built-in defaults | Values shown in the tables below |

### Project-level config (`ca.config`)

Create a `ca.config` file in any project root to override models and settings for that project only. Use `/config init` inside the REPL to generate a commented template.

```toml
# ca.config — project-level overrides
implementer_model = "starcoder2:15b"   # switch from qwen to starcoder for this project
num_ctx           = 4096               # tighter context window
auto_approve      = false              # always ask before editing files
```

Field names are the same as the `CA_*` variable names in lowercase without the prefix.

**Fields that work in `ca.config`:** `implementer_model`, `architect_model`,
`reviewer_model`, `tester_model`, `embed_model`, `ollama_host`, `num_ctx`,
`num_threads`, `num_batch`, `temperature`, `auto_approve`, `debate_enabled`,
`debate_rounds`, `use_pipeline`, `chroma_path`, `chunk_size`, `chunk_overlap`,
`rag_top_k`, `project_context_enabled`, `project_context_file`

**Fields that do NOT work in `ca.config`** (silently ignored — set these in the machine config instead):
`feedback_enabled`, `feedback_dir`, `few_shot_max`, `sessions_dir`, `log_dir`, `log_level`

### Machine-level config (`~/.code-assistant/config.toml`)

Applies to all projects on this machine. Same flat TOML format. All fields accepted here.

```toml
# ~/.code-assistant/config.toml — machine-level defaults
implementer_model = "qwen2.5-coder:14b"
num_threads       = 12
feedback_enabled  = true
few_shot_max      = 5
log_level         = "INFO"
```

### Models

| Variable | Default | Description |
|---|---|---|
| `CA_ARCHITECT_MODEL` | `qwen2.5-coder:7b` | Architect / planner model |
| `CA_IMPLEMENTER_MODEL` | `qwen2.5-coder:14b` | Coder model |
| `CA_REVIEWER_MODEL` | `qwen2.5-coder:7b` | Code reviewer model |
| `CA_TESTER_MODEL` | `qwen2.5-coder:7b` | QA tester model |
| `CA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for RAG |
| `CA_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

### Inference tuning

| Variable | Default | Description |
|---|---|---|
| `CA_NUM_THREADS` | (all cores) | Threads for inference |
| `CA_NUM_CTX` | `8192` | Context window size (tokens) |
| `CA_NUM_BATCH` | `512` | Prompt processing batch size |
| `CA_TEMPERATURE` | `0.2` | Sampling temperature (lower = more deterministic) |

### Behaviour

| Variable | Default | Description |
|---|---|---|
| `CA_AUTO_APPROVE` | `true` | Auto-approve all file/shell tool calls |
| `CA_DEBATE_ENABLED` | `true` | Enable dual-agent debate for complex tasks |
| `CA_DEBATE_ROUNDS` | `2` | Max architect↔implementer rounds |
| `CA_USE_PIPELINE` | `false` | Enable 4-persona pipeline by default |

### RAG

| Variable | Default | Description |
|---|---|---|
| `CA_CHROMA_PATH` | `.chroma` | RAG index directory |
| `CA_CHUNK_SIZE` | `1500` | Characters per code chunk |
| `CA_CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `CA_RAG_TOP_K` | `5` | Chunks injected per query |

### Feedback / few-shot learning

> **Machine-level only.** Set these in `~/.code-assistant/config.toml`, not `ca.config`.
> Feedback must accumulate from all projects into a shared pool.

| Variable | Default | Description |
|---|---|---|
| `CA_FEEDBACK_ENABLED` | `true` | Enable feedback collection and few-shot injection |
| `CA_FEEDBACK_DIR` | `~/.code-assistant/feedback` | Directory for `feedback.jsonl` |
| `CA_FEW_SHOT_MAX` | `3` | Max past examples injected per system prompt |

### Per-project memory

| Variable | Default | Description |
|---|---|---|
| `CA_PROJECT_CONTEXT_ENABLED` | `true` | Enable `code_assistant.md` generation |
| `CA_PROJECT_CONTEXT_FILE` | `code_assistant.md` | Filename for the AI memory file |

### Persistence & logging

| Variable | Default | Description |
|---|---|---|
| `CA_SESSIONS_DIR` | `~/.code-assistant/sessions` | Session storage directory |
| `CA_MAX_HISTORY_CHARS` | `24000` | Trigger history compaction above this threshold |
| `CA_LOG_LEVEL` | `DEBUG` | `DEBUG` \| `INFO` \| `ERROR` |
| `CA_LOG_DIR` | `ca_logs` | Log file directory (relative to working directory) |

### Example configs

**Project-level** (`ca.config` in the project root):

```toml
# Switch to a different implementer for this specific project
implementer_model = "deepseek-coder-v2:16b"
num_ctx           = 4096
auto_approve      = false
```

**Machine-level** (`~/.code-assistant/config.toml`):

```toml
# Applies to all projects on this machine
implementer_model = "qwen2.5-coder:14b"
num_threads       = 12
temperature       = 0.1
feedback_enabled  = true
few_shot_max      = 5
log_level         = "INFO"
```

**Environment / `.env`** (still valid — takes priority over TOML files):

```env
CA_IMPLEMENTER_MODEL=qwen2.5-coder:32b-instruct-q4_K_M
CA_NUM_CTX=4096
CA_AUTO_APPROVE=false
CA_LOG_LEVEL=INFO
```

---

## Benchmarks

The benchmark harness measures the pipeline against Claude API across multiple requirements.

### Setup

```bash
pip install -e ".[benchmark]"
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run a single benchmark

```bash
python -m benchmarks.harness --req benchmarks/req_01_calculator.txt
```

This runs both the local pipeline and Claude API on the same requirement, evaluates the outputs, and writes results to `benchmarks/results/<timestamp>/`.

### Run all requirements

```bash
for req in benchmarks/req_*.txt; do
    python -m benchmarks.harness --req "$req"
done
```

### Compare multiple runs

```bash
python benchmarks/compare.py \
    --dirs benchmarks/results/20260314_114231 \
           benchmarks/results/20260314_152828 \
           benchmarks/results/20260314_205148 \
           benchmarks/results/20260314_205146 \
           benchmarks/results/20260314_221802 \
           benchmarks/results/20260314_223607
```

### Benchmark results (req_01 — calculator app)

| Metric | Code Assistant (CPU) | Claude API |
|---|---|---|
| Total time | 29m 50s | 18m 57s |
| API calls | — | 24 |
| Tokens in | — | 315,513 |
| Tokens out | — | 74,322 |
| Est. cost | free | ~$2.06 |
| Files written | — | 14 |
| Total lines (py) | — | 2,235 |
| Syntax errors | 1 | 0 |
| Tests passed | 0 | 218 |
| Tests failed | 0 | 2 |

> **Note:** Claude wins on output quality (220 tests vs 0, 30–50× more code). The local pipeline wins on cost ($0 vs $2–$9 per run) and privacy (100% local). The feedback loop is specifically designed to close this quality gap over time.

---

## Project layout

```
code-assistant/
├── pyproject.toml              Build config and dependency list
├── README.md                   This file
├── CLAUDE.md                   Context for Claude Code (conventions, patterns, gotchas)
├── src/code_assistant/
│   ├── __init__.py
│   ├── main.py                 REPL entry point, CLI args, slash commands
│   ├── config.py               Layered config: pydantic-settings + TOML sources
│   ├── project_context.py      ProjectScanner + ProjectContext — per-project AI memory
│   ├── logger.py               Rotating file logger setup
│   ├── agents/
│   │   ├── base.py             Core Agent class: streaming, tool loop, token counting
│   │   ├── architect.py        Planner persona + spec-architect variant
│   │   ├── implementer.py      Coder persona (full tool access)
│   │   ├── reviewer.py         Reviewer persona (read-only)
│   │   ├── tester.py           QA tester persona (read + shell)
│   │   ├── quick.py            Minimal agent for --quick mode
│   │   ├── orchestrator.py     Mode selection: single / debate / pipeline
│   │   └── pipeline.py         7-phase sequential pipeline
│   ├── tools/
│   │   ├── registry.py         Tool schemas (JSON) and dispatch table
│   │   ├── file_ops.py         read / write / edit / list_dir / glob
│   │   └── shell.py            run_shell with confirmation gate
│   ├── rag/
│   │   ├── indexer.py          File chunking → embeddings → ChromaDB
│   │   └── retriever.py        Semantic query + context formatting
│   ├── session/
│   │   ├── history.py          Conversation history with auto-compaction
│   │   └── persistence.py      Save / load sessions as JSON
│   ├── feedback/
│   │   ├── collector.py        Extract mistake→correction pairs from pipeline state
│   │   ├── enricher.py         Load examples, inject few-shot block into prompts
│   │   └── export.py           Export feedback.jsonl → ChatML JSONL for LoRA
│   └── ui/
│       ├── console.py          Rich streaming panels, agent headers, confirmations
│       └── diff.py             Coloured unified diffs for file edits
├── benchmarks/
│   ├── harness/                Runner + evaluator modules
│   ├── compare.py              Multi-run comparison report generator
│   ├── req_01_calculator.txt   Benchmark spec: calculator CLI + REPL
│   ├── req_02_todo_webapp.txt  Benchmark spec: todo web app
│   ├── req_03_log_analyser.txt Benchmark spec: log analysis tool
│   └── results/                Timestamped benchmark outputs
└── projects/
    └── calculator/             Sample output project (from a benchmark run)
```

---

## Troubleshooting

### Ollama is not running

```
Error: [Errno 111] Connection refused
```

Start Ollama:

```bash
ollama serve &
```

Or ensure the `ollama` daemon is running as a system service.

### Model not found

```
Error: model 'qwen2.5-coder:14b' not found
```

Pull the model:

```bash
ollama pull qwen2.5-coder:14b
```

### Context window exceeded

The model stops responding mid-task or produces truncated output. Reduce context usage:

```bash
CA_NUM_CTX=4096    # halve the context window
CA_RAG_TOP_K=3     # inject fewer RAG chunks
```

Or use a model with a larger native context window.

### Pydantic validation error on startup

```
ValidationError: Extra inputs are not permitted
```

The `.env` file contains a key that is not prefixed with `CA_` (e.g. `ANTHROPIC_API_KEY`). This is handled automatically since `config.py` sets `"extra": "ignore"`. If you see this error, verify you are running the current version.

### Tool call parsing failures

Some smaller or older models do not reliably emit tool calls via the API. The assistant includes a fallback JSON parser that detects tool calls embedded as text in the response. If the fallback is triggering frequently, switch to a model with better function-calling support (`qwen2.5-coder:14b` or later).

### Pipeline runs out of context on long tasks

On a 32 GB machine with `num_ctx=8192`, very long pipeline runs (many files, many review cycles) can exhaust the context window. Options:

1. Reduce `CA_NUM_CTX` slightly and accept some truncation.
2. Switch to a model with larger native context.
3. Split the requirement into smaller sub-tasks and run the pipeline multiple times.

### Feedback examples not appearing in prompts

Check that `feedback.jsonl` exists and has content:

```bash
wc -l ~/.code-assistant/feedback/feedback.jsonl
```

If empty, the pipeline has not completed a run that generated extractable examples yet. Run the pipeline at least once and look for the summary line:

```
[info] Feedback: N new example(s) saved → ~/.code-assistant/feedback/feedback.jsonl
```

If you want to disable the feature entirely: `CA_FEEDBACK_ENABLED=false` (or set in `~/.code-assistant/config.toml`).

### `code_assistant.md` not being created

Check that the feature is enabled:

```bash
# Verify the setting
ca  # then /config — look for project_context_enabled
```

If `CA_PROJECT_CONTEXT_ENABLED=false` is set in your environment or machine config, unset it. Also note that `ensure()` is idempotent — if `code_assistant.md` already exists it will not be regenerated on launch; only a pipeline run triggers a full update.

### ca.config settings not taking effect

Restart `ca` — the config is loaded once at startup. Also verify the field name matches
exactly (no `CA_` prefix, lowercase, underscores):

```toml
# CORRECT
implementer_model = "starcoder2:15b"

# WRONG (prefix)
CA_IMPLEMENTER_MODEL = "starcoder2:15b"
```

Fields related to feedback, sessions, and logging are silently ignored in `ca.config`
by design — set them in `~/.code-assistant/config.toml` instead.
