import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Ollama connection
    ollama_host: str = "http://localhost:11434"

    # Models — defaults sized for 32 GB CPU machine
    # Architect: fast 7B for planning/reasoning
    # Implementer: sharper 14B for writing and reviewing code
    architect_model: str = "qwen2.5-coder:7b"
    implementer_model: str = "qwen2.5-coder:14b"
    embed_model: str = "nomic-embed-text"

    # CPU inference tuning
    num_threads: int = Field(default_factory=lambda: os.cpu_count() or 8)
    num_ctx: int = 8192       # context window — don't go higher on CPU
    num_batch: int = 512      # chunked prompt processing
    temperature: float = 0.2  # low for deterministic code generation

    # Reviewer and Tester models (fast 7B models are fine for these roles)
    reviewer_model: str = "qwen2.5-coder:7b"
    tester_model: str = "qwen2.5-coder:7b"

    # Debate mode
    debate_enabled: bool = True
    debate_rounds: int = 2    # max architect↔implementer back-and-forth rounds

    # Pipeline mode — 4-persona sequential flow with keep_alive=0 (one model at a time)
    use_pipeline: bool = False

    # RAG
    chroma_path: str = ".chroma"
    chunk_size: int = 1500    # characters per code chunk
    chunk_overlap: int = 200
    rag_top_k: int = 5        # chunks to inject per query

    # Session persistence
    sessions_dir: str = str(Path.home() / ".code-assistant" / "sessions")

    # History management — compact when total chars exceeds this
    max_history_chars: int = 24000

    # Logging
    # Level: DEBUG (log everything) | INFO (lifecycle events only) | ERROR (failures only)
    # Default is DEBUG while the application is in active development.
    # Switch to INFO once stable.
    log_level: str = "DEBUG"
    log_dir: str = "ca_logs"   # relative to cwd where `ca` is launched

    model_config = {"env_prefix": "CA_", "env_file": ".env"}


# Singleton — import this everywhere
config = Config()
