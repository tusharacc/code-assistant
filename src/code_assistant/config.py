import os
import tomllib
from pathlib import Path
from typing import Any, Tuple

from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


# ---------------------------------------------------------------------------
# Custom TOML settings source
# ---------------------------------------------------------------------------

# Fields that must NEVER be overridden in a project-level ca.config.
# They are machine-scope concerns: feedback accumulates across ALL projects
# into a shared pool (200+ examples needed for LoRA fine-tuning), and
# session/log directories are per-machine infrastructure, not per-project.
# These fields are silently dropped when the project TomlFileSource is used.
_PROJECT_EXCLUDED_FIELDS: frozenset[str] = frozenset({
    "feedback_enabled",
    "feedback_dir",
    "few_shot_max",
    "sessions_dir",
    "log_dir",
    "log_level",
    # Web / search — credentials and global toggles are machine-scope
    "serper_api_key",
})


class TomlFileSource(PydanticBaseSettingsSource):
    """
    Read settings from a flat TOML file.

    Both files use the same flat format — field names match the Config
    attributes directly (no nested sections needed):

        implementer_model = "starcoder2:15b"
        num_ctx           = 4096
        auto_approve      = false

    The file is loaded once at instantiation; missing files are silently
    ignored (the source contributes an empty dict).

    Parameters
    ----------
    exclude_fields : frozenset[str] | None
        Field names that this source will never supply, even if present in
        the TOML file.  Used by the project-level source to block settings
        that must remain machine-scope (feedback_dir, sessions_dir, etc.).
    """

    def __init__(
        self,
        settings_cls: type,
        toml_path: Path,
        exclude_fields: "frozenset[str] | None" = None,
    ) -> None:
        super().__init__(settings_cls)
        self._path = toml_path
        self._exclude = exclude_fields or frozenset()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return tomllib.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        if field_name in self._exclude:
            return None, field_name, False
        val = self._data.get(field_name)
        return val, field_name, self.field_is_complex(field)

    def __call__(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self._data.items()
            if v is not None and k not in self._exclude
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config(BaseSettings):
    # Ollama connection
    ollama_host: str = "http://localhost:11434"

    # Device mode — controls GPU offloading passed to Ollama as num_gpu.
    # "auto"  — let Ollama decide (default; uses Metal on Apple Silicon automatically)
    # "cpu"   — force CPU only (num_gpu=0); useful for debugging or memory-limited runs
    # "metal" — explicitly enable Metal GPU offloading (num_gpu=-1, all layers on GPU)
    # "cuda"  — explicitly enable CUDA GPU offloading (num_gpu=-1, all layers on GPU)
    # Per-device model size presets below are activated when device="metal" or "cuda".
    device: str = "auto"

    # Models — defaults sized for 32 GB CPU machine
    # Architect: fast 7B for planning/reasoning
    # Implementer: sharper 14B for writing and reviewing code
    architect_model: str = "qwen2.5-coder:7b"
    implementer_model: str = "qwen2.5-coder:14b"
    embed_model: str = "nomic-embed-text"

    # Per-GPU model size presets — used when device="metal" or device="cuda".
    # On GPU, larger models fit in VRAM and inference is fast enough to be practical.
    # Set to "" to fall back to the base architect_model / implementer_model above.
    gpu_architect_model: str = "qwen2.5-coder:32b"
    gpu_implementer_model: str = "qwen2.5-coder:32b"

    # Classification model — used for intent routing (conversational/implementation/complex).
    # Defaults to architect_model when empty. Decouple this if you upgrade the architect
    # to a larger reasoning model and want classification to stay on a small fast model.
    classification_model: str = ""

    # CPU inference tuning
    num_threads: int = Field(default_factory=lambda: os.cpu_count() or 8)
    num_ctx: int = 32768      # context window — raised from 8192; pipeline runs easily hit 20k tokens
    num_batch: int = 512      # chunked prompt processing
    temperature: float = 0.2  # low for deterministic code generation

    # Reviewer and Tester models (fast 7B models are fine for these roles)
    reviewer_model: str = "qwen2.5-coder:7b"
    tester_model: str = "qwen2.5-coder:7b"

    # Tool confirmation — auto-approve all write_file / edit_file / run_shell calls by default.
    # Set to False (or CA_AUTO_APPROVE=false / --no-auto-approve) to be prompted for each action.
    auto_approve: bool = True

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
    rag_always_query: bool = False  # if True, query RAG on every input ≥ _RAG_MIN_CHARS

    # AST symbol index
    ast_path: str = ".ca_ast"  # directory for SQLite symbol index (relative to cwd)

    # Web tools
    # fetch_url is always available (stdlib urllib, no key needed).
    # web_search requires this toggle plus either a serper_api_key or
    # the duckduckgo-search package (pip install duckduckgo-search).
    #
    # web_search_enabled can be set in ca.config (project) or
    # ~/.code-assistant/config.toml (machine).
    #
    # serper_api_key MUST be set in ~/.code-assistant/config.toml or as
    # CA_SERPER_API_KEY env var — never in ca.config (it's a credential).
    # If omitted, web_search falls back to DuckDuckGo (free, no key).
    web_search_enabled: bool = False
    serper_api_key: str = ""   # machine-level only — see _PROJECT_EXCLUDED_FIELDS

    # Session persistence
    sessions_dir: str = str(Path.home() / ".code-assistant" / "sessions")

    # History management — compact when total chars exceeds this
    max_history_chars: int = 24000

    # Feedback / few-shot learning
    # After each pipeline run, mistake→correction pairs are extracted from the
    # message history and saved to feedback_dir/feedback.jsonl.  On subsequent
    # runs the most recent few_shot_max examples are injected into the
    # implementer and tester system prompts so the model avoids repeating
    # the same mistakes — no GPU or weight updates needed.
    #
    # ⚠  MACHINE-LEVEL SETTINGS — set these in ~/.code-assistant/config.toml,
    #    NOT in a project-level ca.config.  Feedback must accumulate from ALL
    #    projects into one pool; 200+ examples are needed before LoRA fine-tuning
    #    becomes worthwhile.  Per-project overrides would fragment the dataset.
    feedback_enabled: bool = True
    feedback_dir: str = str(Path.home() / ".code-assistant" / "feedback")
    few_shot_max: int = 3   # max examples injected per system prompt

    # Per-project AI memory file
    # On first `ca` launch in any directory, a code_assistant.md file is
    # auto-generated by scanning project metadata (pyproject.toml, package.json,
    # directory structure) — no LLM call required.  The file is loaded into
    # context at the start of every session so the AI remembers the project.
    # It is fully regenerated after every pipeline run and after ca --spec /finalize.
    project_context_enabled: bool = True
    project_context_file: str = "code_assistant.md"   # relative to cwd where ca is launched

    # ca_memory/ — project memory system.
    # Maintains file_registry.md, task_log.md, and archived requirement files so
    # every pipeline agent knows which files pre-existed the current run.
    ca_memory_enabled: bool = True
    ca_memory_dir:     str  = "ca_memory"             # relative to cwd

    # Extra context files to load into every session.
    # Paths are relative to the directory where `ca` is launched (or absolute).
    # Set in ca.config for per-project defaults; override per-run with --context.
    # Example in ca.config:  context_files = ["README.md", "docs/api.md"]
    context_files: list[str] = Field(default_factory=list)

    # Logging
    # Level: DEBUG (log everything) | INFO (lifecycle events only) | ERROR (failures only)
    # Default is DEBUG while the application is in active development.
    # Switch to INFO once stable.
    log_level: str = "DEBUG"
    log_dir: str = "ca_logs"   # relative to cwd where `ca` is launched

    # ------------------------------------------------------------------
    # Device / model helpers
    # ------------------------------------------------------------------

    def effective_architect_model(self) -> str:
        """Return the architect model for the current device."""
        if self.device in ("metal", "cuda") and self.gpu_architect_model:
            return self.gpu_architect_model
        return self.architect_model

    def effective_implementer_model(self) -> str:
        """Return the implementer model for the current device."""
        if self.device in ("metal", "cuda") and self.gpu_implementer_model:
            return self.gpu_implementer_model
        return self.implementer_model

    def effective_classification_model(self) -> str:
        """Return the model used for intent classification.

        Falls back to architect_model (never the GPU preset — classification
        is a 1-token call and should always stay on the smallest fast model).
        """
        return self.classification_model or self.architect_model

    def ollama_num_gpu(self) -> int | None:
        """Return num_gpu value for Ollama options, or None to omit the field.

        None (auto) — Ollama decides; on Apple Silicon it uses Metal automatically.
        0           — CPU only.
        -1          — All layers on GPU (Metal or CUDA).
        """
        if self.device == "cpu":
            return 0
        if self.device in ("metal", "cuda"):
            return -1
        return None  # auto — do not pass num_gpu to Ollama

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("feedback_dir", mode="before")
    @classmethod
    def _resolve_feedback_dir(cls, v: str) -> str:
        """
        Always resolve feedback_dir to an absolute path.

        This prevents a ca.config entry like:
            feedback_dir = "feedback"
        from silently redirecting examples into the project directory instead
        of the intended machine-level location.
        """
        return str(Path(v).expanduser().resolve())

    # ------------------------------------------------------------------
    # Source customisation — layered config priority (highest → lowest):
    #   1. init kwargs (runtime mutations: CLI flags, /model command)
    #   2. CA_* environment variables
    #   3. .env file
    #   4. ./ca.config          (project-level TOML)
    #   5. ~/.code-assistant/config.toml  (machine-level TOML)
    #   6. built-in defaults
    # ------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: "type[Config]",
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> "tuple[PydanticBaseSettingsSource, ...]":
        return (
            init_settings,       # runtime mutations / __init__ kwargs
            env_settings,        # CA_* environment variables
            dotenv_settings,     # .env file
            # Project-level: excludes machine-only fields so feedback, sessions,
            # and logging always come from the machine config or built-in defaults.
            TomlFileSource(
                settings_cls,
                Path.cwd() / "ca.config",
                exclude_fields=_PROJECT_EXCLUDED_FIELDS,
            ),
            # Machine-level: all fields accepted (no exclusions)
            TomlFileSource(
                settings_cls,
                Path.home() / ".code-assistant" / "config.toml",
            ),
            file_secret_settings,
        )

    def config_sources(self) -> dict[str, str]:
        """
        Return a {field_name: source_label} dict describing where each
        config value came from.

        Labels: "project" | "machine" | "env" | "default"

        Used by the /config slash command to display source annotations.
        """
        machine_data: dict[str, Any] = {}
        project_data: dict[str, Any] = {}

        machine_path = Path.home() / ".code-assistant" / "config.toml"
        if machine_path.exists():
            try:
                machine_data = tomllib.loads(
                    machine_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        project_path = Path.cwd() / "ca.config"
        if project_path.exists():
            try:
                project_data = tomllib.loads(
                    project_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        # Collect field names that were set via CA_* environment variables
        env_fields: set[str] = {
            k.lower().removeprefix("ca_")
            for k in os.environ
            if k.upper().startswith("CA_")
        }

        result: dict[str, str] = {}
        for field_name in self.model_fields:
            if field_name in env_fields:
                result[field_name] = "env"
            elif field_name in project_data:
                result[field_name] = "project"
            elif field_name in machine_data:
                result[field_name] = "machine"
            else:
                result[field_name] = "default"
        return result

    model_config = {"env_prefix": "CA_", "env_file": ".env", "extra": "ignore"}


# Singleton — import this everywhere
config = Config()
