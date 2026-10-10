"""
Model discovery and recommendation system.

Queries Ollama's library for available coding models, detects local hardware
constraints (RAM, GPU, CPU), ranks models by multiple quality signals, and
recommends optimal models for each persona role.

Cache TTL: 24 hours (to avoid spamming the Ollama API).
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

from rich.table import Table
from rich.text import Text

from .config import config
from .logger import get_logger

_log = get_logger(__name__)

# Cache location (machine-level, 24h TTL)
_CACHE_DIR = Path.home() / ".code-assistant"
_CACHE_FILE = _CACHE_DIR / "model_cache.json"
_CACHE_TTL_HOURS = 24

# Ollama library API endpoint
_OLLAMA_LIBRARY_API = "https://ollama.com/api/tags"

# Coding model name patterns to match (others filtered out)
# We match against the full model name with partial string matching
_CODING_MODEL_PATTERNS = {
    "qwen2.5-coder",
    "qwen-coder",
    "codeqwen",
    "codegemma",
    "codellama",
    "deepseek-coder",
    "deepseek-v",  # deepseek-v3, v4, etc. are strong coding models
    "starcoder2",
    "starcoder",
    "granite-code",
    "coder",  # Generic catch-all for models with "coder" in the name
}


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    has_gpu: bool
    gpu_vram_gb: float = 0.0
    gpu_type: str = ""  # "metal", "cuda", "none"

    def __str__(self) -> str:
        gpu_str = f"{self.gpu_type.upper()} GPU ({self.gpu_vram_gb:.0f} GB VRAM)" if self.has_gpu else "CPU only"
        return f"{self.total_ram_gb:.0f} GB RAM, {gpu_str}, {self.cpu_count} cores"


@dataclass
class ModelInfo:
    """Information about an Ollama model."""

    name: str  # e.g. "qwen2.5-coder:14b"
    family: str  # e.g. "qwen2.5-coder"
    size_gb: float  # model size in GB
    param_count: str  # e.g. "14b", "7b", "32b"
    release_date: Optional[str] = None  # ISO format or None if unknown
    downloads: int = 0
    benchmark_score: float = 0.0  # HumanEval or similar (0-100)
    tags: list[str] = field(default_factory=list)

    # Computed fields
    rank_score: float = 0.0  # Overall quality score (0-100)
    fits_in_ram: bool = True
    reason: str = ""  # Human-readable justification


@dataclass
class RecommendationResult:
    """Full recommendation output."""

    hardware: HardwareProfile
    current_models: dict[str, str]  # role -> model name
    recommendations: dict[str, list[ModelInfo]]  # role -> ranked list
    timestamp: str
    cache_hit: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Hardware detection
# ══════════════════════════════════════════════════════════════════════════════


def detect_hardware() -> HardwareProfile:
    """
    Detect system hardware capabilities.

    Returns a HardwareProfile with RAM, CPU, and GPU information.
    Falls back to conservative estimates if psutil is unavailable.
    """
    import os

    # CPU count
    cpu_count = os.cpu_count() or 8

    # RAM detection
    if psutil:
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)
    else:
        # Fallback: assume 16 GB total, 8 GB available (conservative)
        _log.warning("psutil not installed; using conservative RAM estimates")
        total_ram_gb = 16.0
        available_ram_gb = 8.0

    # GPU detection (platform-specific)
    has_gpu = False
    gpu_vram_gb = 0.0
    gpu_type = "none"

    system = platform.system()

    if system == "Darwin":
        # macOS: check for Apple Silicon (Metal)
        machine = platform.machine()
        if machine in ("arm64", "aarch64"):
            has_gpu = True
            gpu_type = "metal"
            # On Apple Silicon, unified memory is shared between CPU/GPU
            # Estimate usable VRAM as 60% of total RAM (conservative)
            gpu_vram_gb = total_ram_gb * 0.6
            _log.debug("Detected Apple Silicon with Metal GPU | vram_estimate=%.1f GB", gpu_vram_gb)

    elif system == "Linux":
        # Linux: check for NVIDIA GPU via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                vram_mb = float(result.stdout.strip().split("\n")[0])
                has_gpu = True
                gpu_type = "cuda"
                gpu_vram_gb = vram_mb / 1024
                _log.debug("Detected NVIDIA GPU via nvidia-smi | vram=%.1f GB", gpu_vram_gb)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            _log.debug("nvidia-smi not found or failed; assuming CPU-only")

    elif system == "Windows":
        # Windows: check for NVIDIA GPU via nvidia-smi (same as Linux)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                vram_mb = float(result.stdout.strip().split("\n")[0])
                has_gpu = True
                gpu_type = "cuda"
                gpu_vram_gb = vram_mb / 1024
                _log.debug("Detected NVIDIA GPU via nvidia-smi | vram=%.1f GB", gpu_vram_gb)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            _log.debug("nvidia-smi not found or failed; assuming CPU-only")

    return HardwareProfile(
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_count=cpu_count,
        has_gpu=has_gpu,
        gpu_vram_gb=gpu_vram_gb,
        gpu_type=gpu_type,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Ollama library API client
# ══════════════════════════════════════════════════════════════════════════════


def query_ollama_library() -> list[ModelInfo]:
    """
    Query local Ollama instance for available coding models.

    Returns a list of ModelInfo objects. Returns empty list on failure.
    Filters to coding model families only.
    """
    try:
        import ollama

        _log.debug("Querying local Ollama instance for available models")

        # Query local Ollama instance
        response = ollama.list()

        # ollama.list() returns a ListResponse object with .models attribute
        local_models = response.models if hasattr(response, 'models') else []

        if not local_models:
            _log.warning("No models found in local Ollama instance")
            return []

        models: list[ModelInfo] = []

        for item in local_models:
            # Model objects have .model attribute (not .name)
            name = item.model if hasattr(item, 'model') else ""
            if not name:
                continue

            # Check if this is a coding model by matching patterns
            name_lower = name.lower()
            is_coding_model = any(pattern in name_lower for pattern in _CODING_MODEL_PATTERNS)

            if not is_coding_model:
                continue

            # Extract family (base name before colon)
            # Examples: "qwen2.5-coder:14b" -> "qwen2.5-coder"
            #           "deepseek-coder:33b-instruct" -> "deepseek-coder"
            family = name.split(":")[0] if ":" in name else name

            # Extract parameter count from the tag after colon
            # "qwen2.5-coder:14b" -> "14b"
            # "deepseek-coder:33b-instruct" -> "33b"
            if ":" in name:
                tag = name.split(":")[1]
                # Extract just the number + 'b' part
                import re
                match = re.search(r'(\d+\.?\d*b)', tag.lower())
                param_count = match.group(1) if match else "unknown"
            else:
                param_count = "unknown"

            # Parse size (bytes -> GB)
            size_bytes = item.size if hasattr(item, 'size') else 0
            size_gb = size_bytes / (1024**3) if size_bytes else 0.0

            # Extract metadata
            # Local Ollama provides modified_at as datetime object
            modified_at = item.modified_at if hasattr(item, 'modified_at') else None
            release_date = modified_at.isoformat() + "Z" if modified_at else None

            # Downloads not available from local instance, use 0
            downloads = 0

            # Benchmark score estimation
            benchmark_score = _estimate_benchmark_score(family, param_count)

            # Tags/details from local Ollama
            details = item.details if hasattr(item, 'details') else None
            if details:
                tags = [
                    getattr(details, 'family', ''),
                    getattr(details, 'parameter_size', '')
                ]
                tags = [t for t in tags if t]  # Remove empty strings
            else:
                tags = []

            models.append(
                ModelInfo(
                    name=name,
                    family=family,
                    size_gb=size_gb,
                    param_count=param_count,
                    release_date=release_date,
                    downloads=downloads,
                    benchmark_score=benchmark_score,
                    tags=tags,
                )
            )

        _log.info("Fetched %d coding models from Ollama library", len(models))
        return models

    except Exception as e:
        _log.error("Failed to query Ollama library: %s", e)
        return []


def _estimate_benchmark_score(family: str, param_count: str) -> float:
    """
    Estimate benchmark score based on known model characteristics.

    This is a placeholder until real benchmark data is integrated.
    Returns a score from 0-100.
    """
    # Base scores by family (from public benchmarks)
    # Higher scores = better coding performance (HumanEval, MBPP, etc.)
    family_lower = family.lower()

    # Match by partial string for flexibility
    if "qwen2.5-coder" in family_lower or "qwen-coder" in family_lower:
        base_score = 75.0
    elif "deepseek-v4" in family_lower or "deepseek-v3" in family_lower:
        base_score = 80.0  # DeepSeek V3/V4 are very strong
    elif "deepseek-coder" in family_lower:
        base_score = 70.0
    elif "codellama" in family_lower:
        base_score = 60.0
    elif "starcoder" in family_lower:
        base_score = 65.0
    elif "codegemma" in family_lower:
        base_score = 58.0
    elif "granite-code" in family_lower:
        base_score = 55.0
    else:
        base_score = 50.0

    # Adjust by parameter count (larger models generally score higher)
    param_multipliers = {
        "0.5b": 0.6,
        "1.3b": 0.65,
        "1.5b": 0.7,
        "2b": 0.75,
        "3b": 0.8,
        "6.7b": 0.85,
        "7b": 0.9,
        "flash": 0.9,  # Flash variants are typically 7b-equivalent
        "13b": 0.95,
        "14b": 1.0,
        "15b": 1.02,
        "32b": 1.1,
        "33b": 1.1,
        "34b": 1.1,
        "pro": 1.1,  # Pro variants are typically 32b+
        "70b": 1.15,
        "236b": 1.2,  # Very large models
    }

    multiplier = param_multipliers.get(param_count.lower(), 1.0)

    return min(base_score * multiplier, 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# Ranking algorithm
# ══════════════════════════════════════════════════════════════════════════════


def rank_models(
    models: list[ModelInfo],
    hw_profile: HardwareProfile,
    role: str = "implementer",
) -> list[ModelInfo]:
    """
    Rank models by multi-factor scoring.

    Factors:
    - Release date (30%): newer models preferred
    - Benchmark score (30%): higher HumanEval/MBPP preferred
    - Parameter count (20%): larger models preferred (if they fit in RAM)
    - Downloads (20%): popularity signal

    Returns models sorted by rank_score (descending), with fits_in_ram flag set.
    """
    if not models:
        return []

    # Determine RAM budget based on role and hardware
    ram_budget_gb = _get_ram_budget(hw_profile, role)

    # Normalize factors to 0-100 scale
    max_downloads = max((m.downloads for m in models), default=1)
    newest_date = max((m.release_date for m in models if m.release_date), default="2020-01-01")

    for model in models:
        # Factor 1: Release date (0-100)
        if model.release_date:
            try:
                model_date = datetime.fromisoformat(model.release_date.replace("Z", "+00:00"))
                newest_dt = datetime.fromisoformat(newest_date.replace("Z", "+00:00"))
                days_diff = (newest_dt - model_date).days
                recency_score = max(0, 100 - (days_diff / 365) * 50)  # Decay over 2 years
            except Exception:
                recency_score = 50.0  # Unknown date = neutral
        else:
            recency_score = 50.0

        # Factor 2: Benchmark score (already 0-100)
        benchmark_score = model.benchmark_score

        # Factor 3: Parameter count (0-100, scaled by size fitting in RAM)
        param_map = {"1.5b": 15, "3b": 30, "7b": 50, "14b": 70, "32b": 90, "70b": 100}
        param_score = param_map.get(model.param_count, 50.0)

        # Penalize if model doesn't fit in RAM
        model.fits_in_ram = model.size_gb <= ram_budget_gb
        if not model.fits_in_ram:
            param_score *= 0.3  # Heavy penalty

        # Factor 4: Downloads (0-100)
        download_score = (model.downloads / max_downloads) * 100 if max_downloads > 0 else 50.0

        # Weighted combination
        model.rank_score = (
            recency_score * 0.30
            + benchmark_score * 0.30
            + param_score * 0.20
            + download_score * 0.20
        )

        # Generate human-readable reason
        model.reason = _generate_reason(model, recency_score, benchmark_score, param_score)

    # Sort by rank_score (descending)
    models.sort(key=lambda m: m.rank_score, reverse=True)

    return models


def _get_ram_budget(hw_profile: HardwareProfile, role: str) -> float:
    """
    Calculate RAM budget for a model based on hardware and role.

    Rules:
    - GPU: use VRAM budget (models run on GPU)
    - CPU: use 60% of available RAM (leave headroom for OS)
    - Architect: allow smaller models (fast inference)
    - Implementer: allow larger models (quality matters)
    """
    if hw_profile.has_gpu and hw_profile.gpu_vram_gb > 0:
        # GPU mode: use VRAM
        # 90% leaves ~2GB headroom on a 22GB system (enough for OS overhead)
        budget = hw_profile.gpu_vram_gb * 0.90  # 90% of VRAM
    else:
        # CPU mode: use available RAM
        budget = hw_profile.available_ram_gb * 0.6  # 60% of available

    # Role-based adjustment
    if role == "architect":
        budget *= 0.7  # Prefer smaller, faster models
    elif role == "implementer":
        budget *= 1.0  # Use full budget for quality

    return budget


def _generate_reason(model: ModelInfo, recency: float, benchmark: float, param: float) -> str:
    """Generate a human-readable reason for the model's ranking."""
    reasons = []

    if recency > 80:
        reasons.append("recently released")
    if benchmark > 70:
        reasons.append(f"{benchmark:.0f}% HumanEval")
    if param > 70 and model.fits_in_ram:
        reasons.append(f"{model.param_count} params")
    elif not model.fits_in_ram:
        reasons.append("⚠️  may not fit in RAM")

    return ", ".join(reasons) if reasons else "stable choice"


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation engine
# ══════════════════════════════════════════════════════════════════════════════


def query_ollama_library_web() -> list[ModelInfo]:
    """
    Query Ollama's web library for available models to pull.

    This queries https://ollama.com/library to find models available
    for download, not just locally installed models.
    """
    try:
        import urllib.request

        _log.debug("Querying Ollama web library for available models")

        # Common coding model families to check
        families_to_check = [
            "qwen2.5-coder",
            "deepseek-coder",
            "codellama",
            "starcoder2",
            "codegemma",
        ]

        models = []

        for family in families_to_check:
            try:
                url = f"https://ollama.com/library/{family}/tags"
                req = urllib.request.Request(url, headers={"User-Agent": "code-assistant"})

                with urllib.request.urlopen(req, timeout=3) as response:
                    html = response.read().decode('utf-8')

                    # Parse common tag formats from the HTML
                    # Look for patterns like ">14b<", ">7b<", ">32b<" in HTML tags
                    import re
                    tag_pattern = r'>(\d+\.?\d*b(?:-[a-z]+)?)<'
                    all_tags = re.findall(tag_pattern, html)

                    # Filter out invalid tags (e.g., timestamps that end in 'b')
                    # Valid tags: "7b", "14b", "32b", "1.5b", "14b-instruct"
                    # Invalid: "72881160103b" (too many digits)
                    tags = []
                    for tag in all_tags:
                        # Extract just the number part before 'b'
                        num_part = tag.split('b')[0]
                        # Valid if it's a reasonable param count (0.1 to 999)
                        try:
                            num_val = float(num_part)
                            if 0.1 <= num_val <= 999:
                                tags.append(tag)
                        except ValueError:
                            continue

                    # Add found models
                    for tag in set(tags):  # Deduplicate
                        model_name = f"{family}:{tag}"

                        # Extract param count
                        param_match = re.search(r'(\d+\.?\d*b)', tag)
                        param_count = param_match.group(1) if param_match else "unknown"

                        # Estimate size based on param count (rough estimates for Q4_K_M quantized models)
                        # Formula: size_gb ≈ param_count_billions * 0.58 (Q4_K_M quantization)
                        size_estimates = {
                            "0.5b": 0.5,
                            "1.3b": 1.0,
                            "1.5b": 1.0,
                            "2b": 1.5,
                            "3b": 2.0,
                            "6.7b": 4.0,
                            "7b": 4.5,
                            "13b": 8.0,
                            "14b": 8.5,
                            "15b": 9.0,
                            "32b": 18.5,
                            "33b": 19.0,
                            "34b": 19.5,
                            "70b": 40.0,
                        }

                        # If not in dict, estimate using formula
                        if param_count in size_estimates:
                            size_gb = size_estimates[param_count]
                        else:
                            # Try to parse number from param_count and estimate
                            try:
                                num = float(param_count.rstrip('b'))
                                size_gb = num * 0.58  # Q4_K_M quantization ratio
                            except (ValueError, AttributeError):
                                size_gb = 5.0  # Fallback

                        benchmark_score = _estimate_benchmark_score(family, param_count)

                        models.append(ModelInfo(
                            name=model_name,
                            family=family,
                            size_gb=size_gb,
                            param_count=param_count,
                            release_date=None,
                            downloads=0,
                            benchmark_score=benchmark_score,
                            tags=["available-to-pull"],
                        ))

            except Exception as e:
                _log.debug("Failed to query %s: %s", family, e)
                continue

        _log.info("Found %d models available in web library", len(models))
        return models

    except Exception as e:
        _log.error("Failed to query Ollama web library: %s", e)
        return []


def get_recommendations(use_cache: bool = True) -> Optional[RecommendationResult]:
    """
    Get model recommendations for all roles.

    Returns a RecommendationResult with ranked suggestions, or None on failure.
    Uses 24h cache if available and fresh.
    """
    # Check cache first
    if use_cache:
        cached = load_cache()
        if cached:
            _log.info("Using cached recommendations (age: %s)", _cache_age_str())
            return cached

    # Detect hardware
    hw_profile = detect_hardware()
    _log.info("Hardware profile: %s", hw_profile)

    # Query local Ollama models (already pulled)
    local_models = query_ollama_library()
    _log.info("Found %d local models", len(local_models))

    # Query web library for available models to pull
    web_models = query_ollama_library_web()
    _log.info("Found %d web library models", len(web_models))

    # Merge: local models + web models (deduplicate by name)
    seen_names = set()
    models = []

    # Local models first (mark them as installed)
    for m in local_models:
        models.append(m)
        seen_names.add(m.name)

    # Add web models that aren't already local
    for m in web_models:
        if m.name not in seen_names:
            models.append(m)
            seen_names.add(m.name)

    if not models:
        _log.warning("No models found (local or web)")
        return None

    # Current models from config
    current_models = {
        "architect": config.effective_architect_model(),
        "implementer": config.effective_implementer_model(),
        "reviewer": config.reviewer_model,
        "tester": config.tester_model,
    }

    # Rank models for each role
    recommendations = {}
    for role in ["architect", "implementer", "reviewer", "tester"]:
        ranked = rank_models(models.copy(), hw_profile, role)
        # Keep top 5 recommendations
        recommendations[role] = ranked[:5]

    # Build result
    result = RecommendationResult(
        hardware=hw_profile,
        current_models=current_models,
        recommendations=recommendations,
        timestamp=datetime.utcnow().isoformat() + "Z",
        cache_hit=False,
    )

    # Save to cache
    save_cache(result)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Cache management
# ══════════════════════════════════════════════════════════════════════════════


def load_cache() -> Optional[RecommendationResult]:
    """
    Load cached recommendations if fresh (< 24h old).

    Returns None if cache doesn't exist, is expired, or is invalid.
    """
    if not _CACHE_FILE.exists():
        return None

    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))

        # Check TTL
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        age = datetime.utcnow().replace(tzinfo=timestamp.tzinfo) - timestamp
        if age > timedelta(hours=_CACHE_TTL_HOURS):
            _log.debug("Cache expired (age: %s)", age)
            return None

        # Reconstruct result
        result = RecommendationResult(
            hardware=HardwareProfile(**data["hardware"]),
            current_models=data["current_models"],
            recommendations={
                role: [ModelInfo(**m) for m in models]
                for role, models in data["recommendations"].items()
            },
            timestamp=data["timestamp"],
            cache_hit=True,
        )

        return result

    except Exception as e:
        _log.warning("Failed to load cache: %s", e)
        return None


def save_cache(result: RecommendationResult) -> None:
    """Save recommendations to cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        data = {
            "hardware": asdict(result.hardware),
            "current_models": result.current_models,
            "recommendations": {
                role: [asdict(m) for m in models]
                for role, models in result.recommendations.items()
            },
            "timestamp": result.timestamp,
        }

        _CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _log.debug("Saved recommendations to cache: %s", _CACHE_FILE)

    except Exception as e:
        _log.warning("Failed to save cache: %s", e)


def _cache_age_str() -> str:
    """Return human-readable cache age string."""
    if not _CACHE_FILE.exists():
        return "N/A"

    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        age = datetime.utcnow().replace(tzinfo=timestamp.tzinfo) - timestamp

        hours = int(age.total_seconds() / 3600)
        if hours < 1:
            return "< 1 hour"
        elif hours == 1:
            return "1 hour"
        else:
            return f"{hours} hours"
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# UI formatting
# ══════════════════════════════════════════════════════════════════════════════


def format_recommendations_table(result: RecommendationResult) -> Table:
    """
    Format recommendations as a Rich Table.

    Shows current vs recommended models for each role.
    Indicates which models are local vs need to be pulled.
    """
    table = Table(title="🔍 Model Recommendations", show_header=True, header_style="bold cyan")

    table.add_column("Role", style="dim", width=12)
    table.add_column("Current", style="yellow", width=25)
    table.add_column("Recommended", style="green", width=50)

    for role in ["architect", "implementer", "reviewer", "tester"]:
        current = result.current_models.get(role, "unknown")
        top_recs = result.recommendations.get(role, [])

        if not top_recs:
            table.add_row(role.capitalize(), current, "[dim]No recommendations[/dim]")
            continue

        # Top recommendation
        top = top_recs[0]

        # Check if current model is already optimal
        if current == top.name:
            rec_text = Text("✅ Already optimal", style="green")
        else:
            rec_text = Text()

            # Check if model needs to be pulled
            is_local = "available-to-pull" not in top.tags
            prefix = "📦" if is_local else "⬇️ "

            rec_text.append(f"{prefix} {top.name}\n", style="bold green")
            rec_text.append(f"   {top.reason}", style="dim")

            # Add pull instruction if needed
            if not is_local and top.fits_in_ram:
                rec_text.append(f"\n   [cyan]Pull: ollama pull {top.name}[/cyan]", style="dim cyan")

        table.add_row(role.capitalize(), current, rec_text)

    # Add hardware footer
    table.caption = f"Hardware: {result.hardware}"
    if result.cache_hit:
        table.caption += f" (cached, age: {_cache_age_str()})"

    return table
