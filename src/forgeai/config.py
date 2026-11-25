"""Runtime settings, read from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv()  # also allow a .env in the CWD


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    base_url: str = field(default_factory=lambda: os.getenv("FORGE_BASE_URL", "http://localhost:11434/v1").rstrip("/"))
    api_key: str = field(default_factory=lambda: os.getenv("FORGE_API_KEY", "ollama"))
    model: str = field(default_factory=lambda: os.getenv("FORGE_MODEL", "qwen2.5:3b"))
    judge_model: str = field(default_factory=lambda: os.getenv("FORGE_JUDGE_MODEL", "") or os.getenv("FORGE_MODEL", "qwen2.5:3b"))
    embed_model: str = field(default_factory=lambda: os.getenv("FORGE_EMBED_MODEL", "nomic-embed-text"))
    temperature: float = field(default_factory=lambda: _f("FORGE_TEMPERATURE", 0.1))
    request_timeout: float = field(default_factory=lambda: _f("FORGE_REQUEST_TIMEOUT", 300))
    max_retries: int = field(default_factory=lambda: int(_f("FORGE_MAX_RETRIES", 3)))
    max_tokens: int = field(default_factory=lambda: int(_f("FORGE_MAX_TOKENS", 0)))
    max_seconds: float = field(default_factory=lambda: _f("FORGE_MAX_SECONDS", 0))
    min_gain: float = field(default_factory=lambda: _f("FORGE_MIN_GAIN", 0.02))
    prompts_dir: Path = field(default_factory=lambda: Path(os.getenv("FORGE_PROMPTS_DIR", REPO_ROOT / "prompts")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("FORGE_CACHE_DIR", REPO_ROOT / ".forge_cache")))

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


def get_settings() -> Settings:
    return Settings()
