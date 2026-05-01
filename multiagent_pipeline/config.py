"""Centralised configuration for the multi-agent pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_anthropic_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY")


def get_anthropic_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")


def get_use_llm(default: bool = False) -> bool:
    return _to_bool(os.getenv("USE_LLM"), default)


def get_dry_run(default: bool = False) -> bool:
    return _to_bool(os.getenv("DRY_RUN"), default)

