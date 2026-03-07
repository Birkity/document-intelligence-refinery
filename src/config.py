"""Centralized configuration for the Document Intelligence Refinery.

Loads settings from environment variables (with .env support via
python-dotenv) and exposes them as typed, validated Pydantic models.

Usage::

    from src.config import get_settings
    cfg = get_settings()
    print(cfg.openrouter_api_key)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

# Project root — two levels up from src/config.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Try loading .env from project root (best-effort)
_dotenv_path = _PROJECT_ROOT / ".env"


class RefinerySettings(BaseSettings):
    """All configurable settings for the refinery pipeline.

    Values are read from environment variables.  A ``.env`` file in the
    project root is loaded automatically if present.
    """

    # ── OpenRouter API ──────────────────────────────────────────────
    openrouter_api_key: str = Field(
        default="", description="OpenRouter API key"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL",
    )

    # ── Model names ─────────────────────────────────────────────────
    vision_model: str = Field(
        default="google/gemma-3-27b-it:free",
        description="Vision model for extreme extraction (OpenRouter only)",
    )
    reasoning_model: str = Field(
        default="qwen/qwen3-coder:480b-cloud",
        description="OpenRouter reasoning model (kept for vision fallback only)",
    )
    extraction_model: str = Field(
        default="deepseek/deepseek-v3.1:671b-cloud",
        description="OpenRouter extraction model (kept for vision fallback only)",
    )

    # ── Ollama (local LLM server) ────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama OpenAI-compatible API base URL",
    )
    ollama_model: str = Field(
        default="qwen3-coder:480b-cloud",
        description="Local Ollama model for PageIndex summaries, QueryAgent reasoning, and FactTable extraction",
    )

    # ── Budget ──────────────────────────────────────────────────────
    budget_max_per_document: float = Field(
        default=0.50, description="Max USD per document"
    )
    budget_max_vision_calls: int = Field(
        default=5, description="Max vision LLM calls per document"
    )
    budget_max_llm_calls: int = Field(
        default=20, description="Max LLM calls per document"
    )

    # ── Confidence thresholds ───────────────────────────────────────
    confidence_fast_text_min: float = Field(
        default=0.6, description="Min confidence for fast-text"
    )
    confidence_layout_min: float = Field(
        default=0.5, description="Min confidence for layout strategy"
    )
    confidence_review_flag: float = Field(
        default=0.4, description="Flag for review below this"
    )

    # ── Feature flags ───────────────────────────────────────────────
    enable_vision_extraction: bool = Field(
        default=False, description="Enable vision LLM extraction"
    )
    enable_llm_fact_extraction: bool = Field(
        default=False, description="Enable LLM-assisted fact extraction"
    )
    enable_page_level_escalation: bool = Field(
        default=True, description="Escalate per-page instead of full doc"
    )
    enable_ocr_fallback: bool = Field(
        default=True, description="Enable RapidOCR fallback"
    )

    # ── Paths ───────────────────────────────────────────────────────
    refinery_db_path: str = Field(
        default=".refinery/refinery.db",
        description="SQLite database path (relative to project root)",
    )
    refinery_chroma_dir: str = Field(
        default=".refinery/chroma_store",
        description="ChromaDB persistence directory",
    )
    refinery_ledger_path: str = Field(
        default=".refinery/extraction_ledger.jsonl",
        description="Extraction ledger JSONL path",
    )
    refinery_runs_dir: str = Field(
        default=".refinery/runs",
        description="Pipeline run artifacts directory",
    )

    # ── Logging ─────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Log level")

    model_config = {
        "env_file": str(_dotenv_path),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ── Derived paths (resolve relative to project root) ────────────

    def resolve_path(self, relative: str) -> Path:
        """Resolve a relative path against the project root."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p

    @property
    def db_path(self) -> Path:
        return self.resolve_path(self.refinery_db_path)

    @property
    def chroma_dir(self) -> Path:
        return self.resolve_path(self.refinery_chroma_dir)

    @property
    def ledger_path(self) -> Path:
        return self.resolve_path(self.refinery_ledger_path)

    @property
    def runs_dir(self) -> Path:
        return self.resolve_path(self.refinery_runs_dir)


@lru_cache(maxsize=1)
def get_settings() -> RefinerySettings:
    """Return the singleton settings instance (cached)."""
    return RefinerySettings()


def get_project_root() -> Path:
    """Return the project root path."""
    return _PROJECT_ROOT
