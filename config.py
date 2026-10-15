"""Central configuration for Oncora Nexus.

Every path and tunable lives here so that scripts, the API and the test suite
share one source of truth. Values can be overridden with environment variables
prefixed ONCORA_ (for example ONCORA_LLM_MODEL=qwen2.5:7b).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"ONCORA_{name}", default)


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = Path(_env("RAW_DIR", str(ROOT / "data" / "raw")))
    processed: Path = Path(_env("PROCESSED_DIR", str(ROOT / "data" / "processed")))
    index: Path = Path(_env("INDEX_DIR", str(ROOT / "data" / "index")))
    models: Path = Path(_env("MODEL_DIR", str(ROOT / "data" / "models")))
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"
    knowledge_cards: Path = ROOT / "oncora" / "data" / "knowledge_cards.json"
    demo_bundle: Path = ROOT / "frontend" / "public" / "demo"
    audit_db: Path = Path(_env("AUDIT_DB", str(ROOT / "data" / "audit.sqlite3")))

    def ensure(self) -> Paths:
        for p in (self.raw, self.processed, self.index, self.models, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_target_tokens: int = int(_env("CHUNK_TOKENS", "180"))
    chunk_overlap_sentences: int = 1
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    dense_dimensions: int = int(_env("DENSE_DIMS", "256"))
    candidate_pool: int = 60
    rrf_k: int = 60
    final_k: int = int(_env("FINAL_K", "8"))
    min_support_score: float = 0.18
    embedding_backend: str = _env("EMBEDDINGS", "lsa")  # lsa or sentence_transformers
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


@dataclass(frozen=True)
class LLMConfig:
    provider: str = _env("LLM_PROVIDER", "ollama")  # ollama or extractive
    base_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    model: str = _env("LLM_MODEL", "llama3.1:8b")
    temperature: float = float(_env("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(_env("LLM_MAX_TOKENS", "900"))
    timeout_seconds: float = float(_env("LLM_TIMEOUT", "180"))


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    random_seed: int = 20261015
    app_name: str = "Oncora Nexus"
    app_version: str = "1.0.0"
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",")
    )


settings = Settings()
