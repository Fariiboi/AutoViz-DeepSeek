"""Application configuration using Pydantic BaseSettings."""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from autoviz.net import configure_insecure_ssl


def find_env_file() -> Path | None:
    """Find .env file in current or parent directories."""
    current = Path.cwd()
    for path in [current, current.parent]:
        env_file = path / ".env"
        if env_file.exists():
            return env_file
    return None


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=find_env_file(),
        env_ignore_empty=True,
        extra="ignore",
    )

    # === Project ===
    PROJECT_NAME: str = "autoviz"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "local", "staging", "production"] = "local"
    LOG_LEVEL: str = "INFO"

    # === AI Agent (pydantic_ai, DeepSeek) ===
    DEEPSEEK_API_KEY: str = ""
    AI_MODEL: str = "deepseek-chat"
    AI_TEMPERATURE: float = 0.7
    AI_FRAMEWORK: str = "pydantic_ai"
    LLM_PROVIDER: str = "deepseek"
    # Local dev only: bypass TLS verification for a TLS-intercepting proxy.
    DISABLE_SSL_VERIFY: bool = False

    # === CSV pipeline ===
    CSV_SAMPLE_SIZE: int = 5
    # 超过该行数时随机采样, 避免超大数据把 token 打爆 (0 = 不限制).
    CSV_MAX_ANALYSIS_ROWS: int = 2000

    # === PostgreSQL + pgvector (optional semantic search) ===
    # 留空 = 关闭向量库写入/检索; 启用示例:
    # postgresql://autoviz:autoviz@127.0.0.1:5433/autoviz
    DATABASE_URL: str = ""
    EMBEDDING_BACKEND: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIM: int = 512
    EMBEDDING_DEVICE: str = "cpu"
    RAG_TOP_K: int = 5
    # Two-stage retrieval: vector recall, then CrossEncoder rerank.
    RAG_RECALL_K: int = 20
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_DEVICE: str = "cpu"
    RERANK_MAX_LENGTH: int = 512
    RERANK_BATCH_SIZE: int = 8
    # Optional HuggingFace mirror, e.g. https://hf-mirror.com
    HF_ENDPOINT: str = ""

    # === Modal Sandbox (feature engineering + visualization code execution) ===
    MODAL_TOKEN_ID: str = ""
    MODAL_TOKEN_SECRET: str = ""
    MODAL_APP_NAME: str = "autoviz"

    # === Visualization workspace ===
    VIZ_WORKSPACE_DIR: str = "workspace/viz"
    VIZ_HISTORY_FILE: str = "workspace/history.md"
    VIZ_TARGET_CHARTS: int = 10
    VIZ_CONCURRENCY: int = 10
    VIZ_TURNS: int = 1
    VIZ_CRITIC_MAX_CODE_ROUNDS: int = 3
    VIZ_CRITIC_MAX_SEMANTIC_REPLANS: int = 2

    # === Visualization Selector (task feasibility + IG filter) ===
    VIZ_SELECTOR_ENABLED: bool = False
    VIZ_SELECTOR_MAX_MISSING_RATE: float = 0.7
    VIZ_SELECTOR_MIN_IG_SCORE: float = 0.2

    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"


settings = Settings()
if settings.HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)
configure_insecure_ssl(settings.DISABLE_SSL_VERIFY)
