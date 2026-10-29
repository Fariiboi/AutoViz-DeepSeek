"""Local text embedding backends for semantic chart search.

Default backend is ``sentence-transformers`` with a Chinese BGE model
(``BAAI/bge-small-zh-v1.5``). A deterministic hashing backend is provided for
offline tests and CI; it must not be used for production retrieval quality.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from autoviz.config import settings

# Keep model weights inside the project workspace and silence the Windows
# symlink warning from huggingface_hub (Developer Mode is often disabled).
# Only do this for a source checkout; wheel installs keep the platform default.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if (_PROJECT_ROOT / "pyproject.toml").exists():
    os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "workspace" / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def model_is_cached(model_name: str) -> bool:
    """Return True when the HF snapshot for ``model_name`` already exists."""
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return False
    repo_dir = Path(hf_home) / "hub" / f"models--{model_name.replace('/', '--')}"
    return repo_dir.is_dir()


class Embedder(Protocol):
    """Minimal embedding interface used by the store and the UI."""

    model_name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalised vector per input text."""


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return a unit-length copy of ``vector`` (zero vectors stay zero)."""
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests/CI.

    Uses hashed word tokens plus CJK character bigrams, so it is language
    agnostic and stable across processes. Retrieval quality is intentionally
    modest; production runs use :class:`SentenceTransformerEmbedder`.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.dim = dim
        self.model_name = f"hash-{dim}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in _TOKEN_RE.findall(text.lower()):
                features = (
                    [token[i : i + 2] for i in range(max(1, len(token) - 1))]
                    if _CJK_RE.search(token)
                    else [token]
                )
                for feature in features:
                    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                    index = int.from_bytes(digest[:4], "little") % self.dim
                    vector[index] += 1.0 if digest[4] & 1 else -1.0
            vectors.append(l2_normalize(vector))
        return vectors


class SentenceTransformerEmbedder:
    """Sentence-Transformers backend (BGE/M3E, CPU by default)."""

    def __init__(
        self,
        model_name: str,
        dim: int,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "sentence-transformers 未安装。请执行：uv sync --extra rag "
                "（CPU 环境可加 --torch-backend cpu）"
            ) from exc

        self.model_name = model_name
        self.dim = dim
        self._batch_size = batch_size
        cached = model_is_cached(model_name)
        try:
            self._model = SentenceTransformer(
                model_name, device=device, local_files_only=cached
            )
        except Exception:
            # A partially cached snapshot should be repaired from the mirror.
            if not cached:
                raise
            self._model = SentenceTransformer(
                model_name, device=device, local_files_only=False
            )

        getter = getattr(self._model, "get_embedding_dimension", None)
        actual = getter() if getter else self._model.get_sentence_embedding_dimension()
        if actual and int(actual) != dim:
            raise ValueError(
                f"EMBEDDING_DIM={dim} 与模型 {model_name} 的实际维度 {actual} 不一致；"
                "请同步修改 .env 中的 EMBEDDING_DIM 与数据库 vector 列维度。"
            )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        rows = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in rows]


_embedder_cache: dict[tuple[str, str, int, str], Embedder] = {}


def build_embedder(
    *,
    backend: str | None = None,
    model_name: str | None = None,
    dim: int | None = None,
    device: str | None = None,
) -> Embedder:
    """Build (and cache) the configured embedder."""
    backend = (backend or settings.EMBEDDING_BACKEND).strip().lower()
    model_name = model_name or settings.EMBEDDING_MODEL
    dim = int(dim or settings.EMBEDDING_DIM)
    device = device or settings.EMBEDDING_DEVICE

    if backend in {"hash", "hashing"}:
        key = ("hash", "", dim, "")
    elif backend in {"sentence-transformers", "sentence_transformers", "st", "bge"}:
        key = ("st", model_name, dim, device)
    else:
        raise ValueError(f"未知 EMBEDDING_BACKEND: {backend!r}")

    cached = _embedder_cache.get(key)
    if cached is not None:
        return cached

    embedder: Embedder
    if key[0] == "hash":
        embedder = HashingEmbedder(dim)
    else:
        embedder = SentenceTransformerEmbedder(model_name, dim, device=device)
    _embedder_cache[key] = embedder
    return embedder


def reset_embedder_cache() -> None:
    """Clear the embedder cache (used by tests)."""
    _embedder_cache.clear()
