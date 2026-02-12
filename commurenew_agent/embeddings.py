from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


EmbeddingBackend = Literal["llamaindex", "simple"]


@dataclass
class EmbeddingConfig:
    dim: int = 512
    text_weight: float = 0.7
    image_weight: float = 0.3
    backend: EmbeddingBackend = "llamaindex"
    clip_text_chunk_chars: int = 120


class LlamaIndexMultimodalEmbedder:
    """Multimodal embedding via LlamaIndex CLIP embedding wrapper."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        # Import lazily so project can still run with fallback backend when CLIP deps are missing.
        try:
            from llama_index.embeddings.clip import ClipEmbedding
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            raise ImportError(
                "LlamaIndex CLIP backend requires `llama-index-embeddings-clip` and CLIP runtime deps. "
                "Install extras (e.g. `pip install git+https://github.com/openai/CLIP.git torch`) "
                "or use embedding_backend='simple'."
            ) from exc

        self.clip = ClipEmbedding()

    def embed_text(self, text: str) -> np.ndarray:
        # CLIP text context is short (77 tokens); long pages are chunked and averaged.
        return self._embed_text_with_clip_limit(text)

    def _embed_text_with_clip_limit(self, text: str) -> np.ndarray:
        try:
            vec = np.array(self.clip.get_text_embedding(text), dtype=np.float32)
            return self._normalize(vec)
        except RuntimeError as exc:
            if "too long for context length" not in str(exc):
                raise

        chunk_vecs: list[np.ndarray] = []
        for chunk in self._chunk_text_for_clip(text):
            chunk_vec = self._embed_text_chunk_safely(chunk)
            if chunk_vec is not None:
                chunk_vecs.append(chunk_vec)

        if not chunk_vecs:
            raise RuntimeError("CLIP text embedding failed for all chunks after length fallback")

        return self._normalize(np.mean(np.stack(chunk_vecs), axis=0))

    def _embed_text_chunk_safely(self, chunk: str) -> np.ndarray | None:
        # If one chunk still overflows tokenizer context, iteratively shorten until it fits.
        current = chunk.strip()
        while current:
            try:
                return self._normalize(np.array(self.clip.get_text_embedding(current), dtype=np.float32))
            except RuntimeError as exc:
                if "too long for context length" not in str(exc):
                    raise
                current = current[: max(1, len(current) // 2)]
        return None

    def _chunk_text_for_clip(self, text: str) -> list[str]:
        clean = " ".join(text.split())
        if not clean:
            return [" "]

        max_chars = max(16, self.config.clip_text_chunk_chars)
        if len(clean) <= max_chars:
            return [clean]

        chunks = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + max_chars)
            chunks.append(clean[start:end])
            start = end
        return chunks

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            # Missing image is treated as zero contribution to keep pipeline robust.
            return np.zeros(self.config.dim, dtype=np.float32)
        # Delegate image encoding to CLIP image encoder via LlamaIndex adapter.
        vec = np.array(self.clip.get_image_embedding(str(path)), dtype=np.float32)
        return self._normalize(vec)

    def fuse(self, text_embedding: np.ndarray, image_embeddings: Sequence[np.ndarray]) -> np.ndarray:
        # Weighted late fusion keeps text dominant while still injecting visual context.
        if image_embeddings:
            image_mean = np.mean(np.stack(image_embeddings), axis=0)
            fused = self.config.text_weight * text_embedding + self.config.image_weight * image_mean
            return self._normalize(fused)
        return self._normalize(text_embedding)

    def embed_node(self, main_text: str, image_paths: Iterable[str | Path]) -> np.ndarray:
        # Node embedding = method/policy text + all page images.
        text_vec = self.embed_text(main_text)
        image_vecs = [self.embed_image(path) for path in image_paths]
        return self.fuse(text_vec, image_vecs)

    def embed_query(self, text: str, image_paths: Iterable[str | Path] | None = None) -> np.ndarray:
        # Query embedding follows same fusion strategy as indexing for vector-space consistency.
        text_vec = self.embed_text(text)
        image_vecs = [self.embed_image(path) for path in (image_paths or [])]
        return self.fuse(text_vec, image_vecs)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)


class SimpleMultimodalEmbedder:
    """Deterministic fallback embedder for offline/local smoke tests."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig(backend="simple")

    def embed_text(self, text: str) -> np.ndarray:
        # Hashing trick: deterministic pseudo-embedding without external model dependencies.
        vec = np.zeros(self.config.dim, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i : i + 2], "little") % self.config.dim
                vec[idx] += 1.0
        return self._normalize(vec)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        # Fallback image signal uses path hash (cheap + deterministic) for plumbing validation.
        vec = np.zeros(self.config.dim, dtype=np.float32)
        path = Path(image_path)
        digest = hashlib.sha256(str(path).encode("utf-8")).digest()
        for i in range(0, len(digest), 2):
            idx = int.from_bytes(digest[i : i + 2], "little") % self.config.dim
            vec[idx] += 1.0
        return self._normalize(vec)

    def fuse(self, text_embedding: np.ndarray, image_embeddings: Sequence[np.ndarray]) -> np.ndarray:
        # Keep same fusion math as llamaindex backend to simplify behavior comparison.
        if image_embeddings:
            image_mean = np.mean(np.stack(image_embeddings), axis=0)
        else:
            image_mean = np.zeros(self.config.dim, dtype=np.float32)
        return self._normalize(self.config.text_weight * text_embedding + self.config.image_weight * image_mean)

    def embed_node(self, main_text: str, image_paths: Iterable[str | Path]) -> np.ndarray:
        text_vec = self.embed_text(main_text)
        image_vecs = [self.embed_image(path) for path in image_paths]
        return self.fuse(text_vec, image_vecs)

    def embed_query(self, text: str, image_paths: Iterable[str | Path] | None = None) -> np.ndarray:
        text_vec = self.embed_text(text)
        image_vecs = [self.embed_image(path) for path in (image_paths or [])]
        return self.fuse(text_vec, image_vecs)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)


def get_embedder(config: EmbeddingConfig | None = None):
    cfg = config or EmbeddingConfig()
    if cfg.backend == "simple":
        # Explicit override for fully-offline environments.
        return SimpleMultimodalEmbedder(cfg)

    try:
        # Preferred production path.
        return LlamaIndexMultimodalEmbedder(cfg)
    except ImportError as exc:
        # Graceful degradation: keep pipeline alive even when CLIP runtime stack is incomplete.
        warnings.warn(
            f"Falling back to simple embedding backend because llamaindex backend is unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return SimpleMultimodalEmbedder(EmbeddingConfig(dim=cfg.dim, text_weight=cfg.text_weight, image_weight=cfg.image_weight, backend="simple"))
