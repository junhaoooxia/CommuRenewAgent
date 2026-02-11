from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from llama_index.embeddings.clip import ClipEmbedding


EmbeddingBackend = Literal["llamaindex", "simple"]


@dataclass
class EmbeddingConfig:
    dim: int = 512
    text_weight: float = 0.7
    image_weight: float = 0.3
    backend: EmbeddingBackend = "llamaindex"


class LlamaIndexMultimodalEmbedder:
    """Multimodal embedding via LlamaIndex CLIP embedding wrapper."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.clip = ClipEmbedding()

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.array(self.clip.get_text_embedding(text), dtype=np.float32)
        return self._normalize(vec)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            return np.zeros(self.config.dim, dtype=np.float32)
        vec = np.array(self.clip.get_image_embedding(str(path)), dtype=np.float32)
        return self._normalize(vec)

    def fuse(self, text_embedding: np.ndarray, image_embeddings: Sequence[np.ndarray]) -> np.ndarray:
        if image_embeddings:
            image_mean = np.mean(np.stack(image_embeddings), axis=0)
            fused = self.config.text_weight * text_embedding + self.config.image_weight * image_mean
            return self._normalize(fused)
        return self._normalize(text_embedding)

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


class SimpleMultimodalEmbedder:
    """Deterministic fallback embedder for offline/local smoke tests."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig(backend="simple")

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.config.dim, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i : i + 2], "little") % self.config.dim
                vec[idx] += 1.0
        return self._normalize(vec)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        vec = np.zeros(self.config.dim, dtype=np.float32)
        path = Path(image_path)
        digest = hashlib.sha256(str(path).encode("utf-8")).digest()
        for i in range(0, len(digest), 2):
            idx = int.from_bytes(digest[i : i + 2], "little") % self.config.dim
            vec[idx] += 1.0
        return self._normalize(vec)

    def fuse(self, text_embedding: np.ndarray, image_embeddings: Sequence[np.ndarray]) -> np.ndarray:
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
        return SimpleMultimodalEmbedder(cfg)
    return LlamaIndexMultimodalEmbedder(cfg)
