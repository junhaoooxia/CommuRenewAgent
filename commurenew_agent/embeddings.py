from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image


@dataclass
class EmbeddingConfig:
    dim: int = 512
    text_weight: float = 0.7
    image_weight: float = 0.3


class SimpleMultimodalEmbedder:
    """Lightweight deterministic embedder for local/offline flows.

    This is a practical fallback for environments without dedicated embedding models.
    For production, replace with CLIP + text embedding providers.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.config.dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i : i + 2], "little") % self.config.dim
                vec[idx] += 1.0
        return self._normalize(vec)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            return np.zeros(self.config.dim, dtype=np.float32)
        with Image.open(path) as img:
            rgb = img.convert("RGB").resize((64, 64))
            arr = np.array(rgb, dtype=np.float32)
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=32, range=(0, 255), density=True)
            hist_parts.append(hist)
        base = np.concatenate(hist_parts)
        vec = np.zeros(self.config.dim, dtype=np.float32)
        vec[: len(base)] = base[: self.config.dim]
        return self._normalize(vec)

    def fuse(self, text_embedding: np.ndarray, image_embeddings: Sequence[np.ndarray]) -> np.ndarray:
        if image_embeddings:
            image_mean = np.mean(np.stack(image_embeddings), axis=0)
        else:
            image_mean = np.zeros(self.config.dim, dtype=np.float32)
        fused = self.config.text_weight * text_embedding + self.config.image_weight * image_mean
        return self._normalize(fused)

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
            return vec
        return (vec / norm).astype(np.float32)
