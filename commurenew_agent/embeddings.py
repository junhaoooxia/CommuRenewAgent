from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


EmbeddingBackend = Literal["openai_qwen", "llamaindex", "llamaindex_zh", "simple"]


@dataclass
class EmbeddingConfig:
    dim: int = 1536
    text_weight: float = 0.7
    image_weight: float = 0.3
    backend: EmbeddingBackend = "openai_qwen"
    clip_text_chunk_chars: int = 120
    zh_text_model_name: str = "BAAI/bge-m3"
    openai_text_model_name: str = "text-embedding-3-small"
    qwen_image_model_name: str = "qwen3-vl-embedding"


class OpenAIQwenMultimodalEmbedder:
    """Multimodal embedding via OpenAI text-embedding-3 + Qwen vision embedding."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()

        from openai import OpenAI

        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        try:
            import dashscope
        except Exception as exc:  # pragma: no cover
            raise ImportError("`dashscope` is required for Qwen vision embedding backend") from exc

        self.dashscope = dashscope

    def embed_text(self, text: str) -> np.ndarray:
        response = self.openai_client.embeddings.create(input=text, model=self.config.openai_text_model_name)
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        return self._fit_dim(self._normalize(vec))

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            return np.zeros(self.config.dim, dtype=np.float32)

        image_payload = self._to_qwen_image_payload(path)
        resp = self._call_qwen_embedding(image_payload)
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(f"Qwen image embedding failed: status_code={getattr(resp, 'status_code', None)}")

        emb_list = resp.output["embeddings"]
        vec = np.array(emb_list[0]["embedding"], dtype=np.float32)
        return self._fit_dim(self._normalize(vec))

    def _call_qwen_embedding(self, image_payload: str):
        # Try explicit dimension first to align vector space with OpenAI text embeddings.
        resp = self.dashscope.MultiModalEmbedding.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=self.config.qwen_image_model_name,
            input=[{"image": image_payload}],
            dimension=self.config.dim,
        )
        if getattr(resp, "status_code", None) == 200:
            return resp

        # Some SDK versions may not accept `dimension`; retry without it.
        return self.dashscope.MultiModalEmbedding.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=self.config.qwen_image_model_name,
            input=[{"image": image_payload}],
        )

    @staticmethod
    def _to_qwen_image_payload(path: Path) -> str:
        # Local images are converted to data-URI so the embedding API can read them directly.
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

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

    def _fit_dim(self, vec: np.ndarray) -> np.ndarray:
        if vec.shape[0] == self.config.dim:
            return vec.astype(np.float32)
        if vec.shape[0] > self.config.dim:
            return vec[: self.config.dim].astype(np.float32)
        out = np.zeros(self.config.dim, dtype=np.float32)
        out[: vec.shape[0]] = vec
        return out

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)


class LlamaIndexMultimodalEmbedder:
    """Multimodal embedding via LlamaIndex CLIP embedding wrapper."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
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
        return self._embed_text_with_clip_limit(text)

    def _embed_text_with_clip_limit(self, text: str) -> np.ndarray:
        try:
            vec = np.array(self.clip.get_text_embedding(text), dtype=np.float32)
            return self._fit_dim(self._normalize(vec))
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
        current = chunk.strip()
        while current:
            try:
                return self._fit_dim(self._normalize(np.array(self.clip.get_text_embedding(current), dtype=np.float32)))
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
            return np.zeros(self.config.dim, dtype=np.float32)
        vec = np.array(self.clip.get_image_embedding(str(path)), dtype=np.float32)
        return self._fit_dim(self._normalize(vec))

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

    def _fit_dim(self, vec: np.ndarray) -> np.ndarray:
        if vec.shape[0] == self.config.dim:
            return vec.astype(np.float32)
        if vec.shape[0] > self.config.dim:
            return vec[: self.config.dim].astype(np.float32)
        out = np.zeros(self.config.dim, dtype=np.float32)
        out[: vec.shape[0]] = vec
        return out

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)


class LlamaIndexChineseHybridEmbedder(LlamaIndexMultimodalEmbedder):
    """Chinese-friendly hybrid: HF text encoder + CLIP image encoder."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        super().__init__(config=config)
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.text_embedder = HuggingFaceEmbedding(model_name=self.config.zh_text_model_name)

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.array(self.text_embedder.get_text_embedding(text), dtype=np.float32)
        return self._fit_dim(self._normalize(vec))


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

    try:
        if cfg.backend == "openai_qwen":
            return OpenAIQwenMultimodalEmbedder(cfg)
        if cfg.backend == "llamaindex_zh":
            return LlamaIndexChineseHybridEmbedder(cfg)
        return LlamaIndexMultimodalEmbedder(cfg)
    except Exception as exc:
        warnings.warn(
            f"Falling back to simple embedding backend because {cfg.backend} backend initialization failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return SimpleMultimodalEmbedder(
            EmbeddingConfig(dim=cfg.dim, text_weight=cfg.text_weight, image_weight=cfg.image_weight, backend="simple")
        )
