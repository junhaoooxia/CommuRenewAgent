from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


EmbeddingBackend = Literal["openai_qwen", "simple"]


@dataclass
class EmbeddingConfig:
    dim: int = 2560
    backend: EmbeddingBackend = "openai_qwen"
    qwen_model_name: str = "qwen3-vl-embedding"


class QwenMultimodalEmbedder:
    """Qwen3-VL embedding for both text and image."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        try:
            import dashscope
        except Exception as exc:  # pragma: no cover
            raise ImportError("`dashscope` is required for Qwen embedding backend") from exc

        self.dashscope = dashscope
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ImportError("DASHSCOPE_API_KEY is required for openai_qwen backend")

    def embed_text(self, text: str) -> np.ndarray:
        safe_text = (text or "").strip() or "<empty>"
        resp = self.dashscope.MultiModalEmbedding.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=self.config.qwen_model_name,
            input=[{"text": safe_text}],
            dimension=self.config.dim,
        )
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(f"Qwen text embedding failed: status_code={getattr(resp, 'status_code', None)}")
        vec = np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
        return self._fit_dim(self._normalize(vec))

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            return np.zeros(self.config.dim, dtype=np.float32)

        payload = self._to_qwen_image_payload(path)
        resp = self.dashscope.MultiModalEmbedding.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=self.config.qwen_model_name,
            input=[{"image": payload}],
            dimension=self.config.dim,
        )
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(f"Qwen image embedding failed: status_code={getattr(resp, 'status_code', None)}")
        vec = np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
        return self._fit_dim(self._normalize(vec))

    @staticmethod
    def _to_qwen_image_payload(path: Path, max_kb: int = 5070) -> str:
        raw = path.read_bytes()
        max_bytes = max_kb * 1024
        if len(raw) <= max_bytes:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"

        from PIL import Image

        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        quality = 92
        current = img
        while True:
            buf = io.BytesIO()
            save_img = current.convert("RGB") if current.mode != "RGB" else current
            save_img.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return f"data:image/jpeg;base64,{base64.b64encode(data).decode('utf-8')}"
            if quality > 70:
                quality -= 8
                continue
            ratio = (max_bytes / max(len(data), 1)) ** 0.5 * 0.98
            nw = max(1, int(current.width * ratio))
            nh = max(1, int(current.height * ratio))
            if nw == current.width and nh == current.height:
                nw, nh = max(1, nw - 1), max(1, nh - 1)
            current = current.resize((nw, nh))

    def _fit_dim(self, vec: np.ndarray) -> np.ndarray:
        if vec.shape[0] == self.config.dim:
            return vec
        if vec.shape[0] > self.config.dim:
            return vec[: self.config.dim]
        out = np.zeros(self.config.dim, dtype=np.float32)
        out[: vec.shape[0]] = vec
        return out

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)


class SimpleMultimodalEmbedder:
    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig(backend="simple")

    def embed_text(self, text: str) -> np.ndarray:
        safe_text = (text or "").strip() or "<empty>"
        vec = np.zeros(self.config.dim, dtype=np.float32)
        for token in safe_text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i : i + 2], "little") % self.config.dim
                vec[idx] += 1.0
        return self._normalize(vec)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        vec = np.zeros(self.config.dim, dtype=np.float32)
        digest = hashlib.sha256(str(Path(image_path)).encode("utf-8")).digest()
        for i in range(0, len(digest), 2):
            idx = int.from_bytes(digest[i : i + 2], "little") % self.config.dim
            vec[idx] += 1.0
        return self._normalize(vec)

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
        return QwenMultimodalEmbedder(cfg)
    except Exception as exc:
        warnings.warn(
            f"Falling back to simple embedding backend because {cfg.backend} backend initialization failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return SimpleMultimodalEmbedder(EmbeddingConfig(dim=cfg.dim, backend="simple"))
