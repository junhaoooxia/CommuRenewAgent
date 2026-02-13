from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .embeddings import EmbeddingConfig, get_embedder
from .models import PerceptionInput, RetrievalResult
from .vector_store import SQLiteVectorStore


def retrieve_relevant_nodes(
    perception: PerceptionInput,
    db_path: str | Path = "data/knowledge.db",
    top_k: int = 15,
    embedding_backend: str = "openai_qwen",
) -> RetrievalResult:
    # Text-plan retrieval is text-only: do not use knowledge/perception images for this recall stage.
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    query_emb = embedder.embed_query(perception.to_text_block(), image_paths=[])

    store = SQLiteVectorStore(db_path=db_path)
    retrieved = store.search(query_embedding=query_emb, top_k=top_k)
    store.close()

    result = RetrievalResult()
    for node in retrieved:
        if node.type == "design_method":
            result.retrieved_methods.append(node)
        elif node.type == "policy":
            result.retrieved_policies.append(node)
        elif node.type == "trend_strategy":
            result.retrieved_trend_strategies.append(node)
    return result


def rank_site_images_for_scene(
    scene_text: str,
    representative_images: Iterable[str],
    embedding_backend: str = "openai_qwen",
    top_k: int = 2,
) -> list[str]:
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    text_emb = embedder.embed_text(scene_text)

    scored: list[tuple[float, str]] = []
    for img_path in representative_images:
        p = Path(img_path)
        if not p.exists():
            continue
        img_emb = embedder.embed_image(str(p))
        score = float(np.dot(text_emb, img_emb))
        scored.append((score, str(p)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def rank_method_images_for_scene(
    scene_text: str,
    retrieval: RetrievalResult,
    referenced_ids: Iterable[str],
    embedding_backend: str = "openai_qwen",
    top_k: int = 3,
) -> list[str]:
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    text_emb = embedder.embed_text(scene_text)

    ref_set = set(referenced_ids)
    candidates = retrieval.retrieved_methods + retrieval.retrieved_trend_strategies
    if ref_set:
        candidates = [n for n in candidates if n.id in ref_set]

    scored: list[tuple[float, str]] = []
    for node in candidates:
        for img_path in node.images:
            p = Path(img_path)
            if not p.exists():
                continue
            img_emb = embedder.embed_image(str(p))
            score = float(np.dot(text_emb, img_emb))
            scored.append((score, str(p)))

    scored.sort(key=lambda x: x[0], reverse=True)
    deduped: list[str] = []
    for _, path in scored:
        if path not in deduped:
            deduped.append(path)
        if len(deduped) >= top_k:
            break
    return deduped
