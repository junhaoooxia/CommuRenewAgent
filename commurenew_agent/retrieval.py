from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .embeddings import EmbeddingConfig, get_embedder
from .models import PerceptionInput, RetrievalResult, RetrievedNode
from .vector_store import SQLiteVectorStore


def _safe_text(text: str) -> str:
    return (text or "").strip() or "<empty>"


def _collapse_children_to_parents(nodes: list[RetrievedNode]) -> list[RetrievedNode]:
    grouped: dict[str, RetrievedNode] = {}
    for node in nodes:
        parent_id = node.metadata.get("parent_id") or node.id
        parent_text = node.metadata.get("parent_text") or node.text
        if parent_id not in grouped or node.score > grouped[parent_id].score:
            merged_metadata = dict(node.metadata)
            merged_metadata["matched_child_id"] = node.id
            grouped[parent_id] = RetrievedNode(
                id=parent_id,
                type=node.type,
                title=node.title,
                text=parent_text,
                images=node.images,
                score=node.score,
                metadata=merged_metadata,
            )
    return sorted(grouped.values(), key=lambda n: n.score, reverse=True)


def retrieve_relevant_nodes(
    perception: PerceptionInput,
    db_path: str | Path = "data/knowledge.db",
    top_k: int = 20,
    embedding_backend: str = "openai_qwen",
) -> RetrievalResult:
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    query_emb = embedder.embed_text(_safe_text(perception.to_text_block()))

    store = SQLiteVectorStore(db_path=db_path)
    retrieved_children = store.search_text(query_embedding=query_emb, top_k=max(top_k * 9, 120))
    store.close()

    retrieved = _collapse_children_to_parents(retrieved_children)

    result = RetrievalResult()
    for node in retrieved:
        if node.type == "design_method":
            result.retrieved_methods.append(node)
        elif node.type == "policy":
            result.retrieved_policies.append(node)
        elif node.type == "trend_strategy":
            result.retrieved_trend_strategies.append(node)
    result.retrieved_methods = result.retrieved_methods[:top_k]
    result.retrieved_policies = result.retrieved_policies[:top_k]
    result.retrieved_trend_strategies = result.retrieved_trend_strategies[:top_k]
    return result


def rank_site_images_for_scene(
    scene_text: str,
    site_images: Iterable[str],
    embedding_backend: str = "openai_qwen",
    top_k: int = 2,
) -> list[str]:
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    text_emb = embedder.embed_text(_safe_text(scene_text))

    scored: list[tuple[float, str]] = []
    for img_path in site_images:
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
    db_path: str | Path = "data/knowledge.db",
    embedding_backend: str = "openai_qwen",
    top_k: int = 3,
) -> list[str]:
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    text_emb = embedder.embed_text(_safe_text(scene_text))

    ref_set = set(referenced_ids)
    candidates = retrieval.retrieved_methods + retrieval.retrieved_trend_strategies
    candidate_ids = [n.id for n in candidates if not ref_set or n.id in ref_set]

    store = SQLiteVectorStore(db_path=db_path)
    image_rows = store.get_image_embeddings(candidate_ids)
    store.close()

    scored: list[tuple[float, str]] = []
    for _, img_path, img_emb in image_rows:
        score = float(np.dot(text_emb, img_emb))
        scored.append((score, img_path))

    scored.sort(key=lambda x: x[0], reverse=True)
    deduped: list[str] = []
    for _, path in scored:
        if path not in deduped:
            deduped.append(path)
        if len(deduped) >= top_k:
            break
    return deduped
