from __future__ import annotations

from pathlib import Path

from .embeddings import EmbeddingConfig, get_embedder
from .models import PerceptionInput, RetrievalResult
from .vector_store import SQLiteVectorStore


def retrieve_relevant_nodes(
    perception: PerceptionInput,
    db_path: str | Path = "data/knowledge.db",
    top_k: int = 15,
    embedding_backend: str = "llamaindex",
) -> RetrievalResult:
    # Encode both site text and representative images into a single query vector.
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    query_emb = embedder.embed_query(perception.to_text_block(), perception.representative_images)

    # Query persistent store and close connection immediately after read.
    store = SQLiteVectorStore(db_path=db_path)
    retrieved = store.search(query_embedding=query_emb, top_k=top_k)
    store.close()

    # Split mixed retrieval results into explicit buckets consumed by reasoning prompt.
    result = RetrievalResult()
    for node in retrieved:
        if node.type == "design_method":
            result.retrieved_methods.append(node)
        elif node.type == "policy":
            result.retrieved_policies.append(node)
        elif node.type == "trend_strategy":
            result.retrieved_trend_strategies.append(node)
    return result
