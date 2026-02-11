from __future__ import annotations

from pathlib import Path

from .knowledge_ingestion import build_knowledge_base
from .models import GenerationOutput, PerceptionInput
from .reasoning import generate_schemes_with_reasoning
from .retrieval import retrieve_relevant_nodes


def index_knowledge_base(
    pdf_specs: list[dict],
    db_path: str | Path = "data/knowledge.db",
    embedding_backend: str = "llamaindex",
) -> int:
    """Offline: parse PDFs and persist multimodal knowledge nodes + embeddings."""
    # Offline entrypoint: called when source PDFs change.
    return build_knowledge_base(pdf_specs=pdf_specs, db_path=db_path, embedding_backend=embedding_backend)


def generate_design_schemes(
    perception: PerceptionInput,
    db_path: str | Path = "data/knowledge.db",
    top_k: int = 15,
    model: str = "gpt-4.1",
    embedding_backend: str = "llamaindex",
) -> tuple[dict, GenerationOutput]:
    """Online: retrieve relevant knowledge and generate three design schemes."""
    # First stage: RAG retrieval conditioned on project perception input.
    retrieval = retrieve_relevant_nodes(
        perception=perception,
        db_path=db_path,
        top_k=top_k,
        embedding_backend=embedding_backend,
    )
    # Second stage: reasoning/generation over retrieved context.
    generated = generate_schemes_with_reasoning(perception=perception, retrieval=retrieval, model=model)
    # Return JSON-serializable retrieval payload for UI/debugging.
    retrieval_payload = {
        "retrieved_methods": [node.__dict__ for node in retrieval.retrieved_methods],
        "retrieved_policies": [node.__dict__ for node in retrieval.retrieved_policies],
        "retrieved_trend_strategies": [node.__dict__ for node in retrieval.retrieved_trend_strategies],
    }
    return retrieval_payload, generated
