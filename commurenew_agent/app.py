from __future__ import annotations

from pathlib import Path

from .image_generation import edit_image_with_gemini_nanobanana
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
    model: str = "gpt-5.2",
    embedding_backend: str = "llamaindex",
    generate_images: bool = False,
    image_model: str = "gemini-3-pro-image-preview",
    image_output_dir: str | Path = "data/generated_images",
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

    if generate_images:
        output_root = Path(image_output_dir)
        for scheme_idx, scheme in enumerate(generated.scheme_list, start=1):
            for scene_idx, scene in enumerate(scheme.node_scenes, start=1):
                for src_idx, src in enumerate(scene.selected_representative_images, start=1):
                    out_path = output_root / f"scheme_{scheme_idx}" / f"scene_{scene_idx}_src_{src_idx}.png"
                    edited = edit_image_with_gemini_nanobanana(
                        prompt=scene.desired_image_prompt,
                        source_image_path=src,
                        output_path=out_path,
                        model=image_model,
                    )
                    scene.generated_images.append(edited)
    # Return JSON-serializable retrieval payload for UI/debugging.
    retrieval_payload = {
        "retrieved_methods": [node.__dict__ for node in retrieval.retrieved_methods],
        "retrieved_policies": [node.__dict__ for node in retrieval.retrieved_policies],
        "retrieved_trend_strategies": [node.__dict__ for node in retrieval.retrieved_trend_strategies],
    }
    return retrieval_payload, generated
