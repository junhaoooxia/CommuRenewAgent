from __future__ import annotations

from pathlib import Path

from .image_generation import edit_image_with_gemini_nanobanana
from .knowledge_ingestion import build_knowledge_base
from .models import GenerationOutput, PerceptionInput
from .reasoning import generate_schemes_with_reasoning
from .retrieval import rank_method_images_for_scene, rank_site_images_for_scene, retrieve_relevant_nodes
from .vision_evidence import build_visual_evidence


def index_knowledge_base(
    source_specs: list[dict],
    db_path: str | Path = "data/knowledge.db",
    embedding_backend: str = "openai_qwen",
) -> int:
    """Offline: parse PDFs/JSONL and persist multimodal knowledge nodes + embeddings."""
    # Offline entrypoint: called when knowledge sources change.
    return build_knowledge_base(source_specs=source_specs, db_path=db_path, embedding_backend=embedding_backend)




def _post_rank_scene_images(generated: GenerationOutput, retrieval, perception: PerceptionInput, embedding_backend: str, db_path: str | Path) -> None:
    # Stage-2 image recall: (1) node->site images from perception, (2) node->method images from retrieved method/strategy images.
    for scheme in generated.scheme_list:
        for scene in scheme.node_scenes:
            scene_query = f"{scene.node_name}\n{scene.description}\n{scene.desired_image_prompt}"
            scene.selected_representative_images = rank_site_images_for_scene(
                scene_text=scene_query,
                representative_images=perception.representative_images,
                embedding_backend=embedding_backend,
                top_k=2,
            )
            scene.reference_example_images = rank_method_images_for_scene(
                scene_text=scene_query,
                retrieval=retrieval,
                referenced_ids=scheme.referenced_methods,
                embedding_backend=embedding_backend,
                db_path=db_path,
                top_k=3,
            )

def generate_design_schemes(
    perception: PerceptionInput,
    db_path: str | Path = "data/knowledge.db",
    top_k: int = 20,
    model: str = "gpt-5.2",
    embedding_backend: str = "openai_qwen",
    generate_images: bool = False,
    image_model: str = "gemini-3-pro-image-preview",
    image_output_dir: str | Path = "data/generated_images",
) -> tuple[dict, GenerationOutput]:
    """主入口：如果 perception.visual_evidence 已存在则直接复用；否则基于 site_images 构建。"""
    if perception.site_images:
        perception.visual_evidence = build_visual_evidence(
            site_images=perception.site_images,
            existing=perception.visual_evidence,
            model=model,
            district_name=perception.district_name,
            current_description=perception.current_description,
        )

    # First stage: RAG retrieval conditioned on project perception input.
    retrieval = retrieve_relevant_nodes(
        perception=perception,
        db_path=db_path,
        top_k=top_k,
        embedding_backend=embedding_backend,
    )
    # Second stage: reasoning/generation over retrieved context.
    generated = generate_schemes_with_reasoning(perception=perception, retrieval=retrieval, model=model)
    _post_rank_scene_images(generated, retrieval, perception, embedding_backend, db_path)

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
