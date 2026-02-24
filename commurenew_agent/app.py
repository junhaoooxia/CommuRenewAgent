from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .image_generation import edit_image_with_gemini_nanobanana
from .knowledge_ingestion import build_knowledge_base
from .models import GenerationOutput, PerceptionInput
from .reasoning import generate_schemes_with_reasoning
from .retrieval import rank_method_images_for_scene, retrieve_relevant_nodes
from .vision_evidence import build_visual_evidence

logger = logging.getLogger(__name__)


def _discover_site_images(default_dir: str | Path = "inputs/siteImgs") -> list[str]:
    root = Path(default_dir)
    if not root.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [str(p) for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _site_images_signature(site_images: list[str]) -> str:
    hasher = hashlib.sha256()
    for img in site_images:
        p = Path(img)
        hasher.update(str(p.resolve()).encode("utf-8", errors="ignore"))
        if p.exists():
            stat = p.stat()
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return hasher.hexdigest()


def _load_visual_evidence_cache(cache_path: Path, signature: str) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("signature") == signature and isinstance(payload.get("visual_evidence"), dict):
        return payload["visual_evidence"]
    return None


def _save_visual_evidence_cache(cache_path: Path, signature: str, visual_evidence: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"signature": signature, "visual_evidence": visual_evidence}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def index_knowledge_base(
    source_specs: list[dict],
    db_path: str | Path = "data/knowledge.db",
    embedding_backend: str = "openai_qwen",
) -> int:
    """Offline: parse PDFs/JSONL and persist multimodal knowledge nodes + embeddings."""
    # Offline entrypoint: called when knowledge sources change.
    count = build_knowledge_base(source_specs=source_specs, db_path=db_path, embedding_backend=embedding_backend)
    logger.info("[app] knowledge indexing finished. nodes=%s", count)
    return count


def _resolve_selected_site_image_paths(selected_images: list[str], site_images: list[str]) -> list[str]:
    """Resolve LLM-selected image labels/filenames to actual site image paths."""
    if not selected_images:
        return []

    resolved: list[str] = []
    site_map_by_name = {Path(p).name.lower(): p for p in site_images}

    for item in selected_images:
        candidate = str(item).strip()
        if not candidate:
            continue

        # 1) direct path exists
        cpath = Path(candidate)
        if cpath.exists():
            resolved.append(str(cpath))
            continue

        # 2) exact entry in known site_images
        if candidate in site_images:
            resolved.append(candidate)
            continue

        # 3) basename lookup (case-insensitive), common when model returns only filename
        hit = site_map_by_name.get(Path(candidate).name.lower())
        if hit:
            resolved.append(hit)
            continue

        # keep original for observability; downstream may still report missing file explicitly
        resolved.append(candidate)

    # de-duplicate while preserving order
    deduped: list[str] = []
    seen = set()
    for pth in resolved:
        key = pth.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pth)
    return deduped


def _post_rank_scene_images(generated: GenerationOutput, retrieval, embedding_backend: str, db_path: str | Path) -> None:
    # Stage-2 image recall: only populate method/strategy reference images.
    # selected_representative_images are chosen by LLM in generate_schemes_with_reasoning from visual_evidence.
    for scheme in generated.scheme_list:
        for scene in scheme.node_scenes:
            scene_query = f"{scene.node_name}\n{scene.description}\n{scene.desired_image_prompt}"
            scene.reference_example_images = rank_method_images_for_scene(
                scene_text=scene_query,
                retrieval=retrieval,
                referenced_ids=scheme.referenced_methods,
                embedding_backend=embedding_backend,
                db_path=db_path,
                top_k=3,
            )
            logger.info("[app] scene post-rank done. scheme=%s node=%s references=%s", scheme.name, scene.node_name, len(scene.reference_example_images))

def generate_design_schemes(
    perception: PerceptionInput,
    db_path: str | Path = "data/knowledge.db",
    top_k: int = 20,
    model: str = "gpt-5.2",
    embedding_backend: str = "openai_qwen",
    generate_images: bool = False,
    image_model: str = "gemini-3-pro-image-preview",
    image_output_dir: str | Path = "output",
    result_timestamp: str | None = None,
) -> tuple[dict, GenerationOutput]:
    """主入口：如果 perception.visual_evidence 已存在则直接复用；否则基于 site_images 构建。"""
    # Backward compatibility: if perception has no site_images attribute, auto discover default folder.
    if not hasattr(perception, "site_images"):
        setattr(perception, "site_images", _discover_site_images())
    if not hasattr(perception, "visual_evidence"):
        setattr(perception, "visual_evidence", {})

    if not perception.site_images:
        perception.site_images = _discover_site_images()
        logger.info("[app] discovered site images count=%s", len(perception.site_images))

    if perception.site_images:
        signature = _site_images_signature(perception.site_images)
        cache_path = Path("output/visual_evidence_cache.json")
        cached = _load_visual_evidence_cache(cache_path=cache_path, signature=signature)
        if cached and not perception.visual_evidence:
            print("Reusing cached visual evidence based on site images signature.")
            perception.visual_evidence = cached
        else:
            perception.visual_evidence = build_visual_evidence(
                site_images=perception.site_images,
                existing=perception.visual_evidence,
                model=model,
                district_name=perception.district_name,
                current_description=perception.current_description,
            )
            _save_visual_evidence_cache(cache_path=cache_path, signature=signature, visual_evidence=perception.visual_evidence)
        logger.info("[app] visual evidence ready. images=%s", len((perception.visual_evidence or {}).get("images", [])))

    # First stage: RAG retrieval conditioned on project perception input.
    retrieval = retrieve_relevant_nodes(
        perception=perception,
        db_path=db_path,
        top_k=top_k,
        embedding_backend=embedding_backend,
    )
    # Second stage: reasoning/generation over retrieved context.
    generated = generate_schemes_with_reasoning(perception=perception, retrieval=retrieval, model=model)
    logger.info("[app] reasoning finished. schemes=%s", len(generated.scheme_list))
    _post_rank_scene_images(generated, retrieval, embedding_backend, db_path)
    logger.info("[app] post rank finished.")

    if generate_images:
        timestamp = result_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(image_output_dir) / f"result_{timestamp}"
        tasks = []
        for scheme_idx, scheme in enumerate(generated.scheme_list, start=1):
            for scene_idx, scene in enumerate(scheme.node_scenes, start=1):
                resolved_sources = _resolve_selected_site_image_paths(scene.selected_representative_images, perception.site_images)
                for src_idx, src in enumerate(resolved_sources, start=1):
                    out_path = output_root / f"scheme_{scheme_idx}" / f"scene_{scene_idx}_src_{src_idx}.png"
                    tasks.append((scheme, scene, scheme_idx, scene_idx, src, out_path))

        logger.info("[app] image generation start. tasks=%s workers=10", len(tasks))

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_map = {
                executor.submit(
                    edit_image_with_gemini_nanobanana,
                    prompt=scene.desired_image_prompt,
                    source_image_path=src,
                    output_path=out_path,
                    model=image_model,
                ): (scheme, scene, scheme_idx, scene_idx, src, out_path)
                for scheme, scene, scheme_idx, scene_idx, src, out_path in tasks
            }

            for fut in as_completed(future_map):
                scheme, scene, scheme_idx, scene_idx, src, out_path = future_map[fut]
                try:
                    edited = fut.result()
                    scene.generated_images.append(edited)
                    logger.info(
                        "[app] generated image saved. scheme=%s scene=%s source=%s output=%s",
                        scheme_idx,
                        scene_idx,
                        src,
                        edited,
                    )
                except Exception as exc:
                    logger.exception(
                        "[app] image generation failed. scheme=%s scene=%s source=%s output=%s err=%s",
                        scheme_idx,
                        scene_idx,
                        src,
                        out_path,
                        exc,
                    )
    # Return JSON-serializable retrieval payload for UI/debugging.
    retrieval_payload = {
        "retrieved_methods": [node.__dict__ for node in retrieval.retrieved_methods],
        "retrieved_policies": [node.__dict__ for node in retrieval.retrieved_policies],
        "retrieved_trend_strategies": [node.__dict__ for node in retrieval.retrieved_trend_strategies],
    }
    logger.info("[app] generate_design_schemes finished.")
    return retrieval_payload, generated
