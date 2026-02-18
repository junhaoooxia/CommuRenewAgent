from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .embeddings import EmbeddingConfig, get_embedder
from .models import KnowledgeNode
from .vector_store import SQLiteVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "untitled"


def extract_nodes_from_pdf(
    pdf_path: str | Path,
    node_type: str,
    output_image_dir: str | Path = "data/extracted_images",
    metadata: dict | None = None,
) -> List[KnowledgeNode]:
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF ingestion. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    pdf_path = Path(pdf_path)
    image_dir = Path(output_image_dir) / pdf_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    nodes: List[KnowledgeNode] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        title = page.get_text("text").strip().split("\n")[0][:120] or f"{pdf_path.stem} page {page_index+1}"
        page_images: List[str] = []
        for img_no, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_img = doc.extract_image(xref)
            ext = base_img.get("ext", "png")
            image_path = image_dir / f"p{page_index+1}_img{img_no+1}.{ext}"
            image_path.write_bytes(base_img["image"])
            page_images.append(str(image_path))

        nodes.append(
            KnowledgeNode(
                id=f"{pdf_path.stem}_{page_index+1}",
                type=node_type,
                title=title,
                main_text=page.get_text("text").strip(),
                images=page_images,
                metadata={**(metadata or {}), "source": "pdf", "pdf": str(pdf_path), "page": page_index + 1},
            )
        )
    return nodes


def extract_nodes_from_word(
    word_path: str | Path,
    node_type: str,
    metadata: dict | None = None,
) -> List[KnowledgeNode]:
    word_path = Path(word_path)
    if word_path.suffix.lower() == ".doc":
        raise RuntimeError(".doc is not directly supported; please convert to .docx for ingestion.")

    try:
        from docx import Document
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "python-docx is required for Word ingestion. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    doc = Document(str(word_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paragraphs)
    title = paragraphs[0][:120] if paragraphs else word_path.stem
    return [
        KnowledgeNode(
            id=f"{word_path.stem}_1",
            type=node_type,
            title=title,
            main_text=text,
            images=[],
            metadata={**(metadata or {}), "source": "word", "word": str(word_path)},
        )
    ]


def _resolve_image_path(image_path: str, _base_dir: Path) -> str:
    normalized = Path(image_path.replace("\\", "/"))
    if normalized.is_absolute():
        return str(normalized)
    ref_root = PROJECT_ROOT / "ref"
    return str((ref_root / normalized).resolve())


def extract_nodes_from_jsonl(
    jsonl_path: str | Path,
    default_type: str = "other",
    metadata: dict | None = None,
) -> List[KnowledgeNode]:
    jsonl_path = Path(jsonl_path)
    base_dir = jsonl_path.parent
    nodes: List[KnowledgeNode] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)

            record_id = str(record.get("id", line_no)).strip() or str(line_no)
            record_type = str(record.get("type") or default_type)
            title = str(record.get("title") or f"{jsonl_path.stem}_{record_id}")
            main_text = str(record.get("main_text") or "")
            raw_images = record.get("images") or []
            images = [_resolve_image_path(img, base_dir) for img in raw_images if str(img).strip()]

            nodes.append(
                KnowledgeNode(
                    id=f"{jsonl_path.stem}_{record_type}_{record_id}",
                    type=record_type,
                    title=title,
                    main_text=main_text,
                    images=images,
                    metadata={**(metadata or {}), "source": "jsonl", "jsonl": str(jsonl_path), "line": line_no},
                )
            )

    return nodes


def _collect_nodes_from_spec(spec: dict) -> List[KnowledgeNode]:
    source_kind = spec.get("source")
    if source_kind == "jsonl" or "jsonl_path" in spec:
        return extract_nodes_from_jsonl(
            jsonl_path=spec["jsonl_path"],
            default_type=spec.get("type", "other"),
            metadata=spec.get("metadata", {}),
        )

    if source_kind in {"word", "docx", "doc"} or "word_path" in spec or "docx_path" in spec:
        return extract_nodes_from_word(
            word_path=spec.get("word_path") or spec.get("docx_path"),
            node_type=spec.get("type", "other"),
            metadata=spec.get("metadata", {}),
        )

    return extract_nodes_from_pdf(
        pdf_path=spec["pdf_path"],
        node_type=spec.get("type", "other"),
        output_image_dir=spec.get("output_image_dir", "data/extracted_images"),
        metadata=spec.get("metadata", {}),
    )


def _build_source_key(spec: dict) -> str:
    source_kind = spec.get("source")
    if source_kind == "jsonl" or "jsonl_path" in spec:
        return f"jsonl::{Path(spec['jsonl_path']).resolve()}"
    if source_kind in {"word", "docx", "doc"} or "word_path" in spec or "docx_path" in spec:
        word_path = spec.get("word_path") or spec.get("docx_path")
        return f"word::{Path(word_path).resolve()}"
    return f"pdf::{Path(spec['pdf_path']).resolve()}"


def _build_source_signature(spec: dict, embedding_backend: str) -> str:
    source_key = _build_source_key(spec)
    _, path_str = source_key.split("::", 1)
    file_path = Path(path_str)
    hasher = hashlib.sha256()
    hasher.update(embedding_backend.encode("utf-8"))
    hasher.update(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    hasher.update(str(file_path).encode("utf-8"))
    if file_path.exists():
        stat = file_path.stat()
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    else:
        hasher.update(b"missing")
    return hasher.hexdigest()


def _build_node_signature(node: KnowledgeNode) -> str:
    hasher = hashlib.sha256()
    payload = {
        "id": node.id,
        "type": node.type,
        "title": node.title,
        "main_text": node.main_text,
        "images": node.images,
    }
    hasher.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _read_cached_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_text_for_embedding(node: KnowledgeNode) -> str:
    text = (node.main_text or "").strip()
    if text:
        return text
    fallback = (node.title or "").strip() or node.id
    return fallback or "empty"


def _embed_text_nodes(nodes: list[KnowledgeNode], embedding_backend: str) -> dict[str, np.ndarray]:
    if not nodes:
        return {}

    def _embed(node: KnowledgeNode) -> tuple[str, np.ndarray]:
        embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
        return node.id, embedder.embed_text(_safe_text_for_embedding(node))

    with ThreadPoolExecutor(max_workers=10) as executor:
        payloads = list(executor.map(_embed, nodes))
    return {node_id: emb for node_id, emb in payloads}


def _embed_images_for_nodes(
    nodes: list[KnowledgeNode],
    embedding_backend: str,
    existing_image_embeddings: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}

    for node in nodes:
        cached = existing_image_embeddings.get(node.id, {})
        merged: dict[str, np.ndarray] = {}

        missing_images: list[str] = []
        for image_path in node.images:
            if image_path in cached:
                merged[image_path] = cached[image_path]
            else:
                missing_images.append(image_path)

        if missing_images:
            embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
            for image_path in missing_images:
                merged[image_path] = embedder.embed_image(image_path)

        output[node.id] = merged

    return output


def _dump_all_nodes(store: SQLiteVectorStore, nodes_dump_path: Path) -> None:
    nodes_dump_path.parent.mkdir(parents=True, exist_ok=True)
    with nodes_dump_path.open("w", encoding="utf-8") as f:
        for node in store.iter_nodes():
            f.write(
                json.dumps(
                    {
                        "id": node.id,
                        "type": node.type,
                        "title": node.title,
                        "main_text": node.main_text,
                        "images": node.images,
                        "metadata": node.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_knowledge_base(
    source_specs: Iterable[dict],
    db_path: str | Path = "data/knowledge.db",
    nodes_dump_path: str | Path = "data/knowledge_nodes.jsonl",
    embedding_backend: str = "openai_qwen",
    state_path: str | Path = "data/index_state.json",
) -> int:
    source_specs = list(source_specs)
    nodes_dump_path = Path(nodes_dump_path)
    state_path = Path(state_path)

    store = SQLiteVectorStore(db_path=db_path)
    cached_state = _read_cached_state(state_path)
    cached_backend = cached_state.get("embedding_backend")
    previous_sources = cached_state.get("sources", {}) if cached_backend == embedding_backend else {}

    if cached_backend and cached_backend != embedding_backend:
        store.clear_all()

    current_source_keys = {_build_source_key(spec) for spec in source_specs}
    removed_keys = set(previous_sources.keys()) - current_source_keys
    for key in removed_keys:
        removed_ids = previous_sources.get(key, {}).get("node_ids", [])
        if removed_ids:
            store.delete_nodes(removed_ids)

    new_state_sources: dict[str, dict] = {}
    for spec in source_specs:
        source_key = _build_source_key(spec)
        source_signature = _build_source_signature(spec, embedding_backend)
        previous_entry = previous_sources.get(source_key, {})

        if previous_entry and previous_entry.get("signature") == source_signature:
            new_state_sources[source_key] = previous_entry
            continue

        nodes = _collect_nodes_from_spec(spec)
        prev_ids = set(previous_entry.get("node_ids", []))
        prev_signatures = previous_entry.get("node_signatures", {})
        current_ids = {node.id for node in nodes}

        removed_ids = sorted(prev_ids - current_ids)
        if removed_ids:
            store.delete_nodes(removed_ids)

        nodes_to_upsert: list[KnowledgeNode] = []
        existing_text_embeddings: dict[str, np.ndarray] = {}
        existing_image_embeddings: dict[str, dict[str, np.ndarray]] = {}

        for node in nodes:
            node_sig = _build_node_signature(node)
            if prev_signatures.get(node.id) == node_sig:
                continue

            existing_text = store.get_node_text_embedding(node.id)
            if existing_text is not None:
                existing_text_embeddings[node.id] = existing_text
                existing_image_embeddings[node.id] = store.get_node_image_embeddings(node.id)
            nodes_to_upsert.append(node)

        text_embeddings = _embed_text_nodes(
            [n for n in nodes_to_upsert if n.id not in existing_text_embeddings],
            embedding_backend=embedding_backend,
        )
        text_embeddings.update(existing_text_embeddings)

        image_embeddings = _embed_images_for_nodes(
            nodes_to_upsert,
            embedding_backend=embedding_backend,
            existing_image_embeddings=existing_image_embeddings,
        )

        for node in nodes_to_upsert:
            store.upsert_node(
                node,
                text_embedding=text_embeddings[node.id],
                image_embeddings=image_embeddings[node.id],
            )

        new_state_sources[source_key] = {
            "signature": source_signature,
            "node_ids": [node.id for node in nodes],
            "node_signatures": {node.id: _build_node_signature(node) for node in nodes},
            "spec": spec,
        }

    _dump_all_nodes(store, nodes_dump_path)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "embedding_backend": embedding_backend,
                "node_count": store.count_nodes(),
                "sources": new_state_sources,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    node_count = store.count_nodes()
    store.close()
    return node_count
