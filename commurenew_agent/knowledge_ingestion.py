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

CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千0-9]+章")
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百千0-9]+节")
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百千0-9]+条")
ITEM_CN_RE = re.compile(r"^[一二三四五六七八九十]+、")
ITEM_PAREN_RE = re.compile(r"^（[一二三四五六七八九十]+）")
ITEM_NUM_RE = re.compile(r"^\d+[\.、]")

CHILD_CHARS = 1100
CHILD_OVERLAP = 220
PARENT_MAX_CHARS = 4200


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "untitled"


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").replace("\r", "\n")).strip()


def _looks_garbled(text: str) -> bool:
    if not text:
        return True
    weird = sum(1 for ch in text if ord(ch) < 9 or (14 <= ord(ch) <= 31) or (127 <= ord(ch) <= 159))
    mojibake = sum(1 for ch in text if 0x0080 <= ord(ch) <= 0x00FF)
    ratio = (weird + mojibake) / max(len(text), 1)
    return ratio > 0.15


def _extract_pdf_text_pages(pdf_path: Path) -> list[dict]:
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF ingestion. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    pages: list[dict] = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        pages.append({"page": i + 1, "text": _normalize_text(page.get_text("text")), "images": page.get_images(full=True)})

    garbled_ratio = sum(1 for p in pages if _looks_garbled(p["text"])) / max(len(pages), 1)
    if garbled_ratio < 0.4:
        return pages

    try:
        import pdfplumber
    except ModuleNotFoundError:
        return pages

    fallback_pages: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            fallback_pages.append({"page": i, "text": _normalize_text(page.extract_text() or ""), "images": []})

    if fallback_pages and sum(1 for p in fallback_pages if not _looks_garbled(p["text"])) > sum(1 for p in pages if not _looks_garbled(p["text"])):
        return fallback_pages
    return pages


def _extract_structural_units(doc_title: str, page_texts: list[dict], base_metadata: dict) -> list[dict]:
    chapter = ""
    section = ""
    article = ""
    item = ""
    units: list[dict] = []

    for item_page in page_texts:
        page_no = item_page["page"]
        text = item_page["text"]
        if not text:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for para in paragraphs:
            heading = para.split("\n", 1)[0].strip()
            if CHAPTER_RE.match(heading):
                chapter, section, article, item = heading, "", "", ""
            elif SECTION_RE.match(heading):
                section, article, item = heading, "", ""
            elif ARTICLE_RE.match(heading):
                article, item = heading, ""
            elif ITEM_CN_RE.match(heading) or ITEM_PAREN_RE.match(heading) or ITEM_NUM_RE.match(heading):
                item = heading

            path_parts = [p for p in [chapter, section, article, item] if p]
            units.append(
                {
                    "text": para,
                    "chapter": chapter,
                    "section": section,
                    "article": article,
                    "item": item,
                    "item_path": " > ".join(path_parts),
                    "page_start": page_no,
                    "page_end": page_no,
                    "metadata": base_metadata,
                    "doc_title": doc_title,
                }
            )

    return units


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _build_parent_child_nodes(
    source_id_prefix: str,
    node_type: str,
    doc_title: str,
    units: list[dict],
    base_metadata: dict,
    images: list[str] | None = None,
) -> list[KnowledgeNode]:
    images = images or []
    if not units:
        return []

    parents: list[dict] = []
    current_group: list[dict] = []
    current_key = None

    def _flush_group(group: list[dict]) -> None:
        if not group:
            return
        raw_text = "\n\n".join(u["text"] for u in group if u["text"].strip())
        parent_parts = _split_long_text(raw_text, max_chars=PARENT_MAX_CHARS, overlap_chars=300)
        for idx, part in enumerate(parent_parts, start=1):
            parent_id = f"{source_id_prefix}_parent_{len(parents)+1}_{idx}"
            chapter = group[-1].get("chapter", "")
            section = group[-1].get("section", "")
            article = group[-1].get("article", "")
            item_path = group[-1].get("item_path", "")
            parents.append(
                {
                    "parent_id": parent_id,
                    "text": part,
                    "chapter": chapter,
                    "section": section,
                    "article": article,
                    "item_path": item_path,
                    "page_start": min(u["page_start"] for u in group),
                    "page_end": max(u["page_end"] for u in group),
                }
            )

    for unit in units:
        key = unit.get("article") or unit.get("section") or unit.get("chapter") or f"p{unit['page_start']}"
        if current_key is None:
            current_key = key
        if key != current_key:
            _flush_group(current_group)
            current_group = []
            current_key = key
        current_group.append(unit)
    _flush_group(current_group)

    nodes: list[KnowledgeNode] = []
    for parent in parents:
        child_parts = _split_long_text(parent["text"], max_chars=CHILD_CHARS, overlap_chars=CHILD_OVERLAP)
        for i, child_text in enumerate(child_parts, start=1):
            child_id = f"{parent['parent_id']}_child_{i}"
            meta = {
                **base_metadata,
                "chunk_level": "child",
                "doc_title": doc_title,
                "chapter": parent["chapter"],
                "section": parent["section"],
                "article": parent["article"],
                "item_path": parent["item_path"],
                "page_range": [parent["page_start"], parent["page_end"]],
                "parent_id": parent["parent_id"],
                "parent_text": parent["text"],
            }
            nodes.append(
                KnowledgeNode(
                    id=child_id,
                    type=node_type,
                    title=doc_title,
                    main_text=child_text,
                    images=images,
                    metadata=meta,
                )
            )
    return nodes


def _extract_region_publish(text: str) -> tuple[str | None, str | None]:
    region_match = re.search(r"(北京市|上海市|天津市|重庆市|[\u4e00-\u9fff]{2,8}(?:省|市|自治区))", text)
    date_match = re.search(r"(20\d{2}[年\-\./]\d{1,2}[月\-\./]\d{1,2}日?)", text)
    region = region_match.group(1) if region_match else None
    publish_date = date_match.group(1) if date_match else None
    return region, publish_date


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

    page_texts = _extract_pdf_text_pages(pdf_path)
    page_images: list[str] = []

    # extract images via fitz even when using pdfplumber text fallback
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_no, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_img = doc.extract_image(xref)
            ext = base_img.get("ext", "png")
            image_path = image_dir / f"p{page_index+1}_img{img_no+1}.{ext}"
            image_path.write_bytes(base_img["image"])
            page_images.append(str(image_path))

    all_text = "\n\n".join(p["text"] for p in page_texts)
    region, publish_date = _extract_region_publish(all_text)
    base_meta = {
        **(metadata or {}),
        "source": "pdf",
        "pdf": str(pdf_path),
        "region": region,
        "publish_date": publish_date,
    }

    units = _extract_structural_units(doc_title=pdf_path.stem, page_texts=page_texts, base_metadata=base_meta)
    if not units:
        units = [
            {
                "text": all_text,
                "chapter": "",
                "section": "",
                "article": "",
                "item_path": "",
                "page_start": 1,
                "page_end": max((p["page"] for p in page_texts), default=1),
            }
        ]

    return _build_parent_child_nodes(
        source_id_prefix=f"{slugify(pdf_path.stem)}_{node_type}",
        node_type=node_type,
        doc_title=pdf_path.stem,
        units=units,
        base_metadata=base_meta,
        images=page_images,
    )


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
    text = "\n\n".join(paragraphs)
    region, publish_date = _extract_region_publish(text)
    base_meta = {**(metadata or {}), "source": "word", "word": str(word_path), "region": region, "publish_date": publish_date}

    page_texts = [{"page": 1, "text": text}]
    units = _extract_structural_units(doc_title=word_path.stem, page_texts=page_texts, base_metadata=base_meta)
    if not units and text:
        units = [{"text": text, "chapter": "", "section": "", "article": "", "item_path": "", "page_start": 1, "page_end": 1}]

    return _build_parent_child_nodes(
        source_id_prefix=f"{slugify(word_path.stem)}_{node_type}",
        node_type=node_type,
        doc_title=word_path.stem,
        units=units,
        base_metadata=base_meta,
        images=[],
    )


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

            region, publish_date = _extract_region_publish(main_text)
            base_meta = {
                **(metadata or {}),
                "source": "jsonl",
                "jsonl": str(jsonl_path),
                "line": line_no,
                "region": region,
                "publish_date": publish_date,
            }
            units = _extract_structural_units(
                doc_title=title,
                page_texts=[{"page": 1, "text": main_text}],
                base_metadata=base_meta,
            )
            if not units and main_text:
                units = [{"text": main_text, "chapter": "", "section": "", "article": "", "item_path": "", "page_start": 1, "page_end": 1}]

            source_prefix = f"{slugify(jsonl_path.stem)}_{record_type}_{slugify(record_id)}"
            nodes.extend(
                _build_parent_child_nodes(
                    source_id_prefix=source_prefix,
                    node_type=record_type,
                    doc_title=title,
                    units=units,
                    base_metadata=base_meta,
                    images=images,
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
        "metadata": node.metadata,
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
