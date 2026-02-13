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
    # Normalize arbitrary titles into stable id-safe tokens for filenames/keys.
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "untitled"


def extract_nodes_from_pdf(
    pdf_path: str | Path,
    node_type: str,
    output_image_dir: str | Path = "data/extracted_images",
    metadata: dict | None = None,
) -> List[KnowledgeNode]:
    # Import fitz lazily so retrieval/reasoning can still run when ingestion deps are absent.
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF ingestion. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    pdf_path = Path(pdf_path)
    # Keep extracted images grouped by source PDF for easier inspection during debugging.
    image_dir = Path(output_image_dir) / pdf_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    nodes: List[KnowledgeNode] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        # Use first text line as a practical page title fallback for retrieval previews.
        title = page.get_text("text").strip().split("\n")[0][:120] or f"{pdf_path.stem} page {page_index+1}"
        page_images: List[str] = []
        for img_no, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_img = doc.extract_image(xref)
            ext = base_img.get("ext", "png")
            image_path = image_dir / f"p{page_index+1}_img{img_no+1}.{ext}"
            image_path.write_bytes(base_img["image"])
            # Store path string so downstream retriever can surface references.
            page_images.append(str(image_path))

        node = KnowledgeNode(
            id=f"{pdf_path.stem}_{page_index+1}",
            type=node_type,
            title=title,
            main_text=page.get_text("text").strip(),
            images=page_images,
            metadata={**(metadata or {}), "source": "pdf", "pdf": str(pdf_path), "page": page_index + 1},
        )
        # One node per PDF page keeps traceability clear during QA.
        nodes.append(node)
    return nodes


def _resolve_image_path(image_path: str, _base_dir: Path) -> str:
    # JSONL may use Windows-style separators; normalize to local filesystem format.
    normalized = Path(image_path.replace("\\", "/"))
    # Keep absolute paths as-is; relative paths are resolved under <repo_root>/ref/<original_relative_path>.
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

            # Prefix id with file stem + type so different files/records avoid key collision.
            node = KnowledgeNode(
                id=f"{jsonl_path.stem}_{record_type}_{record_id}",
                type=record_type,
                title=title,
                main_text=main_text,
                images=images,
                metadata={**(metadata or {}), "source": "jsonl", "jsonl": str(jsonl_path), "line": line_no},
            )
            nodes.append(node)

    return nodes


def _collect_nodes_from_spec(spec: dict) -> List[KnowledgeNode]:
    source_kind = spec.get("source")
    if source_kind == "jsonl" or "jsonl_path" in spec:
        return extract_nodes_from_jsonl(
            jsonl_path=spec["jsonl_path"],
            default_type=spec.get("type", "other"),
            metadata=spec.get("metadata", {}),
        )

    return extract_nodes_from_pdf(
        pdf_path=spec["pdf_path"],
        node_type=spec.get("type", "other"),
        output_image_dir=spec.get("output_image_dir", "data/extracted_images"),
        metadata=spec.get("metadata", {}),
    )




def _iter_files_for_fingerprint(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()], key=lambda x: str(x).lower())


def _compute_index_fingerprint(source_specs: Iterable[dict], embedding_backend: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(embedding_backend.encode("utf-8"))
    hasher.update(json.dumps(list(source_specs), ensure_ascii=False, sort_keys=True).encode("utf-8"))

    for base in [PROJECT_ROOT / "knowledge", PROJECT_ROOT / "ref"]:
        hasher.update(str(base).encode("utf-8"))
        for file_path in _iter_files_for_fingerprint(base):
            stat = file_path.stat()
            hasher.update(str(file_path.relative_to(PROJECT_ROOT)).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    return hasher.hexdigest()


def _read_cached_state(state_path: Path) -> dict | None:
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_existing_node_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(1) FROM knowledge_nodes").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()



def _embed_node_payload(node: KnowledgeNode, embedding_backend: str) -> tuple[KnowledgeNode, np.ndarray, dict[str, np.ndarray]]:
    # Thread worker: each task uses its own embedder instance to avoid shared client/thread-safety issues.
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    text_emb = embedder.embed_text(node.main_text)
    image_embs = {img: embedder.embed_image(img) for img in node.images}
    return node, text_emb, image_embs

def build_knowledge_base(
    source_specs: Iterable[dict],
    db_path: str | Path = "data/knowledge.db",
    nodes_dump_path: str | Path = "data/knowledge_nodes.jsonl",
    embedding_backend: str = "openai_qwen",
    state_path: str | Path = "data/index_state.json",
) -> int:
    source_specs = list(source_specs)
    db_path = Path(db_path)
    nodes_dump_path = Path(nodes_dump_path)
    state_path = Path(state_path)

    current_fingerprint = _compute_index_fingerprint(source_specs, embedding_backend)
    cached_state = _read_cached_state(state_path)
    if (
        cached_state
        and cached_state.get("fingerprint") == current_fingerprint
        and db_path.exists()
        and nodes_dump_path.exists()
    ):
        return _read_existing_node_count(db_path)

    store = SQLiteVectorStore(db_path=db_path)

    all_nodes: List[KnowledgeNode] = []
    for spec in source_specs:
        nodes = _collect_nodes_from_spec(spec)
        all_nodes.extend(nodes)

    # Multithread embedding generation for faster offline indexing on large knowledge bases.
    with ThreadPoolExecutor(max_workers=10) as executor:
        embedded_payloads = list(executor.map(lambda n: _embed_node_payload(n, embedding_backend), all_nodes))

    for node, text_emb, image_embs in embedded_payloads:
        store.upsert_node(node, text_embedding=text_emb, image_embeddings=image_embs)

    nodes_dump_path.parent.mkdir(parents=True, exist_ok=True)
    with nodes_dump_path.open("w", encoding="utf-8") as f:
        for node in all_nodes:
            # Keep an auditable JSONL dump to inspect parser/indexing outputs without DB tools.
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

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "fingerprint": current_fingerprint,
                "embedding_backend": embedding_backend,
                "node_count": len(all_nodes),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    store.close()
    return len(all_nodes)
