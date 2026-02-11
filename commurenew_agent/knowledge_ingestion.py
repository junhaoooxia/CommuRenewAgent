from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

from .embeddings import EmbeddingConfig, get_embedder
from .models import KnowledgeNode
from .vector_store import SQLiteVectorStore


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
            # Store absolute/relative path string so downstream retriever can surface references.
            page_images.append(str(image_path))

        node = KnowledgeNode(
            id=f"{pdf_path.stem}_{page_index+1}",
            type=node_type,
            title=title,
            main_text=page.get_text("text").strip(),
            images=page_images,
            metadata={**(metadata or {}), "pdf": str(pdf_path), "page": page_index + 1},
        )
        # One node per PDF page keeps traceability clear during QA.
        nodes.append(node)
    return nodes


def build_knowledge_base(
    pdf_specs: Iterable[dict],
    db_path: str | Path = "data/knowledge.db",
    nodes_dump_path: str | Path = "data/knowledge_nodes.jsonl",
    embedding_backend: str = "llamaindex",
) -> int:
    # Build embedder once to avoid repeated model initialization overhead.
    embedder = get_embedder(EmbeddingConfig(backend=embedding_backend))
    store = SQLiteVectorStore(db_path=db_path)

    all_nodes: List[KnowledgeNode] = []
    for spec in pdf_specs:
        # Each spec maps one source PDF into typed nodes (policy/method/trend).
        nodes = extract_nodes_from_pdf(
            pdf_path=spec["pdf_path"],
            node_type=spec.get("type", "other"),
            output_image_dir=spec.get("output_image_dir", "data/extracted_images"),
            metadata=spec.get("metadata", {}),
        )
        all_nodes.extend(nodes)

    for node in all_nodes:
        # Embed multimodal node content and persist vectors for later similarity search.
        emb = embedder.embed_node(node.main_text, node.images)
        store.upsert_node(node, emb)

    nodes_dump_path = Path(nodes_dump_path)
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

    store.close()
    return len(all_nodes)
