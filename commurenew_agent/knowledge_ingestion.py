from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

from .embeddings import SimpleMultimodalEmbedder
from .models import KnowledgeNode
from .vector_store import SQLiteVectorStore


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

        node = KnowledgeNode(
            id=f"{pdf_path.stem}_{page_index+1}",
            type=node_type,
            title=title,
            main_text=page.get_text("text").strip(),
            images=page_images,
            metadata={**(metadata or {}), "pdf": str(pdf_path), "page": page_index + 1},
        )
        nodes.append(node)
    return nodes


def build_knowledge_base(
    pdf_specs: Iterable[dict],
    db_path: str | Path = "data/knowledge.db",
    nodes_dump_path: str | Path = "data/knowledge_nodes.jsonl",
) -> int:
    embedder = SimpleMultimodalEmbedder()
    store = SQLiteVectorStore(db_path=db_path)

    all_nodes: List[KnowledgeNode] = []
    for spec in pdf_specs:
        nodes = extract_nodes_from_pdf(
            pdf_path=spec["pdf_path"],
            node_type=spec.get("type", "other"),
            output_image_dir=spec.get("output_image_dir", "data/extracted_images"),
            metadata=spec.get("metadata", {}),
        )
        all_nodes.extend(nodes)

    for node in all_nodes:
        emb = embedder.embed_node(node.main_text, node.images)
        store.upsert_node(node, emb)

    nodes_dump_path = Path(nodes_dump_path)
    nodes_dump_path.parent.mkdir(parents=True, exist_ok=True)
    with nodes_dump_path.open("w", encoding="utf-8") as f:
        for node in all_nodes:
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
