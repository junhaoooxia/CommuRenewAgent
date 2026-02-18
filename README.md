# CommuRenewAgent

Multimodal RAG system for **residential district renewal design**.

This repository provides a modular pipeline that:

1. Builds a persistent multimodal knowledge base from PDF documents and JSONL datasets (policy, design methods, trend strategies).
2. Retrieves relevant knowledge nodes using project-specific perception inputs (text + images).
3. Generates **three distinct renewal design schemes** in machine-parseable JSON (iterative: one run per scheme focus).
4. Optionally performs image-to-image editing per node scene via Gemini to produce updated concept images.

## Project structure

- `commurenew_agent/knowledge_ingestion.py`: PDF/JSONL parsing, image extraction/path normalization, node construction, embedding + persistent indexing.
- `commurenew_agent/retrieval.py`: multimodal query embedding and top-k similarity retrieval.
- `commurenew_agent/reasoning.py`: structured prompt + multimodal LLM reasoning into three schemes (iterative single-scheme generation loop).
- `commurenew_agent/image_generation.py`: optional Gemini img2img helper (using selected representative source images).
- `commurenew_agent/app.py`: end-to-end orchestration.
- `main.py`: simple runnable example.

## Data model

Each knowledge page/method is represented as a node:

```json
{
  "id": "design_methods_17",
  "type": "design_method",
  "title": "Pocket-space activation with slow traffic integration",
  "main_text": "...",
  "images": ["data/extracted_images/design_methods/p17_img1.png"],
  "metadata": {"page": 17, "scale": "community"}
}
```

Embeddings are multimodal:

- Text and image embeddings are generated with **Qwen `qwen3-vl-embedding`** (DashScope) in a unified 2560-d vector space.

The index is persisted to SQLite (`data/knowledge.db`) so it can be reused across sessions.

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Add knowledge sources

Put your source files under `knowledge/` (or provide absolute paths in code):

- `knowledge/policies.pdf`
- `knowledge/design_methods.pdf`
- `knowledge/trend_strategies.pdf`
- `knowledge/design_method.jsonl`
- `knowledge/trend_strategy.jsonl`

### 3) Run example

```bash
python main.py
```

This will:

- Index available PDF + JSONL sources (offline stage)
- Run retrieval with sample perception input (online stage)
- Produce JSON output with:
  - `scheme_list` (3 schemes)
  - per-scheme references to methods/images
  - per-node scene prompts + selected source images (`selected_representative_images`) for downstream img2img generation

## Core API

```python
from commurenew_agent.app import index_knowledge_base, generate_design_schemes
from commurenew_agent.models import PerceptionInput

source_specs = [
    {"source": "pdf", "pdf_path": "knowledge/policies.pdf", "type": "policy"},
    {"source": "jsonl", "jsonl_path": "knowledge/design_method.jsonl", "type": "design_method"},
    {"source": "jsonl", "jsonl_path": "knowledge/trend_strategy.jsonl", "type": "trend_strategy"},
    {"source": "word", "word_path": "knowledge/community_guideline.docx", "type": "policy"},
]

index_knowledge_base(source_specs, embedding_backend="openai_qwen")

perception = PerceptionInput(
    district_name="Example",
    current_description="...",
    problem_summary="...",
    survey_summary="...",
    representative_images=["inputs/site_plan.png", "inputs/node_photo_01.jpg"],
)

retrieval_payload, generation_output = generate_design_schemes(
    perception,
    embedding_backend="openai_qwen",
    generate_images=True,
    image_model="gemini-3-pro-image-preview",
)
```

## Notes

- Default embedding backend is `openai_qwen`: Qwen `qwen3-vl-embedding` is used for both text (`input=[{"text": ...}]`) and image (`input=[{"image": ...}]`) with 2560-d alignment for direct multimodal retrieval.
- The only fallback backend is `simple` (deterministic local embedding) for environments without DashScope credentials.
- JSONL ingestion supports records with `id/type/title/main_text/images`; relative image paths are normalized (including Windows `\` separators) and resolved as absolute paths under `<repo_root>/ref/...` (e.g. `design_method_images\a.jpg` -> `/.../CommuRenewAgent/ref/design_method_images/a.jpg`).
- Word ingestion is supported for `.docx` sources via `{"source": "word", "word_path": "...docx"}` (legacy `.doc` should be converted to `.docx`).
- Image editing uses Gemini API (set `GEMINI_API_KEY` or `GOOGLE_API_KEY`). The reasoning layer selects which files from `representative_images` should be edited for each node scene.
- For `openai_qwen` embeddings, set `DASHSCOPE_API_KEY` (Qwen text+vision embedding).
- For `openai_qwen` embeddings, input images are auto-resized proportionally when they exceed Qwen size limit (5070KB), targeting the upper bound without exceeding it.
- Offline indexing now keeps per-source state and compares current vs previous sources to apply incremental updates (add/update/remove) directly in `knowledge.db`, without re-parsing unchanged PDFs/JSONL/Word files.
- Offline embedding generation uses multithreading (`max_workers=10`) during node indexing to speed up large knowledge-base builds.
- Retrieval is now split by objective: text-plan generation recalls from text embeddings only (policy/method/strategy text), while node-image outputs are post-ranked in two steps (scene->site images from `perception.representative_images`, and scene->method images from already-retrieved method/strategy image pools).
- If `OPENAI_API_KEY` is set, reasoning calls `gpt-5.2` by default and sends `perception.representative_images` as multimodal image inputs (not injected into the prompt text); otherwise a deterministic fallback generator is used.

