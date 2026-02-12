# CommuRenewAgent

Multimodal RAG system for **residential district renewal design**.

This repository provides a modular pipeline that:

1. Builds a persistent multimodal knowledge base from PDF documents (policy, design methods, trend strategies).
2. Retrieves relevant knowledge nodes using project-specific perception inputs (text + images).
3. Generates **three distinct renewal design schemes** in machine-parseable JSON (iterative: one run per scheme focus).
4. Optionally performs image-to-image editing per node scene via Gemini to produce updated concept images.
3. Generates **three distinct renewal design schemes** in machine-parseable JSON (iterative: one run per scheme focus).
4. Optionally performs image-to-image editing per node scene via Gemini to produce updated concept images.


## Project structure

- `commurenew_agent/knowledge_ingestion.py`: PDF parsing, image extraction, node construction, embedding + persistent indexing.
- `commurenew_agent/retrieval.py`: multimodal query embedding and top-k similarity retrieval.
- `commurenew_agent/reasoning.py`: structured prompt + multimodal LLM reasoning into three schemes (iterative single-scheme generation loop).
- `commurenew_agent/image_generation.py`: optional Gemini img2img helper (using selected representative source images).
- `commurenew_agent/reasoning.py`: structured prompt + multimodal LLM reasoning into three schemes (iterative single-scheme generation loop).
- `commurenew_agent/image_generation.py`: optional Gemini img2img helper (using selected representative source images).


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

- Text and image embeddings are generated with **LlamaIndex CLIP** (`llama_index.embeddings.clip.ClipEmbedding`)
- Weighted fusion is applied (default text 0.7, image 0.3)

The index is persisted to SQLite (`data/knowledge.db`) so it can be reused across sessions.

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Add PDFs

Put your source PDFs under `knowledge/` (or provide absolute paths in code):

- `knowledge/policies.pdf`
- `knowledge/design_methods.pdf`
- `knowledge/trend_strategies.pdf`

### 3) Run example

```bash
python main.py
```

This will:

- Index available PDFs (offline stage)
- Run retrieval with sample perception input (online stage)
- Produce JSON output with:
  - `scheme_list` (3 schemes)
  - per-scheme references to methods/images
  - per-node scene prompts + selected source images (`selected_representative_images`) for downstream img2img generation
  - per-node scene prompts + selected source images (`selected_representative_images`) for downstream img2img generation


## Core API

```python
from commurenew_agent.app import index_knowledge_base, generate_design_schemes
from commurenew_agent.models import PerceptionInput

pdf_specs = [
    {"pdf_path": "knowledge/policies.pdf", "type": "policy"},
    {"pdf_path": "knowledge/design_methods.pdf", "type": "design_method"},
    {"pdf_path": "knowledge/trend_strategies.pdf", "type": "trend_strategy"},
]

index_knowledge_base(pdf_specs, embedding_backend="llamaindex")

perception = PerceptionInput(
    district_name="Example",
    current_description="...",
    problem_summary="...",
    constraints_and_needs="...",
    survey_summary="...",
    representative_images=["inputs/site_plan.png", "inputs/node_photo_01.jpg"],
)

retrieval_payload, generation_output = generate_design_schemes(
    perception,
    embedding_backend="llamaindex",
    generate_images=True,
    image_model="gemini-3-pro-image-preview",
    generate_images=True,
    image_model="gemini-2.5-flash-image-preview",
)
```

## Notes

- Default embedding backend is `llamaindex` (CLIP via LlamaIndex). If CLIP runtime dependencies are missing, the code automatically falls back to `simple` and emits a warning. You can also force fallback with `embedding_backend="simple"`.
- Image editing uses Gemini API (set `GEMINI_API_KEY` or `GOOGLE_API_KEY`). The reasoning layer selects which files from `representative_images` should be edited for each node scene.
- If `OPENAI_API_KEY` is set, reasoning calls `gpt-5.2` by default and sends `perception.representative_images` as multimodal image inputs (not injected into the prompt text); otherwise a deterministic fallback generator is used.
- Image editing uses Gemini API (set `GEMINI_API_KEY` or `GOOGLE_API_KEY`). The reasoning layer selects which files from `representative_images` should be edited for each node scene.
- If `OPENAI_API_KEY` is set, reasoning can call an OpenAI model for richer scheme generation; otherwise a deterministic fallback generator is used.



> If you want true CLIP embeddings via LlamaIndex, install extra runtime deps: `torch` and OpenAI CLIP (`pip install git+https://github.com/openai/CLIP.git`).
