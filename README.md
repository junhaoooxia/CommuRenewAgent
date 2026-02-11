# CommuRenewAgent

Multimodal RAG system for **residential district renewal design**.

This repository provides a modular pipeline that:

1. Builds a persistent multimodal knowledge base from PDF documents (policy, design methods, trend strategies).
2. Retrieves relevant knowledge nodes using project-specific perception inputs (text + images).
3. Generates **three distinct renewal design schemes** in machine-parseable JSON.
4. Optionally turns node-level scene prompts into generated concept images.

## Project structure

- `commurenew_agent/knowledge_ingestion.py`: PDF parsing, image extraction, node construction, embedding + persistent indexing.
- `commurenew_agent/retrieval.py`: multimodal query embedding and top-k similarity retrieval.
- `commurenew_agent/reasoning.py`: structured prompt + multimodal LLM reasoning into three schemes.
- `commurenew_agent/image_generation.py`: optional image generation helper.
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

- Text embedding from `main_text`
- Image embeddings from node images
- Weighted fusion (default text 0.7, image 0.3)

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
  - per-node scene prompts for downstream image generation

## Core API

```python
from commurenew_agent.app import index_knowledge_base, generate_design_schemes
from commurenew_agent.models import PerceptionInput

pdf_specs = [
    {"pdf_path": "knowledge/policies.pdf", "type": "policy"},
    {"pdf_path": "knowledge/design_methods.pdf", "type": "design_method"},
    {"pdf_path": "knowledge/trend_strategies.pdf", "type": "trend_strategy"},
]

index_knowledge_base(pdf_specs)

perception = PerceptionInput(
    district_name="Example",
    current_description="...",
    problem_summary="...",
    constraints_and_needs="...",
    survey_summary="...",
    representative_images=["inputs/site_plan.png", "inputs/node_photo_01.jpg"],
)

retrieval_payload, generation_output = generate_design_schemes(perception)
```

## Notes

- The included embedder is a deterministic lightweight fallback suitable for local testing and plumbing validation.
- For production quality retrieval, swap `SimpleMultimodalEmbedder` with a stronger model stack (e.g., CLIP + dedicated text embedding model).
- If `OPENAI_API_KEY` is set, reasoning can call an OpenAI model for richer scheme generation; otherwise a deterministic fallback generator is used.
