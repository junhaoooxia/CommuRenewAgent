from __future__ import annotations

import json
from pathlib import Path

from commurenew_agent.app import generate_design_schemes, index_knowledge_base
from commurenew_agent.models import PerceptionInput


if __name__ == "__main__":
    # 1) Offline indexing (run once whenever knowledge PDFs change)
    pdf_specs = [
        {"pdf_path": "knowledge/policies.pdf", "type": "policy", "metadata": {"city": "demo_city"}},
        {"pdf_path": "knowledge/design_methods.pdf", "type": "design_method"},
        {"pdf_path": "knowledge/trend_strategies.pdf", "type": "trend_strategy"},
    ]

    existing_pdfs = [spec for spec in pdf_specs if Path(spec["pdf_path"]).exists()]
    if existing_pdfs:
        # Build/update persistent KB only when source PDFs are present.
        count = index_knowledge_base(existing_pdfs)
        print(f"Indexed {count} nodes")
    else:
        print("No PDFs found under ./knowledge. Skipping indexing.")

    # 2) Online per-project generation
    perception = PerceptionInput(
        district_name="Xinyuan Community",
        current_description="A 1990s residential district with aging open spaces and fragmented circulation.",
        problem_summary="Public activity spaces are insufficient, pedestrian-vehicle conflicts are frequent, and facilities are unevenly distributed.",
        constraints_and_needs="Need low-cost phased renewal, maintain fire access, preserve mature trees, and satisfy elderly + child-friendly use.",
        survey_summary="Residents prioritize safer walking, more shaded seating, and better package/delivery management.",
        representative_images=[],
    )

    retrieval, output = generate_design_schemes(
        perception=perception,
        # Set to True after configuring GEMINI_API_KEY/GOOGLE_API_KEY for img2img outputs.
        generate_images=False,
    )

    # Dump retrieval groups to verify RAG relevance during debugging.
    print("\n=== Retrieval ===")
    print(json.dumps(retrieval, ensure_ascii=False, indent=2))

    # Dump machine-parseable scheme payload for downstream UI/image generation steps.
    print("\n=== Generated Schemes ===")
    print(
        json.dumps(
            {
                "scheme_list": [scheme.__dict__ | {"node_scenes": [scene.__dict__ for scene in scheme.node_scenes]} for scheme in output.scheme_list]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
