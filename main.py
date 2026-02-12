from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from commurenew_agent.app import generate_design_schemes, index_knowledge_base
from commurenew_agent.models import PerceptionInput


def _setup_logger() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("commurenew_agent.main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized. Log file: %s", log_file)
    return logger


def _to_serializable_generation(output) -> dict:
    return {
        "scheme_list": [
            scheme.__dict__ | {"node_scenes": [scene.__dict__ for scene in scheme.node_scenes]}
            for scheme in output.scheme_list
        ]
    }


def _save_output(retrieval: dict, output_dict: dict, logger: logging.Logger) -> Path:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {"retrieval": retrieval, "generation": output_dict}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved final result to: %s", output_path)
    return output_path


if __name__ == "__main__":
    logger = _setup_logger()

    logger.info("Step 1/4: Preparing PDF specs for offline indexing")
    pdf_specs = [
        {"pdf_path": "knowledge/policies.pdf", "type": "policy", "metadata": {"city": "demo_city"}},
        {"pdf_path": "knowledge/design_methods.pdf", "type": "design_method"},
        {"pdf_path": "knowledge/trend_strategies.pdf", "type": "trend_strategy"},
    ]

    existing_pdfs = [spec for spec in pdf_specs if Path(spec["pdf_path"]).exists()]
    logger.info("Detected %d/%d PDF files", len(existing_pdfs), len(pdf_specs))

    if existing_pdfs:
        logger.info("Step 2/4: Running knowledge indexing")
        count = index_knowledge_base(existing_pdfs)
        logger.info("Indexed %d nodes", count)
    else:
        logger.warning("No PDFs found under ./knowledge. Skipping indexing.")

    logger.info("Step 3/4: Building perception input and running retrieval + reasoning")
    perception = PerceptionInput(
        district_name="Xinyuan Community",
        current_description="A 1990s residential district with aging open spaces and fragmented circulation.",
        problem_summary="Public activity spaces are insufficient, pedestrian-vehicle conflicts are frequent, and facilities are unevenly distributed.",
        constraints_and_needs="Need low-cost phased renewal, maintain fire access, preserve mature trees, and satisfy elderly + child-friendly use.",
        survey_summary="Residents prioritize safer walking, more shaded seating, and better package/delivery management.",
        representative_images=["inputs/masterPlan.png", "inputs/centerGround.JPG"],
    )

    retrieval, output = generate_design_schemes(
        perception=perception,
        # Set to True after configuring GEMINI_API_KEY/GOOGLE_API_KEY for img2img outputs.
        generate_images=False,
    )

    logger.info(
        "Retrieval done. methods=%d, policies=%d, strategies=%d",
        len(retrieval.get("retrieved_methods", [])),
        len(retrieval.get("retrieved_policies", [])),
        len(retrieval.get("retrieved_trend_strategies", [])),
    )
    logger.info("Reasoning done. generated_schemes=%d", len(output.scheme_list))

    logger.info("Step 4/4: Saving results to output directory")
    output_dict = _to_serializable_generation(output)
    saved_path = _save_output(retrieval, output_dict, logger)

    print("\n=== Retrieval ===")
    print(json.dumps(retrieval, ensure_ascii=False, indent=2))

    print("\n=== Generated Schemes ===")
    print(json.dumps(output_dict, ensure_ascii=False, indent=2))

    print(f"\nResult written to: {saved_path}")
