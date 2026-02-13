from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from .models import DesignScheme, GenerationOutput, PerceptionInput, RetrievalResult, SchemeNodeScene


SCHEME_FOCI = [
    ("Scheme A - Public Space Vitality", "public space quality and social life"),
    ("Scheme B - Mobility & Smart Logistics", "traffic organization and smart logistics"),
    ("Scheme C - Interface Renewal & Activation", "building interface and frontage activation"),
]


class SceneSchema(BaseModel):
    node_name: str
    description: str
    desired_image_prompt: str
    reference_example_images: List[str] = Field(default_factory=list)
    selected_representative_images: List[str] = Field(default_factory=list)


class SchemeSchema(BaseModel):
    name: str
    overall_concept: str
    key_strategies: List[str]
    referenced_methods: List[str]
    referenced_example_images: List[str]
    node_scenes: List[SceneSchema]


def _format_retrieved_nodes(retrieval: RetrievalResult) -> dict:
    def fmt(nodes):
        return [
            {
                "id": n.id,
                "title": n.title,
                "text": n.text[:600],
                "images": n.images,
                "score": round(n.score, 4),
            }
            for n in nodes
        ]

    return {
        "retrieved_methods": fmt(retrieval.retrieved_methods),
        "retrieved_policies": fmt(retrieval.retrieved_policies),
        "retrieved_trend_strategies": fmt(retrieval.retrieved_trend_strategies),
    }


def _build_single_scheme_prompt(
    perception: PerceptionInput,
    retrieval: RetrievalResult,
    scheme_name: str,
    scheme_focus: str,
) -> str:
    payload = {
        "task": "Generate one residential renewal scheme in strict JSON.",
        "scheme_name": scheme_name,
        "scheme_focus": scheme_focus,
        "constraints": [
            "Focus on public space and outdoor environment renewal.",
            "Use retrieval method IDs in referenced_methods.",
            "For each node scene, provide node-focused scene description and desired_image_prompt; image recall/ranking is handled by downstream retrieval.",
            "Return exactly one scheme JSON object matching the output schema.",
        ],
        "perception": {
            "district_name": perception.district_name,
            "current_description": perception.current_description,
            "problem_summary": perception.problem_summary,
            "constraints_and_needs": perception.constraints_and_needs,
            "survey_summary": perception.survey_summary,
        },
        "retrieval": _format_retrieved_nodes(retrieval),
        "output_schema": SchemeSchema.model_json_schema(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_multimodal_user_input(prompt: str, representative_images: List[str]) -> List[dict]:
    # Keep image paths out of the textual prompt and pass perception images via multimodal input parts.
    content: List[dict] = [{"type": "input_text", "text": prompt}]
    for idx, image_path in enumerate(representative_images, start=1):
        p = Path(image_path)
        if not p.exists():
            continue
        mime_type = mimetypes.guess_type(p.name)[0] or "image/png"
        # Give the model an explicit path label adjacent to each image so it can return exact selections.
        content.append({"type": "input_text", "text": f"Representative image {idx} path: {image_path}"})
        content.append({"type": "input_image", "image_url": f"data:{mime_type};base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"})
    return content


def _generate_single_scheme_with_openai(
    perception: PerceptionInput,
    retrieval: RetrievalResult,
    scheme_name: str,
    scheme_focus: str,
    model: str,
) -> tuple[DesignScheme, str]:
    from openai import OpenAI

    prompt = _build_single_scheme_prompt(perception, retrieval, scheme_name, scheme_focus)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are an urban renewal planning assistant. Output strict JSON only.",
            },
            {
                "role": "user",
                "content": _build_multimodal_user_input(prompt, perception.representative_images),
            },
        ],
        temperature=0.4,
    )
    text = response.output_text
    parsed = SchemeSchema.model_validate_json(text)

    scheme = DesignScheme(
        name=parsed.name,
        overall_concept=parsed.overall_concept,
        key_strategies=parsed.key_strategies,
        referenced_methods=parsed.referenced_methods,
        referenced_example_images=parsed.referenced_example_images,
        node_scenes=[
            SchemeNodeScene(
                node_name=n.node_name,
                description=n.description,
                desired_image_prompt=n.desired_image_prompt,
                reference_example_images=n.reference_example_images,
                selected_representative_images=n.selected_representative_images,
            )
            for n in parsed.node_scenes
        ],
    )
    return scheme, text


def _fallback_generation(perception: PerceptionInput, retrieval: RetrievalResult) -> GenerationOutput:
    method_ids = [n.id for n in retrieval.retrieved_methods[:6]]
    kb_image_ids = [img for n in retrieval.retrieved_methods[:3] for img in n.images[:1]]
    rep_images = perception.representative_images[:2]

    schemes = []
    for name, focus in SCHEME_FOCI:
        schemes.append(
            DesignScheme(
                name=name,
                overall_concept=f"For {perception.district_name}, prioritize {focus} with phased low-impact renewal.",
                key_strategies=[
                    f"Apply problem-oriented methods for {focus}.",
                    "Respect policy constraints and residents' survey priorities.",
                    "Introduce trend strategies where they improve operations and livability.",
                ],
                referenced_methods=method_ids,
                referenced_example_images=kb_image_ids,
                node_scenes=[
                    SchemeNodeScene(
                        node_name="Community Entrance",
                        description="Reorganize frontage, greenery, and slow traffic sharing.",
                        desired_image_prompt="Edit the source node photo into an upgraded residential entrance with clearer pedestrian priority, planting and integrated smart-delivery elements.",
                        reference_example_images=kb_image_ids[:2],
                        selected_representative_images=rep_images[:1],
                    ),
                    SchemeNodeScene(
                        node_name="Central Open Space",
                        description="Create age-inclusive public activity area with shading and flexible seating.",
                        desired_image_prompt="Edit the source node photo into a central plaza renewal with diverse seating, canopy shading, children and elderly activity zones.",
                        reference_example_images=kb_image_ids[1:3],
                        selected_representative_images=rep_images[1:2],
                    ),
                ],
            )
        )

    return GenerationOutput(scheme_list=schemes, raw_response=None)


def generate_schemes_with_reasoning(
    perception: PerceptionInput,
    retrieval: RetrievalResult,
    model: str = "gpt-5.2",
) -> GenerationOutput:
    """Generate three schemes iteratively (one call per scheme focus) for better controllability."""
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_generation(perception, retrieval)

    scheme_list: List[DesignScheme] = []
    raw_chunks: List[dict] = []
    for scheme_name, scheme_focus in SCHEME_FOCI:
        scheme, raw_text = _generate_single_scheme_with_openai(
            perception=perception,
            retrieval=retrieval,
            scheme_name=scheme_name,
            scheme_focus=scheme_focus,
            model=model,
        )
        scheme_list.append(scheme)
        raw_chunks.append({"scheme_name": scheme_name, "raw": raw_text})

    return GenerationOutput(scheme_list=scheme_list, raw_response=json.dumps(raw_chunks, ensure_ascii=False))
