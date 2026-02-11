from __future__ import annotations

import json
import os
from typing import List

from pydantic import BaseModel, Field

from .models import DesignScheme, GenerationOutput, PerceptionInput, RetrievalResult, SchemeNodeScene


class SceneSchema(BaseModel):
    node_name: str
    description: str
    desired_image_prompt: str
    reference_example_images: List[str] = Field(default_factory=list)


class SchemeSchema(BaseModel):
    name: str
    overall_concept: str
    key_strategies: List[str]
    referenced_methods: List[str]
    referenced_example_images: List[str]
    node_scenes: List[SceneSchema]


class OutputSchema(BaseModel):
    scheme_list: List[SchemeSchema]


def _build_prompt(perception: PerceptionInput, retrieval: RetrievalResult) -> str:
    def fmt_nodes(nodes):
        # Trim long text bodies to keep token usage predictable.
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

    payload = {
        # Structured payload helps model output strict machine-parseable JSON.
        "task": "Generate exactly three distinct residential renewal schemes in JSON.",
        "constraints": [
            "Focus on public space and outdoor environment renewal.",
            "Explicitly cite referenced method IDs from retrieval.",
            "Include policies and trend strategies where relevant.",
            "Each scheme must have a distinct emphasis.",
        ],
        "perception": {
            "district_name": perception.district_name,
            "current_description": perception.current_description,
            "problem_summary": perception.problem_summary,
            "constraints_and_needs": perception.constraints_and_needs,
            "survey_summary": perception.survey_summary,
            "representative_images": perception.representative_images,
        },
        "retrieval": {
            "retrieved_methods": fmt_nodes(retrieval.retrieved_methods),
            "retrieved_policies": fmt_nodes(retrieval.retrieved_policies),
            "retrieved_trend_strategies": fmt_nodes(retrieval.retrieved_trend_strategies),
        },
        "output_schema": OutputSchema.model_json_schema(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def generate_schemes_with_reasoning(
    perception: PerceptionInput,
    retrieval: RetrievalResult,
    model: str = "gpt-4.1",
) -> GenerationOutput:
    # Serialize all constraints/context into a single prompt blob.
    prompt = _build_prompt(perception, retrieval)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Deterministic fallback keeps API usable in offline or keyless environments.
        return _fallback_generation(perception, retrieval)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    # Request strict JSON so frontend/image backend can consume output directly.
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are an urban renewal planning assistant. Output strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    text = response.output_text
    # Validate model JSON against schema before returning to caller.
    parsed = OutputSchema.model_validate_json(text)
    return GenerationOutput(
        scheme_list=[
            DesignScheme(
                name=s.name,
                overall_concept=s.overall_concept,
                key_strategies=s.key_strategies,
                referenced_methods=s.referenced_methods,
                referenced_example_images=s.referenced_example_images,
                node_scenes=[
                    SchemeNodeScene(
                        node_name=n.node_name,
                        description=n.description,
                        desired_image_prompt=n.desired_image_prompt,
                        reference_example_images=n.reference_example_images,
                    )
                    for n in s.node_scenes
                ],
            )
            for s in parsed.scheme_list
        ],
        raw_response=text,
    )


def _fallback_generation(perception: PerceptionInput, retrieval: RetrievalResult) -> GenerationOutput:
    # Reuse top retrieved methods/images as explicit references in fallback schemes.
    method_ids = [n.id for n in retrieval.retrieved_methods[:6]]
    image_ids = [img for n in retrieval.retrieved_methods[:3] for img in n.images[:1]]

    templates = [
        ("Scheme A - Public Space Vitality", "public space quality and social life"),
        ("Scheme B - Mobility & Smart Logistics", "traffic organization and smart logistics"),
        ("Scheme C - Interface Renewal & Activation", "building interface and frontage activation"),
    ]

    schemes = []
    for name, focus in templates:
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
                referenced_example_images=image_ids,
                node_scenes=[
                    SchemeNodeScene(
                        node_name="Community Entrance",
                        description="Reorganize frontage, greenery, and slow traffic sharing.",
                        desired_image_prompt="Concept rendering of upgraded residential entrance with greenery, clear pedestrian priority, integrated smart delivery lockers.",
                        reference_example_images=image_ids[:2],
                    ),
                    SchemeNodeScene(
                        node_name="Central Open Space",
                        description="Create age-inclusive public activity area with shading and flexible seating.",
                        desired_image_prompt="Landscape concept image of central plaza renewal with diverse seating, canopy shading, children's and elderly zones.",
                        reference_example_images=image_ids[1:3],
                    ),
                ],
            )
        )

    return GenerationOutput(scheme_list=schemes, raw_response=None)
