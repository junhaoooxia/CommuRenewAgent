from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Literal, Optional

NodeType = Literal["policy", "design_method", "trend_strategy", "other"]


@dataclass
class KnowledgeNode:
    id: str
    type: NodeType
    title: str
    main_text: str
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionInput:
    district_name: str
    current_description: str
    problem_summary: str
    survey_summary: str
    site_images: List[str] = field(default_factory=list)
    visual_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_text_block(self) -> str:
        # Flatten project context into a single textual query prompt for embedding/retrieval.
        blocks = [
            f"District: {self.district_name}",
            f"Current Situation: {self.current_description}",
            f"Problems: {self.problem_summary}",
            f"Survey Summary: {self.survey_summary}",
        ]
        if self.visual_evidence:
            blocks.append("Visual Evidence JSON: " + json.dumps(self.visual_evidence, ensure_ascii=False))
        return "\n".join(blocks)


@dataclass
class RetrievedNode:
    id: str
    type: str
    title: str
    text: str
    images: List[str]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    retrieved_methods: List[RetrievedNode] = field(default_factory=list)
    retrieved_policies: List[RetrievedNode] = field(default_factory=list)
    retrieved_trend_strategies: List[RetrievedNode] = field(default_factory=list)


@dataclass
class SchemeNodeScene:
    node_name: str
    description: str
    desired_image_prompt: str
    reference_example_images: List[str]
    selected_representative_images: List[str] = field(default_factory=list)
    generated_images: List[str] = field(default_factory=list)


@dataclass
class DesignScheme:
    name: str
    renewal_mode: str
    target_goal: List[str]
    overall_concept: str
    scheme_text: str
    expected_effect: List[str]
    key_strategies: List[str]
    referenced_methods: List[str]
    node_scenes: List[SchemeNodeScene]


@dataclass
class GenerationOutput:
    scheme_list: List[DesignScheme]
    raw_response: Optional[str] = None
