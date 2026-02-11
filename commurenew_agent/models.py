from __future__ import annotations

from dataclasses import dataclass, field
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
    constraints_and_needs: str
    survey_summary: str
    representative_images: List[str] = field(default_factory=list)

    def to_text_block(self) -> str:
        # Flatten project context into a single textual query prompt for embedding/retrieval.
        return "\n".join(
            [
                f"District: {self.district_name}",
                f"Current Situation: {self.current_description}",
                f"Problems: {self.problem_summary}",
                f"Constraints & Needs: {self.constraints_and_needs}",
                f"Survey Summary: {self.survey_summary}",
            ]
        )


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


@dataclass
class DesignScheme:
    name: str
    overall_concept: str
    key_strategies: List[str]
    referenced_methods: List[str]
    referenced_example_images: List[str]
    node_scenes: List[SchemeNodeScene]


@dataclass
class GenerationOutput:
    scheme_list: List[DesignScheme]
    raw_response: Optional[str] = None
