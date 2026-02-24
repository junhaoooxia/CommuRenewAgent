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
    renewal_mode: str
    target_goal: List[str]
    overall_concept: str
    scheme_text: str
    expected_effect: List[str]
    key_strategies: List[str]
    referenced_methods: List[str]
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
    retrieval_json = json.dumps(_format_retrieved_nodes(retrieval), ensure_ascii=False, indent=2)
    output_schema_json = json.dumps(SchemeSchema.model_json_schema(), ensure_ascii=False, indent=2)
    prompt = f"""
        你是一名具有城市更新与住区改造经验的综合设计顾问。当前任务是基于给定住区的现状信息、问题总结、居民调查结果与现实约束条件，按照住区更新的一般实践逻辑，完成一次“更新模式判定 + 核心需求提炼 + 空间更新策略生成”的全过程推理。

        本任务不是直接生成完整设计图纸，而是在明确更新模式与更新目标的前提下，提出多套具有空间逻辑的更新建议，并对关键节点进行示意性空间演绎。

        ====================
        一、项目基本信息
        ====================

        【更新对象】
        {perception.district_name}

        【现状与空间条件】
        {perception.current_description}

        【问题总结】
        {perception.problem_summary}

        【居民调查结果】
        {perception.survey_summary}

        ====================
        二、第一阶段任务：更新模式判定与核心需求重构
        ====================

        在生成空间方案前，请先完成以下思考：

        1）更新模式判定
        结合住区建成年代、建筑结构状况、问题类型与现实约束，自主判断本次住区更新更适合以下哪种模式（或其组合）：
        - 微更新（局部改善、设施补齐）
        - 综合整治（系统提升公共空间与基础设施）
        - 拆改结合（部分重构）
        - 整体重建

        请明确说明：
        - 为什么选择该模式；
        - 为什么不选择其他模式；
        - 本次更新的尺度与干预强度应控制在何种范围。

        2）核心更新目标再提炼
        请基于现状问题、居民满意度结构、趋势接受度与现实约束，自行提出 3–5 条本次住区更新的“核心目标”，并进行优先级排序，作为后续空间设计的指导原则。
        注意：核心目标不应简单重复输入内容，而应经过综合判断形成更高层级的归纳。

        ====================
        三、第二阶段任务：生成三套空间更新方案
        ====================

        在明确更新模式与核心目标后，请生成三套具有明显差异化侧重点的更新方案。每套方案需包含：

        （一）总体策略定位
        - 方案名称
        - 更新主题与优先方向
        - 如何回应前述“核心目标”

        （二）重点空间更新策略（以空间为核心）
        请围绕以下空间系统展开更新构思：
        - 出入口空间重构
        - 步行系统与铺装更新
        - 非机动车“停放+充电”系统重构
        - 公共活动空间升级（含室内公共空间补充）
        - 楼栋首层与大堂空间整治
        - 老幼友好优化
        - 智慧物流与无人配送嵌入方式

        每一项需说明：
        - 空间组织如何改变；
        - 是否可分期实施；
        - 与既有空间的衔接方式；
        - 对居民日常生活影响是否可控。

        ====================
        四、趋势场景嵌入原则
        ====================

        在引入智慧物流或低空经济相关设施时，必须遵守：
        1. 地面无人车/机器人配送优先；
        2. 无人机配送仅在满足严格安全条件下考虑；
        3. 设施选址需回应居民偏好（靠近单元门口或可控节点）；
        4. 必须提出隐私保护、安全隔离与噪音控制措施；
        5. 不允许生成大规模科幻化布设。

        ====================
        五、输出形式要求
        ====================

        1）文字方案输出
        - 三套方案结构清晰；
        - 必须先写更新模式与核心目标；
        - 再写空间策略；
        - 不生成完整平面图；
        - 不进行详细工程预算。

        2）关键节点示意生成说明
        每套方案选择 3–5 个关键节点，提供：
        - 更新前问题概述；
        - 更新后空间组织描述；
        - 用于图像生成的空间描述语句（真实、可落地）。

        ====================
        六、必须遵守的现实边界
        ====================

        - 不得进行整体拆除重建；
        - 不得破坏成熟乔木与大面积绿地；
        - 不得影响消防通道与登高面；
        - 不得脱离既有人车分流格局；
        - 必须考虑施工干扰可控。

        请严格按照以上流程展开推理与生成。开始。

        ====================
        【补充约束】
        ====================

        - 当前仅生成单方案，方向提示为：{scheme_name}
        - 该方案重点方向：{scheme_focus}
        - 方案名需要你自行拟定，要求更有设计感与表达力，不要直接照抄方向提示。
        - 必须在 selected_representative_images 中从 visual_evidence 的图片里选择与节点最相关的图片。
        - selected_representative_images 里必须填写可用于图改图的图片路径。
        - 必须使用检索结果中的方法ID写入 referenced_methods。
        - 输出必须是严格 JSON，且只能输出一个 SchemeSchema 对象（不得包含额外文本）。

        【检索结果（仅文本知识用于方案推理）】
        {retrieval_json}

        【视觉证据（来自现场图片识别）】
        {json.dumps(perception.visual_evidence, ensure_ascii=False, indent=2) if perception.visual_evidence else "{}"}
        注意：visual_evidence.images[*].issues[*].severity 为 1–5 的整数，数值越大表示问题越严重（5 为最严重）。
        在判断更新优先级和提出空间策略时，应优先回应严重程度高的问题。

        【输出 JSON Schema】
        {output_schema_json}
        """
    return prompt



def _build_multimodal_user_input(prompt: str, site_images: List[str]) -> List[dict]:
    # Keep image paths out of the textual prompt and pass site images via multimodal input parts.
    content: List[dict] = [{"type": "input_text", "text": prompt}]
    for idx, image_path in enumerate(site_images, start=1):
        p = Path(image_path)
        if not p.exists():
            continue
        mime_type = mimetypes.guess_type(p.name)[0] or "image/png"
        # Give the model an explicit path label adjacent to each image so it can return exact selections.
        content.append({"type": "input_text", "text": f"Site image {idx} path: {image_path}"})
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
                "content": prompt,
            },
        ],
        temperature=0.8,
    )
    text = response.output_text
    parsed = SchemeSchema.model_validate_json(text)

    scheme = DesignScheme(
        name=parsed.name,
        renewal_mode=parsed.renewal_mode,
        target_goal=parsed.target_goal,
        overall_concept=parsed.overall_concept,
        scheme_text=parsed.scheme_text,
        expected_effect=parsed.expected_effect,
        key_strategies=parsed.key_strategies,
        referenced_methods=parsed.referenced_methods,
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
    site_images = perception.site_images[:2]

    schemes = []
    for name, focus in SCHEME_FOCI:
        schemes.append(
            DesignScheme(
                name=name,
                renewal_mode="综合整治 + 微更新",
                target_goal=[
                    "提升步行与公共空间品质",
                    "完善非机动车停放与充电秩序",
                    "在可控条件下嵌入智慧物流",
                ],
                overall_concept=f"For {perception.district_name}, prioritize {focus} with phased low-impact renewal.",
                scheme_text=f"该方案以{focus}为主线，先判定更新模式为综合整治与微更新结合，再围绕核心目标组织分期更新策略，并将节点改造与运营管理协同推进。",
                expected_effect=[
                    "公共空间使用效率与舒适性提升",
                    "交通冲突与无序停放问题缓解",
                    "居民对更新成效感知增强",
                ],
                key_strategies=[
                    f"Apply problem-oriented methods for {focus}.",
                    "Respect policy constraints and residents' survey priorities.",
                    "Introduce trend strategies where they improve operations and livability.",
                ],
                referenced_methods=method_ids,
                node_scenes=[
                    SchemeNodeScene(
                        node_name="Community Entrance",
                        description="Reorganize frontage, greenery, and slow traffic sharing.",
                        desired_image_prompt="Edit the source node photo into an upgraded residential entrance with clearer pedestrian priority, planting and integrated smart-delivery elements.",
                        reference_example_images=kb_image_ids[:2],
                        selected_representative_images=site_images[:1],
                    ),
                    SchemeNodeScene(
                        node_name="Central Open Space",
                        description="Create age-inclusive public activity area with shading and flexible seating.",
                        desired_image_prompt="Edit the source node photo into a central plaza renewal with diverse seating, canopy shading, children and elderly activity zones.",
                        reference_example_images=kb_image_ids[1:3],
                        selected_representative_images=site_images[1:2],
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
