from __future__ import annotations

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import mimetypes
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from .models import DesignScheme, GenerationOutput, PerceptionInput, RetrievalResult, SchemeNodeScene

logger = logging.getLogger(__name__)


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
        你是一名具有“城市更新 + 住区改造 + 空间重构 + 施工可实施性”经验的综合设计顾问。你的任务并不是直接生成设计图，而是在系统推理的基础上，提出基于真实可落地逻辑的住区更新策略与空间演绎说明。

        本任务严格按照“更新模式判定 → 核心目标重构 → 单方案生成 → 空间系统策略 → 节点演绎”的顺序展开，禁止跳步骤推理。

        ==================================================
        一、项目输入信息（用于推理，不得简单复述）
        ==================================================

        【更新对象】
        {perception.district_name}

        【现状与空间条件】
        {perception.current_description}

        【问题总结】
        {perception.problem_summary}

        【居民调查结果】
        {perception.survey_summary}

        【视觉证据（来自现场图像识别）】
        {json.dumps(perception.visual_evidence, ensure_ascii=False, indent=2) if perception.visual_evidence else "{}"}

        注意：
        - visual_evidence.images[*].issues[*].severity 为 1–5，可作为问题参考，但不作为主要优先级排序依据。
        - 主要优先级排序依据：现状痛点、居民满意度结构、调查结果与现实约束。
        - 图像仅用于辅助理解场地空间状况和选择典型节点。

        ==================================================
        二、阶段一任务：更新模式判定 + 核心目标重构（必须先写）
        ==================================================

        请严格按照以下 **推理链框架** 展开，而不是直接给结论。

        ----------------------------------------
        【1）更新模式判定（必须按“推理矩阵”展开）】
        ----------------------------------------

        ① 场地“不可改变因素”
        - 消防登高面
        - 主干结构
        - 成熟乔木与大面积绿地
        - 既有人车分流格局
        - 禁止整体拆除重建

        ② 场地“可适度调整因素”
        - 路径微调
        - 功能节点迁移或优化
        - 铺装更新
        - 楼栋前场品质提升
        - 停车/非机动停放组织优化

        ③ 场地“严禁触碰因素”
        - 危及安全的开口
        - 破坏消防通道
        - 长期封闭导致日常通行瘫痪
        - 科幻化、大尺度虚构设施

        ④ “模式选择可行性矩阵”
        从以下方面进行评估（低/中/高）：
        - 成本可控性
        - 施工干扰程度
        - 居民接受度
        - 技术落地难度

        ⑤ 最终更新模式判定（可组合）
        - 微更新
        - 综合整治
        - 拆改结合（部分重构）
        - 禁止整体重建

        请阐述：
        - 为什么选择该模式  
        - 为什么不选择其他模式  
        - 建议干预尺度（轻度/中度/中偏强）

        ----------------------------------------
        【2）核心更新目标（必须“结构跃迁”）】
        ----------------------------------------
        要求：
        - 根据：现状痛点 + 居民满意度 + 调查结果 + 视觉证据 + 场地约束 进行综合判断
        - 提炼 3–5 条“上位抽象型”目标（如：安全提升、可达性优化、活动丰富度增强、老幼友好、韧性提升等）
        - 禁止简单重复现状或问题原句
        - 必须按优先级排序

        ==================================================
        三、阶段二任务：生成单方案（根据代码变量）
        ==================================================

        当前任务仅生成 **一套方案**：
        - 方向提示：{scheme_name}
        - 方案重点方向：{scheme_focus}

        你必须：
        - 自主拟定具有设计感的方案名称（不得照抄方向提示）。

        ==================================================
        四、方案内容结构（必须保持此顺序）
        ==================================================

        ----------------------------------------
        （一）总体策略定位
        ----------------------------------------
        内容包含：
        - 方案名称
        - 更新主题（一个短语）
        - 优先方向
        - 逐条回应核心更新目标的方式与力度

        ----------------------------------------
        （二）重点空间更新策略（按系统分项输出）
        ----------------------------------------
        必须按以下顺序逐项展开：

        1. 出入口空间重构  
        2. 步行系统与铺装更新  
        3. 非机动车“停放+充电”系统重构  
        4. 公共活动空间升级（含室内公共空间补充）  
        5. 楼栋首层与大堂空间整治  
        6. 老幼友好优化  
        7. 智慧物流与无人配送嵌入方式  

        每一项需说明：
        - 空间组织如何改善  
        - 是否可分期实施（分期逻辑）  
        - 与既有空间的衔接方式  
        - 对居民日常生活影响是否可控（施工干扰、通行保障等）

        ----------------------------------------
        （三）智慧物流与低空经济嵌入原则（不可科幻）
        ----------------------------------------
        必须遵守：
        1. 地面无人车/机器人配送优先  
        2. 无人机配送仅在满足可控安全条件下作为补充  
        3. 场地设施选址应靠近单元入口或可控节点  
        4. 必须提出隐私保护、安全隔离、噪音控制原则  
        5. 禁止大规模、科幻化或漂浮式设施  

        ==================================================
        五、关键节点示意生成（图改图用，需从“更新后描述”转写）
        ==================================================

        请提供 **5–10 个关键节点**，每个节点包含：

        1）节点名称  

        2）更新前问题概述  
        - 根据现状与视觉证据，简要说明该节点存在的主要问题  
        - 可引用问题类型与典型现象（如：路面破损、占停车位、空间压迫感等）

        3）更新后空间组织描述  
        - 以文字清晰描述更新后的空间结构、功能分区、流线关系与使用方式  
        - 要求真实可落地，不夸张，不科幻  
        - 不需要写成图纸说明，但需要逻辑自洽、便于后续转为图像

        4）selected_representative_images  
        - 必须从 visual_evidence 中选择  
        - 数量为 3–5 张  
        - 必须填写可用于图改图的真实图片路径  
        - 应选择与该节点最相关、最能体现问题与场景特征的图片

        5）用于图像生成的提示语（基于“更新后空间组织描述”转写）  
        - 将本节点的“更新后空间组织描述”转写为一段单独的图像生成提示语  
        - 该提示语应紧密对应更新后的空间设想
        - 使用自然语言描述该类最终图像中应该呈现的空间、人物与环境  

        ==================================================
        六、必须遵守的现实边界（硬性要求）
        ==================================================
        - 不得整体拆除重建  
        - 不得破坏成熟乔木与大面积绿地  
        - 不得影响消防通道与登高面  
        - 不得改变既有人车分流格局  
        - 必须保证施工干扰可控  

        ==================================================
        七、补充检索知识（仅用于方案推理参考）
        ==================================================

        【检索结果（用于知识补全，仅作参考，不必在输出中复述全文）】
        {retrieval_json}

        在推理过程中，你可以借鉴检索结果中出现的成熟做法和策略类型，用于完善更新逻辑和空间策略，但不需要逐条复述检索内容。

        ==================================================
        八、JSON 输出要求（必须严格遵守）
        ==================================================
        - 最终输出必须是单个 SchemeSchema JSON 对象  
        - 不得包含额外文字或说明  
        - JSON 字符串必须闭合  
        - 字段名与结构必须与下述 schema 完全一致  
        - selected_representative_images 为数组（3–5 个图片路径）  
        - referenced_methods 必须使用检索结果中的方法 ID  

        ==================================================
        【输出 JSON Schema】
        {output_schema_json}
        ==================================================

        请严格按以上步骤推理后，再按照 Schema 输出最终 JSON。
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
    logger.info("[reasoning] request single scheme. direction=%s focus=%s model=%s", scheme_name, scheme_focus, model)
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
    logger.info("[reasoning] parsed single scheme. name=%s scenes=%s", parsed.name, len(parsed.node_scenes))

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
        logger.warning("[reasoning] OPENAI_API_KEY not set, using fallback generation.")
        return _fallback_generation(perception, retrieval)

    logger.info("[reasoning] concurrent single-scheme generation start. workers=3")
    indexed_foci = list(enumerate(SCHEME_FOCI))
    results: list[tuple[int, str, DesignScheme, str]] = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(
                _generate_single_scheme_with_openai,
                perception=perception,
                retrieval=retrieval,
                scheme_name=scheme_name,
                scheme_focus=scheme_focus,
                model=model,
            ): (idx, scheme_name)
            for idx, (scheme_name, scheme_focus) in indexed_foci
        }

        for fut in as_completed(future_map):
            idx, scheme_name = future_map[fut]
            scheme, raw_text = fut.result()
            results.append((idx, scheme_name, scheme, raw_text))
            logger.info("[reasoning] scheme finished. direction=%s output_name=%s", scheme_name, scheme.name)

    results.sort(key=lambda x: x[0])
    scheme_list: List[DesignScheme] = [item[2] for item in results]
    raw_chunks: List[dict] = [{"scheme_name": item[1], "raw": item[3]} for item in results]

    logger.info("[reasoning] all schemes generated. count=%s", len(scheme_list))
    return GenerationOutput(scheme_list=scheme_list, raw_response=json.dumps(raw_chunks, ensure_ascii=False))
