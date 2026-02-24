from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


NODE_TYPE_CHOICES: List[str] = [
    "社区出入口",
    "景观绿化",
    "室外活动场地",
    "楼栋首层",
    "楼栋外立面",
    "楼栋公共空间",
    "车行道路",
    "人行道路",
    "机动车停车位",
    "非机动车停车位",
    "垃圾回收站",
    "门禁治安",
    "社区边界",
]


def _encode_image_to_data_url(image_path: str) -> Optional[str]:
    """将本地图片转为 data URL，用于多模态输入。"""
    p = Path(image_path)
    if not p.exists():
        return None
    mime_type = mimetypes.guess_type(p.name)[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def _parse_file_name(image_path: str) -> Dict[str, str]:
    """
    从文件名中解析 image_id 与 node_type 默认值。
    """
    p = Path(image_path)
    file_name = p.name
    stem = p.stem

    # 新格式："类型-编号"，例如 "室外活动场地-4"
    parts = re.split(r"[-_－]", stem, maxsplit=1)
    node_type_hint = parts[0] if len(parts) > 0 else ""
    image_id = parts[1] if len(parts) > 1 else ""

    m = re.match(r"(\d+)", image_id)
    if m:
        image_id = m.group(1)

    return {
        "image_id": image_id,
        "file_name": file_name,
        "node_type_hint": node_type_hint,
    }


def _build_image_evidence_schema_example() -> Dict[str, Any]:
    return {
        "image_id": "001",
        "file_name": "001-社区出入口.jpg",
        "node_type": "社区出入口",
        "caption": "对该图片中住区空间的完整中文描述。",
        "issues": [
            {
                "kind": "非机动车占用人行道",
                "severity": 4,
                "description": "入口附近大量非机动车停在人行通道上，影响行人通行与消防疏散。",
            }
        ],
    }


def _normalize_evidence_object(obj: Dict[str, Any], info: Dict[str, str]) -> Dict[str, Any]:
    if "image_id" not in obj or not str(obj.get("image_id", "")).strip():
        obj["image_id"] = info["image_id"]
    else:
        obj["image_id"] = str(obj["image_id"])

    if "file_name" not in obj or not str(obj.get("file_name", "")).strip():
        obj["file_name"] = info["file_name"]

    node_type = str(obj.get("node_type") or "").strip()
    hint = info.get("node_type_hint", "").strip()
    if node_type not in NODE_TYPE_CHOICES:
        obj["node_type"] = hint if hint in NODE_TYPE_CHOICES else "社区出入口"

    if not isinstance(obj.get("caption"), str):
        obj["caption"] = ""

    raw_issues = obj.get("issues")
    if not isinstance(raw_issues, list):
        raw_issues = []
    normalized_issues: List[Dict[str, Any]] = []
    for issue in raw_issues:
        if not isinstance(issue, dict):
            continue
        kind = str(issue.get("kind") or "").strip()
        description = str(issue.get("description") or "").strip()
        severity = issue.get("severity", 3)
        try:
            severity = int(severity)
        except Exception:
            severity = 3
        severity = max(1, min(5, severity))
        if kind:
            normalized_issues.append({"kind": kind, "severity": severity, "description": description})
    obj["issues"] = normalized_issues
    return obj


def _build_single_image_evidence(
    client: OpenAI,
    model: str,
    image_path: str,
    cnt: int,
    district_name: str = "",
    current_description: str = "",
) -> Optional[Dict[str, Any]]:
    info = _parse_file_name(image_path)
    info["image_id"] = str(cnt)
    data_url = _encode_image_to_data_url(image_path)
    if not data_url:
        return None

    schema_example = json.dumps(_build_image_evidence_schema_example(), ensure_ascii=False, indent=2)
    node_type_choices_str = "、".join(NODE_TYPE_CHOICES)

    system_msg = (
        "你是一名从住区现场照片中提取空间与问题信息的助手。"
        "你必须输出严格的 JSON 对象，不允许输出任何额外文字。"
    )

    user_text = f"""
现在有一张住区现场照片，请你根据图片内容输出一个 JSON 对象，字段说明如下：

- image_id: 使用本次输入图片的顺序编号（从 1 开始递增）的字符串。当前这张图的 image_id 应为 "{cnt}"。
- file_name: 原始文件名，例如 "001-社区出入口.jpg"。
- node_type: 使用中文类别，从以下枚举中选择一个，除非你有特别理由，否则优先使用文件名中的类型作为默认值：
  [{node_type_choices_str}]
  例如文件名为 "001-社区出入口.jpg"，默认 node_type 为 "社区出入口"。
- caption: 用较完整的中文句子描述该图片中的住区空间现状，至少包括：
  * 空间类型（例如社区出入口、车行道路、楼栋首层等）；
  * 可见的构成要素（道路、绿化、围墙、建筑立面、车辆、设施等）；
  * 使用状态（停车、人流、活动、管理状态等）；
  * 明显的问题或潜在的改善机会。
- issues: 针对该图片中可见问题的数组（可以为空数组 []），每个元素是一个对象，包含：
  * kind: 用简短中文概括问题类型，例如 "非机动车占用人行道"、"楼栋立面破损"、"垃圾堆放外溢" 等；
  * severity: 问题严重程度的打分，为 1 到 5 的整数，数值越大表示问题越严重（5 为最严重，1 为轻微问题）；
  * description: 用中文简要说明该问题，对应的空间位置、影响对象或影响类型。

请注意：
1. 所有字段名必须与上述完全一致。
2. kind 和 node_type 必须使用中文。
3. severity 必须是 1、2、3、4、5 其中之一，不要使用字符串。
4. 你的最终输出必须是一个单独的 JSON 对象，不能是数组，也不能包含多余的注释或文字。

下面是一个示例结构（仅供字段参考，内容请根据实际图片填写）：
{schema_example}

当前图片的文件名为：
{info["file_name"]}

当前项目小区名称为：
{district_name}

当前项目现状描述为：
{current_description}

其中：
- 本次顺序编号（cnt）是 "{cnt}"，请将其作为 image_id。
- 文件名里的类型部分是 "{info["node_type_hint"]}"（如果它正好属于可选的 node_type 枚举，请优先使用它作为 node_type 的默认值）。
"""

    input_parts = [
        {"type": "input_text", "text": user_text},
        {"type": "input_image", "image_url": data_url},
    ]

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
            {"role": "user", "content": input_parts},
        ],
        temperature=0.1,
    )

    text = resp.output_text
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    return _normalize_evidence_object(obj, info)


def build_visual_evidence(
    site_images: List[str],
    existing: Optional[Dict[str, Any]] = None,
    model: str = "gpt-5.2",
    district_name: str = "",
    current_description: str = "",
) -> Dict[str, Any]:
    """
    构建（或复用）视觉证据 JSON。
    """
    if existing and isinstance(existing, dict) and existing.get("images"):
        return existing

    if not site_images:
        return {"images": []}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"images": []}

    def _worker(payload: tuple[int, str]) -> Optional[Dict[str, Any]]:
        cnt, image_path = payload
        client = OpenAI(api_key=api_key)
        evidence = _build_single_image_evidence(
            client=client,
            model=model,
            image_path=image_path,
            cnt=cnt,
            district_name=district_name,
            current_description=current_description,
        )
        if evidence:
            evidence["image_id"] = str(cnt)
        return evidence

    indexed_images = [(idx, image_path) for idx, image_path in enumerate(site_images, start=1)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        raw_results = list(executor.map(_worker, indexed_images))

    results: List[Dict[str, Any]] = [item for item in raw_results if item]

    return {"images": results}
