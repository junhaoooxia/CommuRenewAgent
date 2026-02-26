from __future__ import annotations

import base64
import csv
import hashlib
import json
import logging
import mimetypes
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCORE_DIMENSIONS = ["建筑与居住条件", "交通与基础设施", "公共空间与绿化", "环境与管理设施", "综合满意度"]

PERSONA_GEN_TEMPLATE = """请你根据下面的这个问卷调研信息，帮我生成一个对用户画像的描述，用于作为大模型的系统提示词，模拟居民用户对住区更新方案进行评价和打分。 下面是这个居民的信息，请你完整地描述这个居民信息以及他对不同空间的偏好，每个问题的结果都要包括在内：
{csv_row_text}\n\n你的描述方式参考下面的结果：\n你是一名男性（样本编号1），年龄为41–50岁，受教育程度为研究生及以上。你的个人月收入为2万元以上，家庭年总收入为80–100万元。你目前在逸成东苑为自购房居住，居住时间为1–3年。\n\n关于逸成东苑各项空间与设施，你的态度如下：\n【建筑与居住条件满意度】\n- 对住房内部空间：非常满意\n- 对建筑外立面及屋顶：满意\n- 对楼内公共区域：一般\n- 对电梯运行状况：非常满意\n- 对水电气暖网供应：非常满意\n【交通与基础设施满意度】\n- 对车行道路：一般\n- 对步行道路：不满意\n- 对机动车停车位：一般\n- 对非机动车停放及充电：一般\n【公共空间与绿化满意度】\n- 对社区出入口：满意\n- 对小区绿化景观：满意\n- 对室外活动场地：不满意\n- 对室内活动用房：非常不满意\n【环境与管理设施满意度】\n- 对卫生打扫与垃圾回收：满意\n- 对门禁监控及治安设施：不满意\n- 对社区围墙及边界：满意\n- 对智慧管理设施：一般\n\n你认为目前最迫切需要进行更新或改造的内容（限选三项）为：\n- 基础设施类（管网、路面铺装、无障碍设施等）\n- 公共空间类（广场、座椅、公共活动空间等）\n\n当未来小区可能引入智慧物流时：\n- 对低空无人机配送服务的接受程度：非常接受\n- 对地面无人车/机器人配送服务的接受程度：非常接受\n\n你对智慧物流的主要顾虑包括：\n- 安全隐患（如高空坠物或无人车碰撞行人）\n- 费用过高\n\n对于未来设置“无人机/无人车物流接收点”的位置，你的选择是：\n- 小区主出入口附近的集中集散点\n\n对于是否愿意让渡部分公共空间以建设相关智慧社区设施，你的态度是：\n- 无所谓\n\n现在将向你提供一份针对住区现有问题的更新方案，包括方案文本和相关更新图片。请你根据自己的真实立场，对该方案从以下五个维度进行评价并打分（1-5），每个维度需给出数值评分及原因说明，最终输出结果需要结构化呈现：建筑与居住条件、交通与基础设施、公共空间与绿化、环境与管理设施、综合满意度\n输出示例：\n{{\n  \"建筑与居住条件\": {{\n    \"评分\": 4,\n    \"原因\": \"给出评分的具体原因，需联系自身情况\"\n  }},\n  \"交通与基础设施\": {{\n    \"评分\": 3,\n    \"原因\": \"给出评分的具体原因，需联系自身情况\"\n  }},\n  \"公共空间与绿化\": {{\n    \"评分\": 3,\n    \"原因\": \"给出评分的具体原因，需联系自身情况\"\n  }},\n  \"环境与管理设施\": {{\n    \"评分\": 4,\n    \"原因\": \"给出评分的具体原因，需联系自身情况\"\n  }},\n  \"综合满意度\": {{\n    \"评分\": 4,\n    \"原因\": \"给出评分的具体原因，需联系自身情况\"\n  }}\n}}\n"""

EVAL_USER_TEXT = (
    "请你作为上述居民用户，对当前输入的住区更新方案进行评价。"
    "注意：你将看到的每张图片都是‘更新前+更新后’拼接图，左边是更新前，右边是更新后。"
    "请严格输出 JSON 对象，且必须包含五个维度："
    "建筑与居住条件、交通与基础设施、公共空间与绿化、环境与管理设施、综合满意度。"
    "每个维度下包含：评分(1-5整数) 和 原因(中文)。"
)

MAX_IMAGE_BYTES = 2 * 1024 * 1024


@dataclass
class ResidentPersona:
    resident_id: str
    system_prompt: str
    raw_row: dict[str, str]


def _safe_json_parse(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def _score_to_int(v: Any) -> int:
    try:
        n = int(v)
    except Exception:
        n = 3
    return max(1, min(5, n))


def _compress_image_to_target_bytes(path: Path, target_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    raw = path.read_bytes()
    if len(raw) <= target_bytes:
        return raw

    try:
        from PIL import Image
    except Exception:
        return raw

    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    quality = 90
    scale = 1.0
    for _ in range(10):
        from io import BytesIO

        buf = BytesIO()
        working = img
        if scale < 1.0:
            w = max(1, int(img.width * scale))
            h = max(1, int(img.height * scale))
            working = img.resize((w, h), Image.Resampling.LANCZOS)
        working.save(buf, format="JPEG", quality=max(30, quality), optimize=True)
        data = buf.getvalue()
        if len(data) <= target_bytes:
            return data
        quality -= 10
        if quality < 40:
            scale *= 0.85
            quality = 85
    return data


def _image_to_data_url(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = _compress_image_to_target_bytes(path, target_bytes=MAX_IMAGE_BYTES)
    mime_type = "image/jpeg"
    if payload == path.read_bytes():
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('utf-8')}"


def _configure_chinese_font(matplotlib_module: Any) -> bool:
    """Configure a CJK-capable font if available to avoid missing glyph warnings."""
    try:
        from matplotlib import font_manager

        preferred = [
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "Source Han Sans SC",
            "SimHei",
            "Microsoft YaHei",
            "PingFang SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in preferred:
            if name in available:
                matplotlib_module.rcParams["font.sans-serif"] = [name] + list(
                    matplotlib_module.rcParams.get("font.sans-serif", [])
                )
                matplotlib_module.rcParams["axes.unicode_minus"] = False
                return True
        matplotlib_module.rcParams["axes.unicode_minus"] = False
        return False
    except Exception:
        logger.warning("[evaluation] failed to configure chinese font; chart text may not render correctly")
        return False


def _locate_survey_csv(path: str | Path | None = None) -> Path:
    if path:
        p = Path(path)
        if p.exists():
            return p
    candidates = [Path("inputs/serveyData.csv"), Path("inputs/surveyData.csv")]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find survey CSV. Tried inputs/serveyData.csv and inputs/surveyData.csv")


def _csv_row_text(headers: list[str], row: dict[str, str]) -> str:
    values = [row.get(h, "") for h in headers]
    return f"标题行: {headers}\n数据行: {values}"


def _survey_signature(csv_path: Path) -> str:
    st = csv_path.stat()
    return hashlib.sha256(f"{csv_path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()


def generate_resident_personas(
    survey_csv_path: str | Path | None = None,
    model: str = "gpt-5.2",
    max_workers: int = 10,
    output_dir: str | Path = "output",
) -> list[ResidentPersona]:
    csv_path = _locate_survey_csv(survey_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = _survey_signature(csv_path)
    cache_path = out_dir / f"resident_prompts_{sig[:12]}.json"

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        personas = [ResidentPersona(**p) for p in payload.get("personas", [])]
        logger.info("[evaluation] reuse resident prompts cache: %s count=%s", cache_path, len(personas))
        return personas

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for persona generation")

    from openai import OpenAI

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []

    logger.info("[evaluation] generating resident prompts. rows=%s workers=%s", len(rows), max_workers)

    def worker(idx_row: tuple[int, dict[str, str]]) -> ResidentPersona:
        idx, row = idx_row
        resident_id = str(idx + 1)
        client = OpenAI(api_key=api_key)
        prompt = PERSONA_GEN_TEMPLATE.format(csv_row_text=_csv_row_text(headers, row))
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            temperature=0.2,
        )
        system_prompt = (resp.output_text or "").strip()
        if not system_prompt:
            system_prompt = prompt
        return ResidentPersona(resident_id=resident_id, system_prompt=system_prompt, raw_row=row)

    personas: list[ResidentPersona] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, item) for item in enumerate(rows)]
        for fut in as_completed(futures):
            personas.append(fut.result())

    personas.sort(key=lambda p: int(p.resident_id))
    cache_path.write_text(
        json.dumps(
            {
                "survey_csv": str(csv_path),
                "signature": sig,
                "personas": [
                    {"resident_id": p.resident_id, "system_prompt": p.system_prompt, "raw_row": p.raw_row}
                    for p in personas
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("[evaluation] resident prompts saved: %s", cache_path)
    return personas


def _scheme_text_payload(scheme: dict[str, Any]) -> str:
    parts = [
        f"方案名称: {scheme.get('name', '')}",
        f"更新模式: {scheme.get('renewal_mode', '')}",
        f"核心目标: {json.dumps(scheme.get('target_goal', []), ensure_ascii=False)}",
        f"总体概念: {scheme.get('overall_concept', '')}",
        f"方案文本: {scheme.get('scheme_text', '')}",
        f"预期效果: {json.dumps(scheme.get('expected_effect', []), ensure_ascii=False)}",
        f"关键策略: {json.dumps(scheme.get('key_strategies', []), ensure_ascii=False)}",
    ]
    node_scenes = scheme.get("node_scenes", []) or []
    parts.append("节点信息:")
    for i, scene in enumerate(node_scenes, start=1):
        parts.append(f"- 节点{i}: {scene.get('node_name','')} | {scene.get('description','')}")
    return "\n".join(parts)


def _scheme_images(scheme: dict[str, Any], max_per_scene: int = 2) -> list[Path]:
    imgs: list[Path] = []
    for scene in scheme.get("node_scenes", []) or []:
        srcs = scene.get("generated_images") or scene.get("selected_representative_images") or []
        for p in srcs[:max_per_scene]:
            imgs.append(Path(p))
    return imgs


def _evaluate_single(
    persona: ResidentPersona,
    scheme: dict[str, Any],
    model: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_content: list[dict[str, Any]] = [
        {"type": "input_text", "text": EVAL_USER_TEXT + "\n\n方案信息如下：\n" + _scheme_text_payload(scheme)}
    ]
    for img in _scheme_images(scheme, max_per_scene=2):
        data_url = _image_to_data_url(img)
        if not data_url:
            continue
        user_content.append({"type": "input_text", "text": f"图片路径: {str(img)}"})
        user_content.append({"type": "input_image", "image_url": data_url})

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": persona.system_prompt}]},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.8,
            )
            obj = _safe_json_parse(resp.output_text or "{}")
            normalized: dict[str, Any] = {}
            for d in SCORE_DIMENSIONS:
                raw = obj.get(d, {}) if isinstance(obj, dict) else {}
                normalized[d] = {
                    "评分": _score_to_int((raw or {}).get("评分", 3)),
                    "原因": str((raw or {}).get("原因", "")),
                }
            return normalized
        except Exception as exc:
            last_exc = exc
            time.sleep(min(2 ** (attempt - 1), 4))

    raise RuntimeError(f"evaluate failed: {last_exc}")



def plot_eval_radar_from_csv(
    summary_csv_path: str | Path,
    output_dir: str | Path = "output",
) -> Path:
    """Draw radar chart for scheme-level average scores from eval_result CSV."""
    csv_path = Path(summary_csv_path)
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in summary CSV: {csv_path}")

    labels = SCORE_DIMENSIONS
    metric_cols = [f"{d}_均分" for d in labels]
    for col in metric_cols:
        if col not in rows[0]:
            raise ValueError(f"Missing score column in summary CSV: {col}")

    import math
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_cjk_font = _configure_chinese_font(matplotlib)
    display_labels = labels if has_cjk_font else [
        "Building",
        "Traffic",
        "Public Space",
        "Env & Mgmt",
        "Overall",
    ]

    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    scheme_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=150, subplot_kw={"projection": "polar"})
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    for i, row in enumerate(rows[:3], start=1):
        ax = axes[i - 1]
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_labels, fontsize=14)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=12)
        values = []
        for c in metric_cols:
            try:
                values.append(float(row.get(c, 0) or 0))
            except Exception:
                values.append(0.0)
        values += values[:1]
        color = scheme_colors[(i - 1) % len(scheme_colors)]
        ax.plot(angles, values, linewidth=2.5, color=color)
        ax.fill(angles, values, color=color, alpha=0.16)
        ax.set_title(f"方案 {i}" if has_cjk_font else f"Scheme {i}", fontsize=18, pad=22)

        for j, score in enumerate(values[:-1]):
            angle = angles[j]
            ax.text(
                angle,
                min(5.0, score + 0.18),
                f"{score:.1f}",
                fontsize=11,
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

    for idx in range(min(len(rows), 3), 3):
        axes[idx].set_axis_off()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    png_path = out_dir / f"eval_radar_{ts}.png"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("[evaluation] radar chart saved: %s", png_path)
    return png_path


def evaluate_result_json(
    result_json_path: str | Path,
    survey_csv_path: str | Path | None = None,
    output_dir: str | Path = "output",
    model: str = "gpt-5.2",
    max_workers: int = 10,
) -> Path:
    result_json_path = Path(result_json_path)
    payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    generation = payload.get("generation", {})
    schemes = generation.get("scheme_list", [])
    if not schemes:
        raise ValueError(f"No scheme_list found in {result_json_path}")

    personas = generate_resident_personas(
        survey_csv_path=survey_csv_path,
        model=model,
        max_workers=max_workers,
        output_dir=output_dir,
    )

    logger.info(
        "[evaluation] start scoring. schemes=%s residents=%s workers=%s",
        len(schemes),
        len(personas),
        max_workers,
    )

    tasks: list[tuple[int, dict[str, Any], ResidentPersona]] = []
    for s_idx, scheme in enumerate(schemes, start=1):
        for persona in personas:
            tasks.append((s_idx, scheme, persona))

    details: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {
            ex.submit(_evaluate_single, persona=persona, scheme=scheme, model=model): (s_idx, scheme, persona)
            for s_idx, scheme, persona in tasks
        }
        for fut in as_completed(fut_map):
            s_idx, scheme, persona = fut_map[fut]
            res = fut.result()
            details.append(
                {
                    "scheme_index": s_idx,
                    "scheme_name": scheme.get("name", f"scheme_{s_idx}"),
                    "resident_id": persona.resident_id,
                    "scores": res,
                }
            )

    # aggregate mean per scheme
    agg_rows: list[dict[str, Any]] = []
    for s_idx, scheme in enumerate(schemes, start=1):
        rows = [d for d in details if d["scheme_index"] == s_idx]
        out: dict[str, Any] = {
            "result_file": result_json_path.name,
            "scheme_index": s_idx,
            "scheme_name": scheme.get("name", f"scheme_{s_idx}"),
            "respondent_count": len(rows),
        }
        for dim in SCORE_DIMENSIONS:
            vals = [r["scores"][dim]["评分"] for r in rows]
            out[f"{dim}_均分"] = round(statistics.mean(vals), 4) if vals else ""
        agg_rows.append(out)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_result_{ts}.csv"
    fieldnames = ["result_file", "scheme_index", "scheme_name", "respondent_count"] + [
        f"{d}_均分" for d in SCORE_DIMENSIONS
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)

    # save details for traceability
    detail_path = out_dir / f"eval_detail_{ts}.json"
    detail_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_csv_path = out_dir / f"eval_scores_detail_{ts}.csv"
    detail_fieldnames = ["scheme_index", "scheme_name", "resident_id"] + SCORE_DIMENSIONS
    with detail_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        for row in sorted(details, key=lambda x: (x.get("scheme_index", 0), x.get("resident_id", ""))):
            record = {
                "scheme_index": row.get("scheme_index"),
                "scheme_name": row.get("scheme_name"),
                "resident_id": row.get("resident_id"),
            }
            for dim in SCORE_DIMENSIONS:
                record[dim] = _score_to_int((((row.get("scores", {}) or {}).get(dim, {}) or {}).get("评分", 3)))
            writer.writerow(record)

    radar_path = plot_eval_radar_from_csv(csv_path, output_dir=out_dir)

    logger.info(
        "[evaluation] finished. summary=%s detail_json=%s detail_csv=%s radar=%s",
        csv_path,
        detail_path,
        detail_csv_path,
        radar_path,
    )
    return csv_path


def export_scores_from_detail_json(detail_json_path: str | Path, output_dir: str | Path = "output") -> tuple[Path, Path]:
    detail_path = Path(detail_json_path)
    details = json.loads(detail_path.read_text(encoding="utf-8"))
    if not isinstance(details, list):
        raise ValueError("detail json must be a list")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    detail_csv = out_dir / f"eval_scores_detail_{ts}.csv"
    fieldnames = ["scheme_index", "scheme_name", "resident_id"] + SCORE_DIMENSIONS
    with detail_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            out = {
                "scheme_index": row.get("scheme_index"),
                "scheme_name": row.get("scheme_name"),
                "resident_id": row.get("resident_id"),
            }
            scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
            for dim in SCORE_DIMENSIONS:
                out[dim] = _score_to_int((scores.get(dim, {}) or {}).get("评分", 3))
            writer.writerow(out)

    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in details:
        key = (row.get("scheme_index"), row.get("scheme_name"))
        grouped.setdefault(key, []).append(row)

    summary_csv = out_dir / f"eval_result_{ts}.csv"
    summary_fields = ["scheme_index", "scheme_name", "respondent_count"] + [f"{d}_均分" for d in SCORE_DIMENSIONS]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for (scheme_index, scheme_name), rows in sorted(grouped.items(), key=lambda x: x[0][0]):
            out: dict[str, Any] = {
                "scheme_index": scheme_index,
                "scheme_name": scheme_name,
                "respondent_count": len(rows),
            }
            for dim in SCORE_DIMENSIONS:
                vals = [_score_to_int(((r.get("scores", {}) or {}).get(dim, {}) or {}).get("评分", 3)) for r in rows]
                out[f"{dim}_均分"] = round(statistics.mean(vals), 4) if vals else ""
            writer.writerow(out)

    logger.info("[evaluation] exported detail scores csv=%s summary csv=%s", detail_csv, summary_csv)
    return detail_csv, summary_csv
