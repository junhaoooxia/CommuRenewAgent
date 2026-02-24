from __future__ import annotations

import base64
import io
import logging
import mimetypes
import os
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


PROMPT_GUARDRAIL = (
    "请尽可能保持原图构图、视角、主体位置与材质不变，仅根据需求进行最小必要修改，"
    "避免大幅重绘、避免风格突变、避免替换无关元素。"
)


def _iter_response_parts(response):
    parts = getattr(response, "parts", None)
    if parts:
        return parts
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        return getattr(candidates[0].content, "parts", None) or []
    return []


def _part_to_image(part) -> Image.Image | None:
    if not (getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None)):
        return None

    try:
        return part.as_image()
    except Exception:
        data = part.inline_data.data
        if isinstance(data, (bytes, bytearray)):
            image_bytes = bytes(data)
        else:
            image_bytes = base64.b64decode(data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _concat_side_by_side(original_image: Image.Image, edited_image: Image.Image) -> Image.Image:
    # Keep same height for side-by-side concat: old on left, new on right.
    target_h = max(original_image.height, edited_image.height)

    def resize_to_h(img: Image.Image, h: int) -> Image.Image:
        if img.height == h:
            return img
        w = max(1, int(img.width * (h / img.height)))
        return img.resize((w, h), Image.Resampling.LANCZOS)

    left = resize_to_h(original_image.convert("RGB"), target_h)
    right = resize_to_h(edited_image.convert("RGB"), target_h)

    canvas = Image.new("RGB", (left.width + right.width, target_h), color=(255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def edit_image_with_gemini_nanobanana(
    prompt: str,
    source_image_path: str | Path,
    output_path: str | Path,
    model: str = "gemini-3-pro-image-preview",
) -> str:
    """Image-to-image editing via Gemini image generation API.

    Saves a side-by-side comparison image: original(left) + edited(right).
    """
    logger.info("[image_generation] start edit model=%s source=%s", model, source_image_path)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini image editing")

    source_image_path = Path(source_image_path)
    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_image_path}")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install `google-genai` to use Gemini image editing") from exc

    client = genai.Client(api_key=api_key)
    img_bytes = source_image_path.read_bytes()
    mime_type = mimetypes.guess_type(source_image_path.name)[0] or "image/png"

    merged_prompt = f"{PROMPT_GUARDRAIL}\n\n需求：{prompt}"

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=merged_prompt),
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(image_size="2K"),
        ),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = Image.open(source_image_path).convert("RGB")
    for part in _iter_response_parts(response):
        if getattr(part, "text", None):
            logger.info("[image_generation] Gemini text response: %s", part.text)

        edited = _part_to_image(part)
        if edited is not None:
            merged = _concat_side_by_side(original, edited)
            merged.save(output_path)
            logger.info("[image_generation] saved side-by-side image -> %s", output_path)
            return str(output_path)

    raise RuntimeError("Gemini did not return an edited image payload")
