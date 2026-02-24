from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _iter_response_parts(response):
    parts = getattr(response, "parts", None)
    if parts:
        return parts
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        return getattr(candidates[0].content, "parts", None) or []
    return []


def edit_image_with_gemini_nanobanana(
    prompt: str,
    source_image_path: str | Path,
    output_path: str | Path,
    model: str = "gemini-3-pro-image-preview",
) -> str:
    """Image-to-image editing via Gemini image generation API."""
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

    # Per Gemini docs, provide prompt + image and request IMAGE modality.
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for part in _iter_response_parts(response):
        if getattr(part, "text", None):
            logger.info("[image_generation] Gemini text response: %s", part.text)

        if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
            # Prefer SDK helper for correctness (avoids bad base64 handling causing unreadable images).
            try:
                pil_image = part.as_image()
                pil_image.save(output_path)
                logger.info("[image_generation] saved image via part.as_image -> %s", output_path)
                return str(output_path)
            except Exception:
                data = part.inline_data.data
                if isinstance(data, (bytes, bytearray)):
                    image_bytes = bytes(data)
                else:
                    image_bytes = base64.b64decode(data)
                output_path.write_bytes(image_bytes)
                logger.info("[image_generation] saved image via inline_data bytes -> %s", output_path)
                return str(output_path)

    raise RuntimeError("Gemini did not return an edited image payload")
