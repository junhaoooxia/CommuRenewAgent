from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path


def edit_image_with_gemini_nanobanana(
    prompt: str,
    source_image_path: str | Path,
    output_path: str | Path,
    model: str = "gemini-3-pro-image-preview",
) -> str:
    """Image-to-image editing via Gemini ("nanobanana" style workflow).

    The source image is taken from perception.site_images selected by reasoning.
    """
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

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    image_bytes = None
    parts = getattr(response, "parts", None)
    if not parts and getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts

    for part in parts or []:
        if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
            image_bytes = base64.b64decode(part.inline_data.data)
            break

    if not image_bytes:
        raise RuntimeError("Gemini did not return an edited image payload")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return str(output_path)
