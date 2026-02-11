from __future__ import annotations

import os
from pathlib import Path


def generate_concept_image(prompt: str, output_path: str | Path, model: str = "gpt-image-1") -> str:
    """Optional helper to generate image assets from node-level scene prompts."""
    # Keep image generation optional to avoid hard dependency during non-visual debugging.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for image generation")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    # Generate a single concept render from synthesized scene prompt.
    result = client.images.generate(model=model, prompt=prompt, size="1536x1024")
    image_b64 = result.data[0].b64_json

    import base64

    output_path = Path(output_path)
    # Persist artifact for frontend review / downstream pipelines.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(image_b64))
    return str(output_path)
