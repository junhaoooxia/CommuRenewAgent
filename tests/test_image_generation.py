import base64
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from commurenew_agent.image_generation import edit_image_with_gemini_nanobanana


class _FakeInlineData:
    def __init__(self, data: str):
        self.data = data


class _FakePart:
    def __init__(self, inline_data=None):
        self.inline_data = inline_data

    def as_image(self):
        return Image.new("RGB", (10, 10), color=(255, 0, 0))


class _FakeResponse:
    def __init__(self, image_payload: bytes):
        self.parts = [_FakePart(_FakeInlineData(base64.b64encode(image_payload).decode("utf-8")))]


class _FakeModels:
    def __init__(self, image_payload: bytes):
        self._image_payload = image_payload

    def generate_content(self, **kwargs):
        return _FakeResponse(self._image_payload)


class _FakeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.models = _FakeModels(b"fake-generated-image")


class _FakeTypesPart:
    @staticmethod
    def from_text(text: str):
        return {"kind": "text", "text": text}

    @staticmethod
    def from_bytes(data: bytes, mime_type: str):
        return {"kind": "bytes", "size": len(data), "mime_type": mime_type}


class _FakeGenerateContentConfig:
    def __init__(self, response_modalities=None, image_config=None):
        self.response_modalities = response_modalities
        self.image_config = image_config


class _FakeImageConfig:
    def __init__(self, image_size=None, aspect_ratio=None):
        self.image_size = image_size
        self.aspect_ratio = aspect_ratio


class EditImageWithGeminiTest(unittest.TestCase):
    def test_edit_image_with_mocked_gemini(self):
        fake_genai = types.SimpleNamespace(Client=_FakeClient)
        fake_types = types.SimpleNamespace(
            Part=_FakeTypesPart,
            GenerateContentConfig=_FakeGenerateContentConfig,
            ImageConfig=_FakeImageConfig,
        )

        fake_google_pkg = types.ModuleType("google")
        fake_google_pkg.genai = fake_genai

        fake_google_genai_mod = types.ModuleType("google.genai")
        fake_google_genai_mod.types = fake_types

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.png"
            output = Path(tmpdir) / "result.png"
            Image.new("RGB", (10, 10), color=(0, 255, 0)).save(source)

            with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}, clear=False), patch.dict(
                "sys.modules",
                {
                    "google": fake_google_pkg,
                    "google.genai": fake_google_genai_mod,
                },
            ):
                out_path = edit_image_with_gemini_nanobanana(
                    prompt="mock edit",
                    source_image_path=source,
                    output_path=output,
                )

            self.assertEqual(out_path, str(output))
            self.assertTrue(output.exists())
            with Image.open(output) as merged:
                self.assertEqual(merged.height, 10)
                self.assertEqual(merged.width, 20)


if __name__ == "__main__":
    unittest.main()
