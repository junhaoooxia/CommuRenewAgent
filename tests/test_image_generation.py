import base64
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from commurenew_agent.image_generation import edit_image_with_gemini_nanobanana


class _FakeInlineData:
    def __init__(self, data: str):
        self.data = data


class _FakePart:
    def __init__(self, inline_data=None):
        self.inline_data = inline_data


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
    def __init__(self, response_modalities=None):
        self.response_modalities = response_modalities


class EditImageWithGeminiTest(unittest.TestCase):
    def test_edit_image_with_mocked_gemini(self):
        fake_genai = types.SimpleNamespace(Client=_FakeClient)
        fake_types = types.SimpleNamespace(
            Part=_FakeTypesPart,
            GenerateContentConfig=_FakeGenerateContentConfig,
        )

        fake_google_pkg = types.ModuleType("google")
        fake_google_pkg.genai = fake_genai

        fake_google_genai_mod = types.ModuleType("google.genai")
        fake_google_genai_mod.types = fake_types

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.png"
            output = Path(tmpdir) / "result.png"
            source.write_bytes(b"raw-source-image")

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
            self.assertEqual(output.read_bytes(), b"fake-generated-image")


if __name__ == "__main__":
    unittest.main()
