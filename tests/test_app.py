import io
import os
import sys
import tempfile
import types
import unittest
from importlib import reload
from unittest.mock import patch

from PIL import Image


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        os.environ["DATABASE_URL"] = self.db_path

        import app as app_module
        import database.db as db_module

        reload(db_module)
        reload(app_module)
        self.client = app_module.app.test_client()
        self.client.testing = True

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_index_page_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tebak Wajah", response.data)

    def test_history_api_returns_json(self):
        response = self.client.get("/api/history")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("history", data)

    def test_predict_api_accepts_upload(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (120, 140), color=(255, 0, 0)).save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = self.client.post(
            "/api/predict",
            data={"file": (image_bytes, "face.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)

    def test_map_personality_uses_google_ai_when_available(self):
        import app as app_module

        class FakeModels:
            def generate_content(self, *args, **kwargs):
                return types.SimpleNamespace(
                    text='{"summary": "Google AI profile", "traits": ["Cerdas"], "strengths": ["Adaptif"], "challenges": ["Perlu fokus"]}'
                )

        class FakeClient:
            def __init__(self, api_key=None):
                self.models = FakeModels()

        fake_genai_module = types.SimpleNamespace(Client=lambda api_key=None: FakeClient(api_key=api_key))
        fake_google_module = types.SimpleNamespace(genai=fake_genai_module)

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}, clear=False), patch.dict(
            sys.modules,
            {
                "google": fake_google_module,
                "google.genai": fake_genai_module,
            },
        ):
            profile = app_module.map_personality("Bulat / Persegi", {})

        self.assertEqual(profile["summary"], "Google AI profile")
        self.assertIn("traits", profile)

    def test_personality_api_accepts_upload(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (120, 140), color=(255, 0, 0)).save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = self.client.post(
            "/api/personality",
            data={"file": (image_bytes, "face.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("personality", data)
        self.assertIn("traits", data["personality"])


if __name__ == "__main__":
    unittest.main()
