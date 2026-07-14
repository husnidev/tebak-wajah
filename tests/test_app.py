import io
import os
import tempfile
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

    def test_map_personality_falls_back_on_openai_quota_error(self):
        import app as app_module
        from openai import RateLimitError

        class FakeCompletions:
            def create(self, *args, **kwargs):
                raise RateLimitError("quota exceeded", response=None, body=None)

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=False), patch("openai.OpenAI", return_value=FakeClient()):
            profile = app_module.map_personality("Bulat / Persegi", {})

        self.assertIn("summary", profile)
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
