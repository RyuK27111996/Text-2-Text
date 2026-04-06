import os
from fastapi.testclient import TestClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from app.main import app  # noqa: E402


def test_generate_validation_rejects_empty_prompt():
    with TestClient(app) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "   ",
                "temperature": 0.2,
            },
        )
        assert response.status_code == 422
