import os
from fastapi.testclient import TestClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from app.main import app  # noqa: E402


def test_health_endpoint():
    """Verify the health endpoint reports the active Gemma backend."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["active_backend"] == "gemma"
        assert body["gemini_configured"] is True
