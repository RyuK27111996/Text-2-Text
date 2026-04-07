from app.config import Settings


def test_ollama_is_disabled_by_default(monkeypatch):
    """Verify Ollama is opt-in when no base URL is configured."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url is None
