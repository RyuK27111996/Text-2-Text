import asyncio
import pytest
from fastapi import HTTPException
import httpx

from app.config import Settings
from app.schemas import GenerateRequest
from app.services.gemini import GeminiService


def test_extract_text_returns_joined_text():
    """Verify text parts are combined into a single response string."""
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "hello"},
                        {"text": "world"},
                    ]
                }
            }
        ]
    }

    assert GeminiService._extract_text(data) == "hello\nworld"


def test_extract_text_explains_max_tokens_without_visible_text():
    """Verify token exhaustion without visible text produces a helpful error."""
    data = {
        "candidates": [
            {
                "content": {"role": "model"},
                "finishReason": "MAX_TOKENS",
            }
        ],
        "usageMetadata": {
            "thoughtsTokenCount": 16,
        },
    }

    with pytest.raises(HTTPException) as exc_info:
        GeminiService._extract_text(data)

    assert exc_info.value.status_code == 502
    assert "Increase max_output_tokens and retry" in exc_info.value.detail


def test_build_request_body_includes_system_instruction_and_json_mode():
    """Verify Gemini request bodies include structured-output configuration."""
    service = GeminiService(
        settings=Settings(gemini_api_key="test-key", ollama_base_url=None, _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
    )
    payload = GenerateRequest(
        prompt="Return JSON",
        system_instruction="Respond with valid JSON only.",
        temperature=0,
        max_output_tokens=256,
        top_p=0.9,
        top_k=32,
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )

    body = service._build_request_body(payload)

    assert body["systemInstruction"]["role"] == "system"
    assert body["systemInstruction"]["parts"][0]["text"] == "Respond with valid JSON only."
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["maxOutputTokens"] == 256
    assert body["generationConfig"]["topP"] == 0.9
    assert body["generationConfig"]["topK"] == 32
    assert body["generationConfig"]["responseSchema"]["type"] == "object"
    assert "additionalProperties" not in body["generationConfig"]["responseSchema"]


def test_extract_error_detail_prefers_nested_google_error_message():
    """Verify Gemini error extraction prefers the nested Google API message."""
    response = httpx.Response(
        400,
        json={
            "error": {
                "code": 400,
                "message": "Invalid JSON payload received.",
                "status": "INVALID_ARGUMENT",
            }
        },
    )

    detail = GeminiService._extract_error_detail(response)

    assert detail == "Invalid JSON payload received. (status=INVALID_ARGUMENT, code=400)"


def test_response_json_schema_requires_json_mode():
    """Verify schemas are only accepted when JSON response mode is enabled."""
    with pytest.raises(ValueError):
        GenerateRequest(
            prompt="hi",
            response_mime_type="text/plain",
            response_json_schema={"type": "object"},
        )


def test_validate_structured_output_rejects_schema_mismatch():
    """Verify local schema validation rejects incompatible JSON output."""
    service = GeminiService(
        settings=Settings(gemini_api_key="test-key", ollama_base_url=None, _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
    )
    payload = GenerateRequest(
        prompt="Return JSON",
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        service._validate_structured_output(payload, '{"ok": true}')

    assert exc_info.value.status_code == 502
    assert "response_json_schema" in exc_info.value.detail


def test_validate_structured_output_accepts_matching_schema():
    """Verify local schema validation accepts matching JSON output."""
    service = GeminiService(
        settings=Settings(gemini_api_key="test-key", ollama_base_url=None, _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
    )
    payload = GenerateRequest(
        prompt="Return JSON",
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["status", "count"],
            "additionalProperties": False,
        },
    )

    service._validate_structured_output(payload, '{"status": "ok", "count": 1}')


def test_post_with_retries_retries_transient_503():
    """Verify transient upstream failures are retried before succeeding."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a transient failure first, then a successful Gemini payload."""
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    service = GeminiService(
        settings=Settings(
            gemini_api_key="test-key",
            ollama_base_url=None,
            gemini_max_retries=1,
            gemini_retry_base_delay_seconds=0,
            _env_file=None,
        ),
        client=client,
        semaphore=asyncio.Semaphore(1),
    )

    async def run_test():
        """Run the retry test inside an event loop."""
        response = await service._post_with_retries("https://example.test", {"prompt": "hi"})
        assert response.status_code == 200
        assert attempts["count"] == 2
        await client.aclose()

    asyncio.run(run_test())


def test_generate_retries_then_returns_text():
    """Verify the full generate path retries and returns validated JSON text."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a rate-limit response first, then a successful Gemini payload."""
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"status":"ok"}'}],
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    service = GeminiService(
        settings=Settings(
            gemini_api_key="test-key",
            ollama_base_url=None,
            gemini_max_retries=1,
            gemini_retry_base_delay_seconds=0,
            _env_file=None,
        ),
        client=client,
        semaphore=asyncio.Semaphore(1),
    )
    payload = GenerateRequest(
        prompt="Return JSON",
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )

    async def run_test():
        """Run the end-to-end retry test inside an event loop."""
        result = await service.generate(payload)
        assert result.text == '{"status":"ok"}'
        assert attempts["count"] == 2
        await client.aclose()

    asyncio.run(run_test())
