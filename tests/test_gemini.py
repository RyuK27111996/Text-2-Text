import asyncio
import base64
import pytest
from fastapi import HTTPException
import httpx

from app.config import Settings
from app.schemas import ImageToTextRequest, TextToTextRequest
from app.services.gemini import GeminiService
from app.services.model_router import ModelRouter, build_default_model_router


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


def test_build_text_request_body_includes_system_instruction_and_json_mode():
    """Verify text request bodies include structured-output configuration."""
    service = GeminiService(
        settings=Settings(gemini_api_key="test-key", _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
        model_router=build_default_model_router(),
    )
    payload = TextToTextRequest(
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

    body = service._build_text_request_body(payload)

    assert body["systemInstruction"]["role"] == "system"
    assert body["systemInstruction"]["parts"][0]["text"] == "Respond with valid JSON only."
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["maxOutputTokens"] == 256
    assert body["generationConfig"]["topP"] == 0.9
    assert body["generationConfig"]["topK"] == 32
    assert body["generationConfig"]["responseSchema"]["type"] == "object"
    assert "additionalProperties" not in body["generationConfig"]["responseSchema"]


def test_build_image_request_body_includes_inline_image_data():
    """Verify image-to-text requests send the image as inline data."""
    service = GeminiService(
        settings=Settings(gemini_api_key="test-key", _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
        model_router=build_default_model_router(),
    )
    payload = ImageToTextRequest(
        prompt="Describe this image.",
        image_base64=base64.b64encode(b"png-bytes").decode(),
        image_mime_type="image/png",
    )

    body = service._build_image_request_body(payload)

    assert body["contents"][0]["parts"][0]["inlineData"]["mimeType"] == "image/png"
    assert body["contents"][0]["parts"][0]["inlineData"]["data"] == payload.image_base64
    assert body["contents"][0]["parts"][1]["text"] == "Describe this image."


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
        TextToTextRequest(
            prompt="hi",
            response_mime_type="text/plain",
            response_json_schema={"type": "object"},
        )


def test_validate_structured_output_rejects_schema_mismatch():
    """Verify local schema validation rejects incompatible JSON output."""
    service = GeminiService(
        settings=Settings(gemini_api_key="test-key", _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
        model_router=build_default_model_router(),
    )
    payload = TextToTextRequest(
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
        settings=Settings(gemini_api_key="test-key", _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
        model_router=build_default_model_router(),
    )
    payload = TextToTextRequest(
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
            gemini_max_retries=1,
            gemini_retry_base_delay_seconds=0,
            _env_file=None,
        ),
        client=client,
        semaphore=asyncio.Semaphore(1),
        model_router=build_default_model_router(),
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
            gemini_max_retries=1,
            gemini_retry_base_delay_seconds=0,
            _env_file=None,
        ),
        client=client,
        semaphore=asyncio.Semaphore(1),
        model_router=build_default_model_router(),
    )
    payload = TextToTextRequest(
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


def test_get_routing_candidates_prefers_requested_model():
    """Verify explicit models are tried first within the selected fallback pool."""
    router = build_default_model_router()
    service = GeminiService(
        settings=Settings(gemini_api_key="test-key", _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
        model_router=router,
    )
    payload = TextToTextRequest(
        prompt="hello",
        model="gemma-3-27b",
        routing_profile="text_fast",
    )

    async def run_test():
        candidates = await service._get_routing_candidates(payload)
        assert candidates[0] == "gemma-3-27b-it"

    asyncio.run(run_test())


def test_image_requests_use_vision_pool_when_auto_routing():
    """Verify image-to-text requests automatically use the vision Gemma pool."""
    router = build_default_model_router()
    service = GeminiService(
        settings=Settings(gemini_api_key="test-key", _env_file=None),
        client=httpx.AsyncClient(),
        semaphore=None,
        model_router=router,
    )
    payload = ImageToTextRequest(
        prompt="What is in this image?",
        image_base64=base64.b64encode(b"png-bytes").decode(),
    )

    async def run_test():
        candidates = await service._get_routing_candidates(payload)
        assert candidates == ["gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it"]

    asyncio.run(run_test())


def test_generate_falls_back_to_next_model_on_rate_limit():
    """Verify a rate-limited primary model automatically falls back to the next model."""
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/models/")[1].split(":generateContent")[0]
        attempts.append(model)
        if model == "gemma-3-27b-it":
            return httpx.Response(429, json={"error": {"message": "rate limited", "status": "RESOURCE_EXHAUSTED", "code": 429}})
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "fallback ok"}]}}]},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    router = ModelRouter(
        model_pools={
            "auto": ["gemma-3-27b-it", "gemma-3-12b-it"],
            "text_fast": ["gemma-3-27b-it", "gemma-3-12b-it"],
            "text_quality": ["gemma-3-12b-it"],
            "cheap_text": ["gemma-3-4b-it"],
            "vision_text": ["gemma-3-27b-it"],
        },
        model_aliases={},
    )
    service = GeminiService(
        settings=Settings(
            gemini_api_key="test-key",
            gemini_max_retries=0,
            gemini_retry_base_delay_seconds=0,
            _env_file=None,
        ),
        client=client,
        semaphore=asyncio.Semaphore(1),
        model_router=router,
    )
    payload = TextToTextRequest(prompt="hello", routing_profile="text_fast", allow_model_fallback=True)

    async def run_test():
        result = await service.generate_text(payload)
        assert result.model == "gemma-3-12b-it"
        assert result.text == "fallback ok"
        assert attempts == ["gemma-3-27b-it", "gemma-3-12b-it"]
        await client.aclose()

    asyncio.run(run_test())
