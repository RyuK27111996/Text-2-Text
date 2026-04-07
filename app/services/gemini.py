from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import GenerateRequest


@dataclass(slots=True)
class GeminiResult:
    """Normalized result returned from a Gemini generation request."""

    model: str
    text: str
    provider_latency_ms: float


class GeminiService:
    """Wrapper around the Gemini `generateContent` API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> None:
        """Store shared dependencies used to call Gemini."""
        self.settings = settings
        self.client = client
        self.semaphore = semaphore

    async def generate(self, payload: GenerateRequest) -> GeminiResult:
        """Generate text through Gemini and validate structured output when requested."""
        if not self.settings.gemini_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GEMINI_API_KEY is not configured on the server.",
            )

        model = self._resolve_model(payload)
        body = self._build_request_body(payload)
        endpoint = self._build_endpoint(model)

        started = time.perf_counter()
        async with self.semaphore:
            response = await self._post_with_retries(endpoint=endpoint, body=body)

        provider_latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini rate limit reached. Retry later.",
            )

        if response.status_code >= 400:
            detail = self._extract_error_detail(response)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini upstream returned {response.status_code}: {detail}",
            )

        data = self._decode_json(response)
        text = self._extract_text(data)
        self._validate_structured_output(payload, text)
        return GeminiResult(model=model, text=text, provider_latency_ms=provider_latency_ms)

    def _resolve_model(self, payload: GenerateRequest) -> str:
        """Choose the request model or fall back to the configured default."""
        return payload.model or self.settings.gemini_default_model

    def _resolve_max_output_tokens(self, payload: GenerateRequest) -> int:
        """Clamp the requested output token budget to the server maximum."""
        requested = payload.max_output_tokens or self.settings.default_max_output_tokens
        return min(requested, self.settings.max_output_tokens_limit)

    def _build_headers(self) -> dict[str, str]:
        """Build headers required by Gemini REST requests."""
        return {
            "x-goog-api-key": self.settings.gemini_api_key or "",
            "Content-Type": "application/json",
        }

    def _build_endpoint(self, model: str) -> str:
        """Construct the Gemini `generateContent` endpoint for a model."""
        return (
            f"{self.settings.gemini_base_url.rstrip('/')}/"
            f"{self.settings.gemini_api_version}/models/{model}:generateContent"
        )

    def _build_request_body(self, payload: GenerateRequest) -> dict[str, Any]:
        """Translate the API request model into a Gemini request payload."""
        generation_config: dict[str, Any] = {
            "temperature": payload.temperature,
            "maxOutputTokens": self._resolve_max_output_tokens(payload),
            "responseMimeType": payload.response_mime_type,
        }
        if payload.response_json_schema is not None:
            generation_config["responseSchema"] = self._to_gemini_response_schema(payload.response_json_schema)

        body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": payload.prompt}],
                }
            ],
            "generationConfig": generation_config,
        }

        if payload.top_p is not None:
            body["generationConfig"]["topP"] = payload.top_p
        if payload.top_k is not None:
            body["generationConfig"]["topK"] = payload.top_k
        if payload.system_instruction:
            body["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": payload.system_instruction}],
            }

        return body

    @classmethod
    def _to_gemini_response_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Strip local-only JSON Schema fields that Gemini does not accept upstream."""
        allowed_keys = {"type", "format", "description", "nullable", "enum", "items", "properties", "required"}
        translated: dict[str, Any] = {}

        for key, value in schema.items():
            if key not in allowed_keys:
                continue
            if key == "properties" and isinstance(value, dict):
                translated[key] = {
                    prop_name: cls._to_gemini_response_schema(prop_schema)
                    for prop_name, prop_schema in value.items()
                    if isinstance(prop_schema, dict)
                }
            elif key == "items" and isinstance(value, dict):
                translated[key] = cls._to_gemini_response_schema(value)
            else:
                translated[key] = value

        return translated

    async def _post_with_retries(self, endpoint: str, body: dict[str, Any]) -> httpx.Response:
        """Send a Gemini request with retry/backoff for transient failures."""
        attempts = self.settings.gemini_max_retries + 1
        last_error: Exception | None = None
        last_response: httpx.Response | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = await self.client.post(
                    endpoint,
                    headers=self._build_headers(),
                    json=body,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt == attempts:
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail="Gemini upstream timed out after retries.",
                    ) from exc
                await asyncio.sleep(self._retry_delay_seconds(attempt))
                continue
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == attempts:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Gemini upstream connection failed after retries: {exc}",
                    ) from exc
                await asyncio.sleep(self._retry_delay_seconds(attempt))
                continue

            if not self._should_retry_response(response) or attempt == attempts:
                return response

            last_response = response
            await asyncio.sleep(self._retry_delay_seconds(attempt, response))

        if last_response is not None:
            return last_response
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini upstream failed before a response was returned: {last_error}",
        )

    def _retry_delay_seconds(self, attempt: int, response: httpx.Response | None = None) -> float:
        """Return the delay to wait before the next retry attempt."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    pass
        return self.settings.gemini_retry_base_delay_seconds * (2 ** (attempt - 1))

    @staticmethod
    def _should_retry_response(response: httpx.Response) -> bool:
        """Return whether an HTTP response should trigger a retry."""
        return response.status_code in {
            status.HTTP_429_TOO_MANY_REQUESTS,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_502_BAD_GATEWAY,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_504_GATEWAY_TIMEOUT,
        }

    def _validate_structured_output(self, payload: GenerateRequest, text: str) -> None:
        """Validate JSON output and optional schema constraints for structured responses."""
        if payload.response_mime_type != "application/json":
            return

        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini response_mime_type was application/json but output was not valid JSON.",
            ) from exc

        if payload.response_json_schema is not None:
            try:
                self._validate_json_schema_instance(parsed, payload.response_json_schema)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini JSON output did not match response_json_schema: {exc}",
                ) from exc

    @classmethod
    def _validate_json_schema_instance(cls, value: Any, schema: dict[str, Any], path: str = "$") -> None:
        """Validate a JSON value against the supported subset of JSON Schema."""
        schema_type = schema.get("type")

        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path} must be one of {schema['enum']}")

        if schema_type is None:
            return

        if schema_type == "object":
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")

            required = schema.get("required", [])
            for key in required:
                if key not in value:
                    raise ValueError(f"{path}.{key} is required")

            properties = schema.get("properties", {})
            for key, item in value.items():
                if key in properties:
                    cls._validate_json_schema_instance(item, properties[key], f"{path}.{key}")
                elif schema.get("additionalProperties") is False:
                    raise ValueError(f"{path}.{key} is not allowed")
            return

        if schema_type == "array":
            if not isinstance(value, list):
                raise ValueError(f"{path} must be an array")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    cls._validate_json_schema_instance(item, item_schema, f"{path}[{index}]")
            return

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "null": type(None),
        }
        expected_type = type_map.get(schema_type)
        if expected_type is None:
            return

        if schema_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{path} must be an integer")
            return

        if schema_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path} must be a number")
            return

        if not isinstance(value, expected_type):
            raise ValueError(f"{path} must be of type {schema_type}")

    @staticmethod
    def _decode_json(response: httpx.Response) -> dict[str, Any]:
        """Decode and validate a Gemini JSON response body."""
        try:
            data = response.json()
        except ValueError as exc:
            snippet = response.text[:1000]
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini returned invalid JSON: {snippet}",
            ) from exc

        if not isinstance(data, dict):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned a non-object JSON payload.",
            )
        return data

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Extract visible model text from a successful Gemini response payload."""
        candidates = data.get("candidates") or []
        if not candidates:
            raise HTTPException(status_code=502, detail="Gemini returned no candidates.")

        first_candidate = candidates[0]
        content = first_candidate.get("content") or {}
        parts = content.get("parts") or []
        texts: list[str] = []
        for part in parts:
            text = part.get("text")
            if text:
                texts.append(text)

        if not texts:
            finish_reason = first_candidate.get("finishReason")
            usage = data.get("usageMetadata") or {}
            thoughts_tokens = usage.get("thoughtsTokenCount")
            if finish_reason == "MAX_TOKENS":
                detail = "Gemini returned no visible text before hitting max_output_tokens"
                if thoughts_tokens is not None:
                    detail += f" (thoughtsTokenCount={thoughts_tokens})"
                detail += ". Increase max_output_tokens and retry."
                raise HTTPException(status_code=502, detail=detail)
            raise HTTPException(status_code=502, detail="Gemini returned no text output.")
        return "\n".join(texts).strip()

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        """Extract a readable upstream error message from a Gemini failure response."""
        try:
            data = response.json()
        except ValueError:
            return response.text[:1000]

        if not isinstance(data, dict):
            return json.dumps(data)[:1000]

        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            status_name = error.get("status")
            code = error.get("code")
            if message and status_name and code:
                return f"{message} (status={status_name}, code={code})"
            if message:
                return str(message)

        return json.dumps(data)[:1000]
