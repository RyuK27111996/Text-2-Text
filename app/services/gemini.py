from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import BaseGemmaRequest, ImageToTextRequest, TextToTextRequest
from app.services.model_router import ModelRouter


@dataclass(slots=True)
class GeminiResult:
    """Normalized result returned from a Gemini generation request."""

    model: str
    text: str
    provider_latency_ms: float


class GeminiService:
    """Wrapper around the Gemini `generateContent` API."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        model_router: ModelRouter,
    ) -> None:
        """Store shared dependencies used to call Gemini."""
        self.settings = settings
        self.client = client
        self.semaphore = semaphore
        self.model_router = model_router

    async def generate_text(self, payload: TextToTextRequest) -> GeminiResult:
        """Generate text from a text prompt using Gemma fallback pools."""
        return await self._generate(payload, self._build_text_request_body)

    async def generate_image_to_text(self, payload: ImageToTextRequest) -> GeminiResult:
        """Generate text from an image and prompt using Gemma vision fallback pools."""
        return await self._generate(payload, self._build_image_request_body)

    async def generate(self, payload: TextToTextRequest) -> GeminiResult:
        """Backward-compatible wrapper for text-to-text generation."""
        return await self.generate_text(payload)

    async def _generate(
        self,
        payload: BaseGemmaRequest,
        body_builder,
    ) -> GeminiResult:
        """Generate text through Gemma and validate structured output when requested."""
        if not self.settings.gemini_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GEMINI_API_KEY is not configured on the server.",
            )

        candidates = await self._get_routing_candidates(payload)
        last_http_error: HTTPException | None = None

        for model in candidates:
            started = time.perf_counter()
            await self.model_router.mark_in_flight(model, 1)
            try:
                body = body_builder(payload)
                endpoint = self._build_endpoint(model)
                async with self.semaphore:
                    response = await self._post_with_retries(endpoint=endpoint, body=body)
            finally:
                await self.model_router.mark_in_flight(model, -1)

            provider_latency_ms = (time.perf_counter() - started) * 1000.0

            if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                detail = self._extract_error_detail(response)
                await self.model_router.mark_rate_limited(model, detail)
                last_http_error = HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Gemini rate limit reached for {model}: {detail}",
                )
                if payload.allow_model_fallback:
                    continue
                raise last_http_error

            if response.status_code >= 400:
                detail = self._extract_error_detail(response)
                await self.model_router.mark_failure(model, detail)
                last_http_error = HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini upstream returned {response.status_code} for {model}: {detail}",
                )
                if payload.allow_model_fallback and response.status_code in {
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    status.HTTP_502_BAD_GATEWAY,
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    status.HTTP_504_GATEWAY_TIMEOUT,
                }:
                    continue
                raise last_http_error

            data = self._decode_json(response)
            text = self._extract_text(data)
            self._validate_structured_output(payload, text)
            await self.model_router.mark_success(model)
            return GeminiResult(model=model, text=text, provider_latency_ms=provider_latency_ms)

        if last_http_error is not None:
            raise last_http_error

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemma routing did not find any compatible models for the request.",
        )

    def _resolve_model(self, payload: BaseGemmaRequest) -> str:
        """Choose the request model or fall back to the configured default."""
        if payload.model:
            return self.model_router.normalize_model_name(payload.model)
        return "gemma-3-27b-it"

    async def _get_routing_candidates(self, payload: BaseGemmaRequest) -> list[str]:
        """Return Gemma models in the order they should be attempted."""
        preferred_model = self._resolve_model(payload) if payload.model else None
        if not payload.allow_model_fallback:
            return [preferred_model or self._resolve_model(payload)]

        profile = payload.routing_profile
        if isinstance(payload, ImageToTextRequest) and profile == "auto":
            profile = "vision_text"
        elif profile == "auto":
            profile = self.model_router.resolve_profile(payload.model, payload.response_mime_type == "application/json")
        return await self.model_router.get_candidates(profile, preferred_model)

    def _resolve_max_output_tokens(self, payload: BaseGemmaRequest) -> int:
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

    def _build_generation_config(self, payload: BaseGemmaRequest) -> dict[str, Any]:
        """Translate shared generation options into Gemini generation config."""
        generation_config: dict[str, Any] = {
            "temperature": payload.temperature,
            "maxOutputTokens": self._resolve_max_output_tokens(payload),
            "responseMimeType": payload.response_mime_type,
        }
        if payload.response_json_schema is not None:
            generation_config["responseSchema"] = self._to_gemini_response_schema(payload.response_json_schema)
        if payload.top_p is not None:
            generation_config["topP"] = payload.top_p
        if payload.top_k is not None:
            generation_config["topK"] = payload.top_k
        return generation_config

    def _build_text_request_body(self, payload: TextToTextRequest) -> dict[str, Any]:
        """Translate a text-to-text request into a Gemini request payload."""
        body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": payload.prompt}],
                }
            ],
            "generationConfig": self._build_generation_config(payload),
        }

        if payload.system_instruction:
            body["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": payload.system_instruction}],
            }

        return body

    def _build_image_request_body(self, payload: ImageToTextRequest) -> dict[str, Any]:
        """Translate an image-to-text request into a Gemini request payload."""
        body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": payload.image_mime_type,
                                "data": payload.image_base64,
                            }
                        },
                        {"text": payload.prompt},
                    ],
                }
            ],
            "generationConfig": self._build_generation_config(payload),
        }
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
