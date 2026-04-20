from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException, status
from pydantic import ValidationError

from app.config import Settings
from app.schemas import SalesRecord
from app.services.gemini import GeminiService


EXTRACTION_PROMPT = """\
You are an expert accountant assistant specializing in digitizing handwritten business sales records.

Examine the provided image carefully. Extract ALL visible sales data including item descriptions, quantities, unit prices, and totals.

Rules:
1. Extract every line item you can identify, even partial ones.
2. If a value is crossed out, skip it and add a note in the item's `notes` field.
3. If a value is illegible, use null for that field and add a warning to the `warnings` array.
4. Currency symbols found anywhere in the document should be captured in each item's `currency` field.
5. Dates may appear in any format (DD/MM/YYYY, MM-DD-YY, "Jan 5", etc.) — preserve exactly as written in `date`.
6. If text appears in multiple languages, transliterate names to Latin script and include the original in the item's `notes`.
7. Set `confidence` to:
   - "high": image is sharp and all values are clearly legible
   - "medium": some values required interpretation or the image is slightly blurry
   - "low": significant portions are illegible, the image is very blurry, or major ambiguity exists
8. In `warnings`, list every specific issue found (e.g. "line 3 quantity illegible", "image partially cut off").
9. Capture the verbatim text you read in `raw_text` before any interpretation.
10. Do NOT invent or hallucinate values. Use null for any field you cannot determine with confidence.

Return ONLY a valid JSON object — no markdown fences, no explanation text.
"""

# Gemini responseSchema sent as generationConfig.responseSchema.
# Only allowed keys: type, format, description, nullable, enum, items, properties, required.
_SALES_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "date": {"type": "string", "nullable": True},
        "seller": {"type": "string", "nullable": True},
        "raw_text": {"type": "string", "nullable": True},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "subtotal": {"type": "number", "nullable": True},
        "tax": {"type": "number", "nullable": True},
        "grand_total": {"type": "number", "nullable": True},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "line_total": {"type": "number"},
                    "currency": {"type": "string", "nullable": True},
                    "notes": {"type": "string", "nullable": True},
                },
                "required": ["description", "quantity", "unit_price", "line_total"],
            },
        },
    },
    "required": ["confidence", "warnings", "items"],
}


@dataclass(slots=True)
class DigitizerResult:
    """Normalized result returned from a vision extraction request."""

    model: str
    sales_record: SalesRecord
    provider_latency_ms: float


def _detect_mime_type(image_bytes: bytes, fallback: str) -> str:
    """Detect MIME type from magic bytes; fall back to the client-supplied value."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if image_bytes[:3] == b"GIF":
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return fallback or "image/jpeg"


class DigitizerService:
    """Calls Gemini Vision API to extract structured sales data from an image."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self.settings = settings
        self.client = client
        self.semaphore = semaphore

    async def digitize(self, image_bytes: bytes, client_mime_type: str) -> DigitizerResult:
        """Encode the image, call Gemini Vision, parse and return a SalesRecord."""
        # Validate file first (before hitting API key check) so errors are specific.
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded image file is empty.",
            )

        if len(image_bytes) > self.settings.max_image_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds maximum allowed size ({self.settings.max_image_bytes} bytes).",
            )

        if not self.settings.gemini_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GEMINI_API_KEY is not configured on the server.",
            )

        mime_type = _detect_mime_type(image_bytes, client_mime_type)
        b64_data = base64.b64encode(image_bytes).decode("ascii")
        model = self.settings.gemini_vision_model
        body = self._build_request_body(b64_data, mime_type)
        endpoint = self._build_endpoint(model)

        started = time.perf_counter()
        async with self.semaphore:
            response = await self._post_with_retries(endpoint, body)
        provider_latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini rate limit reached. Retry later.",
            )
        if response.status_code >= 400:
            detail = GeminiService._extract_error_detail(response)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini upstream returned {response.status_code}: {detail}",
            )

        data = GeminiService._decode_json(response)
        text = GeminiService._extract_text(data)
        sales_record = self._parse_and_enrich(text)
        return DigitizerResult(
            model=model,
            sales_record=sales_record,
            provider_latency_ms=provider_latency_ms,
        )

    def _build_endpoint(self, model: str) -> str:
        return (
            f"{self.settings.gemini_base_url.rstrip('/')}/"
            f"{self.settings.gemini_api_version}/models/{model}:generateContent"
        )

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.settings.gemini_api_key or "",
            "Content-Type": "application/json",
        }

    def _build_request_body(self, b64_data: str, mime_type: str) -> dict[str, Any]:
        return {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                        {"text": EXTRACTION_PROMPT},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                "responseSchema": GeminiService._to_gemini_response_schema(_SALES_JSON_SCHEMA),
            },
        }

    def _parse_and_enrich(self, text: str) -> SalesRecord:
        """Parse Gemini JSON, compute server-side total, set total_matches."""
        try:
            raw: dict[str, Any] = json.loads(text)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned non-JSON for digitization response.",
            ) from exc

        computed = round(
            sum(
                item.get("line_total") or 0.0
                for item in raw.get("items", [])
                if isinstance(item.get("line_total"), (int, float))
            ),
            4,
        )
        raw["computed_total"] = computed

        grand_total = raw.get("grand_total")
        total_matches = grand_total is not None and abs(computed - grand_total) < 0.01
        raw["total_matches"] = total_matches

        if not total_matches and grand_total is not None:
            raw.setdefault("warnings", []).append(
                f"Computed total {computed:.2f} does not match written total {grand_total:.2f}"
            )

        try:
            return SalesRecord(**raw)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Extracted data did not match expected schema: {exc}",
            ) from exc

    async def _post_with_retries(self, endpoint: str, body: dict[str, Any]) -> httpx.Response:
        """POST to Gemini with exponential backoff on transient failures."""
        attempts = self.settings.gemini_max_retries + 1
        last_response: httpx.Response | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = await self.client.post(
                    endpoint,
                    headers=self._build_headers(),
                    json=body,
                    timeout=self.settings.vision_timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                if attempt == attempts:
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail="Gemini Vision upstream timed out after retries.",
                    ) from exc
                await asyncio.sleep(self._retry_delay(attempt))
                continue
            except httpx.HTTPError as exc:
                if attempt == attempts:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Gemini Vision connection failed after retries: {exc}",
                    ) from exc
                await asyncio.sleep(self._retry_delay(attempt))
                continue

            if not GeminiService._should_retry_response(response) or attempt == attempts:
                return response

            last_response = response
            await asyncio.sleep(self._retry_delay(attempt, response))

        return last_response  # type: ignore[return-value]

    def _retry_delay(self, attempt: int, response: httpx.Response | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    pass
        return self.settings.gemini_retry_base_delay_seconds * (2 ** (attempt - 1))
