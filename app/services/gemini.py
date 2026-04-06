from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import GenerateRequest


@dataclass(slots=True)
class GeminiResult:
    model: str
    text: str
    provider_latency_ms: float


class GeminiService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> None:
        self.settings = settings
        self.client = client
        self.semaphore = semaphore

    async def generate(self, payload: GenerateRequest) -> GeminiResult:
        if not self.settings.gemini_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GEMINI_API_KEY is not configured on the server.",
            )

        model = payload.model or self.settings.gemini_default_model
        max_output_tokens = payload.max_output_tokens or self.settings.default_max_output_tokens
        max_output_tokens = min(max_output_tokens, self.settings.max_output_tokens_limit)

        body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": payload.prompt}],
                }
            ],
            "generationConfig": {
                "temperature": payload.temperature,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": payload.response_mime_type,
            },
        }

        if payload.top_p is not None:
            body["generationConfig"]["topP"] = payload.top_p
        if payload.top_k is not None:
            body["generationConfig"]["topK"] = payload.top_k
        if payload.system_instruction:
            body["systemInstruction"] = {"parts": [{"text": payload.system_instruction}]}

        endpoint = (
            f"{self.settings.gemini_base_url.rstrip('/')}/"
            f"{self.settings.gemini_api_version}/models/{model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.settings.gemini_api_key,
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        async with self.semaphore:
            try:
                response = await self.client.post(endpoint, headers=headers, json=body)
            except httpx.TimeoutException as exc:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Gemini upstream timed out.",
                ) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini upstream connection failed: {exc}",
                ) from exc

        provider_latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini rate limit reached. Retry later.",
            )

        if response.status_code >= 400:
            detail = self._safe_error_detail(response)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini upstream returned {response.status_code}: {detail}",
            )

        data = response.json()
        text = self._extract_text(data)
        return GeminiResult(model=model, text=text, provider_latency_ms=provider_latency_ms)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            raise HTTPException(status_code=502, detail="Gemini returned no candidates.")

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts: list[str] = []
        for part in parts:
            text = part.get("text")
            if text:
                texts.append(text)

        if not texts:
            raise HTTPException(status_code=502, detail="Gemini returned no text output.")
        return "\n".join(texts).strip()

    @staticmethod
    def _safe_error_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
            return str(data)
        except Exception:
            return response.text[:1000]
