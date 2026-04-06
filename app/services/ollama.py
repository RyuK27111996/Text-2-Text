from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
import logging
from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import GenerateRequest


@dataclass(slots=True)
class OllamaResult:
    model: str
    text: str
    provider_latency_ms: float


class OllamaService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> None:
        self.settings = settings
        self.client = client
        self.semaphore = semaphore

    async def generate(self, payload: GenerateRequest) -> OllamaResult:
        model = payload.model or self.settings.ollama_default_model
        max_output_tokens = payload.max_output_tokens or self.settings.default_max_output_tokens
        max_output_tokens = min(max_output_tokens, self.settings.max_output_tokens_limit)

        # Ollama expects a POST to /api/generate or /api/chat depending on model; use /api/generate
        endpoint = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"

        body: dict[str, Any] = {
            "model": model,
            "prompt": payload.prompt,
            "max_tokens": max_output_tokens,
            "temperature": payload.temperature,
        }

        if payload.system_instruction:
            # Ollama supports system by prefixing or using chat format; simple approach: prepend system to prompt
            body["prompt"] = f"System: {payload.system_instruction}\n\n{payload.prompt}"

        started = time.perf_counter()
        async with self.semaphore:
            try:
                response = await self.client.post(endpoint, json=body)
            except httpx.TimeoutException as exc:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Ollama upstream timed out.",
                ) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Ollama upstream connection failed: {exc}",
                ) from exc

        provider_latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ollama upstream returned {response.status_code}: {detail}",
            )

        # Ollama's /api/generate returns JSON with 'response' or 'output'; extract text robustly
        raw = response.text
        logger = logging.getLogger("app")

        # Try to parse as normal JSON first
        data = None
        try:
            import json

            data = json.loads(raw)
        except Exception:
            # Not a single JSON object — try NDJSON / streaming JSON (one JSON per line)
            parts: list[str] = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    import json as _json

                    obj = _json.loads(line)
                except Exception:
                    # ignore truncated lines (seen in streaming) or non-json lines
                    continue
                # prefer 'response' key used by Ollama streaming
                if isinstance(obj, dict):
                    val = obj.get("response") or obj.get("output") or obj.get("text")
                    if isinstance(val, str):
                        parts.append(val)
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                parts.append(item)
                            elif isinstance(item, dict):
                                for k in ("text", "content", "message"):
                                    t = item.get(k)
                                    if t:
                                        parts.append(t if isinstance(t, str) else str(t))
            if parts:
                text = "".join(parts)
                return OllamaResult(model=model, text=text.strip(), provider_latency_ms=provider_latency_ms)

        # If we parsed a normal JSON object, try to extract text from common keys
        def _extract_from_obj(obj: Any) -> str | None:
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                for key in ("response", "output", "text", "result", "message", "content"):
                    v = obj.get(key)
                    if isinstance(v, str) and v.strip():
                        return v
                    if isinstance(v, list) and v:
                        # join strings
                        strs = []
                        for item in v:
                            if isinstance(item, str):
                                strs.append(item)
                            elif isinstance(item, dict):
                                for k in ("text", "content", "message"):
                                    t = item.get(k)
                                    if t:
                                        strs.append(t if isinstance(t, str) else str(t))
                        if strs:
                            return "\n".join(strs)
            if isinstance(obj, list):
                parts = []
                for item in obj:
                    t = _extract_from_obj(item)
                    if t:
                        parts.append(t)
                if parts:
                    return "\n".join(parts)
            return None

        text = None
        if data is not None:
            text = _extract_from_obj(data)

        if not text:
            # log truncated raw for debugging
            logger.debug("Ollama raw response (truncated): %s", raw[:2000])
            snippet = raw[:1000]
            raise HTTPException(status_code=502, detail=f"Invalid JSON from Ollama: {snippet}")

        return OllamaResult(model=model, text=text.strip(), provider_latency_ms=provider_latency_ms)
