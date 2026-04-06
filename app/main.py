from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.middleware.request_id import RequestContextMiddleware
from app.schemas import ErrorResponse, GenerateRequest, GenerateResponse
from app.services.gemini import GeminiService
try:
    from app.services.ollama import OllamaService
except Exception:  # pragma: no cover - optional dependency
    OllamaService = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")


class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("request_id", self.extra.get("request_id", "-"))
        return msg, kwargs


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)

    app.state.settings = settings
    app.state.http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    app.state.provider_semaphore = asyncio.Semaphore(settings.max_concurrent_provider_calls)

    logger.info(
        "Application startup complete",
        extra={"request_id": "startup"},
    )
    try:
        yield
    finally:
        await app.state.http_client.aclose()
        logger.info(
            "Application shutdown complete",
            extra={"request_id": "shutdown"},
        )


app = FastAPI(
    title="FastAPI Gemini Async API",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request):
    settings = request.app.state.settings
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "gemini_configured": bool(settings.gemini_api_key),
        "max_concurrent_provider_calls": settings.max_concurrent_provider_calls,
    }


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
    responses={
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def generate(request: Request, payload: GenerateRequest):
    request_id = getattr(request.state, "request_id", "-")
    log = RequestLoggerAdapter(logger, {"request_id": request_id})
    settings = request.app.state.settings

    if len(payload.prompt) > settings.max_input_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"prompt exceeds max allowed characters ({settings.max_input_chars})",
        )

    # choose service: Ollama (local) if configured, else Gemini
    if getattr(settings, "ollama_base_url", None):
        if OllamaService is None:
            raise HTTPException(status_code=500, detail="OllamaService not available")
        service = OllamaService(
            settings=settings,
            client=request.app.state.http_client,
            semaphore=request.app.state.provider_semaphore,
        )
    else:
        service = GeminiService(
            settings=settings,
            client=request.app.state.http_client,
            semaphore=request.app.state.provider_semaphore,
        )

    total_started = time.perf_counter()
    log.info("Incoming generation request")
    result = await service.generate(payload)
    total_latency_ms = (time.perf_counter() - total_started) * 1000.0
    log.info("Generation request completed")

    return GenerateResponse(
        request_id=request_id,
        model=result.model,
        output_text=result.text,
        provider_latency_ms=round(result.provider_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )
