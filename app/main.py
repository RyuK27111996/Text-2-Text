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
from app.schemas import ErrorResponse, GenerateResponse, ImageToTextRequest, TextToTextRequest
from app.services.gemini import GeminiService
from app.services.model_router import build_default_model_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")


class RequestLoggerAdapter(logging.LoggerAdapter):
    """Inject the current request id into structured log records."""

    def process(self, msg, kwargs):
        """Populate the `request_id` logging field when absent."""
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("request_id", self.extra.get("request_id", "-"))
        return msg, kwargs


def get_active_backend() -> str:
    """Return the single provider name exposed by this application."""
    return "gemma"


def build_generation_service(request: Request):
    """Instantiate the Gemma service used for incoming requests."""
    settings = request.app.state.settings
    return GeminiService(
        settings=settings,
        client=request.app.state.http_client,
        semaphore=request.app.state.provider_semaphore,
        model_router=request.app.state.model_router,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and clean up shared application resources."""
    settings = get_settings()
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)

    app.state.settings = settings
    app.state.http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    app.state.provider_semaphore = asyncio.Semaphore(settings.max_concurrent_provider_calls)
    app.state.model_router = build_default_model_router()

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
    title="FastAPI Gemma Async API",
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
    """Report service health and the currently selected backend."""
    settings = request.app.state.settings
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "active_backend": get_active_backend(),
        "gemini_configured": bool(settings.gemini_api_key),
        "max_concurrent_provider_calls": settings.max_concurrent_provider_calls,
    }


@app.post(
    "/v1/text-to-text",
    response_model=GenerateResponse,
    responses={
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def text_to_text(request: Request, payload: TextToTextRequest):
    """Validate a text prompt, call the active provider, and normalize the response."""
    request_id = getattr(request.state, "request_id", "-")
    log = RequestLoggerAdapter(logger, {"request_id": request_id})
    settings = request.app.state.settings

    if len(payload.prompt) > settings.max_input_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"prompt exceeds max allowed characters ({settings.max_input_chars})",
        )

    service = build_generation_service(request)

    total_started = time.perf_counter()
    log.info("Incoming text-to-text request")
    result = await service.generate_text(payload)
    total_latency_ms = (time.perf_counter() - total_started) * 1000.0
    log.info("Text-to-text request completed")

    return GenerateResponse(
        request_id=request_id,
        model=result.model,
        output_text=result.text,
        provider_latency_ms=round(result.provider_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )


@app.post(
    "/v1/image-to-text",
    response_model=GenerateResponse,
    responses={
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def image_to_text(request: Request, payload: ImageToTextRequest):
    """Validate an image-to-text request, call the active provider, and normalize the response."""
    request_id = getattr(request.state, "request_id", "-")
    log = RequestLoggerAdapter(logger, {"request_id": request_id})

    service = build_generation_service(request)

    total_started = time.perf_counter()
    log.info("Incoming image-to-text request")
    result = await service.generate_image_to_text(payload)
    total_latency_ms = (time.perf_counter() - total_started) * 1000.0
    log.info("Image-to-text request completed")

    return GenerateResponse(
        request_id=request_id,
        model=result.model,
        output_text=result.text,
        provider_latency_ms=round(result.provider_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
    include_in_schema=False,
)
async def generate(request: Request, payload: TextToTextRequest):
    """Backward-compatible alias for the text-to-text endpoint."""
    return await text_to_text(request, payload)
