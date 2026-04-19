"""Tests for DigitizerService — uses httpx.MockTransport to avoid real API calls."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.services.digitizer import DigitizerService


def _make_gemini_response(payload: dict) -> bytes:
    """Wrap a dict in the Gemini generateContent response envelope."""
    body = {
        "candidates": [
            {
                "content": {"parts": [{"text": json.dumps(payload)}]},
                "finishReason": "STOP",
            }
        ]
    }
    return json.dumps(body).encode()


def _make_settings(**overrides) -> Settings:
    base = {
        "gemini_api_key": "test-key",
        "gemini_vision_model": "gemini-2.0-flash",
        "max_image_bytes": 10 * 1024 * 1024,
        "vision_timeout_seconds": 30.0,
        "gemini_max_retries": 0,
        "gemini_retry_base_delay_seconds": 0.0,
        "gemini_base_url": "https://generativelanguage.googleapis.com",
        "gemini_api_version": "v1beta",
        "max_concurrent_provider_calls": 5,
    }
    base.update(overrides)
    return Settings(**base)


def _make_service(mock_response: httpx.Response) -> tuple[DigitizerService, asyncio.Semaphore]:
    transport = httpx.MockTransport(lambda req: mock_response)
    client = httpx.AsyncClient(transport=transport)
    semaphore = asyncio.Semaphore(5)
    service = DigitizerService(settings=_make_settings(), client=client, semaphore=semaphore)
    return service, semaphore


_VALID_PAYLOAD = {
    "date": "2024-01-15",
    "seller": "John's Shop",
    "items": [
        {"description": "Apple", "quantity": 3, "unit_price": 1.5, "line_total": 4.5, "currency": "$", "notes": None},
        {"description": "Bread", "quantity": 2, "unit_price": 2.0, "line_total": 4.0, "currency": "$", "notes": None},
    ],
    "subtotal": 8.5,
    "tax": 0.5,
    "grand_total": 9.0,
    "confidence": "high",
    "warnings": [],
    "raw_text": "John's Shop 2024-01-15",
}


@pytest.mark.asyncio
async def test_digitize_returns_structured_record():
    mock = httpx.Response(200, content=_make_gemini_response(_VALID_PAYLOAD))
    service, _ = _make_service(mock)
    tiny_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00" * 100
    result = await service.digitize(tiny_jpeg, "image/jpeg")
    assert result.sales_record.date == "2024-01-15"
    assert result.sales_record.seller == "John's Shop"
    assert len(result.sales_record.items) == 2
    assert result.sales_record.items[0].description == "Apple"


@pytest.mark.asyncio
async def test_digitize_computes_server_side_total():
    payload = dict(_VALID_PAYLOAD)
    payload["grand_total"] = 999.99  # intentionally wrong
    mock = httpx.Response(200, content=_make_gemini_response(payload))
    service, _ = _make_service(mock)
    tiny_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00" * 100
    result = await service.digitize(tiny_jpeg, "image/jpeg")
    assert result.sales_record.computed_total == pytest.approx(8.5)
    assert result.sales_record.total_matches is False
    assert any("999.99" in w for w in result.sales_record.warnings)


@pytest.mark.asyncio
async def test_digitize_matching_total():
    mock = httpx.Response(200, content=_make_gemini_response(_VALID_PAYLOAD))
    service, _ = _make_service(mock)
    tiny_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00" * 100
    result = await service.digitize(tiny_jpeg, "image/jpeg")
    assert result.sales_record.computed_total == pytest.approx(8.5)
    assert result.sales_record.total_matches is False  # 8.5 != 9.0


@pytest.mark.asyncio
async def test_digitize_empty_image_raises_422():
    from fastapi import HTTPException
    service, _ = _make_service(httpx.Response(200, content=b"{}"))
    with pytest.raises(HTTPException) as exc_info:
        await service.digitize(b"", "image/jpeg")
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_digitize_oversized_image_raises_413():
    from fastapi import HTTPException
    settings = _make_settings(max_image_bytes=100)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"{}"))
    client = httpx.AsyncClient(transport=transport)
    service = DigitizerService(settings=settings, client=client, semaphore=asyncio.Semaphore(5))
    with pytest.raises(HTTPException) as exc_info:
        await service.digitize(b"\x00" * 200, "image/jpeg")
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_digitize_empty_items_list():
    payload = {
        "items": [],
        "confidence": "low",
        "warnings": ["No items detected"],
        "date": None,
        "seller": None,
        "subtotal": None,
        "tax": None,
        "grand_total": None,
        "raw_text": None,
    }
    mock = httpx.Response(200, content=_make_gemini_response(payload))
    service, _ = _make_service(mock)
    tiny_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00" * 100
    result = await service.digitize(tiny_jpeg, "image/jpeg")
    assert result.sales_record.computed_total == 0.0
    assert result.sales_record.items == []
    assert result.sales_record.confidence == "low"


@pytest.mark.asyncio
async def test_digitize_no_api_key_raises_500():
    from fastapi import HTTPException
    settings = _make_settings(gemini_api_key=None)
    service = DigitizerService(
        settings=settings,
        client=httpx.AsyncClient(),
        semaphore=asyncio.Semaphore(5),
    )
    with pytest.raises(HTTPException) as exc_info:
        await service.digitize(b"\x00" * 100, "image/jpeg")
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_digitize_upstream_error_raises_502():
    from fastapi import HTTPException
    error_body = json.dumps({"error": {"message": "Bad request", "status": "INVALID_ARGUMENT", "code": 400}}).encode()
    mock = httpx.Response(400, content=error_body)
    service, _ = _make_service(mock)
    tiny_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00" * 100
    with pytest.raises(HTTPException) as exc_info:
        await service.digitize(tiny_jpeg, "image/jpeg")
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_digitize_endpoint_rejects_empty_file(tmp_path):
    """Integration test: POST /v1/digitize with empty file returns 422."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        response = client.post(
            "/v1/digitize",
            files={"image": ("test.jpg", b"", "image/jpeg")},
        )
    assert response.status_code == 422
