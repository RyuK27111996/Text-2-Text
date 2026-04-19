"""Tests for SheetsService — mocks google-api-python-client."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.schemas import SaleItem, SalesRecord


def _make_settings(**overrides) -> Settings:
    base = {
        "gemini_api_key": "test-key",
        "google_sheets_credentials_json": json.dumps(
            {
                "type": "service_account",
                "project_id": "test",
                "private_key_id": "key-id",
                "private_key": (
                    "-----BEGIN RSA PRIVATE KEY-----\n"
                    "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xHn/ygWep4v...\n"
                    "-----END RSA PRIVATE KEY-----\n"
                ),
                "client_email": "test@test.iam.gserviceaccount.com",
                "client_id": "123",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        ),
        "google_sheets_default_title": "Test Sales",
    }
    base.update(overrides)
    return Settings(**base)


def _sample_record() -> SalesRecord:
    return SalesRecord(
        date="2024-01-15",
        seller="Test Shop",
        items=[
            SaleItem(description="Apple", quantity=3, unit_price=1.5, line_total=4.5, currency="$", notes=None),
            SaleItem(description="Bread", quantity=2, unit_price=2.0, line_total=4.0, currency="$", notes=None),
        ],
        subtotal=8.5,
        tax=0.5,
        grand_total=8.5,
        computed_total=8.5,
        total_matches=True,
        confidence="high",
        warnings=[],
        raw_text=None,
    )


def test_sheets_service_raises_without_credentials():
    from app.services.sheets import SheetsService
    settings = _make_settings(google_sheets_credentials_json=None)
    with pytest.raises(RuntimeError, match="GOOGLE_SHEETS_CREDENTIALS_JSON"):
        SheetsService(settings=settings)


def test_export_endpoint_returns_503_without_credentials():
    from fastapi.testclient import TestClient
    from app.main import app
    record = _sample_record()
    with TestClient(app) as client:
        response = client.post(
            "/v1/export-to-sheets",
            json={
                "sales_record": record.model_dump(),
                "sheet_id": None,
                "sheet_name": "Sales",
            },
        )
    assert response.status_code == 503
    assert "GOOGLE_SHEETS_CREDENTIALS_JSON" in response.json()["detail"]


def test_build_rows_includes_header_when_empty():
    from app.services.sheets import SheetsService, _HEADER_ROW
    record = _sample_record()

    # Directly construct the service instance without going through __init__
    # (which would require real Google credentials).
    service = SheetsService.__new__(SheetsService)
    service._creds = MagicMock()
    service._build = MagicMock()
    service._default_title = "Test"

    rows_with_header = service._build_rows(record, include_header=True)
    rows_without_header = service._build_rows(record, include_header=False)

    assert rows_with_header[0] == _HEADER_ROW
    assert len(rows_with_header) == len(record.items) + 1  # header + items
    assert len(rows_without_header) == len(record.items)


def test_build_rows_empty_items_still_writes_one_row():
    from app.services.sheets import SheetsService
    record = SalesRecord(
        items=[],
        computed_total=0.0,
        total_matches=True,
        confidence="low",
        warnings=["No items detected"],
    )

    service = SheetsService.__new__(SheetsService)
    service._creds = MagicMock()
    service._build = MagicMock()
    service._default_title = "Test"

    rows = service._build_rows(record, include_header=False)
    assert len(rows) == 1
    assert rows[0][2] == "(no items detected)"
