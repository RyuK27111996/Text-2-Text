from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date

from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import SalesRecord


@dataclass(slots=True)
class SheetsResult:
    """Result returned after writing to Google Sheets."""

    sheet_url: str
    sheet_id: str
    rows_written: int


_HEADER_ROW = [
    "Date",
    "Seller",
    "Description",
    "Quantity",
    "Unit Price",
    "Line Total",
    "Currency",
    "Item Notes",
    "Subtotal",
    "Tax",
    "Grand Total",
    "Confidence",
    "Warnings",
]


class SheetsService:
    """Writes extracted SalesRecord data to Google Sheets via the v4 API."""

    def __init__(self, settings: Settings) -> None:
        if not settings.google_sheets_credentials_json:
            raise RuntimeError(
                "Google Sheets integration is not configured. "
                "Set the GOOGLE_SHEETS_CREDENTIALS_JSON environment variable."
            )
        # Defer imports so the service can still be imported when the
        # google-api-python-client package is present but credentials are absent.
        try:
            from google.oauth2 import service_account  # type: ignore[import]
            from googleapiclient.discovery import build  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "google-api-python-client and google-auth must be installed. "
                "Run: pip install google-api-python-client google-auth"
            ) from exc

        creds_dict = json.loads(settings.google_sheets_credentials_json)
        self._creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
            ],
        )
        self._build = build
        self._default_title = settings.google_sheets_default_title

    async def export(
        self,
        record: SalesRecord,
        sheet_id: str | None,
        sheet_name: str,
    ) -> SheetsResult:
        """Write a SalesRecord to Google Sheets; create a new sheet if sheet_id is None."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._export_sync, record, sheet_id, sheet_name
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Google Sheets API error: {exc}",
            ) from exc

    def _export_sync(
        self,
        record: SalesRecord,
        sheet_id: str | None,
        sheet_name: str,
    ) -> SheetsResult:
        sheets_client = self._build("sheets", "v4", credentials=self._creds)
        spreadsheets = sheets_client.spreadsheets()

        if sheet_id is None:
            sheet_id = self._create_spreadsheet(spreadsheets, sheet_name)

        self._ensure_sheet_tab(spreadsheets, sheet_id, sheet_name)
        needs_header = self._is_sheet_empty(spreadsheets, sheet_id, sheet_name)
        rows = self._build_rows(record, include_header=needs_header)
        self._append_rows(spreadsheets, sheet_id, sheet_name, rows)

        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        data_rows = len(rows) - (1 if needs_header else 0)
        return SheetsResult(sheet_url=url, sheet_id=sheet_id, rows_written=data_rows)

    def _create_spreadsheet(self, spreadsheets, sheet_name: str) -> str:
        title = f"{self._default_title} - {date.today().isoformat()}"
        body = {
            "properties": {"title": title},
            "sheets": [{"properties": {"title": sheet_name}}],
        }
        result = spreadsheets.create(body=body).execute()
        return result["spreadsheetId"]

    def _ensure_sheet_tab(self, spreadsheets, sheet_id: str, sheet_name: str) -> None:
        """Add the named worksheet tab if it doesn't already exist."""
        meta = spreadsheets.get(spreadsheetId=sheet_id).execute()
        existing = {s["properties"]["title"] for s in meta.get("sheets", [])}
        if sheet_name not in existing:
            spreadsheets.batchUpdate(
                spreadsheetId=sheet_id,
                body={
                    "requests": [
                        {"addSheet": {"properties": {"title": sheet_name}}}
                    ]
                },
            ).execute()

    def _is_sheet_empty(self, spreadsheets, sheet_id: str, sheet_name: str) -> bool:
        result = (
            spreadsheets.values()
            .get(spreadsheetId=sheet_id, range=f"{sheet_name}!A1")
            .execute()
        )
        return "values" not in result

    def _build_rows(self, record: SalesRecord, *, include_header: bool) -> list[list]:
        rows: list[list] = []
        if include_header:
            rows.append(_HEADER_ROW)

        warnings_str = "; ".join(record.warnings) if record.warnings else ""

        for item in record.items:
            rows.append(
                [
                    record.date or "",
                    record.seller or "",
                    item.description,
                    item.quantity,
                    item.unit_price,
                    item.line_total,
                    item.currency or "",
                    item.notes or "",
                    record.subtotal if record.subtotal is not None else "",
                    record.tax if record.tax is not None else "",
                    record.grand_total if record.grand_total is not None else "",
                    record.confidence,
                    warnings_str,
                ]
            )

        # If there are no items, still write a summary row so the sheet isn't empty
        if not record.items:
            rows.append(
                [
                    record.date or "",
                    record.seller or "",
                    "(no items detected)",
                    "",
                    "",
                    "",
                    "",
                    "",
                    record.subtotal if record.subtotal is not None else "",
                    record.tax if record.tax is not None else "",
                    record.grand_total if record.grand_total is not None else "",
                    record.confidence,
                    warnings_str,
                ]
            )
        return rows

    def _append_rows(
        self,
        spreadsheets,
        sheet_id: str,
        sheet_name: str,
        rows: list[list],
    ) -> None:
        spreadsheets.values().append(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": rows},
        ).execute()
