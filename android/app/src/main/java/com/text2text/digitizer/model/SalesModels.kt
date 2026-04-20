package com.text2text.digitizer.model

data class SaleItem(
    val description: String,
    val quantity: Double,
    val unit_price: Double,
    val line_total: Double,
    val currency: String?,
    val notes: String?
)

data class SalesRecord(
    val date: String?,
    val seller: String?,
    val items: List<SaleItem>,
    val subtotal: Double?,
    val tax: Double?,
    val grand_total: Double?,
    val computed_total: Double,
    val total_matches: Boolean,
    val confidence: String,
    val warnings: List<String>,
    val raw_text: String?
)

data class DigitizeResponse(
    val request_id: String,
    val model: String,
    val sales_record: SalesRecord,
    val provider_latency_ms: Double,
    val total_latency_ms: Double
)

data class ExportSheetsRequest(
    val sales_record: SalesRecord,
    val sheet_id: String?,
    val sheet_name: String = "Sales"
)

data class ExportSheetsResponse(
    val sheet_url: String,
    val sheet_id: String,
    val rows_written: Int
)

data class ApiError(val detail: String)
