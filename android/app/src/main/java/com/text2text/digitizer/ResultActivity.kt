package com.text2text.digitizer

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import androidx.preference.PreferenceManager
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.gson.Gson
import com.text2text.digitizer.api.RetrofitClient
import com.text2text.digitizer.databinding.ActivityResultBinding
import com.text2text.digitizer.model.DigitizeResponse
import com.text2text.digitizer.model.ExportSheetsRequest
import kotlinx.coroutines.launch

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private lateinit var digitizeResponse: DigitizeResponse
    private var lastSheetId: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val json = intent.getStringExtra("digitize_result") ?: run {
            finish()
            return
        }
        digitizeResponse = Gson().fromJson(json, DigitizeResponse::class.java)
        val record = digitizeResponse.sales_record

        // Confidence banner
        val (colorRes, bannerText) = when (record.confidence) {
            "high" -> Pair(android.R.color.holo_green_dark, "High Confidence")
            "medium" -> Pair(android.R.color.holo_orange_dark, "Medium Confidence — Review Recommended")
            else -> Pair(android.R.color.holo_red_dark, "Low Confidence — Manual Verification Required")
        }
        binding.confidenceBanner.setBackgroundColor(getColor(colorRes))
        binding.confidenceBanner.text = bannerText

        // Total mismatch warning
        if (!record.total_matches && record.grand_total != null) {
            binding.totalMismatchWarning.isVisible = true
            binding.totalMismatchWarning.text =
                "⚠ Written total ${record.grand_total} differs from computed ${record.computed_total}"
        }

        // Header fields
        binding.tvDate.text = "Date: ${record.date ?: "Not found"}"
        binding.tvSeller.text = "Seller: ${record.seller ?: "Not found"}"
        binding.tvComputedTotal.text = "Computed total: ${record.computed_total}"

        // Items RecyclerView
        binding.recyclerItems.layoutManager = LinearLayoutManager(this)
        binding.recyclerItems.adapter = SaleItemAdapter(record.items)

        // Warnings
        if (record.warnings.isNotEmpty()) {
            binding.tvWarnings.isVisible = true
            binding.tvWarnings.text = "Warnings:\n" + record.warnings.joinToString("\n") { "• $it" }
        }

        // Latency
        binding.tvLatency.text = "Extracted in ${digitizeResponse.total_latency_ms.toLong()} ms"

        binding.btnExport.setOnClickListener { exportToSheets() }

        binding.btnCopyJson.setOnClickListener {
            val clipboard = getSystemService(ClipboardManager::class.java)
            clipboard.setPrimaryClip(ClipData.newPlainText("Sales JSON", Gson().toJson(record)))
            Toast.makeText(this, "JSON copied to clipboard", Toast.LENGTH_SHORT).show()
        }
    }

    private fun exportToSheets() {
        binding.btnExport.isEnabled = false
        binding.exportProgress.isVisible = true

        val prefs = PreferenceManager.getDefaultSharedPreferences(this)
        val backendUrl = prefs.getString("backend_url", "http://10.0.2.2:8000") ?: "http://10.0.2.2:8000"

        lifecycleScope.launch {
            try {
                val request = ExportSheetsRequest(
                    sales_record = digitizeResponse.sales_record,
                    sheet_id = lastSheetId,
                    sheet_name = "Sales"
                )
                val response = RetrofitClient.getInstance(backendUrl).exportToSheets(request)
                when {
                    response.isSuccessful -> {
                        val result = response.body()!!
                        lastSheetId = result.sheet_id
                        Toast.makeText(this@ResultActivity, "${result.rows_written} rows written!", Toast.LENGTH_SHORT).show()
                        startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(result.sheet_url)))
                    }
                    response.code() == 503 -> {
                        Toast.makeText(
                            this@ResultActivity,
                            "Google Sheets is not configured on the server. Ask your admin to add GOOGLE_SHEETS_CREDENTIALS_JSON.",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                    else -> {
                        Toast.makeText(this@ResultActivity, "Export failed: HTTP ${response.code()}", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Toast.makeText(this@ResultActivity, "Network error: ${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                binding.btnExport.isEnabled = true
                binding.exportProgress.isVisible = false
            }
        }
    }
}
