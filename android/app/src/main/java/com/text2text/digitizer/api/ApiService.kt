package com.text2text.digitizer.api

import com.text2text.digitizer.model.DigitizeResponse
import com.text2text.digitizer.model.ExportSheetsRequest
import com.text2text.digitizer.model.ExportSheetsResponse
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiService {

    @Multipart
    @POST("v1/digitize")
    suspend fun digitize(
        @Part image: MultipartBody.Part
    ): Response<DigitizeResponse>

    @POST("v1/export-to-sheets")
    suspend fun exportToSheets(
        @Body request: ExportSheetsRequest
    ): Response<ExportSheetsResponse>
}
