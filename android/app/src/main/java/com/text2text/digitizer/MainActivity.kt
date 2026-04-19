package com.text2text.digitizer

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import androidx.preference.PreferenceManager
import coil.load
import com.google.gson.Gson
import com.text2text.digitizer.api.RetrofitClient
import com.text2text.digitizer.databinding.ActivityMainBinding
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else Toast.makeText(this, "Camera permission is required to take photos", Toast.LENGTH_LONG).show()
    }

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { handleSelectedImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnCapture.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                if (imageCapture != null) takePhoto() else startCamera()
            } else {
                requestCameraPermission.launch(Manifest.permission.CAMERA)
            }
        }

        binding.btnGallery.setOnClickListener {
            pickImage.launch("image/*")
        }

        binding.btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        binding.btnUpload.setOnClickListener {
            // Handled per-image in handleSelectedImage
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            imageCapture = capture
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture
                )
                binding.previewView.isVisible = true
                binding.btnCapture.text = "Capture"
            } catch (e: Exception) {
                Toast.makeText(this, "Failed to open camera: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val photoFile = File(externalCacheDir, "sales_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        imageCapture?.takePicture(
            outputOptions, ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    binding.previewView.isVisible = false
                    handleSelectedImage(Uri.fromFile(photoFile))
                }
                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(this@MainActivity, "Capture failed: ${exc.message}", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    private fun handleSelectedImage(uri: Uri) {
        binding.imgPreview.isVisible = true
        binding.imgPreview.load(uri)
        binding.btnUpload.isEnabled = true
        binding.btnUpload.setOnClickListener { uploadImage(uri) }
    }

    private fun uploadImage(uri: Uri) {
        val bytes = contentResolver.openInputStream(uri)?.readBytes()
        if (bytes == null || bytes.isEmpty()) {
            Toast.makeText(this, "Could not read image file", Toast.LENGTH_SHORT).show()
            return
        }
        if (bytes.size > 10 * 1024 * 1024) {
            Toast.makeText(this, "Image too large (max 10 MB). Please use a lower resolution.", Toast.LENGTH_LONG).show()
            return
        }

        val mimeType = contentResolver.getType(uri) ?: "image/jpeg"
        val requestFile = bytes.toRequestBody(mimeType.toMediaTypeOrNull())
        val part = MultipartBody.Part.createFormData("image", "sales.jpg", requestFile)

        binding.progressBar.isVisible = true
        binding.btnUpload.isEnabled = false

        val prefs = PreferenceManager.getDefaultSharedPreferences(this)
        val backendUrl = prefs.getString("backend_url", "http://10.0.2.2:8000") ?: "http://10.0.2.2:8000"

        lifecycleScope.launch {
            try {
                val response = RetrofitClient.getInstance(backendUrl).digitize(part)
                if (response.isSuccessful) {
                    val result = response.body()!!
                    val intent = Intent(this@MainActivity, ResultActivity::class.java).apply {
                        putExtra("digitize_result", Gson().toJson(result))
                    }
                    startActivity(intent)
                } else {
                    val errorBody = response.errorBody()?.string() ?: "Unknown error"
                    Toast.makeText(this@MainActivity, "Server error ${response.code()}: $errorBody", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "Network error: ${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                binding.progressBar.isVisible = false
                binding.btnUpload.isEnabled = true
            }
        }
    }
}
