package com.example.parkinsonsdiseasedetection

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class DetectParkinsonActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var bitmap: Bitmap
    private lateinit var tflite: Interpreter

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detect_parkinson)

        // Enable back button
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        val selectButton: Button = findViewById(R.id.selectButton)
        val classifyButton: Button = findViewById(R.id.classifyButton)

        // Load TFLite Model
        try {
            tflite = Interpreter(loadModelFile())
            Log.d("TFLite", "Model loaded successfully")
        } catch (e: IOException) {
            e.printStackTrace()
            resultTextView.text = "Error loading model"
            Log.e("TFLite", "Error loading model: ${e.message}")
        }

        // Select Image from Gallery
        val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                bitmap = uriToBitmap(it)
                imageView.setImageBitmap(bitmap)
                Log.d("ImageProcessing", "Image selected successfully")
            }
        }

        selectButton.setOnClickListener {
            imagePicker.launch("image/*")
        }

        classifyButton.setOnClickListener {
            if (!::bitmap.isInitialized) {
                Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
                Log.e("TFLite", "No image selected")
                return@setOnClickListener
            }
            classifyImage()
        }
    }

    // Handle back button action
    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }

    private fun uriToBitmap(uri: Uri): Bitmap {
        val inputStream = contentResolver.openInputStream(uri)
        return BitmapFactory.decodeStream(inputStream)!!
    }

    private fun loadModelFile(): ByteBuffer {
        try {
            Log.d("TFLite", "Attempting to load model file...")
            val fileDescriptor = assets.openFd("model2.tflite")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
            Log.d("TFLite", "Model loaded successfully")
            return mappedByteBuffer
        } catch (e: IOException) {
            Log.e("TFLite", "Error loading model: ${e.message}")
            throw RuntimeException("Model file not found or cannot be read: ${e.message}")
        }
    }

    @SuppressLint("SetTextI18n")
    private fun classifyImage() {
        try {
            if (!::bitmap.isInitialized) {
                return
            }

            Log.d("TFLite", "Starting image classification...")

            val inputSize = 224
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(resizedBitmap)

            val normalizedBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
            normalizedBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(inputSize * inputSize)
            resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

            var pixelIndex = 0
            for (i in 0 until inputSize) {
                for (j in 0 until inputSize) {
                    val pixelValue = intValues[pixelIndex++]
                    normalizedBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
                    normalizedBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
                    normalizedBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
                }
            }

            val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, inputSize, inputSize, 3), DataType.FLOAT32)
            inputBuffer.loadBuffer(normalizedBuffer)

            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 3), DataType.FLOAT32)

            tflite.run(inputBuffer.buffer, outputBuffer.buffer)

            val outputArray = outputBuffer.floatArray

            val resultText = when {
                outputArray[1] > outputArray[0] -> "Parkinson Detected"
                else -> "Healthy"
            }

            runOnUiThread {
                resultTextView.text = resultText
            }

        } catch (e: Exception) {
            e.printStackTrace()
            resultTextView.text = "Error processing image"
        }
    }
}
