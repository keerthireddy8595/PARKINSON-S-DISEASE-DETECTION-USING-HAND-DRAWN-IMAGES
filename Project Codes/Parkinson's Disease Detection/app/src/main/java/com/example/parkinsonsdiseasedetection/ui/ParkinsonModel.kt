package com.example.parkinsonsdiseasedetection.ui

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ParkinsonModel(context: Context) {
    private var interpreter: Interpreter

    init {
        interpreter = Interpreter(loadModelFile(context))
    }

    // Load the TFLite model from assets
    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("model2.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    // Make a prediction (input is a 224x224 image)
    fun predict(inputImage: ByteBuffer): Float {
        val output = Array(1) { FloatArray(1) } // Binary classification output
        interpreter.run(inputImage, output)
        return output[0][0] // Prediction probability (closer to 1 → parkinson, closer to 0 → normal)
    }

    // Preprocess image: Convert bitmap to ByteBuffer
    fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4) // Float32
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(224 * 224)
        bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        return inputBuffer
    }
}
