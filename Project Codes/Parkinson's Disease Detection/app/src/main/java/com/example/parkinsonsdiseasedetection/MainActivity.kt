package com.example.parkinsonsdiseasedetection

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val exploreButton: Button = findViewById(R.id.exploreButton)
        val imageView: ImageView = findViewById(R.id.imageView)

        // Set the image
        imageView.setImageResource(R.drawable.parkinson)

        exploreButton.setOnClickListener {
            val intent = Intent(this, ExploreActivity::class.java)
            startActivity(intent)
        }
    }
}
