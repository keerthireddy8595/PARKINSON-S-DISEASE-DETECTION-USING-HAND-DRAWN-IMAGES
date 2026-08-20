package com.example.parkinsonsdiseasedetection

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button

class ExploreActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_explore)  // Set the layout for Explore page

        val knowButton: Button = findViewById(R.id.knowButton)
        val detectButton: Button = findViewById(R.id.detectButton)

        // Navigate to Know Parkinson page
        knowButton.setOnClickListener {
            val intent = Intent(this, KnowParkinsonActivity::class.java)
            startActivity(intent)
        }

        // Navigate to Detect Parkinson page
        detectButton.setOnClickListener {
            val intent = Intent(this, DetectParkinsonActivity::class.java)
            startActivity(intent)
        }
    }
}
