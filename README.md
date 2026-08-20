# PARKINSON-S-DISEASE-DETECTION-USING-HAND-DRAWN-IMAGES

[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![Android](https://img.shields.io/badge/Android-Kotlin-orange)](https://developer.android.com/kotlin)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-blue)](https://www.tensorflow.org/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, two-stage diagnostic framework that detects whether an individual shows signs of Parkinson's Disease by analyzing hand-drawn images (spirals and waves) — **without** requiring external cloud API calls.

The goal of this project is to implement a resource-efficient setup based on custom Convolutional Neural Networks (DeepCNN) and Transfer Learning architectures (VGG16, ResNet50), making the diagnostic system accessible both as a local Tkinter desktop app and as a mobile application running on Android via TensorFlow Lite.

---

## Architecture

The workflow consists of model training, desktop comparison, and a mobile application running local real-time inference:

```
Hand-Drawn Spiral/Wave Image
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage 1: Pre-processing & Feature Ext.  │  ← Normalization & Resize
│ Input: Uploaded Image                   │     64x64 (DeepCNN) or
│ Output: Normalized tensor               │     224x224 (VGG16/ResNet50)
└─────────────────────────────────────────┘
    │ Pre-processed Tensor
    ▼
┌─────────────────────────────────────────┐
│ Stage 2: Deep Learning Classifier       │  ← Mobile: TensorFlow Lite
│ Model Option: VGG16 / DeepCNN / ResNet50│     Desktop: Keras Backend
│ Output: Healthy (0) / Parkinson's (1)   │     Local Real-Time Diagnostics
└─────────────────────────────────────────┘
```

---

## How It Works

Most state-of-the-art diagnostic imaging pipelines require heavy cloud computing and expensive API query latency. This framework brings detection straight to the local device:

1. **Pre-processing:** Hand-drawn images (spiral or wave tests) are loaded, normalized, and resized (either to $64 \times 64$ for DeepCNN or $224 \times 224$ for Transfer Learning models).
2. **Deep Learning Classification:** A selected local neural network analyzes the handwriting artifacts (such as tremors, stroke pressure differences, and line inconsistencies).
3. **Divergence Gap:** The models evaluate diagnostic features. VGG16 maps spatial anomalies with high accuracy, while DeepCNN provides a lightweight alternative that runs quickly with minimal memory footprint.

---

## Features

- **Multi-Model Support:** Includes training pipelines and weights for DeepCNN, VGG16, and ResNet50.
- **Tkinter Desktop Interface:** Allows loading model weights, testing drawing images, and visualizing accuracy/loss curves.
- **Model Comparison Dashboard:** Side-by-side epoch-wise plotting of Precision, Recall, Accuracy, and Loss.
- **Mobile Diagnostic App:** Local Android application built in Kotlin, running offline classification using an embedded TensorFlow Lite model.
- **Interactive Canvas & Camera:** Diagnoses drawings directly from paper photos or canvas inputs.
- **Educational Information Dashboard:** Shows Parkinson's symptoms, causes, early detection facts, and treatment guides.

---

## Results

| Model Architecture | Accuracy | Precision | Recall | Target Resource Target | Key Benefit |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **VGG16 Transfer Learning** | **99.31%** | High | High | Desktop & Mobile | Most accurate; captures minute drawing tremors |
| **DeepCNN (Custom)** | **98.60%** | Medium-High | High | Desktop & Mobile | Very lightweight (~64x64 inputs); fast inference |
| **ResNet50 Transfer Learning** | **77.80%** | Medium | Medium | Desktop | High depth; useful for complex dataset patterns |

---

## Visualizations

### Exploratory Data Analysis
Model performance is tracked during training and cached as Pickle logs (`.pckl`). You can view these metrics directly in the desktop application.

### Comparison Dashboard
The comparison view plots the following subplots for all three models simultaneously across epochs:
1. **Model Precision Comparison**
2. **Model Recall Comparison**
3. **Model Accuracy Comparison**
4. **Model Loss Comparison**

---

## Repository Structure

```text
Parkinson's_Project/
├── README.md                      # This file
├── .gitignore                     # Git configuration to ignore build files/local credentials
├── readme.pdf                     # Project documentation/guide PDF
├── Execution Video.mp4            # Step-by-step walkthrough demo of the systems
│
└── Project Codes/
    ├── ParkinsonPrediction.py     # Main Tkinter Desktop Application
    ├── testtrain.py               # DeepCNN training script (saves to 'model/')
    ├── testtrain1.py              # ResNet50 training script (saves to 'model1/')
    ├── testtrain2.py              # VGG16 training script (saves to 'model2/')
    │
    └── Parkinson's Disease Detection/  # Android Project Directory
        ├── build.gradle.kts       # Project-level Gradle configuration
        └── app/
            ├── build.gradle.kts   # App-level module configuration
            └── src/main/
                ├── assets/
                │   └── model2.tflite  # Mobile TensorFlow Lite model
                └── java/com/example/parkinsonsdiseasedetection/
                    ├── MainActivity.kt          # Landing Screen
                    ├── DetectParkinsonActivity.kt # TFLite Drawing Inference Screen
                    ├── KnowParkinsonActivity.kt   # Symptoms & Causes info
                    └── ExploreActivity.kt         # Health explorer UI
```

---

## Datasets

The training pipeline uses the following dataset structure:

| Split | Location | Categories | Description |
| :--- | :--- | :--- | :--- |
| **Training Set** | `dataset/train/` | `healthy/`, `parkinson/` | Base drawing images (spirals and wave drawings) |
| **Testing Set** | `dataset/test/` | `healthy/`, `parkinson/` | Validation drawings to evaluate model generalization |

---

## Quickstart

### 1. Run the Desktop GUI App
Install the python requirements and run the prediction app:
```bash
pip install opencv-python numpy tensorflow pandas matplotlib pillow scikit-learn
python "Project Codes/ParkinsonPrediction.py"
```

### 2. Build the Android Mobile App
1. Open the folder `Project Codes/Parkinson's Disease Detection` in **Android Studio**.
2. Connect your Android phone (ensure USB Debugging is turned on).
3. Click **Run** (`Shift + F10`) to build the project and install it on your device.

### 3. Run Training Pipelines
To retrain the models on new drawing datasets:
* For Custom DeepCNN: `python "Project Codes/testtrain.py"`
* For ResNet50: `python "Project Codes/testtrain1.py"`
* For VGG16: `python "Project Codes/testtrain2.py"`

---

## Script Sections

| File | Type | Description |
| :--- | :--- | :--- |
| `ParkinsonPrediction.py` | GUI Application | Integrates Tkinter window, model selectors, inference visualizer, and comparative matplotlib plots. |
| `testtrain.py` | Keras / TF script | Prepares $64 \times 64$ dataset, trains DeepCNN model, evaluates performance, and outputs pickle statistics. |
| `testtrain1.py` | Keras / TF script | Prepares $224 \times 224$ dataset, trains ResNet50 base with custom dense layer, and exports performance curves. |
| `testtrain2.py` | Keras / TF script | Prepares $224 \times 224$ dataset, trains VGG16 transfer model, and prints classification reports. |

---

## Key Design Decisions

| Choice | Why |
| :--- | :--- |
| **Offline Inference (TFLite)** | Guarantees instant diagnostic output with zero latency and high data privacy. |
| **Image Resolution Reduction** | Using $64\times64$ inputs for DeepCNN allows fast desktop classification with minimal RAM consumption. |
| **Transfer Learning on VGG16** | Leverages pre-trained ImageNet weights, achieving **99.31% accuracy** even with small handwriting datasets. |
| **Pickled Training Logs** | Enables the desktop UI to display instant comparative graphs without re-evaluating the datasets. |

---

## Requirements

The python environment expects:
* `tensorflow>=2.10.0`
* `opencv-python>=4.6.0`
* `numpy>=1.23.0`
* `matplotlib>=3.6.0`
* `scikit-learn>=1.1.0`
* `pillow>=9.2.0`
* `pandas>=1.5.0`

---

## Author
**keerthireddy8595**

## License
MIT — see [LICENSE](LICENSE) for details.
