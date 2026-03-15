# Air Writing Recognition — Project Documentation

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Running the Application](#running-the-application)
5. [How to Use](#how-to-use)
6. [Project Architecture](#project-architecture)
7. [How It Works (Technical)](#how-it-works-technical)
8. [Retraining the Model (Optional)](#retraining-the-model-optional)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This application lets you **draw letters in the air** using your finger in front of a webcam, and uses a **Convolutional Neural Network (CNN)** to recognize the characters in real time. It tracks your index fingertip using Google's MediaPipe hand landmark detection, renders strokes on a virtual canvas, and automatically predicts the letter once you pause drawing.

**Supported characters:** English uppercase letters A–Z (26 classes).

---

## Prerequisites

- **Python 3.10+** installed on your system
- A **webcam** (built-in or external)
- **Windows OS** (tested on Windows 10/11)

---

## Setup

### Step 1: Clone or download the repository

```bash
git clone <your-repo-url>
cd multilinguistic-air-writing
```

### Step 2: Create a Python virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the virtual environment

```powershell
# PowerShell
venv\Scripts\Activate.ps1

# Command Prompt
venv\Scripts\activate.bat
```

You should see `(venv)` appear at the beginning of your terminal prompt.

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| **TensorFlow** | Deep learning framework for the CNN model |
| **OpenCV** | Computer vision (webcam capture, image processing) |
| **NumPy** | Numerical computing |
| **MediaPipe** | Google's hand tracking library |

### Step 5: Verify the setup

```bash
python -c "import tensorflow, cv2, mediapipe; print('All dependencies installed.')"
```

---

## Running the Application

```bash
python main.py
```

This will:
1. Suppress noisy TensorFlow warnings
2. Load the pre-trained CNN model (~45 MB, loaded once at startup)
3. Open your webcam
4. Display two windows: **Live Feed** and **Canvas**

---

## How to Use

### Step 1: Start the application

Run `python main.py`. Wait for `Loading character model... ready.` to appear.

### Step 2: Position yourself

- Face your webcam
- Hold your hand up so it's visible in the frame
- The green skeleton overlay confirms your hand is being tracked

### Step 3: Draw a letter

The application is completely hands-free.

| Gesture | What happens |
|---------|--------------|
| **Index finger only** (point) | Draws a stroke on the canvas |
| **5 Fingers** (Open Palm) | Pauses drawing securely |
| **2 Fingers** (Peace Sign) | Clears the entire canvas |
| **3 Fingers** | Backspaces the last accepted letter |
| **Thumbs Up** | Accepts the word and Safely Exits |

- To write a multi-stroke letter like **"A"**: point your index finger, draw the left stroke, raise 5 fingers to pause, then point again to draw the right stroke and crossbar.
- You can hover your index finger over the on-screen **HELP** button to view controls at any time.
- A strict **Exponential Moving Average (EMA)** algorithm radically reduces finger jitter for butter-smooth strokes.

### Step 4: The Auto-Accept Engine

When you pause drawing for ~1.5 seconds, the system:
1. Crops your drawing and runs the CNN prediction.
2. If the AI is **> 80% confident**, the letter is **automatically accepted**.
3. The screen freezes for 0.5 seconds, flashes the prediction, and pushes the letter into your floating text ribbon at the bottom of the screen.

### Step 5: Backspace or Clear

If the AI predicts the wrong letter (or is <80% confident):
- **3 Fingers:** Instantly Backspaces the last accepted letter from your text ribbon.
- **2 Fingers:** Manually clears the canvas to wipe the current bad stroke.

### Step 6: Repeat

Build up your word or sentence letter by letter. The AI handles the acceptance cycle automatically as you draw and pause.

### Step 7: Save and exit

Show a **Thumbs Up** to quit (Ensure your thumb is pointing straight up). Your accumulated text is:
- Displayed in a massive confirmation banner
- Printed to the terminal
- Saved to `output/recognized_text.txt`

---

## Project Architecture

```
multilinguistic-air-writing/
├── main.py                      # Entry point — configures environment, launches app
├── src/                         # Core application code
│   ├── __init__.py              # Makes src a Python package
│   ├── app.py                   # Main loop — webcam, hand tracking, recognition
│   ├── constants.py             # Shared config — labels, colors, thresholds
│   ├── ui.py                    # Frosted-glass UI rendering & Text elements
│   └── utils.py                 # Helpers — tracking, classification, EMA smoothing
├── training/                    # Model training (optional)
│   └── train.py                 # CNN training pipeline
├── docs/                        # Documentation
│   └── DOCUMENTATION.md         # This file
├── models/                      # Pre-trained models
│   ├── hand_landmarker.task     # MediaPipe hand landmark model (~7.5 MB)
│   └── model_eng_alphabets.h5   # CNN for A-Z character recognition (~45 MB)
├── output/                      # Generated output
│   └── recognized_text.txt      # Accumulated recognized text
├── requirements.txt             # Python dependencies
├── README.md                    # Quick-start guide
└── .gitignore
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `main.py` | Suppresses TF warnings, prints welcome banner, calls `launch()` |
| `src/constants.py` | All tunable values: label map, skeleton edges, ink color, canvas size, delays, thresholds |
| `src/utils.py` | Hand tracker init, skeleton rendering, finger state detection, smoothing, canvas cropping + CNN inference, stroke reset |
| `src/app.py` | Main event loop — camera capture, hand landmark processing, HUD rendering, auto-recognition timing, keystroke handling, output saving |
| `training/train.py` | Loads CSV dataset, builds CNN architecture, trains, saves weights to `models/` |

---

## How It Works (Technical)

### 1. Hand Tracking (MediaPipe)

The application uses **MediaPipe HandLandmarker** in VIDEO mode to detect 21 hand landmarks per frame. The index fingertip is landmark point **#8**.

- **Drawing mode**: Index finger tip (#8) is above its PIP joint (#6), and middle finger tip (#12) is below its PIP joint (#10).
- **Pause mode**: Any other finger configuration.

A strict **Exponential Moving Average (EMA)** algorithm smooths the fingertip position to absorb jitter and create fluid strokes.

### 2. Canvas Rendering

Strokes are stored as a list of `deque` segments. Each segment represents a continuous stroke. When the user lifts the pen (pause mode), a new segment begins. Both the live frame and a white canvas are rendered simultaneously.

### 3. Character Recognition Pipeline

When the user pauses drawing for 1.5 seconds, the system:

1. **Converts** the canvas to grayscale and applies binary thresholding
2. **Finds contours** to locate the drawn character
3. **Crops** the largest contour's bounding box with padding
4. **Pads** the crop to a square (maintaining aspect ratio)
5. **Resizes** to 28×28 pixels (the CNN input size)
6. **Runs inference** through the cached CNN model
7. **Returns** the predicted letter and confidence score

### 4. CNN Model Architecture

The pre-trained model (`model_eng_alphabets.h5`) uses the following architecture:

```
Input (28×28×1)
  → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
  → Flatten
  → Dense(64, ReLU) → Dense(128, ReLU)
  → Dense(26, Softmax)
```

Trained on the [A-Z Handwritten Alphabets Dataset](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) from Kaggle (370,000+ samples).

---

## Retraining the Model (Optional)

The pre-trained model is included — you do **not** need to retrain unless you want to improve accuracy.

### Step 1: Download the dataset

Download the [A-Z Handwritten Alphabets CSV](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) and place it at:
```
data/eng_alphabets.csv
```

### Step 2: Run training

```bash
python training/train.py True eng_alphabets
```

This will:
1. Load and split the CSV data (80% train / 20% test)
2. Display class distribution and sample images
3. Train the CNN for 10 epochs with early stopping
4. Save the trained model to `models/model_eng_alphabets.h5`

Training takes approximately 10–30 minutes depending on your hardware.

### Step 3: Inspect an existing model

```bash
python training/train.py False eng_alphabets
```

This loads and prints the model summary without retraining.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Webcam not opening** | Close other apps using the camera. Try changing `VideoCapture(0)` to `VideoCapture(1)` in `src/app.py` |
| **Hand not detected** | Ensure good lighting. Keep your hand fully visible in frame |
| **Wrong predictions** | Write letters large and in UPPERCASE style. Pause clearly between strokes |
| **Keys not responding** | Click on the OpenCV window (Live Feed) to give it focus before pressing keys |
| **TensorFlow warnings** | These are suppressed by `main.py`. If they still appear, they're harmless |
| **MediaPipe warnings** | `NORM_RECT` and `inference_feedback_manager` warnings are non-critical — ignore them |
