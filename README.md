# Air Writing Recognition

A CNN-based air-writing recognition system that uses a webcam and **finger tracking** (MediaPipe) to detect and recognize English characters (A-Z) drawn in the air.

## How It Works

1. **Drawing**: MediaPipe tracks your index fingertip. The brush is stabilized by a mathematical **Exponential Moving Average (EMA)** algorithm for smooth handwriting.
2. **Auto-recognition**: When you pause drawing, the CNN evaluates the handwritten strokes. If confidence is **>80%**, it is automatically accepted and injected into the bottom Text Box UI.
3. **Accumulation**: Letters build into words, which are saved to `output/recognized_text.txt` when you exit the app.

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Gesture Controls

The application is completely hands-free and controlled via an advanced heuristic gesture engine:

| Gesture | Action |
|---------|--------|
| **1 Finger (Index)** | **Draw** ink on the canvas & push interactive buttons |
| **2 Fingers (Peace)**| **Clear** the entire canvas manually |
| **3 Fingers** | **Backspace** the last officially accepted letter |
| **5 Fingers (Palm)** | **Pause** drawing securely |
| **Thumbs Up** | **Accept & Save** your text and gracefully exit the app |

*(Note: Making a closed fist safely pauses drawing but explicitly hides any UI elements for a cleaner screen).*

## Project Structure

```
├── main.py                  # Entry point
├── src/
│   ├── app.py               # Core detection + recognition loop
│   ├── constants.py         # Shared configuration values
│   ├── ui.py                # Minimalist frosted-glass UI elements
│   └── utils.py             # Gesture engine & EMA smoothing
├── training/
│   └── train.py             # Model training pipeline (optional)
├── docs/
│   └── DOCUMENTATION.md     # Detailed step-by-step documentation
├── models/
│   ├── hand_landmarker.task # MediaPipe hand model
│   └── model_eng_alphabets.h5  # English A-Z CNN model
├── output/
│   └── recognized_text.txt  # Accumulated output
├── requirements.txt
└── README.md
```

## Documentation

See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for the full step-by-step guide.
