"""
Entry point for the air writing recognition application.
Configures the runtime environment and launches the main application.
"""

import os

# configure tensorflow logging before any tf imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from src.app import launch


def main():
    banner = [
        "=" * 50,
        "  Air Writing Recognition (English A-Z)",
        "=" * 50,
        "",
        "Gestures:",
        "  Point index finger  →  Draw",
        "  Raise all fingers   →  Lift pen",
        "  Pause drawing       →  Auto-predict",
        "",
        "Keys:",
        "  z = Accept    c = Clear    q = Save & Exit",
        "",
    ]
    print("\n".join(banner))
    launch()


if __name__ == "__main__":
    main()
