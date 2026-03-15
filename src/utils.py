"""
Utility functions for hand tracking, drawing, and character recognition.
"""

import os
import math
import numpy as np
import cv2
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python

from src.constants import (
    SKELETON_EDGES, SMOOTHING_WINDOW, MIN_CONTOUR_AREA,
    MODEL_INPUT_DIM, LABEL_MAP,
)


def apply_smoothing(raw_pt, history):
    """
    Reduce jitter on tracked fingertip using an Exponential Moving Average (EMA).
    This deeply stabilizes the stroke brush.
    """
    if not history:
        history.append(raw_pt)
        return raw_pt
        
    alpha = 0.40  # Smoothing factor. Lower = smoother but more lag.
    prev_pt = history[-1]
    
    mx = int(alpha * raw_pt[0] + (1.0 - alpha) * prev_pt[0])
    my = int(alpha * raw_pt[1] + (1.0 - alpha) * prev_pt[1])
    
    smoothed_pt = (mx, my)
    history.append(smoothed_pt)
    
    # Cap history purely for memory, EMA only needs [-1]
    while len(history) > SMOOTHING_WINDOW:
        history.popleft()
        
    return smoothed_pt


def extract_and_classify(canvas_img, classifier):
    """
    Isolate the drawn glyph from the canvas, normalize it to a square,
    resize to the model input dimensions, and run inference.
    """
    grayscale = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY_INV)

    outlines, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not outlines:
        return None, 0.0

    largest = max(outlines, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None, 0.0

    bx, by, bw, bh = cv2.boundingRect(largest)
    margin = max(bw, bh) // 10 + 5
    y_start = max(0, by - margin)
    y_end = min(binary.shape[0], by + bh + margin)
    x_start = max(0, bx - margin)
    x_end = min(binary.shape[1], bx + bw + margin)

    region = binary[y_start:y_end, x_start:x_end]
    rh, rw = region.shape

    # pad to square
    side = max(rh, rw)
    padded = np.zeros((side, side), dtype=np.uint8)
    dy, dx = (side - rh) // 2, (side - rw) // 2
    padded[dy:dy + rh, dx:dx + rw] = region

    normalized = cv2.resize(padded, (MODEL_INPUT_DIM, MODEL_INPUT_DIM))
    tensor = normalized.reshape(1, MODEL_INPUT_DIM, MODEL_INPUT_DIM, 1).astype("float32")

    probabilities = classifier.predict(tensor, verbose=0)[0]
    best_idx = int(np.argmax(probabilities))
    return LABEL_MAP[best_idx], float(probabilities[best_idx])


def init_hand_tracker():
    """Set up MediaPipe hand landmark detector in video mode."""
    task_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task"
    )
    cfg = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_file),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(cfg)


def draw_skeleton(frame, landmarks, fw, fh):
    """Render the hand wireframe on the video feed."""
    for a, b in SKELETON_EDGES:
        pa, pb = landmarks[a], landmarks[b]
        pt_a = (int(pa.x * fw), int(pa.y * fh))
        pt_b = (int(pb.x * fw), int(pb.y * fh))
        cv2.line(frame, pt_a, pt_b, (0, 200, 0), 2)


def finger_raised(landmarks, tip_idx, pip_idx):
    """Check if a finger is extended (tip above the PIP joint)."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def get_gesture(pts):
    """
    Analyzes the 21 MediaPipe landmarks and returns a string representing 
    the current recognized gesture state.
    Returns one of: 'DRAW', 'CLEAR', 'UNDO', 'HELP', 'THUMBS_UP', 'PAUSE'
    """
    # Basic "is finger pointing up" logic (Tip is higher than PIP joint)
    # For thumb, calculate absolute physical distance from index knuckle (MCP 5) to guarantee it's not tucked into a fist.
    thumb_dist = math.hypot(pts[4].x - pts[5].x, pts[4].y - pts[5].y)
    thumb_up = (pts[4].y < pts[3].y - 0.02) and (pts[3].y < pts[2].y) and (thumb_dist > 0.12)
    
    idx_up = finger_raised(pts, 8, 6)
    mid_up = finger_raised(pts, 12, 10)
    ring_up = finger_raised(pts, 16, 14)
    pinky_up = finger_raised(pts, 20, 18)
    
    # 5 Fingers (Open Palm) -> PAUSE
    if idx_up and mid_up and ring_up and pinky_up:
        return 'PAUSE'
        
    # Thumbs Up -> SAVE & EXIT
    elif thumb_up and not idx_up and not mid_up and not ring_up and not pinky_up:
        return 'THUMBS_UP'
            
    # 3 Fingers (Index, Middle, Ring) -> UNDO
    elif idx_up and mid_up and ring_up and not pinky_up:
        return 'UNDO'
        
    # 2 Fingers (Index, Middle / Peace) -> CLEAR
    elif idx_up and mid_up and not ring_up and not pinky_up:
        return 'CLEAR'
        
    # 1 Finger (Index) -> DRAW
    elif idx_up and not mid_up and not ring_up and not pinky_up:
        return 'DRAW'
        
    # Any intermediate weird state -> NONE
    return 'NONE'

def reset_strokes(stroke_data):
    """Clear all drawing state and return a fresh canvas."""
    stroke_data["segments"] = [deque(maxlen=512)]
    stroke_data["seg_idx"] = 0
    stroke_data["surface"][:, :, :] = 255
    stroke_data["jitter_buf"].clear()
    stroke_data["dirty"] = False
    stroke_data["predicted"] = False
    stroke_data["result"] = None
    stroke_data["score"] = 0.0
