"""
Core application loop for real-time air writing detection and recognition.
Combines hand tracking with a pre-trained CNN to convert air-drawn gestures into text.
"""

import os
import time
import numpy as np
import cv2
from collections import deque
import mediapipe as mp
import tensorflow as tf

from src.constants import (
    INK_COLOR, STROKE_WIDTH, CANVAS_SIZE, RECOGNITION_DELAY, SMOOTHING_WINDOW,
)
from src.utils import (
    apply_smoothing, extract_and_classify, init_hand_tracker,
    draw_skeleton, reset_strokes, get_gesture
)
from src.ui import (
    draw_help_menu_pil, draw_bottom_text_box, draw_help_button,
    draw_status_pill, draw_prediction_pill
)


def launch():
    """Run the real-time air writing application."""
    os.makedirs("output", exist_ok=True)

    # load classifier once
    print("Loading character model...", end=" ", flush=True)
    classifier = tf.keras.models.load_model("models/model_eng_alphabets.h5")
    print("ready.")

    tracker = init_hand_tracker()
    camera = cv2.VideoCapture(0)
    
    # Request higher resolution to accommodate future text boxes
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # drawing state
    state = {
        "segments": [deque(maxlen=512)],
        "seg_idx": 0,
        "surface": np.full((*CANVAS_SIZE, 3), 255, dtype=np.uint8),
        "jitter_buf": deque(maxlen=SMOOTHING_WINDOW),
        "dirty": False,
        "predicted": False,
        "result": None,
        "score": 0.0,
    }

    output_chars = []
    pen_active = False
    last_stroke_ts = 0.0
    last_erase_ts = 0.0

    cv2.namedWindow("Canvas", cv2.WINDOW_AUTOSIZE)

    while True:
        ok, raw_frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(raw_frame, 1)
        fh, fw = frame.shape[:2]

        # -- hand detection --
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = tracker.detect_for_video(mp_img, int(time.time() * 1000))

        # -- Help Button UI --
        frame, (hx1, hy1, hx2, hy2) = draw_help_button(frame)

        # -- HUD: Live Prediction (Minimalist) --
        if state["predicted"] and state["result"]:
            pred_text = f"Prediction: {state['result']} ({state['score']:.0%})"
            frame = draw_prediction_pill(frame, pred_text)

        # -- process hand landmarks --
        if detection.hand_landmarks:
            pts = detection.hand_landmarks[0]
            draw_skeleton(frame, pts, fw, fh)

            tip_x = int(pts[8].x * fw)
            tip_y = int(pts[8].y * fh)
            cv2.circle(frame, (tip_x, tip_y), 12, (0, 255, 255), -1)

            gesture = get_gesture(pts)
            
            # Interactive Button collision check
            if gesture == 'DRAW' and hx1 <= tip_x <= hx2 and hy1 <= tip_y <= hy2:
                gesture = 'HELP'

            # ---------------------------------------------
            # STATE 1: DRAW (Index Finger)
            # ---------------------------------------------
            if gesture == 'DRAW':
                frame = draw_status_pill(frame, "Drawing")

                smooth_pt = apply_smoothing((tip_x, tip_y), state["jitter_buf"])

                if tip_y > 65:
                    state["segments"][state["seg_idx"]].appendleft(smooth_pt)
                    state["dirty"] = True
                    state["predicted"] = False
                    state["result"] = None

                pen_active = True
                last_stroke_ts = time.time()
                
            # ---------------------------------------------
            # STATE 2: UNDO (3 Fingers)
            # ---------------------------------------------
            elif gesture == 'UNDO':
                frame = draw_status_pill(frame, "Backspace")
                
                # Backspace logic: remove last accepted letter instead of undoing stroke
                if time.time() - last_erase_ts > 0.6:
                    if output_chars:
                        removed = output_chars.pop()
                        print(f">>> Backspace: Removed '{removed}'  |  Text: {''.join(output_chars)}")
                    
                    reset_strokes(state)
                    last_erase_ts = time.time()
                
                # Ensure we end pen_active safely
                if pen_active:
                    state["segments"].append(deque(maxlen=512))
                    state["seg_idx"] += 1
                    state["jitter_buf"].clear()
                pen_active = False
                last_stroke_ts = time.time()

            # ---------------------------------------------
            # STATE 3: CLEAR (2 Fingers / Peace)
            # ---------------------------------------------
            elif gesture == 'CLEAR':
                frame = draw_status_pill(frame, "Clearing")
                
                if time.time() - last_erase_ts > 0.5:
                    reset_strokes(state)
                    last_erase_ts = time.time()
                    
                if pen_active:
                    state["segments"].append(deque(maxlen=512))
                    state["seg_idx"] += 1
                    state["jitter_buf"].clear()
                pen_active = False
                last_stroke_ts = time.time()
                
            # ---------------------------------------------
            # STATE 4: HELP MENU (5 Fingers / Open Palm)
            # ---------------------------------------------
            elif gesture == 'HELP':
                frame = draw_help_menu_pil(frame)
                
                if pen_active:
                    state["segments"].append(deque(maxlen=512))
                    state["seg_idx"] += 1
                    state["jitter_buf"].clear()
                pen_active = False
                last_stroke_ts = time.time()
                
            # ---------------------------------------------
            # STATE 5: SAVE & EXIT (Thumbs Up)
            # ---------------------------------------------
            elif gesture == 'THUMBS_UP':
                exit_w, exit_h = get_text_size_pil("Saving...", font_size=28)
                cx, cy = fw // 2, fh // 2
                x1, y1 = cx - exit_w // 2 - 30, cy - exit_h // 2 - 20
                x2, y2 = cx + exit_w // 2 + 30, cy + exit_h // 2 + 20
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                frame = draw_text_with_pil(frame, "Saving...", (cx - exit_w // 2, cy - exit_h // 2), 
                                           font_size=28, color=(255, 255, 255))
                                           
                cv2.imshow("Live Feed", frame)
                cv2.waitKey(1)
                time.sleep(1.0)
                break
                
            # ---------------------------------------------
            # STATE 6: PAUSED (Open Palm)
            # ---------------------------------------------
            elif gesture == 'PAUSE':
                frame = draw_status_pill(frame, "Paused")
                if pen_active:
                    state["segments"].append(deque(maxlen=512))
                    state["seg_idx"] += 1
                    state["jitter_buf"].clear()
                pen_active = False
                
            # ---------------------------------------------
            # STATE 7: NONE (Closed Fist or Unknown)
            # ---------------------------------------------
            else:
                if pen_active:
                    state["segments"].append(deque(maxlen=512))
                    state["seg_idx"] += 1
                    state["jitter_buf"].clear()
                pen_active = False

        else:
            if pen_active:
                state["segments"].append(deque(maxlen=512))
                state["seg_idx"] += 1
                state["jitter_buf"].clear()
            pen_active = False

        # -- auto-recognition on pause --
        if (state["dirty"]
                and not pen_active
                and not state["predicted"]
                and time.time() - last_stroke_ts > RECOGNITION_DELAY):
            char, conf = extract_and_classify(state["surface"], classifier)
            if char:
                # ---------------------------------------------
                # THE AUTO-ACCEPT ENGINE (>80% confidence)
                # ---------------------------------------------
                if conf >= 0.80:
                    output_chars.append(char)
                    print(f">>> Auto-Accepted: '{char}' ({conf:.0%})  |  Text: {''.join(output_chars)}")
                    
                    # Visibly show the accepted letter before wiping
                    pred_text = f"Prediction: {char} ({conf:.0%}) [AUTO]"
                    frame = draw_prediction_pill(frame, pred_text)
                    
                    # Force render the updated text box
                    frame = draw_bottom_text_box(frame, "".join(output_chars))
                    cv2.imshow("Live Feed", frame)
                    cv2.waitKey(1)
                    time.sleep(0.5)
                    
                    reset_strokes(state)
                else:
                    # Just show the prediction pill
                    state["result"] = char
                    state["score"] = conf
                    state["predicted"] = True

        # -- render ink strokes --
        for seg in state["segments"]:
            prev = None
            for pt in seg:
                if prev is not None:
                    cv2.line(frame, prev, pt, INK_COLOR, STROKE_WIDTH)
                    cv2.line(state["surface"], prev, pt, INK_COLOR, STROKE_WIDTH)
                prev = pt

        # -- Render Final Accepted Word Box --
        frame = draw_bottom_text_box(frame, "".join(output_chars))

        cv2.imshow("Live Feed", frame)
        cv2.imshow("Canvas", state["surface"])

        # -- keyboard input (Fallbacks only) --
        pressed = cv2.waitKey(1) & 0xFF
        if pressed == ord("q"):
            break

    # -- teardown --
    tracker.close()
    camera.release()
    cv2.destroyAllWindows()

    if output_chars:
        final = "".join(output_chars)
        with open("output/recognized_text.txt", "w") as fout:
            fout.write(final + "\n")
        print(f"\n{'=' * 40}")
        print(f"Final text: {final}")
        print(f"Saved to: output/recognized_text.txt")
        print(f"{'=' * 40}")
    else:
        print("\nNo characters recognized.")
