"""
Shared constants for the air writing application.
"""

# -- Character labels --
LABEL_MAP = dict(enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# -- Hand skeleton connections --
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                                # palm
]

# -- Drawing settings --
INK_COLOR = (0, 0, 0)
STROKE_WIDTH = 25
CANVAS_SIZE = (720, 1280)

# -- Recognition settings --
RECOGNITION_DELAY = 1.5
SMOOTHING_WINDOW = 4
MIN_CONTOUR_AREA = 500
MODEL_INPUT_DIM = 28
