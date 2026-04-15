import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Font helpers — try Segoe UI (premium Windows font) then fallback to Arial
# ---------------------------------------------------------------------------
_FONT_PATHS = [
    "C:/Windows/Fonts/segoeui.ttf",   # Segoe UI Regular
    "C:/Windows/Fonts/segoeuib.ttf",  # Segoe UI Bold
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
_FONT_BOLD_PATHS = [
    "C:/Windows/Fonts/segoeuib.ttf",  # Segoe UI Bold
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load the best available font at the given point size."""
    paths = _FONT_BOLD_PATHS if bold else _FONT_PATHS
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _get_text_size(font, text):
    """Return (width, height) of text rendered with font."""
    try:
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    except AttributeError:
        return font.getsize(text)


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def draw_help_menu_pil(frame):
    """Render the Help overlay card in the center of the frame."""
    fh, fw = frame.shape[:2]
    card_w, card_h = 620, 420
    cx, cy = fw // 2, fh // 2
    x1, y1 = cx - card_w // 2, cy - card_h // 2
    x2, y2 = cx + card_w // 2, cy + card_h // 2

    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)

    # Card background
    overlay_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=28,
        fill=(18, 18, 22, 245), outline=(80, 80, 100, 255), width=1
    )

    font_title = _load_font(34, bold=True)
    font_gesture = _load_font(22, bold=True)
    font_action = _load_font(19, bold=False)

    overlay_draw.text((x1 + 40, y1 + 36), "✋  Gesture Guide", font=font_title, fill=(255, 255, 255, 255))

    # Separator line
    overlay_draw.line([(x1 + 30, y1 + 90), (x2 - 30, y1 + 90)], fill=(60, 60, 80, 200), width=1)

    controls = [
        ("☝️  1 Finger (Index)",     "Draw ink on the canvas"),
        ("✌️  2 Fingers (Peace)",     "Clear the entire canvas"),
        ("🤟  3 Fingers",             "Backspace last letter"),
        ("🖐  Open Palm",             "Pause drawing / Lift pen"),
        ("👍  Thumbs Up (hold)",      "Save text & Exit"),
    ]

    y_pos = y1 + 108
    for gesture, action in controls:
        overlay_draw.text((x1 + 44, y_pos), gesture, font=font_gesture, fill=(220, 220, 255, 255))
        overlay_draw.text((x1 + 330, y_pos + 2), f"→  {action}", font=font_action, fill=(160, 165, 190, 255))
        y_pos += 52

    # Merge overlay with frame
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)


def draw_bottom_text_box(frame, text):
    """Render the recognised-text capsule at the bottom of the frame."""
    fh, fw = frame.shape[:2]
    display_text = text if text else "Waiting for input..."

    font = _load_font(36, bold=True)
    text_w, text_h = _get_text_size(font, display_text)

    box_w = max(text_w + 70, 280)
    box_h = text_h + 34
    x1 = (fw - box_w) // 2
    y1 = fh - box_h - 24
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)

    overlay_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=box_h // 2,
        fill=(16, 16, 20, 235), outline=(100, 110, 160, 255), width=1
    )

    text_x = x1 + (box_w - text_w) // 2
    text_y = y1 + (box_h - text_h) // 2 - 3

    text_color = (255, 255, 255, 255) if text else (110, 115, 140, 255)
    overlay_draw.text((text_x, text_y), display_text, font=font, fill=text_color)

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)


def draw_help_button(frame):
    """Render the Help button pill at top-left and return its bounding box."""
    x1, y1 = 20, 20
    x2, y2 = 130, 62
    fh, fw = frame.shape[:2]

    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)

    overlay_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=16,
        fill=(30, 30, 38, 230), outline=(90, 95, 130, 255), width=1
    )

    font = _load_font(22, bold=True)
    label = "? Help"
    lw, lh = _get_text_size(font, label)
    overlay_draw.text((x1 + (x2 - x1 - lw) // 2, y1 + (y2 - y1 - lh) // 2 - 1),
                      label, font=font, fill=(220, 220, 255, 255))

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    res_frame = cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

    return res_frame, (x1, y1, x2, y2)


def draw_status_pill(frame, text):
    """Render a status pill (Drawing / Paused / etc.) at bottom-right."""
    fh, fw = frame.shape[:2]
    font = _load_font(20, bold=True)
    tw, th = _get_text_size(font, text)

    pill_w = tw + 32
    pill_h = max(th + 18, 36)
    x1 = fw - pill_w - 20
    y1 = fh - pill_h - 90
    x2 = fw - 20
    y2 = y1 + pill_h

    # Colour coding
    colour_map = {
        "Drawing":   (30, 120, 60, 220),
        "Paused":    (80, 60, 30, 220),
        "Clearing":  (120, 30, 30, 220),
        "Backspace": (80, 30, 80, 220),
    }
    fill = colour_map.get(text, (30, 30, 38, 220))

    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)
    overlay_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=pill_h // 2,
        fill=fill, outline=(180, 185, 210, 200), width=1
    )
    overlay_draw.text(
        (x1 + (pill_w - tw) // 2, y1 + (pill_h - th) // 2 - 1),
        text, font=font, fill=(255, 255, 255, 255)
    )

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)


def draw_prediction_pill(frame, text):
    """Render the prediction label pill at top-right."""
    fh, fw = frame.shape[:2]
    font = _load_font(23, bold=True)
    tw, th = _get_text_size(font, text)

    pill_w = tw + 44
    pill_h = th + 22
    x1 = fw - pill_w - 20
    y1 = 20
    x2 = fw - 20
    y2 = 20 + pill_h

    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)
    overlay_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=12,
        fill=(20, 60, 110, 230), outline=(80, 140, 220, 255), width=1
    )
    overlay_draw.text((x1 + 22, y1 + 11), text, font=font, fill=(200, 225, 255, 255))

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)


def draw_save_overlay(frame, progress: float, saving: bool = False):
    """
    Render the save-confirmation overlay while the user holds Thumbs Up.

    Parameters
    ----------
    frame    : BGR numpy array from OpenCV
    progress : float in [0.0, 1.0] — how far through the hold the user is
    saving   : if True, show the final "Saved!" message
    """
    fh, fw = frame.shape[:2]

    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)

    font_large = _load_font(40, bold=True)
    font_sub   = _load_font(22, bold=False)

    # Card dimensions
    card_w, card_h = 460, 180
    cx, cy = fw // 2, fh // 2
    x1, y1 = cx - card_w // 2, cy - card_h // 2
    x2, y2 = cx + card_w // 2, cy + card_h // 2

    # Background card
    overlay_draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=24,
        fill=(12, 14, 20, 240), outline=(60, 80, 160, 255), width=2
    )

    if saving:
        title = "✅  Saved!"
        sub   = "Text written to output/recognized_text.txt"
        title_color = (100, 240, 140, 255)
    else:
        title = "👍  Hold to Save…"
        pct = int(progress * 100)
        sub = f"Keep thumb raised — {pct}%"
        title_color = (200, 210, 255, 255)

    # Title text
    tw, th = _get_text_size(font_large, title)
    overlay_draw.text(
        (cx - tw // 2, y1 + 28),
        title, font=font_large, fill=title_color
    )

    # Sub text
    sw, sh = _get_text_size(font_sub, sub)
    overlay_draw.text(
        (cx - sw // 2, y1 + 28 + th + 10),
        sub, font=font_sub, fill=(160, 165, 200, 220)
    )

    # Progress bar
    if not saving:
        bar_x1 = x1 + 40
        bar_y1 = y2 - 36
        bar_x2 = x2 - 40
        bar_y2 = y2 - 16

        # Track
        overlay_draw.rounded_rectangle(
            [bar_x1, bar_y1, bar_x2, bar_y2], radius=8,
            fill=(40, 40, 55, 255)
        )
        # Fill
        fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * progress)
        if fill_x2 > bar_x1 + 16:
            overlay_draw.rounded_rectangle(
                [bar_x1, bar_y1, fill_x2, bar_y2], radius=8,
                fill=(80, 160, 255, 255)
            )

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
