import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_help_menu_pil(frame):
    # Technical Proportions: 600x400
    fh, fw = frame.shape[:2]
    card_w, card_h = 600, 400
    cx, cy = fw // 2, fh // 2
    x1, y1 = cx - card_w // 2, cy - card_h // 2
    x2, y2 = cx + card_w // 2, cy + card_h // 2
    
    # Create PIL image for the overlay
    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)
    
    # Background Pill (Matte Black with soft white border)
    # RGBA: (20, 20, 20, 240)
    overlay_draw.rounded_rectangle([x1, y1, x2, y2], radius=30, fill=(20, 20, 20, 240), outline=(200, 200, 200, 255), width=1)
    
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 36)
        font_sub = ImageFont.truetype("arialbd.ttf", 24)
        font_body = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font_title = ImageFont.load_default()
        font_sub = font_title
        font_body = font_title
        
    overlay_draw.text((x1 + 40, y1 + 40), "Help Manual", font=font_title, fill=(255, 255, 255, 255))
    
    controls = [
        ("1 Finger (Index)", "Draw ink on the canvas"),
        ("2 Fingers (Peace)", "Clear the entire canvas"),
        ("3 Fingers", "Backspace last letter"),
        ("5 Fingers (Open Palm)", "Pause drawing"),
        ("Thumbs Up", "Accept text & Exit")
    ]
    
    y_pos = y1 + 100
    for gesture, action in controls:
        overlay_draw.text((x1 + 50, y_pos), gesture, font=font_sub, fill=(255, 255, 255, 255))
        overlay_draw.text((x1 + 320, y_pos), f"- {action}", font=font_body, fill=(180, 180, 180, 255))
        y_pos += 45
        
    # Merge overlay with frame
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

def draw_bottom_text_box(frame, text):
    # Technical Proportions: y1 = fh - box_h - 20
    fh, fw = frame.shape[:2]
    display_text = text if text else "Waiting for input..."
    
    try:
        font = ImageFont.truetype("arialbd.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
        
    try:
        left, top, right, bottom = font.getbbox(display_text)
        text_w = right - left
        text_h = bottom - top
    except AttributeError:
        text_w, text_h = font.getsize(display_text)
        
    box_w = max(text_w + 60, 250)
    box_h = text_h + 30
    x1 = (fw - box_w) // 2
    y1 = fh - box_h - 20
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)
    
    # Capsule Fill/Border
    overlay_draw.rounded_rectangle([x1, y1, x2, y2], radius=box_h//2, fill=(25, 25, 25, 230), outline=(180, 180, 180, 255), width=1)
    
    text_x = x1 + (box_w - text_w) // 2
    text_y = y1 + (box_h - text_h) // 2 - 4
    
    text_color = (255, 255, 255, 255) if text else (130, 130, 130, 255)
    overlay_draw.text((text_x, text_y), display_text, font=font, fill=text_color)
    
    # Merge
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

def draw_help_button(frame):
    # Technical Proportions: 20,20 to 120,60
    x1, y1 = 20, 20
    x2, y2 = 120, 60
    fh, fw = frame.shape[:2]
    
    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)
    
    # Button Pill
    overlay_draw.rounded_rectangle([x1, y1, x2, y2], radius=15, fill=(30, 30, 30, 230), outline=(180, 180, 180, 255), width=1)
    
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        font = ImageFont.load_default()
        
    overlay_draw.text((x1 + 32, y1 + 8), "Help", font=font, fill=(255, 255, 255, 255))
    
    # Merge
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    res_frame = cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    return res_frame, (x1, y1, x2, y2)

def draw_status_pill(frame, text):
    """Renders a status pill (Drawing, Paused, etc) at a fixed bottom-right position."""
    fh, fw = frame.shape[:2]
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    try:
        left, top, right, bottom = font.getbbox(text)
        tw = right - left
        th = bottom - top
    except AttributeError:
        tw, th = font.getsize(text)
        
    pill_w = tw + 30
    x1, y1 = fw - pill_w - 20, fh - 90
    x2, y2 = fw - 20, fh - 60
    
    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)
    overlay_draw.rounded_rectangle([x1, y1, x2, y2], radius=15, fill=(30, 30, 30, 230), outline=(180, 180, 180, 255), width=1)
    overlay_draw.text((x1 + 15, y1 + (30 - th)//2 - 2), text, font=font, fill=(255, 255, 255, 255))
    
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

def draw_prediction_pill(frame, text):
    """Renders the prediction score pill at the top-right."""
    fh, fw = frame.shape[:2]
    try:
        font = ImageFont.truetype("arialbd.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        
    try:
        left, top, right, bottom = font.getbbox(text)
        tw = right - left
        th = bottom - top
    except AttributeError:
        tw, th = font.getsize(text)
        
    pill_w = tw + 40
    pill_h = th + 20
    x1, y1 = fw - pill_w - 20, 20
    x2, y2 = fw - 20, 20 + pill_h
    
    overlay_img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_img)
    overlay_draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=(30, 30, 30, 230), outline=(180, 180, 180, 255), width=1)
    overlay_draw.text((x1 + 20, y1 + 10), text, font=font, fill=(255, 255, 255, 255))
    
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    frame_pil.alpha_composite(overlay_img)
    return cv2.cvtColor(np.array(frame_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
