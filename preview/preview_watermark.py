"""Arial preview caption below the camera's original timestamp."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).resolve().parent.parent / 'assets/fonts/Arial.ttf'

class Watermark:
    def __init__(self):
        self.fonts = {}; self.cache_key = None; self.patch = None; self.alpha = None; self.last_layout = None
        if not FONT_PATH.is_file(): raise FileNotFoundError(f'Arial font missing: {FONT_PATH}')

    def draw(self, image, text):
        h, w = image.shape[:2]; size = max(16, round(18 * w / 1280)); x, y = max(4, round(8 * w / 1920)), max(28, round(48 * h / 1080)); key = (text, size)
        if key != self.cache_key:
            if size not in self.fonts: self.fonts[size] = ImageFont.truetype(str(FONT_PATH), size)
            font = self.fonts[size]; stroke = max(1, round(size / 12)); box = font.getbbox(text, stroke_width=stroke)
            overlay = Image.new('RGBA', (box[2] - box[0] + 2, box[3] - box[1] + 2)); ImageDraw.Draw(overlay).text((1 - box[0], 1 - box[1]), text, font=font, fill=(255, 255, 255, 255), stroke_width=stroke, stroke_fill=(24, 24, 24, 220))
            pixels = np.asarray(overlay); self.patch = pixels[:, :, :3][:, :, ::-1].astype(np.float32); self.alpha = pixels[:, :, 3:4].astype(np.float32) / 255; self.cache_key = key
        text_x = x; x = max(0, x - max(1, round(size / 12)) - 1); ph, pw = min(self.patch.shape[0], h - y), min(self.patch.shape[1], w - x); roi = image[y:y + ph, x:x + pw]; alpha = self.alpha[:ph, :pw]
        roi[:] = np.rint(self.patch[:ph, :pw] * alpha + roi * (1 - alpha)).astype(np.uint8)
        self.last_layout = dict(text=text, font='Arial', font_px=size, font_pt=size * 72 / 96, x=x, text_x=text_x, y=y, width=pw, height=ph, background='transparent', color='white', outline=True)
