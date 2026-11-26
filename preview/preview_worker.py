"""Optional, demand-driven visualization of the exact frame and exported boxes.

No inference, network requests or disk history. One pending frame per camera.
"""
from collections import deque
import json
import os
from pathlib import Path
import struct
import threading
import time

import cv2
import numpy as np
from .preview_watermark import Watermark

ROOT = Path(os.environ.get('RSU_PREVIEW_ROOT', '/dev/shm/rsu-detection-preview'))
TRAIL_SECONDS = 5.0
TRAIL_MAX_GAP = 0.5
LABELS = {0: ('Person', (100, 220, 115)), 1: ('Rider', (40, 190, 255)),
          2: ('Car', (245, 170, 80)), 7: ('Truck', (215, 110, 225))}


class PreviewPublisher:
    def __init__(self, cam, stop, model_config=None):
        self.cam = cam
        self.watermark = Watermark()
        config = model_config or json.loads((Path(__file__).resolve().parent.parent / "selected_model.json").read_text())
        self.model_name = str(config["name"])
        self.precision = str(config.get("precision", "FP16"))
        self.model_label = self.model_name + " " + self.precision
        self.stop = stop
        self.fps = max(1, min(15, float(os.environ.get('RSU_PREVIEW_FPS', '10'))))
        self.width = max(640, min(1920, int(os.environ.get('RSU_PREVIEW_WIDTH', '1280'))))
        self.quality = max(50, min(90, int(os.environ.get('RSU_PREVIEW_QUALITY', '78'))))
        ROOT.mkdir(mode=0o750, exist_ok=True)
        self.demand = ROOT / f'{cam}.demand'
        self.target = ROOT / f'{cam}.frame'
        self.pending = None
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.trails = {}
        self.trail_shape = None
        self.next_offer = 0
        self.last_error = 0
        self.thread = threading.Thread(target=self.run, name=f'preview-{cam}', daemon=True)
        self.thread.start()

    def offer(self, frame, payload):
        # The payload consists of copied Python primitives, never native tracker
        # ctypes references that the next inference would overwrite.
        now = time.monotonic()
        shape = frame.shape[:2]
        if self.trail_shape != shape:
            self.trails.clear()
            self.trail_shape = shape
        for track_id, trail in list(self.trails.items()):
            while trail and trail[0][0] < now - TRAIL_SECONDS:
                trail.popleft()
            if not trail:
                del self.trails[track_id]
        for obj in payload['objs']:
            x1, _, x2, y2 = obj['box']
            trail = self.trails.setdefault(obj['i'], deque(maxlen=512))
            if trail and now - trail[-1][0] > TRAIL_MAX_GAP:
                trail.clear()
            trail.append((now, (x1 + x2) * .5, y2))
        if now < self.next_offer:
            return
        self.next_offer = now + 1 / self.fps
        try:
            if time.time() - self.demand.stat().st_mtime > 3:
                return
        except OSError:
            return
        with self.lock:
            trails = {obj['i']: tuple(self.trails[obj['i']]) for obj in payload['objs']}
            self.pending = (frame, payload, trails, now)
        self.event.set()

    def run(self):
        while not self.stop.is_set():
            if not self.event.wait(.5):
                continue
            self.event.clear()
            with self.lock:
                item, self.pending = self.pending, None
            if item is None:
                continue
            frame, payload, trails, offered = item
            if time.monotonic() - offered > .5:
                continue
            try:
                start = time.monotonic()
                h, w = frame.shape[:2]
                scale = min(1, self.width / w)
                image = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
                ih, iw = image.shape[:2]
                drawn_trails = 0
                for obj in payload['objs']:
                    trail = trails[obj['i']]
                    if len(trail) < 2:
                        continue
                    points = np.rint(np.asarray([(x, y) for _, x, y in trail]) * scale).astype(np.int32)
                    np.clip(points, (0, 0), (iw - 1, ih - 1), out=points)
                    _, color = LABELS.get(obj['c'], ('Object', (190, 190, 190)))
                    cv2.polylines(image, [points], False, (18, 23, 29), 4, cv2.LINE_AA)
                    cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
                    drawn_trails += 1
                for obj in payload['objs']:
                    box = obj['box']
                    x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
                    x1, x2 = max(0, min(iw - 1, x1)), max(0, min(iw - 1, x2))
                    y1, y2 = max(0, min(ih - 1, y1)), max(0, min(ih - 1, y2))
                    name, color = LABELS.get(obj['c'], ('Object', (190, 190, 190)))
                    text = f"{name} #{obj['i']} {obj['score']:.0%}"
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0]
                    ty = max(th + 8, y1)
                    tx = min(x1, max(0, iw - tw - 8))
                    cv2.rectangle(image, (tx, ty - th - 8), (tx + tw + 8, ty), color, -1)
                    cv2.putText(image, text, (tx + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, .5, (15, 22, 28), 1, cv2.LINE_AA)
                self.watermark.draw(image, f"CAM {self.cam} | {self.model_label} | {len(payload['objs'])} objects")
                ok, jpg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                meta = dict(model=self.model_name, precision=self.precision, camera=self.cam, frame_seq=payload['frame_seq'], source_ms=int(payload['ts']),
                            encoded_ms=int(time.time() * 1000), width=iw, height=ih,
                            preview_fps_limit=self.fps, encode_ms=round((time.monotonic() - start) * 1000, 2),
                            objects=payload['objs'], trail_seconds=TRAIL_SECONDS, drawn_trails=drawn_trails)
                meta['watermark'] = self.watermark.last_layout
                header = json.dumps(meta, separators=(',', ':')).encode()
                temp = self.target.with_suffix(f'.{os.getpid()}.tmp')
                with temp.open('wb') as file:
                    file.write(struct.pack('!I', len(header)))
                    file.write(header)
                    file.write(jpg.tobytes())
                os.replace(temp, self.target)
            except Exception as error:
                if time.monotonic() - self.last_error > 30:
                    print(f'[Preview {self.cam}] WARNING {type(error).__name__}: {error}', flush=True)
                    self.last_error = time.monotonic()


def create_publisher(cam, stop, model_config=None):
    # An optional preview failure must never stop detection or UDP publishing.
    try:
        return PreviewPublisher(cam, stop, model_config)
    except Exception as error:
        print(f'[Preview {cam}] disabled: {error}', flush=True)
        return None
