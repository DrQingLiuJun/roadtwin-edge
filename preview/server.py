#!/usr/bin/env python3
"""Read-only LAN HTTP/MJPEG endpoint. Uses only the Python standard library."""
import json
import io
import html
import os
from pathlib import Path
import re
import signal
import socket
import socketserver
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from PIL import Image

WEB = Path(__file__).resolve().parent
BASE = WEB.parent
ROOT = Path(os.environ.get('RSU_PREVIEW_ROOT', '/dev/shm/rsu-detection-preview'))
CAMS = json.loads((BASE / 'active_cameras.json').read_text())
FLEET_PATH=WEB/'fleet.json'
FLEET=json.loads(FLEET_PATH.read_text()) if FLEET_PATH.is_file() else {'edges':{'local':{'url':'','cameras':CAMS}},'camera_sources':{},'portal_url':''}
LIMIT = threading.BoundedSemaphore(18)
GUARD = threading.Lock()
VIEWERS = {cam: 0 for cam in CAMS}
CACHE = {}
CACHE_LOCKS = {cam: threading.Lock() for cam in CAMS}
PROFILES = {'wall': (960, 65, 6), 'compact': (640, 60, 5), 'low': (480, 55, 3)}


def read_frame(cam, jpeg=True):
    with CACHE_LOCKS[cam]:
        path = ROOT / f'{cam}.frame';stat = path.stat();key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        cached = CACHE.get(cam)
        if not cached or cached['key'] != key:
            with path.open('rb') as file:
                length = struct.unpack('!I', file.read(4))[0]
                if not 0 < length <= 65536:raise ValueError('invalid metadata length')
                meta = json.loads(file.read(length));data = file.read()
            if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):raise ValueError('incomplete JPEG')
            cached = dict(key=key, meta=meta, data=data, profiles={});CACHE[cam] = cached
        return cached['meta'], cached['data'] if jpeg else None


def preview_frame(cam, profile):
    meta, data = read_frame(cam)
    if profile not in PROFILES:return meta, data
    with CACHE_LOCKS[cam]:
        cached = CACHE[cam]
        if cached['meta'] is not meta:return meta, data
        if profile not in cached['profiles']:
            width, quality, _ = PROFILES[profile]
            with Image.open(io.BytesIO(data)) as image:
                image.thumbnail((width, width), getattr(Image, 'Resampling', Image).BILINEAR)
                output = io.BytesIO();image.save(output, 'JPEG', quality=quality)
            cached['profiles'][profile] = output.getvalue()
        return meta, cached['profiles'][profile]


def current_model():
    config = json.loads((BASE / 'selected_model.json').read_text())
    return dict(name=config['name'], precision=config.get('precision', 'FP16'),
                input_size=config.get('input_size'), pt=Path(config['pt']).name)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32
    allow_reuse_address = True

    def server_bind(self):
        # HTTPServer resolves the local hostname during bind by default. LAN
        # machines may have no working reverse DNS, so avoid blocking startup.
        socketserver.TCPServer.server_bind(self)
        self.server_name = socket.gethostname()
        self.server_port = self.server_address[1]

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection, address


class Handler(BaseHTTPRequestHandler):
    server_version = 'RSUPreview/1.0'

    def log_message(self, fmt, *args):
        # No per-frame/access log growth. systemd keeps service failures.
        pass

    def send(self, data, typ='application/json; charset=utf-8', code=200):
        self.send_response(code)
        self.send_header('Content-Type', typ)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            self.get()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def get(self):
        path = urlsplit(self.path).path
        if path == '/preview-player.js':return self.send((WEB / 'preview-player.js').read_bytes(), 'text/javascript; charset=utf-8')
        if path in ['/', '/all'] or re.fullmatch(r'/camera/\d+', path):
            portal=FLEET.get('portal_url','').rstrip('/')
            styles='<link rel="icon" href="'+html.escape(portal,quote=True)+'/favicon.png"><link rel="stylesheet" href="'+html.escape(portal,quote=True)+'/readable-fonts.css">' if portal else ''
            data=(WEB/'preview.html').read_text().replace('__LOCAL_CAMERAS__',json.dumps(CAMS)).replace('__FLEET_CONFIG__',json.dumps(FLEET).replace('<','\\u003c')).replace('__PORTAL_STYLES__',styles)
            return self.send(data.encode(), 'text/html; charset=utf-8')
        if path == '/api/status':
            model = current_model()
            status = []
            for cam in CAMS:
                row = dict(camera=cam, viewers=VIEWERS[cam], state='idle')
                try:
                    meta, _ = read_frame(cam, False)
                    age = time.time() - meta['source_ms'] / 1000
                    row.update(model=meta.get('model'), precision=meta.get('precision'), frame_seq=meta['frame_seq'], age_ms=round(age * 1000),
                               width=meta['width'], height=meta['height'], objects=len(meta['objects']),
                               encode_ms=meta['encode_ms'], preview_fps_limit=meta['preview_fps_limit'])
                    row['state'] = ('live' if age < 2 and meta.get('model') == model['name'] else 'stale') if row['viewers'] else 'idle'
                except (OSError, ValueError, KeyError, struct.error):
                    if row['viewers']:
                        row['state'] = 'waiting'
                status.append(row)
            return self.send(json.dumps(dict(model=model, cameras=status, server_ms=int(time.time() * 1000))).encode())
        match = re.fullmatch(r'/stream/(\d+)\.mjpg', path)
        if match and int(match.group(1)) in CAMS:
            return self.stream(int(match.group(1)), parse_qs(urlsplit(self.path).query).get('profile', ['native'])[0])
        match = re.fullmatch(r'/snapshot/(\d+)\.jpg', path)
        if match and int(match.group(1)) in CAMS:
            cam = int(match.group(1))
            (ROOT / f'{cam}.demand').touch()
            until = time.monotonic() + 3
            while time.monotonic() < until:
                try:
                    meta, data = read_frame(cam)
                    if time.time() - meta['source_ms'] / 1000 < 2 and meta.get('model') == current_model()['name']:
                        return self.send(data, 'image/jpeg')
                except (OSError, ValueError, struct.error):
                    pass
                time.sleep(.05)
            return self.send(b'No fresh detection frame', 'text/plain', 503)
        return self.send(b'Not found', 'text/plain', 404)

    def stream(self, cam, profile='native'):
        if not LIMIT.acquire(blocking=False):
            return self.send(b'Too many viewers', 'text/plain', 503)
        with GUARD:
            VIEWERS[cam] += 1
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=rsuframe')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            selected_name = current_model()['name']
            previous = None
            last_fresh = time.monotonic()
            touch_at = 0
            next_frame = 0;interval = 1 / PROFILES[profile][2] if profile in PROFILES else 0
            while True:
                now = time.monotonic()
                if now >= touch_at:
                    (ROOT / f'{cam}.demand').touch()
                    touch_at = now + .5
                try:
                    meta, _ = read_frame(cam, False)
                    key = (meta['source_ms'], meta['frame_seq'])
                    fresh = time.time() - meta['source_ms'] / 1000 < 2 and meta.get('model') == selected_name
                    if fresh and key != previous and now >= next_frame:
                        meta, data = preview_frame(cam, profile)
                        key = (meta['source_ms'], meta['frame_seq'])
                        self.wfile.write(b'--rsuframe\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(data)).encode() +
                                         b'\r\nX-Frame-Sequence: ' + str(meta['frame_seq']).encode() + b'\r\nX-Source-Milliseconds: ' +
                                         str(meta['source_ms']).encode() + b'\r\nX-Objects: ' + str(len(meta['objects'])).encode() + b'\r\n\r\n' + data + b'\r\n')
                        self.wfile.flush()
                        previous = key
                        last_fresh = now
                        next_frame = time.monotonic() + interval
                except (FileNotFoundError, ValueError, struct.error):
                    pass
                if now - last_fresh > 8:
                    break
                time.sleep(.025)
        finally:
            with GUARD:
                VIEWERS[cam] -= 1
            LIMIT.release()


if __name__ == '__main__':
    ROOT.mkdir(mode=0o750, exist_ok=True)
    server = Server(('0.0.0.0', 8083), Handler)
    def shutdown(signum, frame):
        # Signal handler runs on the serve_forever thread; shutdown must not.
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f'Detection preview listening on :8083; cameras={CAMS}; read-only', flush=True)
    try:
        server.serve_forever(poll_interval=.3)
    finally:
        server.server_close()
