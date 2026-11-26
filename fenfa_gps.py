#!/usr/bin/env python3
"""Local fenfa_gps processing logic with the deployed real-time transport adapters."""
import argparse, hashlib, json, math, multiprocessing as mp, os, signal, socket, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
BASE = Path(os.environ.get('FENFA_CONFIG_ROOT', str(Path(__file__).resolve().parent)))
PACKAGE = Path(os.environ.get('FENFA_RUNTIME_ROOT', str(BASE / 'runtime')))
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(PACKAGE / 'vendor')); sys.path.insert(0, str(PACKAGE))
from desktop_bytetrack import BYTETracker
from desktop_engine import DesktopDetector, TrackerCapture
from desktop_logic import process_frame
from edge_tracker_config import load as load_tracker_config, revision
from preview.preview_worker import create_publisher
MANIFEST = json.loads((PACKAGE / 'manifest.json').read_text())
CAMERA_MAP = {int(key): value for key, value in MANIFEST['camera_map'].items()}
WIRE_CLASS_MAP = {int(key): value for key, value in MANIFEST['wire_class_map'].items()}
WANTED_CAM_IDS = json.loads((BASE / 'active_cameras.json').read_text())

def selected_model():
    config = json.loads((BASE / 'selected_model.json').read_text())
    if config.get('model_classes') != {'0': 'person', '1': 'car', '2': 'cyclist', '3': 'truck'}:
        raise RuntimeError('Desktop logic requires person/car/cyclist/truck model class order')
    if not (PACKAGE / 'libdesktop_detector.so').is_file():
        raise FileNotFoundError('Desktop detector library is missing')
    for key in ['pt', 'engine', 'library']:
        config[key] = str((BASE / config[key]).resolve())
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    if hashlib.sha256(Path(config['pt']).read_bytes()).hexdigest() != config['pt_sha256']:
        raise RuntimeError('PT 文件与所选引擎的权重记录不一致')
    if hashlib.sha256(Path(config['engine']).read_bytes()).hexdigest() != config['engine_sha256']:
        raise RuntimeError('TensorRT 引擎校验失败')
    return config

class LatestFrame:

    def __init__(self, url, stop):
        self.url = url
        self.stop = stop
        self.condition = threading.Condition()
        self.value = None
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        while not self.stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            try:
                if not cap.isOpened():
                    raise ConnectionError('视频连接失败: ' + self.url)
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    stamp = time.time() * 1000
                    with self.condition:
                        self.sequence += 1
                        self.value = (frame, stamp, self.sequence)
                        self.condition.notify_all()
            except Exception as exc:
                self.error = str(exc)
            finally:
                cap.release()
            self.stop.wait(1)

    def next(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self.stop.is_set() or (self.value is not None and self.value[2] != previous), timeout=2)
            if self.stop.is_set() or self.value is None or self.value[2] == previous:
                return None
            return self.value

def worker(camera, model_config, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN); cv2.setNumThreads(1)
    cam = int(camera['host'].split('.')[-1]); logical_id = CAMERA_MAP[cam]; detector = None; sock = None
    try:
        os.environ['RSU_MODEL_NUM_CLASSES'] = '4'; os.environ['RSU_MODEL_CLASS_MAP'] = '0,1,2,3'
        detector = DesktopDetector(model_config['engine'], PACKAGE / 'libdesktop_detector.so')
        values = load_tracker_config(); tracker_args = dict(track_buffer=values['MAX_LOST_FRAMES'], track_high_thresh=values['HIGH_THRESH'], track_low_thresh=values['LOW_THRESH'], match_thresh=values['MATCH_THRESH'], new_track_thresh=values['BIRTH_SCORE'], fuse_score=True)
        tracker = TrackerCapture(BYTETracker(SimpleNamespace(**tracker_args), frame_rate=30))
        local_camera = dict(camera, camera_id=logical_id); reader = LatestFrame('http://%s:%s%s' % (camera['host'], camera['port'], camera['path']), stop)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); preview = create_publisher(cam, stop, model_config)
        tracker_revision = revision(values); ack = BASE / f'.tracker-worker-{cam}'; temporary = ack.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(os.environ.get('INVOCATION_ID', '') + ':' + tracker_revision); temporary.replace(ack)
        print(f'[Cam {cam}] desktop_source={MANIFEST["source_sha256"][:12]} logical_id={logical_id} tracker=ultralytics-8.3.0 parameters={tracker_args}', flush=True)
        previous = 0; last_log = time.monotonic(); count = 0
        while not stop.is_set():
            current = reader.next(previous)
            if current is None: continue
            frame, stamp, previous = current; start = time.perf_counter(); queue_ms = max(0, time.time() * 1000 - stamp)
            tracker.output = np.empty((0, 8), dtype=np.float32); tracker.input_count = 0
            payload = process_frame(frame, local_camera, detector, tracker, str(int(stamp)))
            tracks = {int(row[4]): row for row in tracker.output}; valid_objects = []
            for obj in payload['objs']:
                if not np.isfinite([obj['lon'], obj['lat']]).all(): continue
                row = tracks[obj['i']]; raw_class = int(obj['c']); obj['c'] = WIRE_CLASS_MAP[raw_class]; obj['local_class'] = raw_class
                obj['score'] = round(float(row[5]), 5); obj['box'] = [round(float(value), 2) for value in row[:4]]; valid_objects.append(obj)
            payload.update(cam=cam, local_camera_id=logical_id, sensor_ip=camera['host'], objs=valid_objects, frame_seq=previous, frame_queue_ms=queue_ms, sent_ms=int(time.time() * 1000), logic_version='desktop-' + MANIFEST['source_sha256'][:12], rider_person_suppressed=max(0, detector.raw_count - tracker.input_count))
            sock.sendto(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode(), (camera['target_ip'], camera['target_port']))
            if preview: preview.offer(frame, payload)
            elapsed = (time.perf_counter() - start) * 1000; count += 1; now = time.monotonic()
            if now - last_log >= 1:
                print('[Cam %d] fps=%.2f infer_ms=%.2f queue_ms=%.2f detections=%d tracks=%d results=%d logic=desktop logical_id=%d' % (cam, count / (now - last_log), elapsed, queue_ms, detector.raw_count, len(tracker.output), len(valid_objects), logical_id), flush=True)
                last_log = now; count = 0
    except Exception as exc:
        print('[Cam %d] ERROR %s: %s' % (cam, type(exc).__name__, exc), flush=True); raise
    finally:
        if detector: detector.close()
        if sock: sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--config', type=Path, default=BASE / 'cameras_config.json')
    args = parser.parse_args()
    config = selected_model()
    cameras = json.loads(args.config.read_text())
    cameras = [c for c in cameras if int(c['host'].split('.')[-1]) in WANTED_CAM_IDS]
    if len(cameras) != len(WANTED_CAM_IDS):
        raise RuntimeError('相机配置不完整')
    print(json.dumps({'selected': config, 'cameras': [{k: c[k] for k in ['camera_id', 'host', 'target_ip', 'target_port']} for c in cameras]}, ensure_ascii=False), flush=True)
    if args.check:
        return
    import fcntl
    lock = open(BASE / '.fenfa_gps.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('fenfa_gps.py 已在运行，请使用 systemctl restart rsu-edge-vision')
    stop = mp.Event()
    shutdown_requested = False

    def shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    processes = [mp.Process(target=worker, args=(c, config, stop), name='camera-' + str(c['camera_id'])) for c in cameras]
    for p in processes:
        p.start()
    failed = False
    try:
        while not shutdown_requested and (not stop.wait(0.25)):
            if any((not p.is_alive() for p in processes)):
                failed = True
                break
    finally:
        stop.set()
        deadline = time.monotonic() + 8
        for p in processes:
            p.join(max(0, deadline - time.monotonic()))
        for p in processes:
            if p.is_alive():
                p.kill()
                p.join()
    if failed:
        raise RuntimeError('相机进程退出，将由服务重新启动')

if __name__ == '__main__': main()
