"""Extracted from the user desktop script; algorithm statements are preserved."""
import numpy as np
TARGET_CLASSES = [0, 1, 2, 3]
CLASS_NAMES = {0: 'person', 1: 'car', 2: 'cyclist', 3: 'truck'}
CROSS_CLASS_IOU_DEDUP = True
IOU_DEDUP_THRESHOLD = 0.75
DEDUP_CLASS_PAIRS = [(0, 2)]
CLASS_COMPENSATION_LOCAL = {0: (0.0, 0), 1: (0.0, 2.5), 2: (0.0, 1.5), 3: (0.0, 0.0)}
CAMERA_CALIBRATION = {1: (1.0, 0.0, 1, 0.0), 2: (1.0267, 1.0326, 0.997, 1.3542), 3: (1.035, -1.7901, 1.0764, 2.1312), 4: (1.0264, -1.9884, 0.9983, 0.8141), 5: (1.0, 0.0, 1, 0.0), 6: (1.0, 0.0, 1.21, 0.0), 7: (1.0, 0.0, 1, 0.0), 8: (1.0, 0.0, 1, 0.0), 9: (1.0, 0.0, 1, 0.0)}

class DetectionsWrapper:

    def __init__(self, xyxy, conf, cls):
        self._xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    @property
    def xywh(self):
        x1, y1, x2, y2 = (self._xyxy[:, 0], self._xyxy[:, 1], self._xyxy[:, 2], self._xyxy[:, 3])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.column_stack([cx, cy, w, h])

    @property
    def xyxy(self):
        return self._xyxy

    def __len__(self):
        return len(self._xyxy)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return SingleDetection(self._xyxy[idx], self.conf[idx], self.cls[idx], self.xywh[idx], int(idx))
        return DetectionsWrapper(self._xyxy[idx], self.conf[idx], self.cls[idx])

class SingleDetection:
    __slots__ = ('xyxy', 'conf', 'cls', 'xywh', 'idx')

    def __init__(self, xyxy, conf, cls, xywh, idx):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.xywh = xywh
        self.idx = idx

def iou(boxA, boxB):
    xA, yA = (max(boxA[0], boxB[0]), max(boxA[1], boxB[1]))
    xB, yB = (min(boxA[2], boxB[2]), min(boxA[3], boxB[3]))
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = float(boxAArea + boxBArea - interArea)
    return interArea / union if union > 0 else 0.0

def deduplicate_detections(boxes_data):
    if not CROSS_CLASS_IOU_DEDUP or len(boxes_data['boxes']) == 0:
        return np.arange(len(boxes_data['boxes']))
    boxes = boxes_data['boxes']
    cls_ids = boxes_data['cls'].astype(int)
    confs = boxes_data['conf']
    to_delete = set()
    for i in range(len(boxes)):
        if i in to_delete:
            continue
        for j in range(i + 1, len(boxes)):
            if j in to_delete:
                continue
            ci, cj = (cls_ids[i], cls_ids[j])
            if not any((ci == a and cj == b or (ci == b and cj == a) for a, b in DEDUP_CLASS_PAIRS)):
                continue
            if iou(boxes[i], boxes[j]) >= IOU_DEDUP_THRESHOLD:
                to_delete.add(j if confs[i] >= confs[j] else i)
    return np.array([i for i in range(len(boxes)) if i not in to_delete])

def pixel_to_gps(u_pix, v_pix, camera_config, cam_id, cls_id):
    if camera_config is None:
        return (None, None)
    calib = camera_config['calibration']
    t1 = np.array(calib['matrix'])
    degree = calib.get('north_angle_deg', 0)
    r_rad = np.radians(degree)
    m = np.dot(t1, np.array([u_pix, v_pix, 1]))
    if m[2] == 0:
        return (None, None)
    u_world, v_world = (m[0] / m[2], m[1] / m[2])
    if cam_id in CAMERA_CALIBRATION:
        xs, xo, ys, yo = CAMERA_CALIBRATION[cam_id]
        lx = u_world * np.cos(r_rad) - v_world * np.sin(r_rad)
        ly = u_world * np.sin(r_rad) + v_world * np.cos(r_rad)
        lx = lx * xs + xo
        ly = ly * ys + yo
        u_world = lx * np.cos(r_rad) + ly * np.sin(r_rad)
        v_world = -lx * np.sin(r_rad) + ly * np.cos(r_rad)
    comp_x, comp_y = CLASS_COMPENSATION_LOCAL.get(int(cls_id), (0, 0))
    comp_e = comp_x * np.cos(r_rad) + comp_y * np.sin(r_rad)
    comp_n = -comp_x * np.sin(r_rad) + comp_y * np.cos(r_rad)
    u_world += comp_e
    v_world += comp_n
    gps = calib['gps']
    R = 6378137.0
    dlat = 180.0 / (np.pi * R)
    dlon = 180.0 / (np.pi * R * np.cos(np.radians(gps['lat'])))
    return (gps['lat'] + v_world * dlat, gps['lon'] + u_world * dlon)

def process_frame(frame, camera_config, model, tracker, timestamp):
    cam_id = camera_config['camera_id']
    payload = {'cam': cam_id, 'ts': timestamp, 'objs': []}
    results = model.predict(frame, classes=TARGET_CLASSES, verbose=False, conf=0.25, imgsz=640)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return payload
    boxes_raw = results[0].boxes.xyxy.cpu().numpy()
    cls_raw = results[0].boxes.cls.cpu().numpy()
    conf_raw = results[0].boxes.conf.cpu().numpy()
    keep_idx = deduplicate_detections({'boxes': boxes_raw, 'cls': cls_raw, 'conf': conf_raw})
    if len(keep_idx) == 0:
        return payload
    dets = DetectionsWrapper(boxes_raw[keep_idx], conf_raw[keep_idx], cls_raw[keep_idx])
    tracked = tracker.update(dets)
    if tracked is None or len(tracked) == 0:
        return payload
    for det in tracked:
        x1, y1, x2, y2, tid, score, cid, _ = det
        tid, cid = (int(tid), int(cid))
        lat, lon = pixel_to_gps((x1 + x2) / 2, y2, camera_config, cam_id, cid)
        if lat is None:
            continue
        payload['objs'].append({'i': tid, 'c': cid, 'lon': round(float(lon), 8), 'lat': round(float(lat), 8)})
    return payload
