import ctypes as C
from types import SimpleNamespace
import numpy as np

class Output(C.Structure):
    _fields_ = [('id', C.c_int), ('cls', C.c_int)] + [(name, C.c_float) for name in ['score', 'x1', 'y1', 'x2', 'y2']]

class ArrayView:
    def __init__(self, value): self.value = value
    def cpu(self): return self
    def numpy(self): return self.value

class Boxes:
    def __init__(self, values):
        self.xyxy = ArrayView(values[:, :4]); self.conf = ArrayView(values[:, 4]); self.cls = ArrayView(values[:, 5]); self.count = len(values)
    def __len__(self): return self.count

class DesktopDetector:
    def __init__(self, engine, library):
        self.library = C.CDLL(str(library)); lib = self.library; self.handle = None; self.raw_count = 0
        lib.desktop_last_error.restype = C.c_char_p; lib.desktop_create.argtypes = [C.c_char_p]; lib.desktop_create.restype = C.c_void_p
        lib.desktop_destroy.argtypes = [C.c_void_p]; lib.desktop_detect.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(Output), C.c_int]; lib.desktop_detect.restype = C.c_int
        self.handle = lib.desktop_create(str(engine).encode()); self.output = (Output * 1024)()
        if not self.handle: raise RuntimeError(lib.desktop_last_error().decode())
    def predict(self, frame, **kwargs):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        n = self.library.desktop_detect(self.handle, frame.ctypes.data, frame.shape[1], frame.shape[0], frame.strides[0], self.output, len(self.output))
        if n < 0: raise RuntimeError(self.library.desktop_last_error().decode())
        self.raw_count = n; values = np.asarray([[t.x1, t.y1, t.x2, t.y2, t.score, t.cls] for t in self.output[:n]], dtype=np.float32).reshape(-1, 6)
        return [SimpleNamespace(boxes=Boxes(values))]
    def close(self):
        if self.handle: self.library.desktop_destroy(self.handle); self.handle = None

class TrackerCapture:
    def __init__(self, tracker): self.tracker = tracker; self.output = np.empty((0, 8), dtype=np.float32); self.input_count = 0
    def update(self, detections): self.input_count = len(detections); self.output = self.tracker.update(detections); return self.output
