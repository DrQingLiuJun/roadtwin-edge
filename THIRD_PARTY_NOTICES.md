# Third-party notices

`runtime/desktop_bytetrack.py` adapts Ultralytics YOLO 8.3.0 tracker, Kalman filter and matching routines, retaining the original AGPL-3.0 notice. Upstream: https://github.com/ultralytics/ultralytics/tree/v8.3.0/ultralytics/trackers . Local changes combine modules and use NumPy/ctypes integration. This project is distributed under AGPL-3.0.

`lap` 0.5.12 is installed separately under its upstream BSD-style terms; the installed runtime vendor package retains its license and package metadata. NumPy, SciPy, Pillow and OpenCV retain their respective licenses. CUDA and TensorRT are separately installed NVIDIA components and are not redistributed in this repository.

Arial is a separately licensed font. Existing devices retain their installed font; the public repository does not redistribute it. Put an appropriately licensed Arial font at `assets/fonts/Arial.ttf` before enabling previews.

Model weights and hardware-specific engines remain on the deployed devices. They are not included in this source repository; obtain authorized model weights and build engines on the target Jetson.
