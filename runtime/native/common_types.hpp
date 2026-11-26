#pragma once
#include <vector>
const std::vector<int> TARGET_CLASSES = {0, 1, 2, 3};
struct DetectedObj { int track_id, class_id; float x_pixel, y_pixel, x1, y1, x2, y2, score; double lat, lon; };
