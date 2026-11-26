#include "desktop_detector.hpp"
#include <cstdio>
struct DesktopOutput { int id, cls; float score, x1, y1, x2, y2; };
static thread_local char desktop_error[1024] = {0};
extern "C" {
const char* desktop_last_error() { return desktop_error; }
void* desktop_create(const char* path) {
    try { cv::setNumThreads(1); return new YoloDetector(path); }
    catch (const std::exception& e) { std::snprintf(desktop_error, sizeof(desktop_error), "%s", e.what()); return nullptr; }
}
void desktop_destroy(void* pointer) { delete static_cast<YoloDetector*>(pointer); }
int desktop_detect(void* pointer, unsigned char* pixels, int width, int height, int stride, DesktopOutput* output, int capacity) {
    try {
        auto result = static_cast<YoloDetector*>(pointer)->detect(cv::Mat(height, width, CV_8UC3, pixels, stride));
        if (static_cast<int>(result.size()) > capacity) throw std::runtime_error("Detection output capacity exceeded");
        int n = 0;
        for (const auto& box : result) output[n++] = {-1, box.class_id, box.score, box.x1, box.y1, box.x2, box.y2};
        return n;
    } catch (const std::exception& e) { std::snprintf(desktop_error, sizeof(desktop_error), "%s", e.what()); return -1; }
}
}
