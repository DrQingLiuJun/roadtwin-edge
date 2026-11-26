#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
opencv_prefix="${OPENCV_PREFIX:-/usr}"
if [[ -f /usr/local/opencv_cuda/include/opencv4/opencv2/opencv.hpp && -z "${OPENCV_PREFIX:-}" ]]; then opencv_prefix=/usr/local/opencv_cuda; fi
cuda_prefix="${CUDA_PREFIX:-/usr/local/cuda}"
output="$project_root/runtime/libdesktop_detector.so"
temporary="${output}.building"
trap 'rm -f -- "$temporary"' EXIT
g++ -std=c++17 -O2 -shared -fPIC -I"$project_root/runtime/native" -I"$opencv_prefix/include/opencv4" -I"$cuda_prefix/include" "$project_root/runtime/native/desktop_detector.cpp" -L"$opencv_prefix/lib" -L"$cuda_prefix/lib64" -Wl,-rpath,"$opencv_prefix/lib:$cuda_prefix/lib64" -lopencv_core -lopencv_imgproc -lopencv_dnn -lcudart -lnvinfer -lpthread -o "$temporary"
mv -- "$temporary" "$output"
echo "Built $output"
