#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <sstream>

#include "common_types.hpp"
#include "logger.hpp"

/**
 * @brief YOLOv11 检测器 (TensorRT 后端)
 *
 * 支持 TensorRT engine 输出为 [1, 84, 8400] 或 [1, 8400, 84]
 *   - 84 = 4 (bbox) + 80 (classes)
 *   - 8400 = 预测框数量
 */
class YoloDetector {
public:
    // 配置参数
    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;
    static constexpr float CONF_THRESHOLD = 0.25f;
    static constexpr float NMS_THRESHOLD = 0.70f;
    static constexpr int NUM_CLASSES = 80;  // COCO 80类

    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };

    YoloDetector(const std::string& engine_path, bool use_cuda = true)
        : use_cuda_(use_cuda) {
        // Raw model labels are adapted once, before tracking/GPS/UDP output.
        if(const char* count=std::getenv("RSU_MODEL_NUM_CLASSES")) num_classes_=std::stoi(count);
        if(num_classes_!=4 && num_classes_!=80) throw std::runtime_error("Only verified 4-class/COCO-80 models are supported");
        if(const char* mapping=std::getenv("RSU_MODEL_CLASS_MAP")) {
            std::istringstream input(mapping);std::string item;
            while(std::getline(input,item,',')) class_map_.push_back(std::stoi(item));
            if(static_cast<int>(class_map_.size())!=num_classes_)throw std::runtime_error("Class mapping length differs from model classes");
            for(int value:class_map_)if(value<0 || value>=80)throw std::runtime_error("Invalid outgoing COCO class");
        } else {
            if(num_classes_!=80)throw std::runtime_error("Custom models require an explicit class mapping");
            class_map_.resize(80);std::iota(class_map_.begin(),class_map_.end(),0);
        }
        LOG_INFO("YOLO", "加载 TensorRT 引擎: " + engine_path);
        loadEngine(engine_path);
        LOG_INFO("YOLO", "引擎加载完成");
    }

    ~YoloDetector() {
        if (buffers_[0]) {
            cudaFree(buffers_[0]);
            buffers_[0] = nullptr;
        }
        if (buffers_[1]) {
            cudaFree(buffers_[1]);
            buffers_[1] = nullptr;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    /**
     * @brief 检测图像中的目标
     * @param frame 输入图像 (BGR格式)
     * @return 检测到的目标列表
     */
    std::vector<DetectedObj> detect(const cv::Mat& frame) {
        // 注意: 每个相机线程拥有独立的 YoloDetector 实例，无需加锁
        std::vector<DetectedObj> results;

        if (frame.empty() || !context_) return results;

        // 1. 预处理 - Letterbox 缩放保持宽高比
        cv::Mat input_blob;
        float scale;
        int dw, dh;
        preprocess(frame, input_blob, scale, dw, dh);

        // 2. 拷贝输入到 GPU
        const size_t input_bytes = input_size_ * sizeof(float);
        checkCuda(cudaMemcpyAsync(buffers_[input_index_], input_blob.ptr<float>(), input_bytes,
                        cudaMemcpyHostToDevice, stream_), "copy input");

        // 3. 推理
        void* bindings[2];
        bindings[input_index_] = buffers_[input_index_];
        bindings[output_index_] = buffers_[output_index_];

        bool ok = context_->enqueueV2(bindings, stream_, nullptr);
        if (!ok) {
            LOG_ERROR("YOLO", "TensorRT 推理失败");
            return results;
        }

        // 4. 拷贝输出回 CPU
        const size_t output_bytes = output_size_ * sizeof(float);
        output_host_.resize(output_size_);
        checkCuda(cudaMemcpyAsync(output_host_.data(), buffers_[output_index_], output_bytes,
                        cudaMemcpyDeviceToHost, stream_), "copy output");
        checkCuda(cudaStreamSynchronize(stream_), "inference synchronization");

        // 5. 后处理
        results = postprocess(output_host_.data(), frame.cols, frame.rows, scale, dw, dh);

        return results;
    }

private:
    template <typename T>
    struct TrtDeleter {
        void operator()(T* obj) const {
            if (obj) obj->destroy();
        }
    };

    void loadEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("无法打开 TensorRT 引擎文件: " + engine_path);
        }

        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(size);
        if (!file.read(engine_data.data(), size)) {
            throw std::runtime_error("读取 TensorRT 引擎文件失败: " + engine_path);
        }

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("创建 TensorRT runtime 失败");
        }

        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine_) {
            throw std::runtime_error("反序列化 TensorRT 引擎失败");
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("创建 TensorRT context 失败");
        }

        const int nb_bindings = engine_->getNbBindings();
        if (nb_bindings != 2) {
            throw std::runtime_error("TensorRT 引擎绑定数量不足");
        }

        for (int i = 0; i < nb_bindings; ++i) {
            if (engine_->bindingIsInput(i)) {
                input_index_ = i;
            } else {
                output_index_ = i;
            }
        }

        for (int i = 0; i < nb_bindings; ++i) {
            if (engine_->getBindingDataType(i) != nvinfer1::DataType::kFLOAT)
                throw std::runtime_error("Only FLOAT input/output bindings are supported");
        }
        // 处理动态输入尺寸
        auto input_dims = engine_->getBindingDimensions(input_index_);
        if (hasDynamicDim(input_dims)) {
            nvinfer1::Dims4 fixed_dims(1, 3, INPUT_HEIGHT, INPUT_WIDTH);
            if (!context_->setBindingDimensions(input_index_, fixed_dims)) {
                throw std::runtime_error("设置输入尺寸失败");
            }
            input_dims = context_->getBindingDimensions(input_index_);
        }

        if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
            input_dims.d[2] < 320 || input_dims.d[3] < 320 ||
            input_dims.d[2] > 1280 || input_dims.d[3] > 1280 ||
            input_dims.d[2] % 32 || input_dims.d[3] % 32)
            throw std::runtime_error("Expected fixed FLOAT input [1,3,H,W], H/W multiples of 32 in [320,1280]");
        input_height_=input_dims.d[2]; input_width_=input_dims.d[3];
        input_size_ = volume(input_dims);

        auto output_dims = engine_->getBindingDimensions(output_index_);
        if (hasDynamicDim(output_dims)) {
            output_dims = context_->getBindingDimensions(output_index_);
        }
        output_size_ = volume(output_dims);
        parseOutputLayout(output_dims);

        checkCuda(cudaStreamCreate(&stream_), "create stream");
        checkCuda(cudaMalloc(&buffers_[input_index_], input_size_ * sizeof(float)), "allocate input");
        checkCuda(cudaMalloc(&buffers_[output_index_], output_size_ * sizeof(float)), "allocate output");
    }

    static bool hasDynamicDim(const nvinfer1::Dims& dims) {
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) return true;
        }
        return false;
    }

    static size_t volume(const nvinfer1::Dims& dims) {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
    }

    static void checkCuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }

    void parseOutputLayout(const nvinfer1::Dims& dims) {
        const int offset = dims.nbDims == 3 ? 1 : 0;
        if ((dims.nbDims != 2 && dims.nbDims != 3) ||
            (dims.nbDims == 3 && dims.d[0] != 1))
            throw std::runtime_error("Unsupported YOLO output rank/batch");
        const int a = dims.d[offset], b = dims.d[offset + 1];
        if (a == num_classes_ + 4 && b > a) {
            output_channel_first_ = true;
            output_rows_ = b;
            output_cols_ = a;
        } else if (b == num_classes_ + 4 && a > b) {
            output_channel_first_ = false;
            output_rows_ = a;
            output_cols_ = b;
        } else {
            throw std::runtime_error("YOLO output channels must equal configured class count + 4");
        }
        LOG_INFO("YOLO", std::string("Output layout: ") +
            (output_channel_first_ ? "channel-first " : "row-major ") +
            std::to_string(output_rows_) + " boxes, " + std::to_string(output_cols_) + " channels");
    }

    /**
     * @brief 预处理 - Letterbox 缩放
     */
    void preprocess(const cv::Mat& frame, cv::Mat& blob, float& scale, int& dw, int& dh) {
        int img_w = frame.cols;
        int img_h = frame.rows;

        // 计算缩放比例
        scale = std::min(static_cast<float>(input_width_) / img_w,
                         static_cast<float>(input_height_) / img_h);

        int new_w = static_cast<int>(img_w * scale);
        int new_h = static_cast<int>(img_h * scale);

        // 计算填充
        dw = (input_width_ - new_w) / 2;
        dh = (input_height_ - new_h) / 2;

        // 缩放
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(new_w, new_h));

        // 创建画布并填充
        cv::Mat canvas(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(canvas(cv::Rect(dw, dh, new_w, new_h)));

        // 转换为 blob (NCHW, float)
        cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);
    }

    /**
     * @brief 后处理 - 解析 YOLOv11 输出
     */
    std::vector<DetectedObj> postprocess(const float* output,
                                         int orig_w, int orig_h,
                                         float scale, int dw, int dh) {
        std::vector<DetectedObj> results;
        std::vector<cv::Rect2d> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<std::pair<float, float>> center_points;  // 保存底部中心点

        boxes.reserve(128);
        confidences.reserve(128);
        class_ids.reserve(128);
        center_points.reserve(128);

        const int rows = output_rows_;
        const int cols = output_cols_;
        if (rows <= 0 || cols < 5) return results;

        for (int i = 0; i < rows; ++i) {
            const auto value = [&](int channel) {
                return output_channel_first_ ? output[channel * rows + i] : output[i * cols + channel];
            };

            // 后80个是类别置信度 - 先快速扫描找最大值

            int class_id = 0;
            float max_score = value(4);
            for (int c = 1; c < num_classes_; ++c) {
                if (value(4 + c) > max_score) {
                    max_score = value(4 + c);
                    class_id = c;
                }
            }

            if (!std::isfinite(max_score) || max_score < CONF_THRESHOLD || max_score > 1.0f) continue;

            class_id=class_map_.at(class_id);

            // 只保留目标类别
            bool is_target = false;
            for (int t : TARGET_CLASSES) {
                if (t == class_id) {
                    is_target = true;
                    break;
                }
            }
            if (!is_target) continue;

            const float cx = value(0);
            const float cy = value(1);
            const float w = value(2);
            const float h = value(3);

            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
                !std::isfinite(h) || w <= 0 || h <= 0) continue;
            // 转换回原图坐标
            float x1 = (cx - w * 0.5f - dw) / scale;
            float y1 = (cy - h * 0.5f - dh) / scale;
            float x2 = (cx + w * 0.5f - dw) / scale;
            float y2 = (cy + h * 0.5f - dh) / scale;

            // 边界检查
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

            const float box_w = x2 - x1;
            const float box_h = y2 - y1;

            if (box_w <= 0 || box_h <= 0) continue;

            boxes.emplace_back(x1, y1, box_w, box_h);
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
            center_points.emplace_back((x1 + x2) * 0.5f, y2);
        }

        std::vector<int> indices;
        // Suppress duplicates within each class only. A rider overlaps a bicycle;
        // class-agnostic NMS was deleting one of these distinct detections.
        for (int cls : TARGET_CLASSES) {
            std::vector<cv::Rect2d> cls_boxes;
            std::vector<float> cls_scores;
            std::vector<int> original, kept;
            for (size_t i=0;i<boxes.size();++i) if(class_ids[i]==cls) {
                cls_boxes.push_back(boxes[i]);cls_scores.push_back(confidences[i]);original.push_back(i);
            }
            cv::dnn::NMSBoxes(cls_boxes,cls_scores,CONF_THRESHOLD,NMS_THRESHOLD,kept);
            for(int i:kept) indices.push_back(original[i]);
        }

        std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) { return confidences[a] > confidences[b]; });
        if (indices.size() > 300) indices.resize(300);
        for (int idx : indices) {
            DetectedObj obj;
            obj.track_id = -1;
            obj.class_id = class_ids[idx];
            obj.x_pixel = center_points[idx].first;
            obj.y_pixel = center_points[idx].second;
            obj.x1 = static_cast<float>(boxes[idx].x);
            obj.y1 = static_cast<float>(boxes[idx].y);
            obj.x2 = static_cast<float>(boxes[idx].x + boxes[idx].width);
            obj.y2 = static_cast<float>(boxes[idx].y + boxes[idx].height);
            obj.score = confidences[idx];
            obj.lat = 0.0;
            obj.lon = 0.0;
            results.push_back(obj);
        }

        return results;
    }

private:
    int num_classes_=80;
    std::vector<int> class_map_;
    int input_width_=640, input_height_=640;
    bool use_cuda_{true};
    Logger logger_{};
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context_;

    void* buffers_[2]{nullptr, nullptr};
    int input_index_{0};
    int output_index_{1};
    size_t input_size_{0};
    size_t output_size_{0};
    bool output_channel_first_{false};
    int output_rows_{0};
    int output_cols_{0};
    std::vector<float> output_host_;
    cudaStream_t stream_{nullptr};
};

#endif // YOLO_DETECTOR_HPP