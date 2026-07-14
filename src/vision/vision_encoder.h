#pragma once

#include <string>
#include <vector>
#include <memory>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

namespace minimind {

/// MNN Interpreter wrapper for vision_encode_proj.mnn.
///
/// Input:  pixel_values  [1, 3, 256, 256] float32, NCHW
/// Output: vision_emb    [1, 64, 768] float32
class VisionEncoder {
public:
    struct Config {
        std::string model_path;
        int num_threads = 4;
        MNNForwardType backend = MNN_FORWARD_CPU;
    };

    explicit VisionEncoder(const Config& cfg);
    ~VisionEncoder();

    bool ready() const { return interpreter_ != nullptr && session_ != nullptr; }

    /// Run vision encoder.
    /// @param pixel_values  float32 [1, 3, 256, 256], NCHW
    /// @param output        float32 [64 * 768], caller-allocated
    bool forward(const float* pixel_values, float* output);

    /// Convenience: returns float32 vector of size 64*768.
    std::vector<float> forward(const float* pixel_values);

    static constexpr int kInputSize  = 1 * 3 * 256 * 256;
    static constexpr int kOutputSize = 64 * 768;

private:
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;
    MNN::Tensor* output_tensor_ = nullptr;
};

}  // namespace minimind
