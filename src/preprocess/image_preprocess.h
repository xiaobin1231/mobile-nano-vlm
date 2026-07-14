#pragma once

#include <vector>
#include <cstdint>

namespace minimind {

/// Bilinear resize + normalize an RGB image for SigLIP2.
///
/// Input:  RGB interleaved uint8, [height * width * 3]
/// Output: float32 planar CHW, [1 * 3 * 256 * 256], values ~[-1, 1]
class ImagePreprocess {
public:
    struct Config {
        int src_width  = 0;
        int src_height = 0;
        int dst_size   = 256;
        float mean[3]  = {0.5f, 0.5f, 0.5f};
        float std[3]   = {0.5f, 0.5f, 0.5f};
    };

    explicit ImagePreprocess(const Config& cfg);

    /// Resize + normalize into pre-allocated float buffer.
    /// @param rgb_u8  input, src_width * src_height * 3 bytes
    /// @param output  float32, 1 * 3 * 256 * 256 = 196608 floats, CHW planar
    void run(const uint8_t* rgb_u8, float* output) const;

    /// Convenience: returns allocated float vector.
    std::vector<float> run(const uint8_t* rgb_u8) const;

    int input_bytes()   const { return src_width_ * src_height_ * 3; }
    int output_floats() const { return 3 * dst_size_ * dst_size_; }

private:
    int src_width_, src_height_, dst_size_;
    float mean_[3], std_[3];
    float scale_;  // 1/255
};

}  // namespace minimind
