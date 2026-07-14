#include "preprocess/image_preprocess.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace minimind {

ImagePreprocess::ImagePreprocess(const Config& cfg)
    : src_width_(cfg.src_width)
    , src_height_(cfg.src_height)
    , dst_size_(cfg.dst_size)
    , scale_(1.0f / 255.0f)
{
    std::copy(cfg.mean, cfg.mean + 3, mean_);
    std::copy(cfg.std, cfg.std + 3, std_);
}

static inline float bilinear_interp(const uint8_t* src, int w, int h,
                                    float x, float y, int c) {
    int x0 = std::max(0, std::min(w - 1, static_cast<int>(x)));
    int y0 = std::max(0, std::min(h - 1, static_cast<int>(y)));
    int x1 = std::min(w - 1, x0 + 1);
    int y1 = std::min(h - 1, y0 + 1);
    float dx = x - x0, dy = y - y0;

    float v00 = src[(y0 * w + x0) * 3 + c];
    float v10 = src[(y0 * w + x1) * 3 + c];
    float v01 = src[(y1 * w + x0) * 3 + c];
    float v11 = src[(y1 * w + x1) * 3 + c];

    float top = v00 * (1.f - dx) + v10 * dx;
    float bot = v01 * (1.f - dx) + v11 * dx;
    return top * (1.f - dy) + bot * dy;
}

void ImagePreprocess::run(const uint8_t* rgb_u8, float* output) const {
    const int dst = dst_size_;
    float scale_x = static_cast<float>(src_width_ - 1) / std::max(dst - 1, 1);
    float scale_y = static_cast<float>(src_height_ - 1) / std::max(dst - 1, 1);

    float* chan_r = output;
    float* chan_g = output + dst * dst;
    float* chan_b = output + 2 * dst * dst;

    for (int y = 0; y < dst; ++y) {
        float src_y = y * scale_y;
        for (int x = 0; x < dst; ++x) {
            float src_x = x * scale_x;
            int idx = y * dst + x;
            chan_r[idx] = (bilinear_interp(rgb_u8, src_width_, src_height_, src_x, src_y, 0) * scale_ - mean_[0]) / std_[0];
            chan_g[idx] = (bilinear_interp(rgb_u8, src_width_, src_height_, src_x, src_y, 1) * scale_ - mean_[1]) / std_[1];
            chan_b[idx] = (bilinear_interp(rgb_u8, src_width_, src_height_, src_x, src_y, 2) * scale_ - mean_[2]) / std_[2];
        }
    }
}

std::vector<float> ImagePreprocess::run(const uint8_t* rgb_u8) const {
    std::vector<float> out(output_floats());
    run(rgb_u8, out.data());
    return out;
}

}  // namespace minimind
