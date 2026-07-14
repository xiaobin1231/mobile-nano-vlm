#pragma once

#include <string>
#include <vector>
#include <memory>
#include "preprocess/image_preprocess.h"
#include "vision/vision_encoder.h"
#include "llm/llm_runtime.h"

namespace minimind {

/// Full MiniMind-V inference pipeline.
class Pipeline {
public:
    Pipeline() = default;

    /// Load vision + LLM models.
    bool load(const std::string& vision_model_path,
              const std::string& llm_config_dir,
              int num_threads = 4);

    bool ready() const { return vision_ && vision_->ready() && llm_ && llm_->ready(); }

    /// Run full pipeline on raw RGB image.
    /// @param rgb_u8     RGB interleaved uint8, width * height * 3
    /// @param width      image width
    /// @param height     image height
    /// @param prompt     text prompt (image placeholder auto-prepended)
    /// @param max_tokens max new tokens to generate
    /// @return generated text
    std::string run(const uint8_t* rgb_u8, int width, int height,
                    const std::string& prompt, int max_tokens = 256);

private:
    std::unique_ptr<VisionEncoder> vision_;
    std::unique_ptr<LlmRuntime>    llm_;

    static void mix_embeddings(const std::vector<int32_t>& input_ids,
                               const float* vision_emb,
                               const float* tok_emb,
                               float* mixed,
                               int seq_len);
};

}  // namespace minimind
