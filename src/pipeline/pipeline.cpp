#include "pipeline/pipeline.h"
#include "common/types.h"
#include <cstring>
#include <sstream>

namespace minimind {

bool Pipeline::load(const std::string& vision_model_path,
                    const std::string& llm_config_dir,
                    int num_threads) {
    VisionEncoder::Config vcfg;
    vcfg.model_path  = vision_model_path;
    vcfg.num_threads = num_threads;
    // Default to Vulkan GPU, fall back to CPU
    vcfg.backend = MNN_FORWARD_VULKAN;
    vision_ = std::make_unique<VisionEncoder>(vcfg);
    if (!vision_->ready()) {
        vcfg.backend = MNN_FORWARD_CPU;
        vision_ = std::make_unique<VisionEncoder>(vcfg);
    }
    if (!vision_->ready()) return false;

    llm_ = std::make_unique<LlmRuntime>();
    if (!llm_->load(llm_config_dir, num_threads)) return false;

    return true;
}

void Pipeline::mix_embeddings(
    const std::vector<int32_t>& input_ids,
    const float* vision_emb,
    const float* tok_emb,
    float* mixed,
    int seq_len) {

    std::memcpy(mixed, tok_emb, seq_len * kHiddenSize * sizeof(float));

    int patch = 0;
    for (int i = 0; i < seq_len && patch < kImageTokenLen; ++i) {
        if (input_ids[i] == kImagePadTokenId) {
            std::memcpy(mixed + i * kHiddenSize,
                        vision_emb + patch * kHiddenSize,
                        kHiddenSize * sizeof(float));
            ++patch;
        }
    }
}

std::string Pipeline::run(const uint8_t* rgb_u8, int width, int height,
                          const std::string& prompt, int max_tokens) {
    if (!ready()) return "";

    // 1. Preprocess image
    ImagePreprocess::Config pcfg;
    pcfg.src_width  = width;
    pcfg.src_height = height;
    ImagePreprocess preproc(pcfg);
    auto pixels = preproc.run(rgb_u8);

    // 2. Vision encode
    auto vision_emb = vision_->forward(pixels.data());

    // 3. Build user content: <|image_pad|>*64 + newline + prompt
    std::ostringstream user_content;
    for (int i = 0; i < kImageTokenLen; ++i) user_content << "<|image_pad|>";
    user_content << "\n" << prompt;

    // 4. Manually construct chat format (MNN lacks JINJA support)
    //    Format: <|im_start|>user\n<content><|im_end|>\n<|im_start|>assistant\n
    std::ostringstream chat_prompt;
    chat_prompt << "<|im_start|>user\n"
                << user_content.str()
                << "<|im_end|>\n"
                << "<|im_start|>assistant\n";

    // 5. Tokenize
    auto input_ids = llm_->tokenize(chat_prompt.str());
    int seq_len = static_cast<int>(input_ids.size());

    // 6. Get token embeddings & mix with vision
    auto tok_emb = llm_->get_embeddings(input_ids);
    std::vector<float> mixed(seq_len * kHiddenSize);
    mix_embeddings(input_ids, vision_emb.data(), tok_emb.data(),
                   mixed.data(), seq_len);

    // 7. Generate from mixed embeddings
    auto out_ids = llm_->generate(mixed.data(), seq_len, max_tokens);

    return llm_->detokenize(out_ids);
}

}  // namespace minimind
