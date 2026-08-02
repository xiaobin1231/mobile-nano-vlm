#include "pipeline/pipeline.h"
#include "common/types.h"
#include <chrono>
#include <cstring>
#include <ostream>
#include <sstream>

namespace minimind {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

bool Pipeline::load(const std::string& vision_model_path,
                    const std::string& llm_config_dir,
                    int num_threads) {
    VisionEncoder::Config vcfg;
    vcfg.model_path  = vision_model_path;
    vcfg.num_threads = num_threads;
    // Offline QNN wrappers are CPU Plugin graphs whose kernels dispatch to HTP.
    vcfg.backend = vision_model_path.find("/vision_qnn/") != std::string::npos
        ? MNN_FORWARD_CPU : MNN_FORWARD_VULKAN;
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
    std::ostringstream output;
    if (!run_vision(rgb_u8, width, height, prompt, output, max_tokens)) {
        return "";
    }
    return output.str();
}

bool Pipeline::run_text(const std::string& prompt, std::ostream& output,
                        int max_tokens) {
    if (!ready() || prompt.empty()) return false;
    last_profile_ = {};

    const auto tokenize_start = Clock::now();
    std::ostringstream chat_prompt;
    chat_prompt << "<|im_start|>user\n"
                << prompt
                << "<|im_end|>\n"
                << "<|im_start|>assistant\n";
    auto input_ids = llm_->tokenize(chat_prompt.str());
    last_profile_.tokenize_ms = elapsed_ms(tokenize_start);

    const auto llm_start = Clock::now();
    const bool ok = llm_->respond(input_ids, output, max_tokens);
    last_profile_.llm_ms = elapsed_ms(llm_start);
    const auto& stats = llm_->last_stats();
    last_profile_.prefill_ms = stats.prefill_ms;
    last_profile_.decode_ms = stats.decode_ms;
    last_profile_.prompt_tokens = stats.prompt_tokens;
    last_profile_.generated_tokens = stats.generated_tokens;
    return ok;
}

bool Pipeline::run_vision(const uint8_t* rgb_u8, int width, int height,
                          const std::string& prompt, std::ostream& output,
                          int max_tokens) {
    if (!ready() || rgb_u8 == nullptr || width <= 0 || height <= 0
        || prompt.empty()) {
        return false;
    }
    last_profile_ = {};

    // 1. Preprocess image
    const auto preprocess_start = Clock::now();
    ImagePreprocess::Config pcfg;
    pcfg.src_width  = width;
    pcfg.src_height = height;
    ImagePreprocess preproc(pcfg);
    auto pixels = preproc.run(rgb_u8);
    last_profile_.preprocess_ms = elapsed_ms(preprocess_start);

    // 2. Vision encode
    const auto vision_start = Clock::now();
    auto vision_emb = vision_->forward(pixels.data());
    last_profile_.vision_ms = elapsed_ms(vision_start);

    // 3. Build user content: <|image_pad|>*64 + newline + prompt
    const auto tokenize_start = Clock::now();
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
    last_profile_.tokenize_ms = elapsed_ms(tokenize_start);

    // 6. Get token embeddings & mix with vision
    const auto embedding_start = Clock::now();
    auto tok_emb = llm_->get_embeddings(input_ids);
    last_profile_.embedding_ms = elapsed_ms(embedding_start);

    const auto mix_start = Clock::now();
    std::vector<float> mixed(seq_len * kHiddenSize);
    mix_embeddings(input_ids, vision_emb.data(), tok_emb.data(),
                   mixed.data(), seq_len);
    last_profile_.mix_ms = elapsed_ms(mix_start);

    // 7. Generate from mixed embeddings
    const auto llm_start = Clock::now();
    const bool ok = llm_->respond(mixed.data(), seq_len, output, max_tokens);
    last_profile_.llm_ms = elapsed_ms(llm_start);
    const auto& stats = llm_->last_stats();
    last_profile_.prefill_ms = stats.prefill_ms;
    last_profile_.decode_ms = stats.decode_ms;
    last_profile_.prompt_tokens = stats.prompt_tokens;
    last_profile_.generated_tokens = stats.generated_tokens;
    return ok;
}

}  // namespace minimind
