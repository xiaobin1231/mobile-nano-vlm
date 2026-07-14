#pragma once

#include <string>
#include <vector>
#include "llm/llm.hpp"  // MNN::Transformer::Llm

namespace minimind {

class LlmRuntime {
public:
    LlmRuntime() = default;
    ~LlmRuntime();

    bool load(const std::string& config_dir, int num_threads = 4, bool low_precision = true);
    bool ready() const { return llm_ != nullptr; }

    // Tokenizer
    std::vector<int32_t> tokenize(const std::string& text);
    std::string detokenize(int32_t token_id);
    std::string detokenize(const std::vector<int32_t>& token_ids);

    // Chat template
    std::string apply_chat_template(const std::string& user_content);

    // Embedding
    std::vector<float> get_embeddings(const std::vector<int32_t>& input_ids);

    // Generation
    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids,
                                  int max_new_tokens = 256);
    std::vector<int32_t> generate(const float* embeds, int seq_len,
                                  int max_new_tokens = 256);

    int hidden_size() const { return 768; }

private:
    MNN::Transformer::Llm* llm_ = nullptr;
};

}  // namespace minimind
