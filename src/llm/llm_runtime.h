#pragma once

#include <string>
#include <vector>
#include <iosfwd>
#include "llm/llm.hpp"  // MNN::Transformer::Llm

namespace minimind {

class LlmRuntime {
public:
    struct GenerationStats {
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
        int prompt_tokens = 0;
        int generated_tokens = 0;
    };

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
                                  int max_new_tokens = 512);
    std::vector<int32_t> generate(const float* embeds, int seq_len,
                                  int max_new_tokens = 512);

    // Stateless streaming generation. Text is flushed to `output` while MNN
    // decodes, so a daemon can forward chunks without waiting for completion.
    bool respond(const std::vector<int32_t>& input_ids, std::ostream& output,
                 int max_new_tokens = 512);
    bool respond(const float* embeds, int seq_len, std::ostream& output,
                 int max_new_tokens = 512);

    const GenerationStats& last_stats() const { return last_stats_; }

    int hidden_size() const { return 768; }

private:
    void update_generation_stats();

    MNN::Transformer::Llm* llm_ = nullptr;
    GenerationStats last_stats_;
};

}  // namespace minimind
