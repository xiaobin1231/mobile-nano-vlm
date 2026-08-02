#include "llm/llm_runtime.h"
#include "common/types.h"
#include <cstring>
#include <fstream>
#include <ostream>
#include <MNN/expr/Expr.hpp>

using namespace MNN::Express;
using namespace MNN::Transformer;

namespace minimind {

LlmRuntime::~LlmRuntime() {
    if (llm_) Llm::destroy(llm_);
}

bool LlmRuntime::load(const std::string& config_dir, int num_threads, bool low_precision) {
    const std::string qnn_config = config_dir + "/llm_config_qnn.json";
    std::ifstream qnn_file(qnn_config);
    const std::string config_path = qnn_file.good()
        ? qnn_config : config_dir + "/llm_config.json";
    llm_ = Llm::createLLM(config_path);
    if (!llm_) return false;

    std::string ov = "{"
        "\"backend_type\": \"cpu\","
        "\"thread_num\": " + std::to_string(num_threads) + ","
        "\"precision\": \"" + (low_precision ? "low" : "normal") + "\","
        "\"use_template\": false,"
        "\"is_visual\": false,"
        "\"hidden_size\": 768,"
        "\"layer_nums\": 8,"
        "\"attention_mask\": \"float\""
        "}";
    llm_->set_config(ov);
    return llm_->load();
}

std::vector<int32_t> LlmRuntime::tokenize(const std::string& text) {
    if (!llm_) return {};
    return llm_->tokenizer_encode(text);
}

std::string LlmRuntime::detokenize(int32_t token_id) {
    if (!llm_) return "";
    return llm_->tokenizer_decode(token_id);
}

std::string LlmRuntime::detokenize(const std::vector<int32_t>& token_ids) {
    std::string result;
    for (auto id : token_ids) {
        if (id == 2) continue;
        result += llm_->tokenizer_decode(id);
    }
    return result;
}

std::string LlmRuntime::apply_chat_template(const std::string& user_content) {
    if (!llm_) return user_content;
    return llm_->apply_chat_template(user_content);
}

std::vector<float> LlmRuntime::get_embeddings(const std::vector<int32_t>& input_ids) {
    if (!llm_ || input_ids.empty()) return {};
    auto emb_varp = llm_->embedding(input_ids);
    auto info = emb_varp->getInfo();
    int total = info->size;
    std::vector<float> result(total);
    std::memcpy(result.data(), emb_varp->readMap<float>(), total * sizeof(float));
    return result;
}

std::vector<int32_t> LlmRuntime::generate(
    const std::vector<int32_t>& input_ids, int max_new_tokens) {
    if (!llm_) return {};
    return llm_->generate(input_ids, max_new_tokens);
}

std::vector<int32_t> LlmRuntime::generate(
    const float* embeds, int seq_len, int max_new_tokens) {
    if (!llm_) return {};
    auto embeds_varp = _Const(embeds, {seq_len, 1, kHiddenSize},
                              NCHW, halide_type_of<float>());
    return llm_->generate(embeds_varp, max_new_tokens);
}

bool LlmRuntime::respond(const std::vector<int32_t>& input_ids,
                         std::ostream& output, int max_new_tokens) {
    if (!llm_ || input_ids.empty()) return false;
    // Every daemon request is independent in protocol v1. Reset explicitly so
    // KV/history/output state cannot leak from the previous client request.
    llm_->reset();
    llm_->response(input_ids, &output, "", max_new_tokens);
    update_generation_stats();
    return output.good();
}

bool LlmRuntime::respond(const float* embeds, int seq_len,
                         std::ostream& output, int max_new_tokens) {
    if (!llm_ || embeds == nullptr || seq_len <= 0) return false;
    auto embeds_varp = _Const(embeds, {seq_len, 1, kHiddenSize},
                              NCHW, halide_type_of<float>());
    llm_->reset();
    llm_->response(embeds_varp, &output, "", max_new_tokens);
    update_generation_stats();
    return output.good();
}

void LlmRuntime::update_generation_stats() {
    last_stats_ = {};
    if (!llm_) return;
    const auto* context = llm_->getContext();
    if (!context) return;
    last_stats_.prefill_ms = context->prefill_us / 1000.0;
    last_stats_.decode_ms = context->decode_us / 1000.0;
    last_stats_.prompt_tokens = context->prompt_len;
    last_stats_.generated_tokens = context->gen_seq_len;
}

}  // namespace minimind
