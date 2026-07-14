#include "pipeline/pipeline.h"
#include "common/types.h"
#include "common/stb_image.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <sstream>

using namespace minimind;

static std::vector<uint8_t> load_image(const std::string& path, int* w, int* h) {
    int channels;
    uint8_t* data = stbi_load(path.c_str(), w, h, &channels, 3);
    if (!data) {
        std::cerr << "Failed to load: " << path << " (" << stbi_failure_reason() << ")" << std::endl;
        return {};
    }
    std::vector<uint8_t> result(data, data + (*w) * (*h) * 3);
    stbi_image_free(data);
    return result;
}

/// Manually build chat prompt (MNN lacks JINJA, template engine doesn't work)
/// Format: <|im_start|>user\n<prompt><|im_end|>\n<|im_start|>assistant\n
static std::string build_chat_prompt(const std::string& user_msg) {
    std::ostringstream oss;
    oss << "<|im_start|>user\n" << user_msg << "<|im_end|>\n<|im_start|>assistant\n";
    return oss.str();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " text <config_dir> <prompt>\n"
                  << "  " << argv[0] << " vision <config_dir> <vision.mnn> <image.jpg> <prompt>\n"
                  << "  " << argv[0] << " test <config_dir>\n";
        return 1;
    }
    std::string mode = argv[1];
    std::string config_dir = (argc > 2) ? argv[2] : "models";

    if (mode == "text") {
        if (argc < 4) { std::cerr << "Usage: text <config_dir> <prompt>" << std::endl; return 1; }
        std::string prompt = argv[3];
        LlmRuntime llm;
        if (!llm.load(config_dir)) { std::cerr << "LLM load failed" << std::endl; return 1; }

        std::string chat = build_chat_prompt(prompt);
        auto ids = llm.tokenize(chat);
        std::cout << "Tokenized: " << ids.size() << " tokens" << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = llm.generate(ids, 512);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Output (" << out.size() << " tokens, " << ms << " ms):\n"
                  << llm.detokenize(out) << std::endl;

    } else if (mode == "vision") {
        if (argc < 6) {
            std::cerr << "Usage: vision <config_dir> <vision.mnn> <image.jpg> <prompt>" << std::endl;
            return 1;
        }
        std::string vision_path = argv[3], image_path = argv[4], prompt = argv[5];
        int w, h; auto img = load_image(image_path, &w, &h);
        if (img.empty()) return 1;
        std::cout << "Image: " << w << "x" << h << std::endl;

        Pipeline pipe;
        if (!pipe.load(vision_path, config_dir)) {
            std::cerr << "Pipeline load failed" << std::endl; return 1;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        auto text = pipe.run(img.data(), w, h, prompt, 512);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Output (" << ms << " ms):\n" << text << std::endl;

    } else if (mode == "test") {
        LlmRuntime llm;
        if (!llm.load(config_dir)) { std::cerr << "LLM load failed" << std::endl; return 1; }
        std::cout << "Loaded." << std::endl;
        std::vector<std::vector<int>> test_inputs = {{1}, {1, 1968, 294, 960}, {42, 100, 500, 777}};
        bool all_same = true; int prev = -1;
        for (auto& ids : test_inputs) {
            auto r = llm.generate(ids, 1);
            int tok = r.empty() ? -1 : r[0];
            std::cout << "  [";
            for (size_t i = 0; i < ids.size(); i++) std::cout << ids[i] << (i+1<ids.size()?",":"");
            std::cout << "] -> " << tok << "(" << llm.detokenize(tok) << ")" << std::endl;
            if (prev >= 0 && tok != prev) all_same = false;
            prev = tok;
        }
        std::cout << "All same: " << (all_same ? "YES" : "NO (model working)") << std::endl;
        auto r = llm.generate({1, 1968, 294}, 8);
        std::cout << "Output: "; for (auto id : r) std::cout << id << "(" << llm.detokenize(id) << ") ";
        std::cout << std::endl;
    }
    return 0;
}
