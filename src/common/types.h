#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace minimind {

// Model constants (MiniMind, hidden_size=768, layers=8)
constexpr int32_t kHiddenSize       = 768;
constexpr int32_t kNumLayers        = 8;
constexpr int32_t kNumAttnHeads     = 8;
constexpr int32_t kNumKvHeads       = 4;
constexpr int32_t kHeadDim          = 96;
constexpr int32_t kVocabSize        = 6400;
constexpr int32_t kImageTokenLen    = 64;       // (256/32)^2 patches
constexpr int32_t kImageSize        = 256;
constexpr int32_t kImageChannels    = 3;
constexpr int32_t kBosTokenId       = 1;
constexpr int32_t kEosTokenId       = 2;
constexpr int32_t kImagePadTokenId  = 12;       // <|image_pad|>

// SigLIP2 normalization
constexpr float kImageMean[] = {0.5f, 0.5f, 0.5f};
constexpr float kImageStd[]  = {0.5f, 0.5f, 0.5f};

using TokenId = int32_t;
using TokenIds = std::vector<TokenId>;

}  // namespace minimind
