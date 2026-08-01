#include "vision/vision_encoder.h"
#include <cstring>
#include <fstream>
#include <limits.h>
#include <unistd.h>

namespace minimind {

VisionEncoder::VisionEncoder(const Config& cfg) {
    interpreter_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(cfg.model_path.c_str()));
    if (!interpreter_) return;

    MNN::ScheduleConfig sched_cfg;
    sched_cfg.type = cfg.backend;
    sched_cfg.numThread = cfg.num_threads;

    // Plugin graph paths are relative in MNN's generated wrapper. CPUPlugin
    // currently receives "." as its model directory, so create the session
    // from the wrapper directory and immediately restore the process cwd.
    char old_cwd[PATH_MAX] = {0};
    bool changed_cwd = false;
    auto model_slash = cfg.model_path.find_last_of("/\\");
    if (cfg.model_path.find("vision_qnn") != std::string::npos
            && model_slash != std::string::npos
            && getcwd(old_cwd, sizeof(old_cwd))) {
        auto model_dir = cfg.model_path.substr(0, model_slash);
        changed_cwd = chdir(model_dir.c_str()) == 0;
    }
    session_ = interpreter_->createSession(sched_cfg);
    if (changed_cwd) chdir(old_cwd);
    if (!session_) return;

    input_tensor_ = interpreter_->getSessionInput(session_, "pixel_values");
    if (!input_tensor_) input_tensor_ = interpreter_->getSessionInput(session_, nullptr);
    position_tensor_ = interpreter_->getSessionInput(session_, "position_embeds");
    output_tensor_ = interpreter_->getSessionOutput(session_, "vision_embeddings");
    if (!output_tensor_) output_tensor_ = interpreter_->getSessionOutput(session_, nullptr);

    // The offline QNN wrapper exposes positional embeddings as a dynamic input
    // so both Add operands receive the same MNN-to-QNN layout conversion.
    if (position_tensor_) {
        auto slash = cfg.model_path.find_last_of("/\\");
        auto dir = slash == std::string::npos ? std::string(".") : cfg.model_path.substr(0, slash);
        std::ifstream in(dir + "/vision_position_f32.bin", std::ios::binary | std::ios::ate);
        if (!in) {
            interpreter_->releaseSession(session_);
            session_ = nullptr;
            return;
        }
        auto bytes = static_cast<size_t>(in.tellg());
        in.seekg(0);
        position_data_.resize(bytes / sizeof(float));
        in.read(reinterpret_cast<char*>(position_data_.data()), bytes);
        if (!in || position_data_.size() != kOutputSize) {
            interpreter_->releaseSession(session_);
            session_ = nullptr;
            return;
        }
    }
}

VisionEncoder::~VisionEncoder() {
    if (session_) interpreter_->releaseSession(session_);
}

bool VisionEncoder::forward(const float* pixel_values, float* output) {
    if (!session_) return false;

    // Copy input
    auto shape = input_tensor_->shape();
    int n_elems = 1;
    for (int d : shape) n_elems *= d;
    std::memcpy(input_tensor_->host<float>(), pixel_values, n_elems * sizeof(float));
    if (position_tensor_) {
        std::memcpy(position_tensor_->host<float>(), position_data_.data(),
                    position_data_.size() * sizeof(float));
    }

    interpreter_->runSession(session_);

    // Copy output via host tensor to handle any internal format (NC4HW4→NCHW)
    auto out_shape = output_tensor_->shape();
    int out_elems = 1;
    for (int d : out_shape) out_elems *= d;

    // Create a host tensor in NCHW format and copy
    std::shared_ptr<MNN::Tensor> host_tensor(
        MNN::Tensor::create<float>(out_shape, nullptr, MNN::Tensor::CAFFE));
    output_tensor_->copyToHostTensor(host_tensor.get());
    std::memcpy(output, host_tensor->host<float>(), out_elems * sizeof(float));

    return true;
}

std::vector<float> VisionEncoder::forward(const float* pixel_values) {
    std::vector<float> out(kOutputSize);
    forward(pixel_values, out.data());
    return out;
}

}  // namespace minimind
