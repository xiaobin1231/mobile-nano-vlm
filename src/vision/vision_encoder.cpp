#include "vision/vision_encoder.h"
#include <cstring>

namespace minimind {

VisionEncoder::VisionEncoder(const Config& cfg) {
    interpreter_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(cfg.model_path.c_str()));
    if (!interpreter_) return;

    MNN::ScheduleConfig sched_cfg;
    sched_cfg.type = cfg.backend;
    sched_cfg.numThread = cfg.num_threads;

    session_ = interpreter_->createSession(sched_cfg);
    if (!session_) return;

    input_tensor_  = interpreter_->getSessionInput(session_, nullptr);
    output_tensor_ = interpreter_->getSessionOutput(session_, nullptr);
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
