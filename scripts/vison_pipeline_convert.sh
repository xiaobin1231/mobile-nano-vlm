#!/bin/bash

export PATH=/home/binxue/workspace/tutorial/mobile-nano-vlm/third_party/MNN/build:$PATH
MNNConvert -f ONNX --modelFile /home/binxue/workspace/tutorial/mobile-nano-vlm/vision_encode_proj.onnx --MNNModel /home/binxue/workspace/tutorial/mobile-nano-vlm/vision_encode_proj.mnn --bizCode MNN --fp16
