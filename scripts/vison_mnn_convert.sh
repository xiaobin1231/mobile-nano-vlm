#!/bin/bash

PROJECT_ROOT=$(pwd)
export PATH=$PROJECT_ROOT/third_party/MNN/build:$PATH

ONNX="vision_encode_proj.onnx"
MNN="vision_encode_proj.mnn"

src=$PROJECT_ROOT/vision_export
dst=$PROJECT_ROOT/vision_export

if [ ! -d $dst ];then
  mkdir $dst
fi

echo "MNNConvert:"
echo "  src: $src/$ONNX"
echo "  dst: $dst/$MNN"

MNNConvert -f ONNX --modelFile $src/$ONNX --MNNModel $dst/$MNN --bizCode MNN --fp16
