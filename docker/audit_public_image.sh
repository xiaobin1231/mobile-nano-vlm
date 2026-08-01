#!/usr/bin/env bash

set -euo pipefail

IMAGE="${1:-${QNN_DOCKER_IMAGE:-mobile-nano-vlm-build:local}}"

docker image inspect "$IMAGE" >/dev/null

history="$(docker history --no-trunc --format '{{.CreatedBy}}' "$IMAGE")"
history_forbidden='/home/[^/]+/|/Users/[^/]+/|sft_vlm_768[.]pth|vision_encode_proj|graph[0-9]+[.]bin'
if printf '%s\n' "$history" | grep -Eiq "$history_forbidden"; then
  printf '错误: 镜像构建历史疑似包含受限或个人内容\n' >&2
  printf '%s\n' "$history" | grep -Ei "$history_forbidden" >&2 || true
  exit 1
fi

matches="$(
  docker run --rm --entrypoint /bin/bash "$IMAGE" -lc \
    "find / -xdev \\( \
       -iname '*qairt*' -o \
       -iname 'libQnn*' -o \
       -iname 'qnn-onnx-converter*' -o \
       -iname '*android-ndk*' -o \
       -iname 'sft_vlm_768.pth' -o \
       -iname 'vision_encode_proj*' -o \
       -iname 'graph?.bin' \
     \\) -print 2>/dev/null" \
  || true
)"
if [ -n "$matches" ]; then
  printf '错误: 镜像文件系统疑似包含受限内容:\n%s\n' "$matches" >&2
  exit 1
fi

docker run --rm -i "$IMAGE" python - <<'PY'
import numpy
import onnx
import torch
import transformers
import yaml
import yaspin
from importlib.metadata import version

print("numpy", numpy.__version__)
print("onnx", onnx.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("pyyaml", yaml.__version__)
print("yaspin", version("yaspin"))
PY

docker run --rm "$IMAGE" test -f \
  /opt/mobile-nano-vlm/public-deps.ready

printf 'Public image audit passed: %s\n' "$IMAGE"
