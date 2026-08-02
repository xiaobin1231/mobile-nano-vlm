#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${1:-${QNN_DOCKER_IMAGE:-mobile-nano-vlm-build:local}}"
PLATFORM_ARGS=()
if [ -n "${DOCKER_PLATFORM:-}" ]; then
  PLATFORM_ARGS+=(--platform "$DOCKER_PLATFORM")
fi

docker build --network host \
  "${PLATFORM_ARGS[@]}" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
  --build-arg CMAKE_VERSION="${CMAKE_VERSION:-4.3.4}" \
  --build-arg CMAKE_SHA256="${CMAKE_SHA256:-ca6f08ccbd5e6b0a9068d33317d0d1aff7278d08cccaed4529b8fbead7942a68}" \
  -t "$IMAGE" \
  -f "$SCRIPT_DIR/Dockerfile" \
  "$SCRIPT_DIR"

printf 'Built: %s\n' "$IMAGE"
