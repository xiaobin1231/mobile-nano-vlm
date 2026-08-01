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
  -t "$IMAGE" \
  -f "$SCRIPT_DIR/Dockerfile" \
  "$SCRIPT_DIR"

printf 'Built: %s\n' "$IMAGE"
