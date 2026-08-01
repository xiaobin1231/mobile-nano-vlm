#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_IMAGE="${SOURCE_IMAGE:-mobile-nano-vlm-build:local}"
TARGET_IMAGE="${1:-${DOCKERHUB_IMAGE:-}}"

if [ -z "$TARGET_IMAGE" ]; then
  cat >&2 <<'EOF'
Usage:
  ./docker/push_public_image.sh <dockerhub-user>/<repository>:<tag>

Example:
  ./docker/push_public_image.sh username/mobile-nano-vlm-build:latest
EOF
  exit 1
fi

case "$TARGET_IMAGE" in
  */*:*) ;;
  *)
    printf '错误: 镜像名必须包含 Docker Hub namespace 和 tag: %s\n' \
      "$TARGET_IMAGE" >&2
    exit 1
    ;;
esac

"$SCRIPT_DIR/audit_public_image.sh" "$SOURCE_IMAGE"
docker tag "$SOURCE_IMAGE" "$TARGET_IMAGE"
"$SCRIPT_DIR/audit_public_image.sh" "$TARGET_IMAGE"

printf '即将公开推送: %s\n' "$TARGET_IMAGE"
docker push "$TARGET_IMAGE"
