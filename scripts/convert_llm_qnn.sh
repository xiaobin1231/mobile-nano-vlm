#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=scripts/_qnn_pipeline_common.sh
source "$SCRIPT_DIR/_qnn_pipeline_common.sh"

DEPLOY=0
PROMOTE=1
DRY_RUN=0
IMAGE_PATH=""
PROMPT="请描述这幅图"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/convert_llm_qnn.sh [options]

Options:
  --deploy           转换后编译、部署完整 QNN VLM
  --image PATH       部署后推送图片并执行完整图文问答
  --prompt TEXT      图文问答提示词，默认“请描述这幅图”
  --no-promote       只生成 build/llm_qnn/package，不更新 artifacts
  --dry-run          只检查主机前置条件并打印执行计划
  -h, --help         显示帮助

Required host environment:
  DEPS_ROOT          依赖根目录，内部包含 qairt/<version>

Required with --deploy:
  ANDROID_NDK
  QNN_SDK_ROOT
  CMAKE_BIN          可选，默认 cmake
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --deploy) DEPLOY=1 ;;
    --image)
      shift
      [ "$#" -gt 0 ] || qnn_die "--image 缺少路径"
      IMAGE_PATH="$1"
      ;;
    --prompt)
      shift
      [ "$#" -gt 0 ] || qnn_die "--prompt 缺少文本"
      PROMPT="$1"
      ;;
    --no-promote) PROMOTE=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) qnn_die "未知参数: $1" ;;
  esac
  shift
done

[ -n "$IMAGE_PATH" ] && DEPLOY=1
if [ "$PROMOTE" = "0" ] && [ "$DEPLOY" = "1" ]; then
  qnn_die "--no-promote 不能和 --deploy/--image 同时使用"
fi

if [ "$DRY_RUN" = "1" ]; then
  qnn_require_file "$ROOT/docker/Dockerfile"
  qnn_require_file "$ROOT/utils/export_minimind_mnn.py"
  qnn_require_file "$ROOT/scripts/generate_qnn.py"
  qnn_require_file "$ROOT/scripts/npu_convert.py"
  printf '%s\n' \
    "LLM pipeline:" \
    "  1. Docker/QAIRT 环境" \
    "  2. PTH -> 临时 HF -> ONNX skeleton -> MNN" \
    "  3. generateIO + compilefornpu 拆分 Prefill/Decode 子图" \
    "  4. QNN SDK 生成 graph0.bin ... graph9.bin" \
    "  5. 生成候选包并按选项更新 artifacts" \
    "  6. 可选 Android 部署和完整图文执行"
  exit 0
fi

if [ "${QNN_PIPELINE_IN_CONTAINER:-0}" != "1" ]; then
  args=()
  [ "$PROMOTE" = "0" ] && args+=(--no-promote)
  qnn_run_in_docker "$ROOT" "/workspace/scripts/convert_llm_qnn.sh" "${args[@]}"
  if [ "$DEPLOY" = "1" ]; then
    qnn_build_deploy_and_run "$ROOT" "$IMAGE_PATH" "$PROMPT"
  fi
  exit 0
fi

cd /workspace
qnn_setup_container_env
qnn_build_mnn_convert
qnn_build_mnn_qnn_tools

qnn_require_file third_party/minimind-v/out/sft_vlm_768.pth
qnn_require_dir third_party/minimind-v/model

LLM_WORK=/workspace/build/llm_qnn
CACHE="$LLM_WORK/cache"
PACKAGE="$LLM_WORK/package"
trap 'qnn_fix_ownership \
  /workspace/build/llm_qnn \
  /workspace/build/artifact_backups \
  /workspace/llm_mnn \
  /workspace/minimind_hf \
  /workspace/src/models \
  /workspace/artifacts/qnn/llm \
  /workspace/artifacts/qnn/llm_config_qnn.json' EXIT

qnn_log 1/4 "导出 CPU LLM MNN 和公共模型资源"
./scripts/run_setup.sh

qnn_log 2/4 "generateIO + compilefornpu 拆分 LLM 子图"
rm -rf "$CACHE" src/models/qnn src/models/config_qnn.json
python3 scripts/generate_qnn.py \
  --model src/models \
  --soc_id "${QNN_SOC_ID:-57}" \
  --dsp_arch "${QNN_DSP_ARCH:-v75}" \
  --mnn_path third_party/MNN/build_qnn_host \
  --cache_path build/llm_qnn/cache \
  --chunk_size "${QNN_CHUNK_SIZE:-128}" \
  --model_name llm.mnn

qnn_log 3/4 "整理 LLM 候选包"
qnn_require_file src/models/qnn/llm.mnn
for index in 0 1 2 3 4 5 6 7 8 9; do
  qnn_require_file "src/models/qnn/graph${index}.bin"
done
qnn_require_file src/models/config_qnn.json

rm -rf "$PACKAGE"
mkdir -p "$PACKAGE"
cp src/models/qnn/llm.mnn "$PACKAGE/llm.mnn"
# MNN LLM runtime always validates an external-weight path. The QNN wrapper
# contains no external weights, so keep the required placeholder empty.
: > "$PACKAGE/llm.mnn.weight"
for index in 0 1 2 3 4 5 6 7 8 9; do
  cp "src/models/qnn/graph${index}.bin" "$PACKAGE/graph${index}.bin"
done
(
  cd "$PACKAGE"
  sha256sum llm.mnn llm.mnn.weight graph*.bin > SHA256SUMS
)
cp src/models/config_qnn.json "$LLM_WORK/llm_config_qnn.json"

if [ "$PROMOTE" = "1" ]; then
  qnn_log 4/4 "更新正式 LLM artifacts"
  backup="/workspace/build/artifact_backups/llm-$(date +%Y%m%d-%H%M%S)"
  qnn_backup_and_replace_dir "$PACKAGE" \
    /workspace/artifacts/qnn/llm \
    "$backup"
  if [ -f /workspace/artifacts/qnn/llm_config_qnn.json ]; then
    cp -a /workspace/artifacts/qnn/llm_config_qnn.json \
      "$backup/llm_config_qnn.json"
  fi
  cp "$LLM_WORK/llm_config_qnn.json" \
    /workspace/artifacts/qnn/llm_config_qnn.json
  qnn_log Done "已更新 LLM artifacts；旧版本备份到 $backup"
else
  qnn_log 4/4 "保留 LLM 候选包，不更新 artifacts"
  qnn_log Done "候选包位于 $PACKAGE"
fi
