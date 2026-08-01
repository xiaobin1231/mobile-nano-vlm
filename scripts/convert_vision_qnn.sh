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
  ./scripts/convert_vision_qnn.sh [options]

Options:
  --deploy           转换后编译、部署完整 QNN VLM
  --image PATH       部署后推送图片并执行图文问答
  --prompt TEXT      图文问答提示词，默认“请描述这幅图”
  --no-promote       只生成 build/vision_qnn/package，不更新 artifacts
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
  qnn_require_file "$ROOT/utils/export_vision_pipeline_onnx.py"
  qnn_require_file "$ROOT/utils/prepare_vision_qnn_onnx.py"
  qnn_require_file "$ROOT/utils/patch_mnn_wrapper_nchw.cpp"
  qnn_require_file "$ROOT/configs/qnn/vision/compilefornpu_input.json"
  qnn_require_file "$ROOT/configs/qnn/vision/compilefornpu_qnn.json"
  printf '%s\n' \
    "Vision pipeline:" \
    "  1. Docker/QAIRT 环境" \
    "  2. 导出 ONNX、原始 MNN、位置编码" \
    "  3. generateIO + compilefornpu 生成 Wrapper" \
    "  4. NC4HW4 -> NCHW" \
    "  5. qnn-onnx-converter -> Model Library -> HTP Context" \
    "  6. 生成候选包并按选项更新 artifacts" \
    "  7. 可选 Android 部署和图文执行"
  exit 0
fi

if [ "${QNN_PIPELINE_IN_CONTAINER:-0}" != "1" ]; then
  args=()
  [ "$PROMOTE" = "0" ] && args+=(--no-promote)
  qnn_run_in_docker "$ROOT" "/workspace/scripts/convert_vision_qnn.sh" "${args[@]}"
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
qnn_require_dir third_party/minimind-v/model/siglip2-base-p32-256-ve

VISION_WORK=/workspace/build/vision_qnn
WRAPPER_WORK="$VISION_WORK/wrapper"
PACKAGE="$VISION_WORK/package"
trap 'qnn_fix_ownership \
  /workspace/build/vision_qnn \
  /workspace/build/artifact_backups \
  /workspace/vision_export \
  /workspace/artifacts/qnn/vision' EXIT

qnn_log 1/9 "导出 Vision ONNX 和位置编码"
python utils/export_vision_pipeline_onnx.py \
  --save_dir out \
  --weight sft_vlm \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --out_dir vision_export

qnn_log 2/9 "ONNX 转原始 Vision MNN"
third_party/MNN/build/MNNConvert \
  -f ONNX \
  --modelFile vision_export/vision_encode_proj.onnx \
  --MNNModel vision_export/vision_encode_proj.mnn \
  --bizCode MNN \
  --fp16

qnn_log 3/9 "generateIO 准备 compilefornpu 输入"
rm -rf "$WRAPPER_WORK"
mkdir -p "$WRAPPER_WORK/testdir"
"$MNN_QNN_HOST/generateIO" \
  vision_export/vision_encode_proj.mnn \
  configs/qnn/vision/compilefornpu_input.json \
  "$WRAPPER_WORK/testdir"

qnn_log 4/9 "compilefornpu 生成 MNN Plugin Wrapper"
(
  cd "$WRAPPER_WORK"
  "$MNN_QNN_HOST/compilefornpu" \
    /workspace/vision_export/vision_encode_proj.mnn \
    qnn/vision_encode_proj.mnn \
    /workspace/configs/qnn/vision/compilefornpu_qnn.json
)

qnn_log 5/9 "Wrapper Input Layout: NC4HW4 -> NCHW"
g++ -std=c++17 \
  -Ithird_party/MNN/schema/current \
  -Ithird_party/MNN/3rd_party/flatbuffers/include \
  utils/patch_mnn_wrapper_nchw.cpp \
  -o "$VISION_WORK/patch_mnn_wrapper_nchw"
"$VISION_WORK/patch_mnn_wrapper_nchw" \
  "$WRAPPER_WORK/qnn/vision_encode_proj.mnn" \
  "$VISION_WORK/vision.mnn"
if ! strings "$VISION_WORK/vision.mnn" | grep -Fxq 't1554'; then
  qnn_die "compilefornpu Wrapper 未声明预期的 QNN 输出 t1554"
fi

qnn_log 6/9 "准备 Wrapper 同名接口并用 QNN SDK 转换 ONNX"
python utils/prepare_vision_qnn_onnx.py \
  --input vision_export/vision_encode_proj.onnx \
  --output "$VISION_WORK/graph0.onnx"
qnn-onnx-converter \
  -i "$VISION_WORK/graph0.onnx" \
  -d t0 1,3,256,256 \
  -d t11 1,768,64 \
  --input_layout t0 NCHW \
  --input_layout t11 NONTRIVIAL \
  --preserve_io \
  --float_bitwidth 32 \
  -o "$VISION_WORK/graph0.cpp"
if ! grep -q '\.name= "t1554"' "$VISION_WORK/graph0.cpp"; then
  qnn_die "QNN SDK Graph 未导出 Wrapper 需要的输出 t1554"
fi

qnn_log 7/9 "生成 x86 QNN Model Library"
rm -rf "$VISION_WORK/model_libs"
qnn-model-lib-generator \
  -c "$VISION_WORK/graph0.cpp" \
  -b "$VISION_WORK/graph0.bin" \
  -t x86_64-linux-clang \
  -l graph0 \
  -o "$VISION_WORK/model_libs"

qnn_log 8/9 "生成目标 HTP Context"
cp configs/qnn/vision/context_config.json \
  "$VISION_WORK/context_config.json"
cp configs/qnn/vision/htp_backend_extensions.json \
  "$VISION_WORK/htp_backend_extensions.json"
python - "$VISION_WORK/htp_backend_extensions.json" <<'PY'
import json
import os
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as file:
    config = json.load(file)
config["devices"][0]["soc_id"] = int(os.environ.get("QNN_SOC_ID", "57"))
config["devices"][0]["dsp_arch"] = os.environ.get("QNN_DSP_ARCH", "v75")
with open(path, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
PY
rm -rf "$VISION_WORK/context"
(
  cd "$VISION_WORK"
  export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
  qnn-context-binary-generator \
    --model ./model_libs/x86_64-linux-clang/libgraph0.so \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --binary_file graph0 \
    --output_dir ./context \
    --config_file ./context_config.json
)

qnn_log 9/9 "整理 Vision 候选包"
rm -rf "$PACKAGE"
mkdir -p "$PACKAGE"
cp "$VISION_WORK/vision.mnn" "$PACKAGE/vision.mnn"
cp "$VISION_WORK/context/graph0.bin" "$PACKAGE/graph0.bin"
cp vision_export/vision_position_f32.bin \
  "$PACKAGE/vision_position_f32.bin"
(
  cd "$PACKAGE"
  sha256sum vision.mnn graph0.bin vision_position_f32.bin > SHA256SUMS
)

if [ "$PROMOTE" = "1" ]; then
  backup="/workspace/build/artifact_backups/vision-$(date +%Y%m%d-%H%M%S)"
  qnn_backup_and_replace_dir "$PACKAGE" \
    /workspace/artifacts/qnn/vision \
    "$backup"
  qnn_log Done "已更新 artifacts/qnn/vision；旧版本备份到 $backup"
else
  qnn_log Done "候选包位于 $PACKAGE"
fi
