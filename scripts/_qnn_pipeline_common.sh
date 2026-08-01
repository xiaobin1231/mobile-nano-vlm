#!/usr/bin/env bash

# Shared implementation for convert_vision_qnn.sh and convert_llm_qnn.sh.
# This file is internal; users should call one of the two public scripts.

set -euo pipefail

qnn_log() {
  printf '\n[%s] %s\n' "$1" "$2"
}

qnn_die() {
  printf '错误: %s\n' "$*" >&2
  exit 1
}

qnn_project_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  cd "$script_dir/.." && pwd
}

qnn_require_file() {
  [ -f "$1" ] || qnn_die "缺少文件: $1"
}

qnn_require_dir() {
  [ -d "$1" ] || qnn_die "缺少目录: $1"
}

qnn_container_sdk_root() {
  printf '/deps/qairt/%s' "${QAIRT_VERSION:-2.48.0.260626}"
}

qnn_build_image() {
  local root="$1"
  local image="${QNN_DOCKER_IMAGE:-mobile-nano-vlm-build:local}"
  if [ "${QNN_REBUILD_IMAGE:-0}" = "1" ] ||
      ! docker image inspect "$image" >/dev/null 2>&1; then
    qnn_log Docker "构建模型转换镜像 $image"
    docker build --network host \
      -t "$image" \
      -f "$root/docker/Dockerfile" \
      "$root/docker"
  fi
}

qnn_run_in_docker() {
  local root="$1"
  local public_script="$2"
  shift 2

  : "${DEPS_ROOT:?请先设置 DEPS_ROOT=/path/to/deps}"
  qnn_require_dir "$DEPS_ROOT"
  qnn_require_dir "$DEPS_ROOT/qairt/${QAIRT_VERSION:-2.48.0.260626}"
  command -v docker >/dev/null 2>&1 || qnn_die "主机找不到 docker"

  qnn_build_image "$root"

  local image="${QNN_DOCKER_IMAGE:-mobile-nano-vlm-build:local}"
  docker run --rm --network host \
    -e QNN_PIPELINE_IN_CONTAINER=1 \
    -e QAIRT_VERSION="${QAIRT_VERSION:-2.48.0.260626}" \
    -e PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    -e QNN_SOC_ID="${QNN_SOC_ID:-57}" \
    -e QNN_DSP_ARCH="${QNN_DSP_ARCH:-v75}" \
    -e QNN_CHUNK_SIZE="${QNN_CHUNK_SIZE:-128}" \
    -e BUILD_JOBS="${BUILD_JOBS:-8}" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -v "$root:/workspace" \
    -v "$DEPS_ROOT:/deps:ro" \
    -w /workspace \
    "$image" \
    bash "$public_script" "$@"
}

qnn_setup_container_env() {
  [ "${QNN_PIPELINE_IN_CONTAINER:-0}" = "1" ] ||
    qnn_die "该函数只能在转换容器内调用"

  local sdk_root
  sdk_root="$(qnn_container_sdk_root)"
  qnn_require_file "$sdk_root/bin/envsetup.sh"

  local public_venv="/opt/mobile-nano-vlm/venv"
  local public_marker="/opt/mobile-nano-vlm/public-deps.ready"
  local venv="/workspace/build/qairt_venv"
  local marker="$venv/.mobile_nano_vlm_qairt_ready"

  if [ -f "$public_marker" ] && [ -x "$public_venv/bin/python" ]; then
    qnn_log Python "复用公开镜像内的预装 Python 环境"
    # shellcheck disable=SC1091
    source "$public_venv/bin/activate"
  elif [ ! -f "$marker" ]; then
    qnn_log Python "创建 QAIRT Python 环境"
    python3 -m venv "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install --force-reinstall \
      numpy==1.26.4 \
      typing-extensions==4.14.0
    "$sdk_root/bin/check-python-dependency"
    python -m pip install \
      onnx==1.19.1 \
      pyyaml==6.0.3 \
      packaging==24.0 \
      torch==2.6.0 \
      transformers==4.57.6
    python -c \
      "import onnx, torch, transformers, yaml; print('Python dependencies OK')"
    touch "$marker"
  else
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
  fi

  # QAIRT 2.48 envsetup.sh 在 nounset 模式下直接读取这两个变量。
  # 先提供空默认值，避免公开镜像的干净环境触发 unbound variable。
  export PYTHONPATH="${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

  # shellcheck disable=SC1090
  source "$sdk_root/bin/envsetup.sh"
  qnn_require_file "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter"
}

qnn_build_mnn_convert() {
  if [ ! -x third_party/MNN/build/MNNConvert ]; then
    qnn_log MNN "构建 Linux MNNConvert"
    cmake -S third_party/MNN \
      -B third_party/MNN/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DMNN_BUILD_CONVERTER=ON \
      -DMNN_BUILD_TOOLS=OFF \
      -DMNN_BUILD_DEMO=OFF \
      -DMNN_BUILD_LLM=OFF
    cmake --build third_party/MNN/build \
      --target MNNConvert \
      -j"${BUILD_JOBS:-8}"
  fi
}

qnn_build_mnn_qnn_tools() {
  local build_dir="/workspace/third_party/MNN/build_qnn_host"
  if [ ! -x "$build_dir/generateIO" ] ||
      [ ! -x "$build_dir/compilefornpu" ]; then
    qnn_log MNN "构建 generateIO 和 compilefornpu"
    cmake -S third_party/MNN \
      -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DMNN_QNN=ON \
      -DMNN_QNN_CONVERT_MODE=ON \
      -DMNN_WITH_PLUGIN=OFF \
      -DMNN_BUILD_TOOLS=ON \
      -DMNN_BUILD_LLM=ON \
      -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
      -DQNN_SDK_ROOT="$QNN_SDK_ROOT"
    cmake --build "$build_dir" \
      --target generateIO compilefornpu \
      -j"${BUILD_JOBS:-8}"
  fi
  export MNN_QNN_HOST="$build_dir"
}

qnn_backup_and_replace_dir() {
  local source_dir="$1"
  local target_dir="$2"
  local backup_root="$3"

  if [ -d "$target_dir" ]; then
    mkdir -p "$backup_root"
    cp -a "$target_dir" "$backup_root/"
  fi
  mkdir -p "$target_dir"
  find "$target_dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  cp -a "$source_dir"/. "$target_dir"/
}

qnn_fix_ownership() {
  [ -n "${HOST_UID:-}" ] || return 0
  [ -n "${HOST_GID:-}" ] || return 0
  chown -R "$HOST_UID:$HOST_GID" "$@" 2>/dev/null || true
}

qnn_shell_quote() {
  local value="$1"
  local escaped
  escaped="$(printf '%s' "$value" | sed "s/'/'\\\\''/g")"
  printf "'%s'" "$escaped"
}

qnn_build_deploy_and_run() {
  local root="$1"
  local image_path="${2:-}"
  local prompt="${3:-请描述这幅图}"

  : "${ANDROID_NDK:?请先设置 ANDROID_NDK=/path/to/android-ndk-r27d}"
  : "${QNN_SDK_ROOT:?请先设置 QNN_SDK_ROOT=/path/to/qairt/2.48.0.260626}"
  export CMAKE_BIN="${CMAKE_BIN:-cmake}"

  command -v adb >/dev/null 2>&1 || qnn_die "主机找不到 adb"
  adb get-state >/dev/null 2>&1 || qnn_die "没有已连接的 adb 设备"

  qnn_log Android "编译 CLI 并部署完整 QNN VLM"
  (
    cd "$root"
    ./scripts/build.sh android --mnn
    ./scripts/deploy_qnn_android.sh
  )

  if [ -n "$image_path" ]; then
    qnn_require_file "$image_path"
    local image_name
    image_name="$(basename "$image_path")"
    local device_root="${DEVICE_ROOT:-/data/local/tmp/mobile-nano-vlm}"
    adb push "$image_path" "$device_root/$image_name"

    local q_root q_image q_prompt
    q_root="$(qnn_shell_quote "$device_root")"
    q_image="$(qnn_shell_quote "$image_name")"
    q_prompt="$(qnn_shell_quote "$prompt")"
    adb shell "cd $q_root &&
      export LD_LIBRARY_PATH=\$PWD/models/qnn/lib &&
      export ADSP_LIBRARY_PATH=\$PWD/models/qnn/lib &&
      ./minimind_cli vision models/ models/vision_qnn/vision.mnn $q_image $q_prompt"
  else
    qnn_log Android "部署完成；未提供 --image，因此跳过图文执行"
  fi
}
