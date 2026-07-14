#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
MNN="$PROJ/third_party/MNN"
SRC="$PROJ/src"

if [ $# -lt 1 ]; then
    echo "Usage: $0 {linux|android} [--mnn]"
    echo ""
    echo "  linux   - 本地 Linux 编译"
    echo "  android - Android arm64-v8a 交叉编译"
    echo "  --mnn   - 同时编译 MNN 库（默认跳过，仅编译推理程序）"
    exit 1
fi

PLATFORM="$1"
BUILD_MNN=false
if [ "${2:-}" = "--mnn" ]; then
    BUILD_MNN=true
fi

JOBS="${BUILD_JOBS:-$(nproc)}"
echo "make -j$JOBS  (MNN: $BUILD_MNN)"

# ---- 设置平台相关变量 ----
case "$PLATFORM" in
  linux)
    BUILD_SUFFIX="build"
    CMAKE_BIN="cmake"
    MNN_EXTRA_FLAGS="-DMNN_USE_METAL=OFF"
    ;;

  android)
    BUILD_SUFFIX="build_android"
    NDK="${ANDROID_NDK:?请设置: export ANDROID_NDK=/path/to/ndk}"
    CMAKE_BIN="${CMAKE_BIN:-cmake}"

    if [ ! -d "$NDK" ]; then
      echo "错误: NDK 未找到: $NDK"
      exit 1
    fi
    if [ ! -x "$CMAKE_BIN" ] && ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
      echo "错误: CMake 未找到: $CMAKE_BIN"
      exit 1
    fi

    TOOLCHAIN=(
      -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
      -DANDROID_ABI=arm64-v8a
      -DANDROID_PLATFORM=android-24
    )
    MNN_EXTRA_FLAGS="
      -DMNN_BUILD_SHARED_LIBS=OFF
      -DMNN_ARM82=ON
      -DMNN_OPENCL=OFF
      -DMNN_VULKAN=ON
      -DMNN_BUILD_FOR_ANDROID_COMMAND=ON
      -DMNN_BUILD_TOOLS=OFF
      -DMNN_BUILD_DEMO=OFF
      -DMNN_BUILD_TRAIN=OFF
      -DMNN_BUILD_CONVERTER=OFF
      -DMNN_EVALUATION=OFF
    "
    ;;

  *)
    echo "Usage: $0 {linux|android} [--mnn]"
    echo ""
    echo "  linux   - 本地 Linux 编译"
    echo "  android - Android arm64-v8a 交叉编译"
    echo "  --mnn   - 同时编译 MNN 库"
    exit 1
    ;;
esac

# ---- Step 1: 编译 MNN（可选） ----
if $BUILD_MNN; then
  echo "=========================================="
  echo " [1/2] Building MNN for $PLATFORM"
  echo "=========================================="

  if [ "$PLATFORM" = "android" ]; then
    rm -rf "$MNN/$BUILD_SUFFIX"
  fi
  mkdir -p "$MNN/$BUILD_SUFFIX" && cd "$MNN/$BUILD_SUFFIX"

  # shellcheck disable=SC2086
  $CMAKE_BIN .. \
    "${TOOLCHAIN[@]}" \
    -DMNN_BUILD_LLM=ON \
    -DMNN_BUILD_LLM_OMNI=ON \
    $MNN_EXTRA_FLAGS

  make -j"$JOBS"
  echo "Done: $MNN/$BUILD_SUFFIX/libMNN.a"
else
  echo "[skip] MNN (use --mnn to rebuild)"
fi

# ---- Step 2: 编译推理程序 ----
echo ""
echo "=========================================="
echo " [$( $BUILD_MNN && echo '2/2' || echo '1/1')] Building minimind_cli for $PLATFORM"
echo "=========================================="

mkdir -p "$SRC/$BUILD_SUFFIX" && cd "$SRC/$BUILD_SUFFIX"

# shellcheck disable=SC2086
$CMAKE_BIN .. "${TOOLCHAIN[@]}"

make -j"$JOBS"
echo ""
echo "Done: $SRC/$BUILD_SUFFIX/minimind_cli"
file "$SRC/$BUILD_SUFFIX/minimind_cli"

echo ""
echo "=========================================="
echo " Build complete ($PLATFORM)"
echo " Binary: $SRC/$BUILD_SUFFIX/minimind_cli"
echo "=========================================="
