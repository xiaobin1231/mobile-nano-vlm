#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
if [ -z "${QNN_SDK_ROOT:-}" ]; then
    echo "请先设置 QNN_SDK_ROOT=/path/to/qairt/2.48.0.260626" >&2
    exit 1
fi
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/mobile-nano-vlm}"
QNN_PACKAGE="${QNN_PACKAGE:-$ROOT/artifacts/qnn}"
LLM_QNN="$QNN_PACKAGE/llm"
VISION_QNN="$QNN_PACKAGE/vision"
VISION_POSITION="$VISION_QNN/vision_position_f32.bin"

require_file() {
    if [ ! -f "$1" ]; then
        echo "缺少文件: $1" >&2
        exit 1
    fi
}

require_file "$ROOT/src/build_android/minimind_cli"
require_file "$ROOT/src/models/embeddings_bf16.bin"
require_file "$ROOT/src/models/tokenizer.txt"
require_file "$ROOT/src/models/llm_config.json"
require_file "$QNN_PACKAGE/llm_config_qnn.json"
require_file "$LLM_QNN/llm.mnn"
require_file "$LLM_QNN/llm.mnn.weight"
require_file "$VISION_QNN/vision.mnn"
require_file "$VISION_QNN/graph0.bin"
require_file "$VISION_POSITION"

QNN_ANDROID_LIB="$QNN_SDK_ROOT/lib/aarch64-android"
QNN_SKEL="$QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"
for file in libQnnHtp.so libQnnSystem.so libQnnHtpV75Stub.so; do
    require_file "$QNN_ANDROID_LIB/$file"
done
require_file "$QNN_SKEL"

adb shell mkdir -p \
    "$DEVICE_ROOT/models/qnn/lib" \
    "$DEVICE_ROOT/models/vision_qnn/qnn"

adb push "$ROOT/src/build_android/minimind_cli" "$DEVICE_ROOT/minimind_cli"
adb shell chmod 755 "$DEVICE_ROOT/minimind_cli"
adb push "$ROOT/src/models/embeddings_bf16.bin" "$DEVICE_ROOT/models/embeddings_bf16.bin"
adb push "$ROOT/src/models/tokenizer.txt" "$DEVICE_ROOT/models/tokenizer.txt"
adb push "$ROOT/src/models/llm_config.json" "$DEVICE_ROOT/models/llm_config.json"
adb push "$QNN_PACKAGE/llm_config_qnn.json" "$DEVICE_ROOT/models/llm_config_qnn.json"

adb push "$LLM_QNN/llm.mnn" "$DEVICE_ROOT/models/qnn/llm.mnn"
adb push "$LLM_QNN/llm.mnn.weight" "$DEVICE_ROOT/models/qnn/llm.mnn.weight"
for index in 0 1 2 3 4 5 6 7 8 9; do
    require_file "$LLM_QNN/graph${index}.bin"
    adb push "$LLM_QNN/graph${index}.bin" "$DEVICE_ROOT/models/qnn/graph${index}.bin"
done

adb push "$VISION_QNN/vision.mnn" "$DEVICE_ROOT/models/vision_qnn/vision.mnn"
adb push "$VISION_QNN/graph0.bin" "$DEVICE_ROOT/models/vision_qnn/qnn/graph0.bin"
adb push "$VISION_POSITION" "$DEVICE_ROOT/models/vision_qnn/vision_position_f32.bin"

adb push "$QNN_ANDROID_LIB/libQnnHtp.so" "$DEVICE_ROOT/models/qnn/lib/libQnnHtp.so"
adb push "$QNN_ANDROID_LIB/libQnnSystem.so" "$DEVICE_ROOT/models/qnn/lib/libQnnSystem.so"
adb push "$QNN_ANDROID_LIB/libQnnHtpV75Stub.so" "$DEVICE_ROOT/models/qnn/lib/libQnnHtpV75Stub.so"
adb push "$QNN_SKEL" "$DEVICE_ROOT/models/qnn/lib/libQnnHtpV75Skel.so"

# These FastRPC libraries come from the connected device and must match its
# vendor image. Keeping them in the app-private runtime directory avoids the
# Android linker namespace restriction for a ProcessBuilder-launched binary.
adb shell cp /vendor/lib64/libcdsprpc.so "$DEVICE_ROOT/models/qnn/lib/libcdsprpc.so"
adb shell cp /vendor/lib64/vendor.qti.hardware.dsp@1.0.so \
    "$DEVICE_ROOT/models/qnn/lib/vendor.qti.hardware.dsp@1.0.so"
adb shell cp /vendor/lib64/libvmmem.so "$DEVICE_ROOT/models/qnn/lib/libvmmem.so"

if [ "${1:-}" = "--install-apk" ]; then
    require_file "$ROOT/android/app/build/outputs/apk/debug/app-debug.apk"
    adb install -r "$ROOT/android/app/build/outputs/apk/debug/app-debug.apk"
fi

adb shell am force-stop com.minimind.vlm >/dev/null 2>&1 || true
adb shell am start -n com.minimind.vlm/.MainActivity >/dev/null 2>&1 || true

echo "QNN 资源已部署到 $DEVICE_ROOT"
echo "文本 smoke test:"
echo "  adb shell am start -n com.minimind.vlm/.MainActivity --es qnn_smoke_prompt hello"
echo "图文 smoke test:"
echo "  adb shell am start -n com.minimind.vlm/.MainActivity --es qnn_smoke_prompt 描述图片 --es qnn_smoke_image /data/local/tmp/example.jpg"
