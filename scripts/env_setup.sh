#!/usr/bin/env bash

# Android 主机编译与 QNN 部署环境。
# 必须通过 `source scripts/env_setup.sh` 加载，使导出的变量保留在当前 Shell。

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "请使用 source 加载此脚本：" >&2
    echo "  export DEPS_ROOT=/path/to/deps" >&2
    echo "  source scripts/env_setup.sh" >&2
    exit 1
fi

mnv_env_error() {
    printf '错误: %s\n' "$*" >&2
    return 1
}

mnv_env_require_dir() {
    [ -d "$1" ] || mnv_env_error "目录不存在: $1"
}

mnv_env_require_file() {
    [ -f "$1" ] || mnv_env_error "文件不存在: $1"
}

mnv_env_setup() {
    if [ -z "${DEPS_ROOT:-}" ]; then
        mnv_env_error "请先设置 DEPS_ROOT=/path/to/deps"
        return 1
    fi
    mnv_env_require_dir "$DEPS_ROOT" || return 1

    # 转成绝对路径，避免用户切换工作目录后环境变量失效。
    DEPS_ROOT="$(cd "$DEPS_ROOT" && pwd)"

    export QAIRT_VERSION="${QAIRT_VERSION:-2.48.0.260626}"
    export ANDROID_NDK="${ANDROID_NDK:-$DEPS_ROOT/android-ndk-r27d}"
    export QNN_SDK_ROOT="${QNN_SDK_ROOT:-$DEPS_ROOT/qairt/$QAIRT_VERSION}"
    export CMAKE_BIN="${CMAKE_BIN:-$DEPS_ROOT/cmake-4.3.4-linux-x86_64/bin/cmake}"
    export DEPS_ROOT

    mnv_env_require_file \
        "$ANDROID_NDK/build/cmake/android.toolchain.cmake" || return 1
    mnv_env_require_dir "$QNN_SDK_ROOT/include/QNN" || return 1

    if [ ! -x "$CMAKE_BIN" ]; then
        mnv_env_error "CMake 不可执行: $CMAKE_BIN"
        return 1
    fi

    printf '%s\n' \
        "Android/QNN 主机环境已就绪：" \
        "  DEPS_ROOT=$DEPS_ROOT" \
        "  ANDROID_NDK=$ANDROID_NDK" \
        "  QNN_SDK_ROOT=$QNN_SDK_ROOT" \
        "  CMAKE_BIN=$CMAKE_BIN" \
        "  $($CMAKE_BIN --version | head -n 1)"
}

if ! mnv_env_setup; then
    unset -f mnv_env_setup mnv_env_require_dir mnv_env_require_file mnv_env_error
    return 1
fi
unset -f mnv_env_setup mnv_env_require_dir mnv_env_require_file mnv_env_error
