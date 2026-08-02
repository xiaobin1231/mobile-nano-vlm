#!/usr/bin/env bash

set -euo pipefail

DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/mobile-nano-vlm}"
VLM_THREADS="${VLM_THREADS:-4}"
LOG_FILE="${VLM_LOG_FILE:-vlm_server.log}"

die() {
    printf '错误: %s\n' "$*" >&2
    exit 1
}

require_device() {
    command -v adb >/dev/null 2>&1 || die "主机找不到 adb"
    adb get-state >/dev/null 2>&1 || die "没有已连接的 adb 设备"
}

client() {
    adb shell "cd '$DEVICE_ROOT' && ./minimind_cli client $*"
}

status() {
    if client ping >/dev/null 2>&1; then
        printf 'VLM daemon: READY\n'
        adb shell "ps -A | grep '[m]inimind_cli' || true"
        return 0
    fi
    printf 'VLM daemon: STOPPED\n'
    return 1
}

start() {
    if status >/dev/null 2>&1; then
        printf 'VLM daemon 已经运行\n'
        return 0
    fi

    adb shell "test -x '$DEVICE_ROOT/minimind_cli'" \
        || die "手机缺少 $DEVICE_ROOT/minimind_cli，请先部署"
    adb shell "test -f '$DEVICE_ROOT/models/qnn/llm.mnn'" \
        || die "手机缺少 QNN LLM 模型，请先部署"
    adb shell "test -f '$DEVICE_ROOT/models/vision_qnn/vision.mnn'" \
        || die "手机缺少 QNN Vision Wrapper，请先部署"

    adb shell "cd '$DEVICE_ROOT'; \
        : >'$LOG_FILE'; \
        export LD_LIBRARY_PATH='$DEVICE_ROOT/models/qnn/lib'; \
        export ADSP_LIBRARY_PATH='$DEVICE_ROOT/models/qnn/lib'; \
        nohup ./minimind_cli server \
          models/ models/vision_qnn/vision.mnn '$VLM_THREADS' \
          >'$LOG_FILE' 2>&1 </dev/null &" >/dev/null

    for _ in $(seq 1 30); do
        if status >/dev/null 2>&1; then
            status
            return 0
        fi
        sleep 1
    done
    adb shell "cd '$DEVICE_ROOT' && cat '$LOG_FILE'" >&2 || true
    die "VLM daemon 未在 30 秒内就绪"
}

stop() {
    if ! status >/dev/null 2>&1; then
        printf 'VLM daemon 未运行\n'
        return 0
    fi
    client shutdown >/dev/null
    for _ in $(seq 1 10); do
        if ! status >/dev/null 2>&1; then
            printf 'VLM daemon 已停止\n'
            return 0
        fi
        sleep 1
    done
    die "VLM daemon 未正常退出"
}

logs() {
    adb shell "cd '$DEVICE_ROOT' && tail -n 100 -f '$LOG_FILE'"
}

require_device
case "${1:-}" in
    start) start ;;
    status) status ;;
    stop) stop ;;
    restart) stop; start ;;
    logs) logs ;;
    *)
        echo "Usage: $0 {start|status|stop|restart|logs}" >&2
        exit 1
        ;;
esac
