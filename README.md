# mobile-nano-vlm

基于 MNN 与 Qualcomm QNN/HTP，在 Android 移动端运行 MiniMind-V。

当前验证环境：Snapdragon 8 Gen 3（SM8650）、HTP v75、QAIRT
2.48.0.260626、Android NDK r27d。模型转换在 Docker 中执行；Android 交叉编译
和 adb 调试在主机执行，不需要修改 `third_party/MNN` 源码。

## 1. 准备

工程需要以下模型文件：

```text
third_party/minimind-v/out/sft_vlm_768.pth
third_party/minimind-v/model/siglip2-base-p32-256-ve/
```

主机依赖目录示例：

```text
/path/to/deps/
├── qairt/2.48.0.260626/
├── android-ndk-r27d/
└── cmake/
```

确认 Docker 和手机连接正常：

```bash
docker --version
adb devices
```

`adb devices` 应显示一台状态为 `device` 的设备。

## 2. 从零转换模型

拉取公开转换镜像并设置依赖：

```bash
docker pull jerry1231/mobile-nano-vlm-build:latest

export QNN_DOCKER_IMAGE=jerry1231/mobile-nano-vlm-build:latest
export DEPS_ROOT=/path/to/deps
```

在工程根目录依次执行：

```bash
./scripts/convert_vision_qnn.sh
./scripts/convert_llm_qnn.sh
```

脚本会自动启动临时 Container、挂载工程与 QAIRT、完成转换并删除 Container；
无需手动执行 `docker run`。Vision 和 LLM 相互独立，LLM 脚本不会重复转换
Vision。

最终产物：

```text
artifacts/qnn/
├── llm_config_qnn.json
├── llm/
│   ├── llm.mnn
│   ├── llm.mnn.weight
│   ├── graph0.bin ... graph9.bin
│   └── SHA256SUMS
└── vision/
    ├── vision.mnn
    ├── graph0.bin
    ├── vision_position_f32.bin
    └── SHA256SUMS
```

检查产物：

```bash
(
  cd artifacts/qnn/vision
  sha256sum -c SHA256SUMS
)

(
  cd artifacts/qnn/llm
  sha256sum -c SHA256SUMS
)
```

如已有完整的 `artifacts/qnn/`，可跳过本节。

## 3. 编译并部署 Android

设置主机编译环境：

```bash
export ANDROID_NDK="$DEPS_ROOT/android-ndk-r27d"
export QNN_SDK_ROOT="$DEPS_ROOT/qairt/2.48.0.260626"
export CMAKE_BIN="$DEPS_ROOT/cmake/bin/cmake"
```

编译并推送程序、模型和 QNN 运行库：

```bash
./scripts/build.sh android --mnn
./scripts/deploy_qnn_android.sh
```

默认部署目录为：

```text
/data/local/tmp/mobile-nano-vlm
```

## 4. 真机运行

在主机推送测试图片：

```bash
adb push /path/to/test_image.jpg \
  /data/local/tmp/mobile-nano-vlm/test_image.jpg
```

进入手机：

```bash
adb shell
```

在手机 Shell 中执行：

```sh
cd /data/local/tmp/mobile-nano-vlm

export LD_LIBRARY_PATH=$PWD/models/qnn/lib
export ADSP_LIBRARY_PATH=$PWD/models/qnn/lib

./minimind_cli vision \
  models/ \
  models/vision_qnn/vision.mnn \
  test_image.jpg \
  '请描述这张图'
```

正常情况下会打印图片尺寸、推理耗时和图片描述。

## 5. 常用入口

| 文件 | 用途 |
|---|---|
| `scripts/convert_vision_qnn.sh` | 自动转换 Vision Wrapper 与 HTP Context |
| `scripts/convert_llm_qnn.sh` | 自动转换 LLM Wrapper 与 10 个 HTP Context |
| `scripts/build.sh` | 编译 Linux 或 Android 程序及 MNN |
| `scripts/deploy_qnn_android.sh` | 通过 adb 部署完整 QNN VLM |
| `docker/build_public_image.sh` | 构建公开转换基础镜像 |
| `docker/audit_public_image.sh` | 检查镜像依赖和敏感内容 |
| `docker/push_public_image.sh` | 审计后推送公开镜像 |

以下文件是转换入口调用的内部实现，不需要用户直接执行：

```text
scripts/_qnn_pipeline_common.sh
scripts/run_setup.sh
scripts/generate_qnn.py
scripts/npu_convert.py
utils/export_minimind_mnn.py
utils/export_vision_pipeline_onnx.py
utils/prepare_vision_qnn_onnx.py
utils/patch_mnn_wrapper_nchw.cpp
```

## 6. 详细文档

- [Vision QNN 转换流程](docs/vision-qnn-conversion.md)
- [LLM QNN 转换流程](docs/llm-qnn-conversion.md)
- [公开模型转换基础镜像](docs/public-docker-image.md)

转换脚本参数可通过以下命令查看：

```bash
./scripts/convert_vision_qnn.sh --help
./scripts/convert_llm_qnn.sh --help
```
