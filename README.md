# mobile-nano-vlm

基于 MNN 与 Qualcomm QNN/HTP，在 Android 移动端运行 MiniMind-V。

当前验证环境：Snapdragon 8 Gen 3（SM8650）、HTP v75、QAIRT
2.48.0.260626、Android NDK r27d。模型转换在 Docker 中执行；Android 交叉编译
和 adb 调试在主机执行。

## 1. 准备

模型文件：

```text
third_party/minimind-v/out/sft_vlm_768.pth
third_party/minimind-v/model/siglip2-base-p32-256-ve/
```

依赖目录`DEPS_ROOT`：

```bash
export DEPS_ROOT=/path/to/deps
mkdir -p "$DEPS_ROOT"
```

下载以下版本到DEPS_ROOT路径下：

| 依赖 | 已验证版本 | 官方入口 |
|---|---|---|
| Qualcomm AI Runtime SDK | 2.48.0.260626 | [Qualcomm Software Center](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community) |
| Android NDK | r27d | [Android NDK Downloads](https://developer.android.com/ndk/downloads) |
| CMake | 4.3.4 Linux x86_64 | [CMake Downloads](https://cmake.org/download/) |
| JDK | 17（仅编译 Android App） | [Eclipse Temurin](https://adoptium.net/temurin/releases/?version=17) |

QAIRT 下载需要登录 Qualcomm Software Center，并接受相应许可。选择
`Qualcomm AI Runtime Community` 的 `2.48.0.260626`，下载文件名应类似
`v2.48.0.260626.zip`。

解压到 `DEPS_ROOT`：

```bash
unzip /path/to/downloads/v2.48.0.260626.zip \
  -d "$DEPS_ROOT"

unzip /path/to/downloads/android-ndk-r27d-linux.zip \
  -d "$DEPS_ROOT"

tar -xzf /path/to/downloads/cmake-4.3.4-linux-x86_64.tar.gz \
  -C "$DEPS_ROOT"
```

解压后的目录必须是：

```text
/path/to/deps/
├── qairt/2.48.0.260626/
├── android-ndk-r27d/
└── cmake-4.3.4-linux-x86_64/
```

`adb` 可从 [Android SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools)
安装并加入 `PATH` 环境变量。确认 Docker 和手机连接正常：

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

构建产物：

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

## 3. 编译并部署 Android

设置主机编译环境：

```bash
export DEPS_ROOT=/path/to/deps
source scripts/env_setup.sh
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

如需图形化 Android App，使用 JDK 17 构建并安装：

```bash
export JAVA_HOME=/path/to/jdk-17
(cd android && ./gradlew assembleDebug)
./scripts/deploy_qnn_android.sh --install-apk
```

App 会以前台服务常驻模型，通过内部 Unix Domain Socket 流式收取结果；用户
不需要配置 Socket 地址或最大生成 token 数。详见
[Android App 常驻服务集成](docs/android-app-integration.md)。

## 4. 启动常驻推理服务

启动后模型和 QNN Context 只加载一次：

```bash
./scripts/vlm_daemon.sh start
./scripts/vlm_daemon.sh status
```

进入手机发送请求：

```sh
cd /data/local/tmp/mobile-nano-vlm

./minimind_cli client text '你好'
./minimind_cli client vision test_image.jpg '请描述这张图片'
```

停止服务：

```bash
./scripts/vlm_daemon.sh stop
```

协议和调试方法见[常驻 VLM 推理服务](docs/vlm-daemon.md)。

## 5. 单次命令运行

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

## 6. 详细说明

- [Vision QNN 转换流程](docs/vision-qnn-conversion.md)
- [LLM QNN 转换流程](docs/llm-qnn-conversion.md)
- [常驻 VLM 推理服务](docs/vlm-daemon.md)
- [端侧性能分析](docs/performance.md)
