# 公开模型转换基础镜像

本工程支持把可再分发的模型转换依赖制作成公开 Docker 镜像。镜像不会包含
Qualcomm QAIRT/QNN SDK、Android NDK、模型权重或工程源码。

## 1. 镜像包含什么

```text
Ubuntu 22.04
Python 3.10虚拟环境
PyTorch 2.6.0 CPU
Transformers 4.57.6
ONNX 1.19.1
NumPy 1.26.4
Yaspin 3.1.0
QAIRT转换器需要的公开Python包
CMake、Clang、G++、Make、Ninja、Java等构建工具
```

大部分 Python 版本清单位于：

```text
docker/requirements-public.txt
```

PyTorch 和 MNN Exporter 使用的 Yaspin 在 `docker/Dockerfile` 中单独固定版本。

## 2. 明确不包含什么

```text
Qualcomm QAIRT/QNN SDK
Android SDK或NDK
MiniMind-V与SigLIP权重
ONNX、MNN、QNN Context等模型产物
工程源码
个人绝对路径或凭据
```

QAIRT/QNN SDK 运行时由用户从本机 `$DEPS_ROOT` 只读挂载。Android NDK 只在
主机交叉编译阶段使用，不进入模型转换容器。

## 3. 本地构建

```bash
./docker/build_public_image.sh
```

默认生成：

```text
mobile-nano-vlm-build:local
```

指定名称：

```bash
./docker/build_public_image.sh \
  username/mobile-nano-vlm-build:latest
```

在 ARM Mac 上构建供 x86_64 QAIRT 使用的镜像：

```bash
DOCKER_PLATFORM=linux/amd64 \
  ./docker/build_public_image.sh \
  username/mobile-nano-vlm-build:latest
```

默认 Python 包使用清华源，PyTorch CPU Wheel 使用 PyTorch 官方索引。可以覆盖：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

## 4. 发布前审计

```bash
./docker/audit_public_image.sh \
  mobile-nano-vlm-build:local
```

审计会检查：

1. 镜像构建历史中没有个人绝对路径或模型文件名。
2. 镜像文件系统中没有 QAIRT、QNN、NDK、模型和 Context 文件。
3. Torch、ONNX、Transformers、NumPy、PyYAML 可以导入。
4. 镜像包含公开依赖完成标记。

审计不能替代法律审查，但能防止构建上下文误带入受限文件。

## 5. 推送 Docker Hub

先登录：

```bash
docker login
```

再执行：

```bash
./docker/push_public_image.sh \
  username/mobile-nano-vlm-build:latest
```

推送脚本会：

1. 审计本地源镜像。
2. 添加 Docker Hub Tag。
3. 再次审计目标 Tag。
4. 执行 `docker push`。

脚本要求显式提供包含 namespace 和 tag 的镜像名，防止误推到错误仓库。

## 6. 转换脚本使用公开镜像

如果使用自己本地构建的默认镜像，无需额外设置：

```bash
export DEPS_ROOT=/path/to/deps
./scripts/convert_vision_qnn.sh
```

使用本工程已经发布的 Docker Hub 镜像：

```bash
docker pull \
  jerry1231/mobile-nano-vlm-build:latest

export QNN_DOCKER_IMAGE=\
jerry1231/mobile-nano-vlm-build:latest
export DEPS_ROOT=/path/to/deps

./scripts/convert_vision_qnn.sh
./scripts/convert_llm_qnn.sh
```

用户不需要手动执行 `docker run` 或进入容器。转换脚本会自动启动临时
Container，并挂载：

```text
$PROJECT_ROOT → /workspace
$DEPS_ROOT → /deps:ro
```

模型导出、MNN Host 工具和 QNN Context 生成都在 Container 内执行。脚本结束后
Container 自动删除；产物保留在主机工程目录。QNN SDK 不会成为镜像层的一部分。

## 7. 版本更新

更新 Python 依赖时：

1. 修改 `docker/requirements-public.txt`。
2. 修改 Dockerfile 中单独安装的 Torch、Yaspin 版本。
3. 使用新 Tag 构建，例如 `v1.1.0`。
4. 执行审计。
5. 不要覆盖已经被其他用户依赖的旧 Tag。

推荐同时发布带日期或版本的不可变 Tag，不要只发布 `latest`。
