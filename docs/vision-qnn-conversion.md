# Vision QNN 转换流程

本文解释 `scripts/convert_vision_qnn.sh` 的每个阶段。普通用户只需要执行脚本，
不需要逐条复制本文命令。

## 1. 一键入口

在主机工程根目录设置依赖目录：

```bash
export DEPS_ROOT=/path/to/deps
```

只转换并更新 `artifacts/qnn/vision/`：

```bash
./scripts/convert_vision_qnn.sh
```

转换、部署并执行图片问答：

```bash
export ANDROID_NDK=/path/to/android-ndk-r27d
export QNN_SDK_ROOT="$DEPS_ROOT/qairt/2.48.0.260626"
export CMAKE_BIN=/path/to/cmake/bin/cmake

./scripts/convert_vision_qnn.sh \
  --deploy \
  --image /path/to/test_image.jpg \
  --prompt '请描述这幅图'
```

只查看计划和检查工程文件：

```bash
./scripts/convert_vision_qnn.sh --dry-run
```

不替换正式 artifacts：

```bash
./scripts/convert_vision_qnn.sh --no-promote
```

## 2. 总体产物链

Vision 使用 MNN 和 QNN SDK 的混合路线：

```text
Wrapper路线：

原始Vision MNN
  → generateIO
  → compilefornpu
  → 原始MNN Plugin Wrapper
  → NC4HW4转NCHW
  → vision.mnn

Context路线：

Vision ONNX
  → 重命名QNN接口
  → qnn-onnx-converter
  → graph0.cpp + 转换权重graph0.bin
  → qnn-model-lib-generator
  → libgraph0.so
  → qnn-context-binary-generator
  → 最终HTP Context graph0.bin
```

最终文件：

```text
artifacts/qnn/vision/
├── vision.mnn
├── graph0.bin
├── vision_position_f32.bin
└── SHA256SUMS
```

## 3. 脚本执行环境

脚本从主机启动，并自动完成：

1. 检查或构建 `mobile-nano-vlm-build:local` 公开依赖镜像。
2. 将工程挂载到 `/workspace`。
3. 将 `$DEPS_ROOT` 只读挂载到 `/deps`。
4. 在 Docker 中完成模型导出和转换。
5. 返回主机后按 `--deploy` 决定是否进行 Android 编译、adb 推送和运行。

Python 包默认使用清华源：

```text
https://pypi.tuna.tsinghua.edu.cn/simple
```

可通过 `PIP_INDEX_URL` 覆盖。QAIRT 版本默认是 `2.48.0.260626`，可通过
`QAIRT_VERSION` 覆盖。

## 4. 阶段 1：导出 ONNX 和 position_embeds

脚本调用：

```bash
python utils/export_vision_pipeline_onnx.py \
  --save_dir out \
  --weight sft_vlm \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --out_dir vision_export
```

输入：

```text
third_party/minimind-v/out/sft_vlm_768.pth
third_party/minimind-v/model/siglip2-base-p32-256-ve/
```

输出：

```text
vision_export/vision_encode_proj.onnx
vision_export/vision_position_f32.bin
```

位置编码在 PyTorch 模型中原本是内部参数。导出脚本为了构造更适合 QNN 的
`[B, C, N] + [1, C, N]`，将它显式导出为第二个输入：

```text
pixel_values:     [1, 3, 256, 256]
position_embeds:  [1, 768, 64]
```

位置编码文件大小：

```text
1 × 768 × 64 × sizeof(float32) = 196608 bytes
```

## 5. 阶段 2：ONNX 转原始 Vision MNN

脚本调用 MNNConvert：

```bash
third_party/MNN/build/MNNConvert \
  -f ONNX \
  --modelFile vision_export/vision_encode_proj.onnx \
  --MNNModel vision_export/vision_encode_proj.mnn \
  --bizCode MNN \
  --fp16
```

这个约 170 MB 的 MNN 是完整 CPU Vision 图，只作为 `compilefornpu` 的输入，
不是最终部署的 944 字节左右的 Plugin Wrapper。

## 6. 阶段 3：generateIO

脚本调用：

```bash
generateIO \
  vision_export/vision_encode_proj.mnn \
  configs/qnn/vision/compilefornpu_input.json \
  build/vision_qnn/wrapper/testdir
```

配置定义两个输入和一个输出：

```text
pixel_values       [1, 3, 256, 256]
position_embeds    [1, 768, 64]
vision_embeddings
```

`generateIO` 用真实 Shape 执行一次原始 MNN，生成：

```text
testdir/0/input.mnn
testdir/0/output.mnn
```

这些数据供 `compilefornpu` 拆图和校验使用。

## 7. 阶段 4：compilefornpu 生成 Wrapper

脚本在 `build/vision_qnn/wrapper/` 下调用：

```bash
compilefornpu \
  /workspace/vision_export/vision_encode_proj.mnn \
  qnn/vision_encode_proj.mnn \
  /workspace/configs/qnn/vision/compilefornpu_qnn.json
```

需要保留：

```text
build/vision_qnn/wrapper/qnn/vision_encode_proj.mnn
```

它记录：

```text
MNN pixel_values      → QNN t0
MNN position_embeds   → QNN t11
QNN t1554             → MNN vision_embeddings
QNN Graph             → graph0
Context相对路径       → qnn/graph0.bin
```

`compilefornpu` 还会生成 `qnn/graph0/graph0.cpp` 和 RAW 权重。本工程不使用
这些文件，也不调用 MNN `npu_convert.py` 生成 Vision Context。原因见第 13 节。

## 8. 阶段 5：NC4HW4 转 NCHW

这是必要的实际转换步骤。

`compilefornpu` 原始 Wrapper 将两个 Input 声明成 NC4HW4，而应用写入普通
NCHW/线性数据。不修正时推理可能不崩溃，但图片描述明显错误。

脚本先编译：

```bash
g++ -std=c++17 \
  -Ithird_party/MNN/schema/current \
  -Ithird_party/MNN/3rd_party/flatbuffers/include \
  utils/patch_mnn_wrapper_nchw.cpp \
  -o build/vision_qnn/patch_mnn_wrapper_nchw
```

再执行：

```bash
build/vision_qnn/patch_mnn_wrapper_nchw \
  build/vision_qnn/wrapper/qnn/vision_encode_proj.mnn \
  build/vision_qnn/vision.mnn
```

预期输出：

```text
Patched 2 input tensor formats to NCHW
```

此工具只修改 Wrapper 中 MNN Input 的 `dataFormat`，不修改 QNN Graph 名、
Tensor 名、Context 路径或模型权重。

## 9. 阶段 6：准备 QNN ONNX 接口

ONNX 原始接口名：

```text
pixel_values
position_embeds
vision_embeddings
```

Wrapper 使用：

```text
t0
t11
t1554
```

脚本调用 `utils/prepare_vision_qnn_onnx.py` 完成重命名，并将 Graph 名设为
`graph0`。这个步骤不改变计算和权重。QNN 输出名必须与 compilefornpu Wrapper
记录的名字完全一致；否则日志会出现 `can't find ... output: t1554`，应用得到的
Vision Embedding 将全部为零。

## 10. 阶段 7：QNN SDK 生成图、模型库和 Context

### 10.1 qnn-onnx-converter

```bash
qnn-onnx-converter \
  -i build/vision_qnn/graph0.onnx \
  -d t0 1,3,256,256 \
  -d t11 1,768,64 \
  --input_layout t0 NCHW \
  --input_layout t11 NONTRIVIAL \
  --preserve_io \
  --float_bitwidth 32 \
  -o build/vision_qnn/graph0.cpp
```

输出：

```text
graph0.cpp       QNN API图结构
graph0.bin       转换阶段权重
graph0_net.json  网络描述
```

### 10.2 qnn-model-lib-generator

将 `graph0.cpp` 和转换阶段权重组合成 x86 Model Library：

```text
build/vision_qnn/model_libs/x86_64-linux-clang/libgraph0.so
```

这个 `.so` 只在主机离线生成 Context 时使用，不部署到 Android。

### 10.3 qnn-context-binary-generator

使用配置：

```text
soc_id   = 57
dsp_arch = v75
graph    = graph0
```

生成：

```text
build/vision_qnn/context/graph0.bin
```

这是最终 HTP Context。

两个同名文件不能混淆：

```text
build/vision_qnn/graph0.bin
  转换阶段权重，不部署

build/vision_qnn/context/graph0.bin
  最终HTP Context，需要部署
```

## 11. 阶段 8：候选包、备份和 artifacts

候选包先生成到：

```text
build/vision_qnn/package/
```

默认情况下，脚本会：

1. 将旧 `artifacts/qnn/vision/` 备份到
   `build/artifact_backups/vision-<时间>/`。
2. 用候选包更新正式 Vision artifacts。
3. 生成 `SHA256SUMS`。

使用 `--no-promote` 可以只保留候选包。

## 12. 阶段 9：可选部署和运行

`--deploy` 在退出 Docker 后执行：

```text
scripts/build.sh android --mnn
scripts/deploy_qnn_android.sh
adb push <测试图片>
minimind_cli vision ...
```

部署完整 VLM 要求 `artifacts/qnn/vision/` 和 `artifacts/qnn/llm/` 都已生成。
