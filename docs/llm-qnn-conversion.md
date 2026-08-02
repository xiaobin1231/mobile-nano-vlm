# LLM QNN 转换流程

本文解释 `scripts/convert_llm_qnn.sh` 的每个阶段。普通用户只需要执行脚本。

## 1. 一键入口

```bash
export DEPS_ROOT=/path/to/deps
./scripts/convert_llm_qnn.sh
```

如果 Vision artifacts 已准备好，可以在 LLM 转换后直接部署完整 VLM：

```bash
export ANDROID_NDK=/path/to/android-ndk-r27d
export QNN_SDK_ROOT="$DEPS_ROOT/qairt/2.48.0.260626"
export CMAKE_BIN=/path/to/cmake/bin/cmake

./scripts/convert_llm_qnn.sh \
  --deploy \
  --image /path/to/test_image.jpg \
  --prompt '请描述这幅图'
```

其他选项：

```bash
./scripts/convert_llm_qnn.sh --dry-run
./scripts/convert_llm_qnn.sh --no-promote
```

## 2. 总体产物链

```text
MiniMind-V PTH
  → 临时Hugging Face目录
  → MNN LLM Exporter生成ONNX skeleton
  → 回填真实权重
  → MNNConvert生成LLM MNN
  → generateIO生成Prefill和Decode输入
  → compilefornpu拆分QNN子图并生成LLM Wrapper
  → QNN SDK生成10个HTP Context
  → artifacts/qnn/llm/
```

最终文件：

```text
artifacts/qnn/
├── llm_config_qnn.json
└── llm/
    ├── llm.mnn
    ├── llm.mnn.weight
    ├── graph0.bin
    ├── graph1.bin
    ├── ...
    ├── graph9.bin
    └── SHA256SUMS
```

## 3. 阶段 1：PTH 转临时 Hugging Face 目录

入口脚本调用内部准备脚本 `scripts/run_setup.sh`，其中
`utils/export_minimind_mnn.py` 加载：

```text
third_party/minimind-v/out/sft_vlm_768.pth
```

并生成临时目录：

```text
minimind_hf/
```

HF 格式不是为了改变模型，而是给 MNN LLM Exporter 提供标准化的：

- 模型配置。
- 权重名称。
- Tokenizer。
- Python 模型入口。

MiniMind-V 原始 checkpoint 不是标准 Transformers 发布目录，MNN LLM
Exporter 不能直接从单独的 `.pth` 得到完整结构信息。

该脚本只导出 LLM 和公共模型资源，不处理 Vision；Vision 由
`scripts/convert_vision_qnn.sh` 独立负责。

## 4. 阶段 2：生成 ONNX skeleton 并回填权重

MNN LLM Exporter 先按模型结构建立 ONNX skeleton，再通过模型映射器把 MiniMind-V
权重填入对应节点。

这里的“skeleton”是计算图骨架，不表示最终模型没有权重。导出脚本会验证真实
权重已经写入 ONNX，然后生成嵌入权重的 ONNX。

这个过程同时处理：

- 8 层 Transformer。
- GQA：8 个 Query Head、4 个 KV Head。
- RMSNorm、RoPE、Attention 和 MLP 的结构映射。
- Tokenizer 和 Embedding 公共资源。

## 5. 阶段 3：ONNX 转 LLM MNN

`utils/export_minimind_mnn.py` 使用 MNNConvert 生成：

```text
llm_mnn/llm_base_final.mnn
```

并复制成：

```text
src/models/llm.mnn
```

`llm.mnn.weight` 当前是空占位文件，因为权重已经嵌入 MNN；MNN LLM Runtime
仍会检查配置中的外部权重路径是否存在，因此候选包和部署目录必须保留它。

脚本还准备：

```text
src/models/tokenizer.txt
src/models/embeddings_bf16.bin
src/models/llm_config.json
```

## 6. 阶段 4：generateIO 准备两种执行形态

LLM 推理分成：

```text
Prefill：一次处理一段输入Token
Decode：每次生成一个新Token
```

当前默认输入：

```text
Prefill sequence length = 128
Decode sequence length  = 1
Hidden size             = 768
```

可通过环境变量修改 Prefill Chunk：

```bash
export QNN_CHUNK_SIZE=128
```

`generateIO` 会生成这两套输入输出，供 `compilefornpu` 拆图和数值校验。

## 7. 阶段 5：compilefornpu 拆分 LLM

入口脚本调用：

```bash
python3 scripts/generate_qnn.py \
  --model src/models \
  --soc_id 57 \
  --dsp_arch v75 \
  --mnn_path third_party/MNN/build_qnn_host \
  --cache_path build/llm_qnn/cache \
  --chunk_size 128 \
  --model_name llm.mnn
```

`scripts/generate_qnn.py` 负责：

1. 生成 Prefill 和 Decode IO。
2. 调用 MNN `compilefornpu` 拆分 QNN 子图。
3. 调用工程侧兼容脚本 `scripts/npu_convert.py`。
4. 整理 Wrapper、Context 和 QNN 配置。

`generateIO` 和 `compilefornpu` 都来自 MNN Host 构建，不是 QNN SDK 命令。

## 8. 阶段 6：QNN SDK 生成 10 个 Context

`scripts/npu_convert.py` 对每个子图执行：

```text
RAW权重
  → 临时权重bin
  → qnn-model-lib-generator
  → x86 Model Library
  → qnn-context-binary-generator
  → 目标HTP Context
```

最终得到：

```text
graph0.bin
graph1.bin
...
graph9.bin
```

LLM 不是简单地按层一层对应一个文件。拆分结果还受 Prefill/Decode、算子支持和
图合并策略影响。`llm.mnn` Wrapper 保存子图之间的调用关系。

## 9. 阶段 7：候选包和 artifacts

候选包：

```text
build/llm_qnn/package/
```

默认情况下脚本会：

1. 检查 `llm.mnn` 和 `graph0.bin`～`graph9.bin`。
2. 生成 `SHA256SUMS`。
3. 备份旧 `artifacts/qnn/llm/`。
4. 更新正式 LLM artifacts。
5. 更新 `artifacts/qnn/llm_config_qnn.json`。

使用 `--no-promote` 时不会修改正式 artifacts。

## 10. 阶段 8：部署和完整 VLM 测试

`--deploy` 会在主机执行：

```text
Android MNN和minimind_cli交叉编译
  → 推送公共资源
  → 推送Vision Wrapper和Context
  → 推送LLM Wrapper和10个Context
  → 推送QNN Runtime
  → 可选执行图片问答
```

因此推荐从零执行：

```bash
./scripts/convert_vision_qnn.sh

./scripts/convert_llm_qnn.sh \
  --deploy \
  --image /path/to/51016.JPG
```

第二条命令完成后会部署 Vision 和 LLM 的正式 artifacts，并运行完整 VLM。

## 11. 可调参数

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `QAIRT_VERSION` | `2.48.0.260626` | QAIRT 目录版本 |
| `QNN_SOC_ID` | `57` | 目标 SoC ID |
| `QNN_DSP_ARCH` | `v75` | HTP 架构 |
| `QNN_CHUNK_SIZE` | `128` | LLM Prefill Chunk |
| `BUILD_JOBS` | `8` | Host 工具编译并行度 |
| `PIP_INDEX_URL` | 清华源 | Python 包源 |

更换 SoC 或 Chunk 后必须重新生成 Context，不能只修改 JSON 配置。
