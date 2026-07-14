#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
SRC="$PROJ/src"
MODELS="$SRC/models"
MINIMIND="$PROJ/third_party/minimind-v"
MNN="$PROJ/third_party/MNN"
PYTHON=python

echo "========================================"
echo " MiniMind-V MNN 导出 & 模型准备"
echo "========================================"

# ---- 检查前置条件 ----
if [ ! -f "$MINIMIND/out/sft_vlm_768.pth" ]; then
    echo "错误: 找不到 $MINIMIND/out/sft_vlm_768.pth"
    exit 1
fi
if [ ! -f "$MNN/build/MNNConvert" ]; then
    echo "错误: 找不到 MNNConvert，请先编译 MNN:"
    echo "  bash scripts/build.sh linux"
    exit 1
fi

# ---- Step 1: 导出 Vision ONNX ----
echo "[1/8] 导出 Vision ONNX ..."
cd "$PROJ"
$PYTHON utils/export_vision_pipeline_onnx.py

# ---- Step 2: Vision ONNX → MNN ----
echo "[2/8] 转换 Vision ONNX → MNN ..."
mkdir -p "$PROJ/vision_export"
"$MNN/build/MNNConvert" -f ONNX \
    --modelFile "$PROJ/vision_export/vision_encode_proj.onnx" \
    --MNNModel "$PROJ/vision_export/vision_encode_proj.mnn" \
    --bizCode MNN --fp16

# ---- Step 3: 导出 LLM MNN ----
echo "[3/8] 导出 LLM MNN ..."
$PYTHON utils/export_minimind_mnn.py

# ---- Step 4: 复制模型文件 ----
echo "[4/8] 复制模型文件 ..."
mkdir -p "$MODELS"
cp "$PROJ/llm_mnn/llm_base_final.mnn" "$MODELS/llm.mnn"
cp "$PROJ/vision_export/vision_encode_proj.mnn" "$MODELS/vision_encode_proj.mnn"
touch "$MODELS/llm.mnn.weight"
echo "  llm.mnn:                $(du -h "$MODELS/llm.mnn" | cut -f1)"
echo "  vision_encode_proj.mnn: $(du -h "$MODELS/vision_encode_proj.mnn" | cut -f1)"

# ---- Step 5: tokenizer.txt ----
echo "[5/8] 生成 tokenizer.txt ..."
$PYTHON -c "
import json, sys
sys.path.insert(0, '$MINIMIND')
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MINIMIND/model/', trust_remote_code=True)
hf = json.load(open('$MINIMIND/model/tokenizer.json'))
vocab = hf['model']['vocab']
merges = hf['model']['merges']
with open('$MODELS/tokenizer.txt', 'w', encoding='utf-8') as f:
    f.write('430 3\n')
    f.write('36 1 0\n')
    f.write(' '.join(str(i) for i in range(36)) + ' 2\n')
    f.write(f'{len(vocab)} {len(merges)}\n')
    for token, _ in sorted(vocab.items(), key=lambda x: x[1]):
        f.write(token + '\n')
    for merge in merges:
        f.write(' '.join(merge) + '\n')
print(f'  tokenizer.txt: {len(vocab)} vocab, {len(merges)} merges')
"

# ---- Step 6: embeddings_bf16.bin ----
echo "[6/8] 生成 embeddings_bf16.bin ..."
$PYTHON -c "
import torch, numpy as np
sd = torch.load('$MINIMIND/out/sft_vlm_768.pth', map_location='cpu')
emb = sd['model.embed_tokens.weight'].float().numpy()
bf16 = (emb.view(np.uint32) >> 16).astype(np.uint16)
bf16.tofile('$MODELS/embeddings_bf16.bin')
print(f'  embeddings_bf16.bin: {bf16.nbytes} bytes ({emb.shape[0]} vocab x {emb.shape[1]} dim)')
"

# ---- Step 7: llm_config.json ----
echo "[7/8] 生成 llm_config.json ..."
$PYTHON -c "
import json
config = {
    'model_type': 'llama',
    'llm_model': 'llm.mnn', 'llm_weight': 'llm.mnn.weight',
    'tokenizer_file': 'tokenizer.txt',
    'backend_type': 'cpu', 'thread_num': 4, 'precision': 'low',
    'max_all_tokens': 4096, 'max_new_tokens': 512,
    'is_single': True, 'is_visual': False,
    'use_template': False, 'use_mmap': False,
    'hidden_size': 768, 'layer_nums': 8,
    'attention_mask': 'float', 'attention_type': 'full',
    'attention_fused': True,
    'key_value_shape': [4, 96],
    'tie_embeddings': [],
    'sampler_type': 'mixed',
    'temperature': 0.7, 'topK': 50, 'topP': 0.85,
    'repetition_penalty': 1.15,
    'bos': '<|im_start|>', 'eos': '<|im_end|>',
    'chat_template': (
        '{% for message in messages %}'
        '{% if message.role == \"system\" %}'
        '<|im_start|>system\n{{ message.content }}<|im_end|>\n'
        '{% elif message.role == \"user\" %}'
        '<|im_start|>user\n{{ message.content }}<|im_end|>\n'
        '{% elif message.role == \"assistant\" %}'
        '<|im_start|>assistant\n{{ message.content }}<|im_end|>\n'
        '{% endif %}{% endfor %}'
        '{% if add_generation_prompt %}'
        '<|im_start|>assistant\n{% endif %}'
    ),
}
json.dump(config, open('$MODELS/llm_config.json', 'w'), indent=2, ensure_ascii=False)
print('  llm_config.json done')
"

# ---- Step 8: stb_image.h ----
echo "[8/8] 下载 stb_image.h ..."
if [ ! -f "$SRC/common/stb_image.h" ]; then
    curl -sL -o "$SRC/common/stb_image.h" \
        https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    echo "  stb_image.h 已下载 ($(wc -l < "$SRC/common/stb_image.h") 行)"
else
    echo "  stb_image.h 已存在"
fi

echo ""
echo "========================================"
echo " 导出 & 模型准备完成！"
echo ""
echo " 模型: $MODELS/"
echo " 编译: bash scripts/build.sh linux|android"
echo "========================================"
