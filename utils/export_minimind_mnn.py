"""Export ORIGINAL GQA model with fixed weight embedding."""
import os, sys, json, shutil, torch, onnx
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINIMIND_ROOT = os.path.join(PROJ, "third_party/minimind-v")
MNN_EXPORT_ROOT = os.path.join(PROJ, "third_party/MNN/transformers/llm/export")
sys.path.insert(0, MINIMIND_ROOT)
sys.path.insert(0, MNN_EXPORT_ROOT)

from model.model_vlm import MiniMindVLM, VLMConfig
from utils.model_mapper import ModelMapper
from utils.mnn_converter import MNNConverter

CKPT = os.path.join(MINIMIND_ROOT, "out/sft_vlm_768.pth")
VISION_PATH = os.path.join(MINIMIND_ROOT, "model/siglip2-base-p32-256-ve")
HF_DIR = os.path.join(PROJ, "minimind_hf")
OUT_DIR = os.path.join(PROJ, "llm_mnn")
MNNCONVERT = os.path.join(PROJ, "third_party/MNN/build/MNNConvert")

# Step 1: Load original model (with GQA, num_kv_heads=4)
print("=" * 60)
print("[1] Loading original GQA model (num_kv_heads=4)")
model = MiniMindVLM(VLMConfig(hidden_size=768, num_hidden_layers=8, use_moe=False), vision_model_path=VISION_PATH)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict({k:v for k,v in sd.items() if "mask" not in k}, strict=False)
state = model.state_dict()

# Save HF format (no weight modification!)
os.makedirs(HF_DIR, exist_ok=True)
cfg = model.config
config_dict = {
    "architectures": ["MiniMindVLM"], "model_type": "minimind-v",
    "auto_map": {"AutoConfig": "modeling_vlm.VLMConfig", "AutoModelForCausalLM": "modeling_vlm.MiniMindVLM"},
    "hidden_size": 768, "num_hidden_layers": 8,
    "num_attention_heads": 8, "num_key_value_heads": 4,
    "head_dim": 96, "vocab_size": 6400,
    "intermediate_size": cfg.intermediate_size, "hidden_act": "silu",
    "max_position_embeddings": cfg.max_position_embeddings,
    "rms_norm_eps": cfg.rms_norm_eps, "rope_theta": cfg.rope_theta,
    "tie_word_embeddings": cfg.tie_word_embeddings,
    "use_moe": False, "dropout": 0.0,
    "bos_token_id": 1, "eos_token_id": 2,
    "torch_dtype": "float32", "transformers_version": "4.57.6",
}
with open(os.path.join(HF_DIR, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)

src_model = os.path.join(MINIMIND_ROOT, "model")
shutil.copy(os.path.join(src_model, "model_vlm.py"), os.path.join(HF_DIR, "modeling_vlm.py"))
shutil.copy(os.path.join(src_model, "model_minimind.py"), os.path.join(HF_DIR, "model_minimind.py"))
open(os.path.join(HF_DIR, "__init__.py"), "w").close()
llm_state = {k:v for k,v in state.items() if not k.startswith("vision_encoder.")}
torch.save(llm_state, os.path.join(HF_DIR, "pytorch_model.bin"))
shutil.copy(os.path.join(src_model, "tokenizer.json"), os.path.join(HF_DIR, "tokenizer.json"))
shutil.copy(os.path.join(src_model, "tokenizer_config.json"), os.path.join(HF_DIR, "tokenizer_config.json"))
print(f"  Saved HF to {HF_DIR}")

# Step 2: Register
print("\n[2] Registering minimind-v")
config_map = {"hidden_size":"hidden_size","head_dim":"head_dim","num_attention_heads":"num_attention_heads","num_hidden_layers":"num_hidden_layers","num_key_value_heads":"num_key_value_heads","rope_theta":"rope_theta","rope_scaling":"rope_scaling","max_position_embeddings":"max_position_embeddings","rms_norm_eps":"rms_norm_eps"}
model_map = {"lm":"lm_head","embed":"model.embed_tokens","blocks":"model.layers","final_layernorm":"model.norm"}
decoder_map = {"self_attn":"self_attn","mlp":"mlp","input_layernorm":"input_layernorm","post_attention_layernorm":"post_attention_layernorm"}
attention_map = {"q_proj":"q_proj","k_proj":"k_proj","v_proj":"v_proj","o_proj":"o_proj","q_norm":"q_norm","k_norm":"k_norm"}
mapper = ModelMapper()
mapper.regist("minimind-v", {"config":config_map,"model":model_map,"decoder":decoder_map,"attention":attention_map})

# Step 3: Export ONNX skeleton
print("\n[3] Exporting ONNX")
os.makedirs(OUT_DIR, exist_ok=True)
from llmexport import LlmExporter
import argparse as _argparse
args = _argparse.Namespace(
    path=HF_DIR, type="minimind-v", tokenizer_path=None,
    eagle_path=None, dflash_path=None, dflash_target_layer_ids=None,
    lora_path=None, gptq_path=None, dst_path=OUT_DIR, verbose=False,
    test=None, export=None, onnx_slim=False, quant_bit=32, quant_block=64,
    visual_quant_bit=None, visual_quant_block=None,
    lm_quant_bit=None, lm_quant_block=None,
    mnnconvert=MNNCONVERT, skip_weight=False, seperate_embed=False,
    seperate_visual=False, seperate_audio_model=False, awq=False, smooth=False,
    omni=False, hqq=False, lora_split=False, lora_split_block=0,
    transformer_c4=False, sym=False, scale_bit=16,
)
exporter = LlmExporter(args)
onnx_sk = exporter.export_onnx()
print(f"  ONNX: {onnx_sk}")

# Step 4: Rebuild ONNX with real weights
print("\n[4] Rebuilding ONNX with real weights")
exporter.onnx_load_param(onnx_sk)

# Step 5: Convert ONNX to embedded format
print("\n[5] Converting to embedded ONNX")
embedded_onnx = onnx_sk.replace('.onnx', '_embedded.onnx')
model_onnx = onnx.load(onnx_sk, load_external_data=True)
onnx.save(model_onnx, embedded_onnx)
for init in model_onnx.graph.initializer:
    if 'q_proj' in init.name and 'weight' in init.name:
        w = onnx.numpy_helper.to_array(init)
        print(f"  Verified: {init.name} shape={w.shape} nonzero={np.count_nonzero(w)}")
        break

# Step 6: Direct MNNConvert (no --saveExternalData, no --fp16)
print("\n[6] Converting ONNX->MNN (weights embedded)")
base_mnn = os.path.join(OUT_DIR, "llm_base.mnn")
cmd = "{} -f ONNX --modelFile {} --MNNModel {} --transformerFuse --allowCustomOp".format(MNNCONVERT, embedded_onnx, base_mnn)
os.system(cmd + " 2>/dev/null")
print(f"  MNN: {base_mnn}")

# Step 7: MNN->JSON
print("\n[7] Converting MNN->JSON")
base_json = base_mnn.replace('.mnn', '.json')
cmd = "{} -f MNN --modelFile {} --JsonFile {}".format(MNNCONVERT, base_mnn, base_json)
os.system(cmd + " 2>/dev/null")

# Step 8: Fix Extra->Attention
print("\n[8] Fixing Extra->Attention")
j = json.load(open(base_json))
new_ops = []
extra_count = 0
for op in j['oplists']:
    if op['type'] == 'Extra':
        attrs = {a['key']: a for a in op['main'].get('attr', [])}
        if 'kv_cache' in attrs or 'output_dim' in attrs:
            kv_cache = attrs.get('kv_cache', {}).get('i', 1)
            layer_idx = attrs.get('layer_index', {}).get('i', -1)
            kv_shared = attrs.get('kv_shared_layer_index', {}).get('i', -1)
            name = attrs.get('name', {}).get('s', op.get('name', ''))
            new_ops.append({
                "inputIndexes": op['inputIndexes'],
                "main_type": "AttentionParam",
                "main": {"kv_cache": bool(kv_cache), "layer_index": layer_idx, "kv_shared_layer_index": kv_shared},
                "name": name, "outputIndexes": op['outputIndexes'],
                "type": "Attention", "defaultDimentionFormat": op.get('defaultDimentionFormat', 'NHWC')
            })
            extra_count += 1
            print(f"  [{layer_idx}] Extra->Attention: {name.split('/')[-1]}")
        else:
            new_ops.append(op)
    else:
        new_ops.append(op)
j['oplists'] = new_ops
print(f"  Converted: {extra_count}/8")

# Step 9: JSON->MNN
print("\n[9] Converting JSON->MNN")
fixed_json = base_mnn.replace('.mnn', '_final.json')
with open(fixed_json, 'w') as f:
    json.dump(j, f)
final_mnn = base_mnn.replace('.mnn', '_final.mnn')
cmd = "{} -f JSON --modelFile {} --MNNModel {}".format(MNNCONVERT, fixed_json, final_mnn)
os.system(cmd + " 2>/dev/null")

# Step 10: Done
print("\n[10] Done!")
print(f"  LLM model: {final_mnn}")
