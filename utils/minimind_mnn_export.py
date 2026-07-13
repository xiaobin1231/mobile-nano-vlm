# -*- coding: utf-8 -*-
"""
Export MiniMind LLM via MNN-LLM's standard llm_export pipeline.

This script:
  1. Loads MiniMind from native .pth checkpoint
  2. Saves it as a HuggingFace-compatible model in a temp directory
  3. Registers 'minimind-v' model type in MNN's ModelMapper
  4. Runs MNN-LLM's LlmExporter to produce ONNX

Usage (run from mobile-nano-vlm/):
    python utils/minimind_mnn_export.py
"""

import os
import sys
import json
import shutil
import argparse
import torch

# ---------------------------------------------------------------------------
# Paths (relative to project root: mobile-nano-vlm/)
# ---------------------------------------------------------------------------
MINIMIND_ROOT = "third_party/minimind-v"
MNN_EXPORT_ROOT = "third_party/MNN/transformers/llm/export"

sys.path.insert(0, MINIMIND_ROOT)
sys.path.insert(0, MNN_EXPORT_ROOT)

from model.model_vlm import MiniMindVLM, VLMConfig


# ---------------------------------------------------------------------------
# Step 1: Register MiniMind in MNN's ModelMapper
# ---------------------------------------------------------------------------
def register_minimind():
    from utils.model_mapper import ModelMapper

    config_map = {
        "hidden_size": "hidden_size",
        "head_dim": "head_dim",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
        "num_key_value_heads": "num_key_value_heads",
        "rope_theta": "rope_theta",
        "rope_scaling": "rope_scaling",
        "max_position_embeddings": "max_position_embeddings",
        "rms_norm_eps": "rms_norm_eps",
    }
    model_map = {
        "lm": "lm_head",
        "embed": "model.embed_tokens",
        "blocks": "model.layers",
        "final_layernorm": "model.norm",
    }
    decoder_map = {
        "self_attn": "self_attn",
        "mlp": "mlp",
        "input_layernorm": "input_layernorm",
        "post_attention_layernorm": "post_attention_layernorm",
    }
    attention_map = {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "q_norm": "q_norm",
        "k_norm": "k_norm",
    }

    minimind_map = {
        "config": config_map,
        "model": model_map,
        "decoder": decoder_map,
        "attention": attention_map,
    }

    mapper = ModelMapper()
    mapper.regist("minimind-v", minimind_map)
    mapper.regist("minimind", minimind_map)
    print("[1/6] Registered minimind-v in MNN ModelMapper")
    return mapper


# ---------------------------------------------------------------------------
# Step 2: Save MiniMind as HuggingFace format
# ---------------------------------------------------------------------------
def save_as_hf(model: MiniMindVLM, hf_dir: str):
    os.makedirs(hf_dir, exist_ok=True)

    # --- config.json ---
    cfg = model.config
    config_dict = {
        "architectures": ["MiniMindVLM"],
        "model_type": "minimind-v",
        "auto_map": {
            "AutoConfig": "modeling_vlm.VLMConfig",
            "AutoModelForCausalLM": "modeling_vlm.MiniMindVLM",
        },
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "intermediate_size": cfg.intermediate_size,
        "hidden_act": cfg.hidden_act,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "rope_scaling": cfg.rope_scaling,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "use_moe": cfg.use_moe,
        "dropout": cfg.dropout,
        "image_special_token": cfg.image_special_token,
        "image_ids": cfg.image_ids,
        "image_hidden_size": cfg.image_hidden_size,
        "image_token_len": cfg.image_token_len,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "torch_dtype": "float32",
        "transformers_version": "4.57.6",
    }
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # --- Copy model code ---
    # model_vlm.py -> modeling_vlm.py (auto_map target)
    # model_minimind.py -> model_minimind.py (imported by modeling_vlm via `from .model_minimind import *`)
    src_model = os.path.join(MINIMIND_ROOT, "model")
    shutil.copy(os.path.join(src_model, "model_vlm.py"),
                os.path.join(hf_dir, "modeling_vlm.py"))
    shutil.copy(os.path.join(src_model, "model_minimind.py"),
                os.path.join(hf_dir, "model_minimind.py"))
    # Make it a proper Python package (needed for relative imports)
    open(os.path.join(hf_dir, "__init__.py"), "w").close()

    # --- Save weights as pytorch_model.bin ---
    state_dict = model.state_dict()
    # Exclude SigLIP vision encoder weights (they live in siglip2-base-p32-256-ve/).
    # Keep vision_proj.* weights — they are part of the .pth checkpoint.
    llm_state = {k: v for k, v in state_dict.items()
                 if not k.startswith("vision_encoder.")}
    torch.save(llm_state, os.path.join(hf_dir, "pytorch_model.bin"))

    # --- Copy tokenizer ---
    shutil.copy(os.path.join(src_model, "tokenizer.json"),
                os.path.join(hf_dir, "tokenizer.json"))
    shutil.copy(os.path.join(src_model, "tokenizer_config.json"),
                os.path.join(hf_dir, "tokenizer_config.json"))

    size_mb = os.path.getsize(os.path.join(hf_dir, "pytorch_model.bin")) / (1024 * 1024)
    print(f"[2/6] Saved HF model to {hf_dir}  ({size_mb:.1f} MB)")
    return hf_dir


# ---------------------------------------------------------------------------
# Step 3: Run MNN LlmExporter → ONNX + rebuild + MNNConvert
# ---------------------------------------------------------------------------
def export_via_mnn(hf_dir: str, out_dir: str, mnnconvert_bin: str, quant_bit_val: int = 8):
    from llmexport import LlmExporter
    import argparse as _argparse

    # Build a bare namespace matching what LlmExporter expects
    args = _argparse.Namespace(
        path=hf_dir,
        type="minimind-v",
        tokenizer_path=None,
        eagle_path=None,
        dflash_path=None,
        dflash_target_layer_ids=None,
        lora_path=None,
        gptq_path=None,
        dst_path=out_dir,
        verbose=False,
        test=None,
        export=None,
        onnx_slim=False,
        quant_bit=quant_bit_val,
        quant_block=64,
        visual_quant_bit=None,
        visual_quant_block=None,
        lm_quant_bit=None,
        lm_quant_block=None,
        mnnconvert=mnnconvert_bin,
        skip_weight=False,
        seperate_embed=False,
        seperate_visual=False,
        seperate_audio_model=False,
        awq=False,
        smooth=False,
        omni=False,
        hqq=False,
        lora_split=False,
        lora_split_block=0,
        transformer_c4=False,
        sym=False,
    )
    exporter = LlmExporter(args)
    print(f"[3/6] Loaded model via LlmModel.from_pretrained()")

    # 3a. Export skeleton ONNX (FakeLinear placeholders)
    onnx_skeleton = exporter.export_onnx()
    print(f"[4/6] Skeleton ONNX exported: {onnx_skeleton}")

    # 3b. Replace FakeLinear with real weights (external data).
    # onnx_load_param saves the full ONNX back to the original path,
    # and returns the external data file path (we don't need it directly).
    exporter.onnx_load_param(onnx_skeleton)
    onnx_full = onnx_skeleton  # same file, now with real weights via external data
    print(f"[5/6] Rebuilt ONNX with real weights")

    # 3c. Convert ONNX → MNN (quantized)
    from utils.mnn_converter import MNNConverter
    converter = MNNConverter(exporter, exporter.unloaded_ops)
    converter.export(onnx_full, quant_bit=quant_bit_val)
    mnn_path = os.path.join(out_dir, "llm.mnn")
    print(f"[6/6] MNN model exported: {mnn_path}")
    return mnn_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MiniMind LLM via MNN-LLM")
    parser.add_argument("--save_dir", default=os.path.join(MINIMIND_ROOT, "out"), type=str)
    parser.add_argument("--weight", default="sft_vlm", type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--hf_dir", default="./minimind_hf_tmp", type=str,
                        help="Temp dir for HF-format model")
    parser.add_argument("--out_dir", default="./mnn_llm_export", type=str,
                        help="Output dir for exported MNN model")
    parser.add_argument("--mnnconvert", default=os.path.join(MNN_EXPORT_ROOT, "../../../build/MNNConvert"), type=str,
                        help="Path to MNNConvert binary")
    parser.add_argument("--quant_bit", default=8, type=int, choices=[4, 8, 16, 32],
                        help="Weight quantization bits (4/8/16/32)")
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    # --- Load native model ---
    register_minimind()

    moe_suffix = "_moe" if args.use_moe else ""
    ckp = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}{moe_suffix}.pth")
    vision_path = os.path.join(MINIMIND_ROOT, "model/siglip2-base-p32-256-ve")

    model = MiniMindVLM(
        VLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        ),
        vision_model_path=vision_path,
    )
    sd = torch.load(ckp, map_location=args.device)
    model.load_state_dict({k: v for k, v in sd.items() if "mask" not in k}, strict=False)

    # --- Save HF format ---
    hf_dir = save_as_hf(model, args.hf_dir)

    # --- Export via MNN-LLM (ONNX → rebuild → MNN) ---
    mnn_path = export_via_mnn(hf_dir, args.out_dir, args.mnnconvert, args.quant_bit)

    print(f"\nDone. MNN model: {mnn_path}")
