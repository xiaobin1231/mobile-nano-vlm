import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.append("third_party/minimind-v")
from model.model_vlm import MiniMindVLM, VLMConfig

class MiniMindLLMPrefill(nn.Module):
    def __init__(self, vlm_model: MiniMindVLM):
        super().__init__()
        cfg = vlm_model.config
        self.model = vlm_model.model
        self.lm_head = vlm_model.lm_head
        self.num_layers = cfg.num_hidden_layers
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

    def forward(self, input_ids, attention_mask):
        hidden_states, presents, _ = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        logits = self.lm_head(hidden_states)
        # Flatten: logits, k0, v0, k1, v1, ...
        outs = [logits]
        for pk, pv in presents:
            outs.append(pk)
            outs.append(pv)
        return tuple(outs)


class MiniMindLLMDecode(nn.Module):
    def __init__(self, vlm_model: MiniMindVLM):
        super().__init__()
        cfg = vlm_model.config
        self.model = vlm_model.model
        self.lm_head = vlm_model.lm_head
        self.num_layers = cfg.num_hidden_layers
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

    def forward(self,
                input_ids,
                attention_mask,
                past_k_0, past_v_0,
                past_k_1, past_v_1,
                past_k_2, past_v_2,
                past_k_3, past_v_3,
                past_k_4, past_v_4,
                past_k_5, past_v_5,
                past_k_6, past_v_6,
                past_k_7, past_v_7):
        past_key_values = [
            (past_k_0, past_v_0), (past_k_1, past_v_1),
            (past_k_2, past_v_2), (past_k_3, past_v_3),
            (past_k_4, past_v_4), (past_k_5, past_v_5),
            (past_k_6, past_v_6), (past_k_7, past_v_7),
        ]

        hidden_states, presents, _ = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = self.lm_head(hidden_states)

        outs = [logits]
        for pk, pv in presents:
            outs.append(pk)
            outs.append(pv)
        return tuple(outs)


KV_DTYPE = torch.float32

def _kv_output_names(num_layers: int):
    names = ["logits"]
    for i in range(num_layers):
        names.append(f"present_k_{i}")
        names.append(f"present_v_{i}")
    return names


def _kv_input_names(num_layers: int):
    names = ["input_ids", "attention_mask"]
    for i in range(num_layers):
        names.append(f"past_k_{i}")
        names.append(f"past_v_{i}")
    return names


def _kv_dynamic_axes(num_layers: int, past_len_name="kv_len", out_len_name="kv_len_out"):
    """Return dynamic_axes dict common to prefill and decode exports."""
    axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len_total"},
        "logits": {0: "batch", 1: "seq_len"},
    }
    for i in range(num_layers):
        axes[f"past_k_{i}"] = {0: "batch", 1: past_len_name}
        axes[f"past_v_{i}"] = {0: "batch", 1: past_len_name}
        axes[f"present_k_{i}"] = {0: "batch", 1: out_len_name}
        axes[f"present_v_{i}"] = {0: "batch", 1: out_len_name}
    return axes


def export_prefill(model: MiniMindVLM, export_path: str, device: str):
    """Export the prefill (prompt processing) ONNX model."""
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim

    llm = MiniMindLLMPrefill(model)
    llm.float().eval().to(device)

    prompt_len = 8
    dummy_ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)
    dummy_mask = torch.ones(1, prompt_len, device=device)

    input_names = ["input_ids", "attention_mask"]
    output_names = _kv_output_names(num_layers)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }
    for i in range(num_layers):
        dynamic_axes[f"present_k_{i}"] = {0: "batch", 1: "kv_len"}
        dynamic_axes[f"present_v_{i}"] = {0: "batch", 1: "kv_len"}

    print(f"Exporting prefill ONNX → {export_path}")
    torch.onnx.export(
        llm,
        (dummy_ids, dummy_mask),
        export_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("  Done.")

def export_decode(model: MiniMindVLM, export_path: str, device: str):
    """Export the decode (single-token) ONNX model."""
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim

    llm = MiniMindLLMDecode(model)
    llm.float().eval().to(device)

    past_len = 8
    dummy_ids = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
    dummy_mask = torch.ones(1, past_len + 1, device=device)
    dummy_kv = torch.randn(1, past_len, num_kv_heads, head_dim, dtype=KV_DTYPE, device=device)

    inputs = (dummy_ids, dummy_mask)
    for _ in range(num_layers):
        inputs = inputs + (dummy_kv, dummy_kv)

    input_names = _kv_input_names(num_layers)
    output_names = _kv_output_names(num_layers)
    dynamic_axes = _kv_dynamic_axes(num_layers)

    print(f"Exporting decode ONNX  → {export_path}")
    torch.onnx.export(
        llm,
        inputs,
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("  Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MiniMind LLM to ONNX (prefill + decode)")
    parser.add_argument("--save_dir", default="out", type=str)
    parser.add_argument("--weight", default="sft_vlm", type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--out_dir", default="onnx_export", type=str, help="Output directory for ONNX files")
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    moe_suffix = "_moe" if args.use_moe else ""
    ckp = f"./third_party/minimind-v/{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
    model = MiniMindVLM(
        VLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        ),
        vision_model_path="./third_party/minimind-v/model/siglip2-base-p32-256-ve",
    )
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(
        {k: v for k, v in state_dict.items() if "mask" not in k}, strict=False
    )

    export_prefill(model, os.path.join(args.out_dir, "llm_prefill.onnx"), args.device)
    export_decode(model, os.path.join(args.out_dir, "llm_decode.onnx"), args.device)

    print(f"\nAll done. ONNX files in: {args.out_dir}/")
