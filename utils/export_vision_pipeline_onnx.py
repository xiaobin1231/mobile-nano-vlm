import argparse
import sys

sys.path.append("third_party/minimind-v")
from model.model_vlm import MiniMindVLM, VLMConfig

import os
import torch
import torch.nn as nn

class VisionPipeline(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        # Transformers 新版 SiglipVisionModel 在外层增加了 vision_model；
        # 旧版则直接暴露 embeddings / encoder / post_layernorm。
        self.encoder = getattr(encoder, "vision_model", encoder)
        self.projector = projector

    def forward(self, pixel_values, position_embeds):
        # SigLIP normally transposes patch embeddings to [B, N, C] and then
        # adds [1, N, C] positional embeddings. MNN's QNN layout conversion
        # keeps the patch tensor as [B, C, N] at that point, which makes the
        # static Add fail QNN validation. Perform the mathematically identical
        # Add before the transpose so both operands are explicitly [B, C, N].
        embeddings = self.encoder.embeddings
        target_dtype = embeddings.patch_embedding.weight.dtype
        patch_embeds = embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))
        hidden_states = (patch_embeds.flatten(2) + position_embeds).transpose(1, 2)

        encoder_outputs = self.encoder.encoder(inputs_embeds=hidden_states)
        image_features = self.encoder.post_layernorm(encoder_outputs.last_hidden_state)
        projected_features = self.projector(image_features)
        return projected_features

def export_vision_to_onnx(model: MiniMindVLM, export_path="vision_encode_proj.onnx"):
    vision_pipeline = VisionPipeline(model.vision_encoder, model.vision_proj)
    vision_pipeline.float().eval()

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    position_input = (
        vision_pipeline.encoder.embeddings.position_embedding.weight
        .transpose(0, 1).unsqueeze(0).detach().float()
    )

    with torch.no_grad():
        reference = model.vision_proj(model.vision_encoder(dummy_input).last_hidden_state)
        rewritten = vision_pipeline(dummy_input, position_input)
        max_diff = (reference - rewritten).abs().max().item()
        print(f"QNN-friendly position Add verification max diff: {max_diff:.8g}")
        if max_diff > 1e-5:
            raise RuntimeError(f"Vision graph rewrite changed output: max diff={max_diff}")

    print("Start export vision pipeline ONNX...")
    torch.onnx.export(
        vision_pipeline,
        (dummy_input, position_input),
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values', 'position_embeds'],
        output_names=['vision_embeddings'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'vision_embeddings': {0: 'batch_size'}
        }
    )
    position_path = os.path.join(os.path.dirname(export_path), "vision_position_f32.bin")
    position_input.contiguous().numpy().tofile(position_path)
    print(f"Position input saved: {position_path} ({position_input.numel() * 4} bytes)")
    print(f"Export success: {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Chat")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径(model=原生torch权重,其他路径=transformers格式)")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀(pretrain_vlm, sft_vlm)")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument("--out_dir", default="vision_export", type=str, help="Output directory for ONNX files")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构(0=否,1=是）")
    args = parser.parse_args()

    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./third_party/minimind-v/{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    model = MiniMindVLM(
        VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
        vision_model_path="./third_party/minimind-v/model/siglip2-base-p32-256-ve"
    )
    state_dict = torch.load(ckp)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)

    os.makedirs(args.out_dir, exist_ok=True)

    export_vision_to_onnx(model, os.path.join(args.out_dir, "vision_encode_proj.onnx"))
