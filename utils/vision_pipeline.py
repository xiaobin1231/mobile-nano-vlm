import argparse

import sys
sys.path.append("third_party/minimind-v")
from model.model_vlm import MiniMindVLM, VLMConfig

import torch
import torch.nn as nn

class VisionPipeline(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, pixel_values):
        encoder_outputs = self.encoder(pixel_values)
        image_features = encoder_outputs.last_hidden_state
        projected_features = self.projector(image_features)
        return projected_features

def export_vision_to_onnx(model: MiniMindVLM, export_path="vision_encode_proj.onnx"):
    vision_pipeline = VisionPipeline(model.vision_encoder, model.vision_proj)
    vision_pipeline.float().eval()

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    print("Start export vision pipeline ONNX...")
    torch.onnx.export(
        vision_pipeline,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['vision_embeddings'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'vision_embeddings': {0: 'batch_size'}
        }
    )
    print(f"Export success: {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Chat")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径(model=原生torch权重,其他路径=transformers格式)")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀(pretrain_vlm, sft_vlm)")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
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

    export_vision_to_onnx(model)
