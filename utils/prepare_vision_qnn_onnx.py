#!/usr/bin/env python3
"""Prepare the exported MiniMind-V vision ONNX graph for its MNN QNN wrapper."""

import argparse
from pathlib import Path

import onnx


TENSOR_RENAMES = {
    "pixel_values": "t0",
    "position_embeds": "t11",
    # Must match the output tensor recorded by compilefornpu in the MNN
    # Plugin Wrapper. A mismatch makes MNN return an all-zero output buffer.
    "vision_embeddings": "t1554",
}


def rename_tensor(graph: onnx.GraphProto, old_name: str, new_name: str) -> None:
    for value in (*graph.input, *graph.output, *graph.value_info):
        if value.name == old_name:
            value.name = new_name

    for initializer in graph.initializer:
        if initializer.name == old_name:
            initializer.name = new_name

    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old_name:
                node.input[index] = new_name
        for index, name in enumerate(node.output):
            if name == old_name:
                node.output[index] = new_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Exported vision ONNX")
    parser.add_argument("--output", required=True, help="QNN-ready ONNX output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model = onnx.load(input_path, load_external_data=True)

    input_names = {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}
    required_inputs = {"pixel_values", "position_embeds"}
    required_outputs = {"vision_embeddings"}
    if not required_inputs.issubset(input_names):
        raise ValueError(
            f"Expected ONNX inputs {sorted(required_inputs)}, got {sorted(input_names)}"
        )
    if not required_outputs.issubset(output_names):
        raise ValueError(
            f"Expected ONNX outputs {sorted(required_outputs)}, got {sorted(output_names)}"
        )

    for old_name, new_name in TENSOR_RENAMES.items():
        rename_tensor(model.graph, old_name, new_name)
    model.graph.name = "graph0"

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)

    print(f"Saved: {output_path}")
    print(f"Graph: {model.graph.name}")
    print("Inputs:", ", ".join(value.name for value in model.graph.input))
    print("Outputs:", ", ".join(value.name for value in model.graph.output))


if __name__ == "__main__":
    main()
