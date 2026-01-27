import os
import argparse
import torch
from nanochat.quantize import export_int8
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, print0, get_base_dir, autodetect_device_type


def main():
    parser = argparse.ArgumentParser(description="Export model to INT8 format")
    parser.add_argument("--model-tag", type=str, required=True, help="Model tag")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step")
    parser.add_argument("--source", type=str, default="base", help="Model source: base|distill")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--bits", type=int, default=8, help="Quantization bits (8 for INT8)")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")
    args = parser.parse_args()
    
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    
    print0(f"Loading model: {args.model_tag} (step {args.step})")
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()
    
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print0(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    
    if args.output is None:
        output_dir = os.path.join(get_base_dir(), "int8_models")
        os.makedirs(output_dir, exist_ok=True)
        step_str = f"_step{args.step}" if args.step else ""
        args.output = os.path.join(output_dir, f"{args.model_tag}{step_str}_int8.pt")
    
    print0(f"Quantizing to {args.bits}-bit...")
    export_data = export_int8(model, args.output, bits=args.bits)
    
    quantized_size = sum(
        v.numel() * (1 if isinstance(v, torch.Tensor) and v.dtype == torch.int8 else 4)
        for v in export_data['quantized_weights'].values()
    )
    quantized_size += len(export_data['scales']) * 4
    
    compression = (1 - quantized_size / original_size) * 100
    print0(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
    print0(f"Compression: {compression:.1f}%")
    print0(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

