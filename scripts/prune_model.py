import os
import argparse
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.prune import prune_model
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import compute_init, print0, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prune a trained model")
    parser.add_argument("--model-tag", type=str, required=True, help="Model tag (e.g., d12, distill_d12_mqa)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (None = latest)")
    parser.add_argument("--source", type=str, default="base", help="Model source: base|distill")
    parser.add_argument("--head-prune-ratio", type=float, default=0.2, help="Fraction of heads to prune (0.0-1.0)")
    parser.add_argument("--neuron-prune-ratio", type=float, default=0.2, help="Fraction of neurons to prune (0.0-1.0)")
    parser.add_argument("--output-tag", type=str, default=None, help="Output model tag (default: {model_tag}_pruned)")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")
    args = parser.parse_args()
    
    # Device setup
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    
    # Load model
    print0(f"Loading model: {args.model_tag} (step {args.step})")
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()
    
    original_params = sum(p.numel() for p in model.parameters())
    print0(f"Original model parameters: {original_params:,}")
    print0(f"Original config: {model.config.n_layer} layers, {model.config.n_embd} dim, {model.config.n_head} heads")
    
    # Prune model
    print0(f"Pruning: {args.head_prune_ratio*100:.1f}% heads, {args.neuron_prune_ratio*100:.1f}% neurons")
    pruned_model, pruned_config = prune_model(
        model,
        head_prune_ratio=args.head_prune_ratio,
        neuron_prune_ratio=args.neuron_prune_ratio
    )
    pruned_model.eval()
    
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = (1 - pruned_params / original_params) * 100
    print0(f"Pruned model parameters: {pruned_params:,} ({reduction:.1f}% reduction)")
    print0(f"Pruned config: {pruned_config.n_layer} layers, {pruned_config.n_embd} dim, {pruned_config.n_head} heads")
    
    # Save pruned model
    output_tag = args.output_tag if args.output_tag else f"{args.model_tag}_pruned"
    
    if args.source == "base":
        checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", output_tag)
    elif args.source == "distill":
        checkpoint_dir = os.path.join(get_base_dir(), "distill_checkpoints", output_tag)
    else:
        checkpoint_dir = os.path.join(get_base_dir(), f"{args.source}_checkpoints", output_tag)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save as step 0 (pruned from original)
    step = 0
    save_checkpoint(
        checkpoint_dir,
        step,
        pruned_model.state_dict(),
        None,  # No optimizer state
        {
            "step": step,
            "val_bpb": None,
            "model_config": {
                "sequence_len": pruned_config.sequence_len,
                "vocab_size": pruned_config.vocab_size,
                "n_layer": pruned_config.n_layer,
                "n_head": pruned_config.n_head,
                "n_kv_head": pruned_config.n_kv_head,
                "n_embd": pruned_config.n_embd,
                "window_pattern": pruned_config.window_pattern,
                "use_mqa": pruned_config.use_mqa,
                "multi_token_n": pruned_config.multi_token_n,
                "draft_n": pruned_config.draft_n,
                "draft_hidden_mult": pruned_config.draft_hidden_mult,
            },
            "pruning_info": {
                "original_model": args.model_tag,
                "original_step": args.step,
                "head_prune_ratio": args.head_prune_ratio,
                "neuron_prune_ratio": args.neuron_prune_ratio,
                "original_params": original_params,
                "pruned_params": pruned_params,
            },
        },
        rank=0,
    )
    
    print0(f"Pruned model saved to: {checkpoint_dir}")

if __name__ == "__main__":
    main()

