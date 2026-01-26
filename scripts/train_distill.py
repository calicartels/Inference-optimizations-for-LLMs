import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.distill_loss import compute_combined_loss
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, load_model
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
print_banner()

# Teacher model
parser.add_argument("--teacher-tag", type=str, default="d34", help="Teacher model tag (e.g., d34)")
parser.add_argument("--teacher-step", type=int, default=None, help="Teacher checkpoint step (None = latest)")
# Student model architecture
parser.add_argument("--student-depth", type=int, default=12, help="Student model depth")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")
parser.add_argument("--use-mqa", action="store_true", help="Use Multi-Query Attention in student")
# Distillation hyperparameters
parser.add_argument("--temperature", type=float, default=4.0, help="Temperature for distillation (higher = softer)")
parser.add_argument("--alpha", type=float, default=0.7, help="Weight for distillation loss (1-alpha for CE)")
# Training
parser.add_argument("--num-iterations", type=int, default=10000, help="number of optimization steps")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters")
parser.add_argument("--weight-decay", type=float, default=0.2, help="weight decay for Muon optimizer")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--warmup-ratio", type=float, default=0.1, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step")
# Evaluation
parser.add_argument("--eval-every", type=int, default=500, help="evaluate val bpb every N steps")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--sample-every", type=int, default=1000, help="sample from model every N steps")
parser.add_argument("--save-every", type=int, default=2000, help="save checkpoints every N steps")
# Logging
parser.add_argument("--run", type=str, default="distill", help="wandb run name")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory")
args = parser.parse_args()
user_config = vars(args).copy()

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# wandb logging
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-distill", name=args.run, config=user_config)

# Flash Attention status
if HAS_FA3:
    print0("Using Flash Attention 3")
else:
    print0("Flash Attention 3 not available, using PyTorch SDPA fallback")

# Load Teacher Model
print0(f"Loading teacher model: {args.teacher_tag}")
teacher_model, tokenizer, teacher_meta = load_model("base", device, phase="eval", model_tag=args.teacher_tag, step=args.teacher_step)
teacher_model.eval()
teacher_step = teacher_meta.get('step', 'unknown')
print0(f"Teacher loaded (step {teacher_step}, vocab_size={teacher_model.config.vocab_size})")

# Tokenizer
vocab_size = tokenizer.get_vocab_size()
token_bytes = get_token_bytes(device=device)
print0(f"Vocab size: {vocab_size:,}")

# Initialize Student Model
num_layers = args.student_depth
base_dim = args.student_depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
num_kv_heads = 1 if args.use_mqa else num_heads
head_dim = model_dim // num_heads

mqa_status = " (MQA)" if args.use_mqa else ""
print0(f"Student model: {num_layers} layers, {model_dim} dim, {num_heads} heads, {num_kv_heads} kv_heads{mqa_status}")

model_config_kwargs = dict(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    window_pattern=args.window_pattern,
    use_mqa=args.use_mqa,
    multi_token_n=0,  # Disabled for distillation (no training signal)
    draft_n=0,  # Disabled for distillation (no training signal)
)

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    student_model = GPT(model_config)

student_model.to_empty(device=device)
student_model.init_weights()

# Resume if requested
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"distill_d{args.student_depth}{'_mqa' if args.use_mqa else ''}"
checkpoint_dir = os.path.join(base_dir, "distill_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    student_model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = student_model
student_model = torch.compile(student_model, dynamic=False)
num_params = sum(p.numel() for p in student_model.parameters())
print0(f"Student model parameters: {num_params:,}")

# Initialize Optimizer
batch_lr_scale = (args.total_batch_size / (2**19)) ** 0.5 if args.total_batch_size != 2**19 else 1.0
if batch_lr_scale != 1.0:
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,}")

weight_decay_scaled = args.weight_decay * (12 / args.student_depth)**2
if args.student_depth != 12:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}")

adam_betas = (args.adam_beta1, args.adam_beta2)
optimizers = student_model.setup_optimizers(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * batch_lr_scale,
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data


# DataLoaders
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device
)
x, y, dataloader_state_dict = next(train_loader)

# Schedulers
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * args.num_iterations)
    warmdown_iters = round(args.warmdown_ratio * args.num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= args.num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (args.num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / args.num_iterations)

# Training loop state
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Training loop
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")

while True:
    last_step = step == args.num_iterations

    # Evaluation
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        student_model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(student_model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "val/bpb": val_bpb,
        })
        student_model.train()

    # Sampling
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        student_model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        student_model.train()

    # Save checkpoint
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        # Student forward
        with autocast_ctx:
            student_logits = student_model(x)
        
        # Teacher forward (no grad, with autocast for efficiency)
        with torch.no_grad(), autocast_ctx:
            teacher_logits = teacher_model(x)
        
        # Combined loss (in fp32 for numerical stability)
        loss, distill_loss, ce_loss = compute_combined_loss(
            student_logits.float(),
            teacher_logits.float(),
            y,
            temperature=args.temperature,
            alpha=args.alpha,
            ignore_index=-1,
            reduction='mean'
        )
        
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
    
    # Optimizer step
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    if isinstance(muon_optimizer, torch.optim.Optimizer):
        for group in muon_optimizer.param_groups:
            group["momentum"] = get_muon_momentum(step)
            group["weight_decay"] = get_weight_decay(step)
    
    adamw_optimizer.step()
    muon_optimizer.step()
    adamw_optimizer.zero_grad()
    muon_optimizer.zero_grad()
    
    synchronize()
    t1 = time.time()
    step_time = t1 - t0
    total_training_time += step_time
    
    # Logging
    smooth_train_loss = 0.99 * smooth_train_loss + 0.01 * train_loss.item()
    if step % 10 == 0:
        print0(f"Step {step:05d} | Loss: {train_loss.item():.4f} (distill: {distill_loss.item():.4f}, ce: {ce_loss.item():.4f}) | LR: {adamw_optimizer.param_groups[0]['lr']:.6f} | Time: {step_time:.3f}s")
    
    wandb_run.log({
        "step": step,
        "train/loss": train_loss.item(),
        "train/distill_loss": distill_loss.item(),
        "train/ce_loss": ce_loss.item(),
        "train/smooth_loss": smooth_train_loss,
        "train/lr": adamw_optimizer.param_groups[0]["lr"],
        "train/step_time": step_time,
    })
    
    step += 1

compute_cleanup()

