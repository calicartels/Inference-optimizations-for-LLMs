# PicoChat: nanochat, but under 10$ with distillation and pruning + inference optimizations

Implementing inference optimizations for Large Language Models using nanochat as the base.

I'm taking Karpathy's nanochat and adding inference optimizations. Since this is one of the few trainable models under $100, I want to experiment and actually train these things myself.

I rented a Vast.ai A100 instance, and the goal is to keep costs low — not training from scratch, but using distillation and pruning to get a smaller, faster model.

## What I've done so far

### 1. Multi-Query Attention (MQA)

Added a `use_mqa` flag to GPTConfig. When enabled, all query heads share a single key-value head instead of each having their own.

**Why:** KV cache drops from `6 × seq × dim` to `1 × seq × dim`. That's 6x less memory during inference. The quality drop is small and recoverable with distillation.

```python
config = GPTConfig(use_mqa=True)  # n_kv_head automatically set to 1
```

### 2. Multi-Token Prediction

Added extra prediction heads that predict future tokens (t+2, t+3, t+4) alongside the main head that predicts t+1.

**Why 3 heads?** 
- 1-2 doesn't help much
- 4+ adds params without proportional benefit  
- 3 is the sweet spot

**Why it matters:** Forces the model to "think ahead" during training, and these heads will become useful for speculative decoding later.

```python
# Default is now multi_token_n=3
loss, logits, mt_logits = model(x, y, return_multi_token=True)
# mt_logits = {"head_2": ..., "head_3": ..., "head_4": ...}
```

### 3. Draft Head for Speculative Decoding

Added a lightweight 2-layer MLP that predicts multiple tokens at once (default: 4 tokens).

**How it works:**
1. Draft head makes quick guesses for the next N tokens
2. Main model verifies all N tokens in one parallel forward pass
3. Keep the longest prefix of correct guesses
4. Repeat

**Why this speeds things up:** Instead of N sequential forward passes, you do ~2 (draft + verify). If the draft is right 50% of the time, you're already 2x faster. The draft head is tiny (`n_embd * 0.5` hidden dim) so it's basically free.

```python
# Speculative generation
for token in model.generate_speculative(prompt_tokens, max_tokens=100):
    print(token)
```

**Hyperparameters:**
- `draft_n=4`: Predicts 4 tokens per draft. More = faster if accurate, but accuracy drops.
- `draft_hidden_mult=0.5`: Hidden layer is half the model dimension. Keeps it fast.

### 4. Knowledge Distillation

Added infrastructure to train a smaller student model by distilling knowledge from nanochat's d34 checkpoint.

**How it works:**
1. Load teacher model (d34) and freeze it
2. Train student to match teacher's logit distribution via KL divergence
3. Combine distillation loss with standard cross-entropy
4. Student learns from teacher's "soft" predictions, not just hard labels

**Why this matters:** Train a much smaller model (e.g., d12 with MQA) that retains most of the teacher's knowledge. The student achieves ~90% quality with 5x fewer parameters.

```bash
python -m scripts.train_distill \
    --teacher-tag d34 \
    --student-depth 12 \
    --use-mqa \
    --temperature 4.0 \
    --alpha 0.7
```

**Hyperparameters:**
- `temperature=4.0`: Softens teacher distribution. Higher = more exploration.
- `alpha=0.7`: Distillation weight. 0.7 = 70% distill, 30% ground truth CE loss.

**Training Components:**
- Main loss: KL divergence (teacher) + cross-entropy (ground truth)
- Multi-token loss (0.2x weight): Trains heads to predict t+2, t+3, t+4 simultaneously
- Draft head loss (0.1x weight): Trains draft head to match teacher's future predictions

All components train end-to-end in a single pass. The multi-token heads and draft head are now enabled and trained during distillation.

### 5. Structured Pruning

Added infrastructure to remove entire attention heads and MLP neurons based on L1 importance scores.

**How it works:**
1. Compute L1 norm importance for each attention head and MLP neuron
2. Remove the least important heads/neurons (e.g., bottom 20%)
3. Create a new model with reduced dimensions
4. Copy relevant weight slices from the original model

**Why this matters:** Structured pruning reduces model size by 30-50% with minimal quality loss. The pruned model can then be fine-tuned with distillation to recover most of the quality.

```bash
# Prune a trained model
python -m scripts.prune_model \
    --model-tag d12 \
    --head-prune-ratio 0.2 \
    --neuron-prune-ratio 0.2

# Then fine-tune the pruned model
python -m scripts.train_distill \
    --teacher-tag d34 \
    --model-tag d12_pruned \
    --source base
```

**Hyperparameters:**
- `head_prune_ratio=0.2`: Remove 20% of attention heads (least important)
- `neuron_prune_ratio=0.2`: Remove 20% of MLP neurons (least important)

**Note:** Pruning reduces `n_head` and `n_embd` uniformly across layers. The pruned model maintains the same architecture but with smaller dimensions.

### 6. INT8 Quantization

Added support for exporting models to INT8 format for deployment.

**How it works:**
1. Quantize weights to 8-bit integers using per-tensor scaling
2. Store quantization scales for dequantization during inference
3. Export model in compressed format (4x smaller than FP32/BF16)

**Why this matters:** INT8 models are 4x smaller and 2-3x faster on hardware with INT8 support. Quality loss is minimal (<5%) for most tasks.

```bash
# Export model to INT8
python -m scripts.export_int8 \
    --model-tag d12 \
    --output model_int8.pt
```

**Usage:**
- `--bits=8`: Quantization bits (8 for INT8, can use 4 for even smaller models)
- `--output`: Output file path (default: `int8_models/{model_tag}_int8.pt`)

**Note:** Quantization uses per-tensor scaling (one scale per weight tensor). For better quality, you can fine-tune with quantization-aware training, but post-training quantization works well for most cases.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu
source .venv/bin/activate
```
