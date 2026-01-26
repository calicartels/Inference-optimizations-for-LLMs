# LLM Inference Optimizations

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

## What's next

- Knowledge distillation from nanochat's d34 checkpoint
- Structured pruning (remove entire heads/neurons)
- INT8 quantization

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu
source .venv/bin/activate
```
