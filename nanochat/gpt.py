"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Multi-Query Attention (MQA) option for maximum KV cache compression
- Multi-Token Prediction heads for improved training signal
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Multi-Query Attention: use single KV head for all query heads that can be shared across multiple query heads, 
    # Reduces KV cache by n_head times 
    use_mqa: bool = False
    # Multi-Token Prediction: extra heads predicting future tokens (t+2, t+3, t+4)
    # Improves training signal and enables speculative decoding
    multi_token_n: int = 3  # predicts 3 future tokens (t+2, t+3, t+4)
    # Draft Head for self-draft speculative decoding
    # Lightweight MLP that predicts multiple tokens at once for fast drafting
    draft_n: int = 4  # number of tokens to draft in one shot
    draft_hidden_mult: float = 0.5  # draft head hidden dim = n_embd * mult (smaller = faster)
    
    def __post_init__(self):
        if self.use_mqa:
            self.n_kv_head = 1


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class DraftHead(nn.Module):
    """
    Lightweight MLP head for self-draft speculative decoding.
    Predicts multiple tokens at once from the last hidden state.
    
    During inference:
    1. Draft head quickly predicts N draft tokens
    2. Main model verifies all N tokens in one parallel forward pass
    3. Accept verified tokens, resample where draft was wrong
    
    This amortizes the cost of autoregressive decoding.
    """
    def __init__(self, n_embd, vocab_size, draft_n, hidden_mult=0.5):
        super().__init__()
        self.draft_n = draft_n
        hidden_dim = int(n_embd * hidden_mult)
        # 2-layer MLP: hidden layer + output layer predicting draft_n * vocab_size
        self.fc1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, draft_n * vocab_size, bias=False)
        self.vocab_size = vocab_size
    
    def forward(self, x):
        """
        Args:
            x: hidden states (B, T, n_embd) or (B, n_embd) for single position
        Returns:
            draft_logits: (B, T, draft_n, vocab_size) or (B, draft_n, vocab_size)
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)  # (B, 1, n_embd)
        
        B, T, _ = x.shape
        h = F.relu(self.fc1(x)) ** 2  # ReLU² like the main MLP
        out = self.fc2(h)  # (B, T, draft_n * vocab_size)
        out = out.view(B, T, self.draft_n, self.vocab_size)
        
        if squeeze:
            out = out.squeeze(1)  # (B, draft_n, vocab_size)
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Multi-token prediction heads: predict tokens at t+2, t+3, etc.
        self.multi_token_heads = nn.ModuleDict()
        for i in range(config.multi_token_n):
            self.multi_token_heads[f"head_{i+2}"] = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Draft head for self-draft speculative decoding
        self.draft_head = None
        if config.draft_n > 0:
            self.draft_head = DraftHead(
                n_embd=config.n_embd,
                vocab_size=config.vocab_size,  # use actual vocab, not padded
                draft_n=config.draft_n,
                hidden_mult=config.draft_hidden_mult
            )
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Multi-token prediction heads (same init as lm_head)
        for head in self.multi_token_heads.values():
            torch.nn.init.normal_(head.weight, mean=0.0, std=0.001)
        # Draft head: small std for fc1 (like other projections), zeros for fc2 (starts neutral)
        if self.draft_head is not None:
            torch.nn.init.normal_(self.draft_head.fc1.weight, mean=0.0, std=n_embd**-0.5)
            torch.nn.init.zeros_(self.draft_head.fc2.weight)  # start with zero output

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        multi_token_params = list(self.multi_token_heads.parameters())
        draft_head_params = list(self.draft_head.parameters()) if self.draft_head is not None else []
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(multi_token_params) + len(draft_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=multi_token_params, lr=unembedding_lr * dmodel_lr_scale),  # same LR as lm_head
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale),  # same LR as token embedding
            dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
            dict(params=x0_params, lr=scalar_lr, betas=(0.96, 0.95)), # higher beta1 for x0 scalars
        ]
        # Add draft head params if present
        if draft_head_params:
            adam_groups.insert(2, dict(params=draft_head_params, lr=unembedding_lr * dmodel_lr_scale))
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_multi_token=False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits
        
        # Multi-token prediction heads (for training with future token prediction)
        multi_token_logits = {}
        if return_multi_token and self.multi_token_heads:
            for name, head in self.multi_token_heads.items():
                mt_logits = head(x)
                mt_logits = mt_logits[..., :self.config.vocab_size]
                mt_logits = mt_logits.float()
                mt_logits = softcap * torch.tanh(mt_logits / softcap)
                multi_token_logits[name] = mt_logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if return_multi_token:
                return loss, logits, multi_token_logits
            return loss
        else:
            # inference: just return the logits directly
            if return_multi_token:
                return logits, multi_token_logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

    @torch.inference_mode()
    def generate_speculative(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Speculative decoding using self-draft.
        
        Algorithm:
        1. Get hidden state for last token
        2. Draft head predicts N tokens quickly
        3. Verify all N+1 positions (original + drafts) in one forward pass
        4. Accept longest prefix where draft matches verification
        5. Yield accepted tokens, repeat
        
        This reduces the effective number of forward passes from max_tokens to ~max_tokens / acceptance_rate.
        """
        assert isinstance(tokens, list)
        assert self.draft_head is not None, "Draft head not available (draft_n=0 in config)"
        device = self.get_device()
        draft_n = self.config.draft_n
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        tokens_generated = 0
        
        while tokens_generated < max_tokens:
            # Forward pass to get hidden states (we need the raw hidden state for draft head)
            # Run trunk to get hidden states
            B, T = ids.size()
            T0 = 0  # no kv cache for simplicity in this version
            cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
            
            x = self.transformer.wte(ids)
            x = norm(x)
            x0 = x
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](ids) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i], None)
            x = norm(x)
            
            # Get hidden state for last position
            last_hidden = x[:, -1, :]  # (B, n_embd)
            
            # Draft N tokens using draft head
            draft_logits = self.draft_head(last_hidden)  # (B, draft_n, vocab_size)
            if temperature > 0:
                draft_logits = draft_logits / temperature
                draft_probs = F.softmax(draft_logits, dim=-1)
                draft_tokens = torch.multinomial(draft_probs.view(-1, draft_probs.size(-1)), num_samples=1, generator=rng)
                draft_tokens = draft_tokens.view(B, draft_n)  # (B, draft_n)
            else:
                draft_tokens = torch.argmax(draft_logits, dim=-1)  # (B, draft_n)
            
            # Prepare verification sequence: original + draft tokens
            verify_ids = torch.cat([ids, draft_tokens], dim=1)  # (B, T + draft_n)
            
            # Verify all draft tokens with full model in one forward pass
            verify_logits = self.forward(verify_ids)  # (B, T + draft_n, vocab_size)
            
            # Sample from verification logits for positions T-1 to T+draft_n-1
            # Position T-1 verifies the first draft token, etc.
            accepted = []
            for i in range(draft_n):
                pos = T - 1 + i  # verification position
                if pos >= verify_logits.size(1):
                    break
                    
                v_logits = verify_logits[:, pos, :]
                if top_k is not None:
                    v, _ = torch.topk(v_logits, min(top_k, v_logits.size(-1)))
                    v_logits[v_logits < v[:, [-1]]] = -float('Inf')
                
                if temperature > 0:
                    v_logits = v_logits / temperature
                    v_probs = F.softmax(v_logits, dim=-1)
                    verified_token = torch.multinomial(v_probs, num_samples=1, generator=rng)
                else:
                    verified_token = torch.argmax(v_logits, dim=-1, keepdim=True)
                
                # Check if draft matches verification
                if i < draft_n and draft_tokens[0, i] == verified_token[0, 0]:
                    accepted.append(verified_token[0, 0].item())
                else:
                    # Draft wrong, accept verified token and stop
                    accepted.append(verified_token[0, 0].item())
                    break
            
            # Yield accepted tokens
            for tok in accepted:
                if tokens_generated >= max_tokens:
                    return
                yield tok
                tokens_generated += 1
            
            # Update ids with accepted tokens
            accepted_tensor = torch.tensor([accepted], dtype=torch.long, device=device)
            ids = torch.cat([ids, accepted_tensor], dim=1)
