"""
The transformer. Everything here is written out so you can read the math.

A decoder-only transformer (the "GPT" family) does exactly this, for every
layer:

    x = x + Attention(RMSNorm(x))      # look at the past
    x = x + MLP(RMSNorm(x))            # think about what we saw

and at the very end:

    logits = Linear(RMSNorm(x))        # predict the next token

Two structural choices matter and are worth understanding:

1. PRE-NORM (norm before the sublayer, not after).
   The residual stream `x` is a highway that carries information from the
   embedding all the way to the output. If we normalized after each
   sublayer, every sublayer would also be normalizing the accumulated
   signal, and deep stacks become hard to optimize. Pre-norm keeps the
   highway clean and stable. (The original "post-norm" Transformer needed
   a very careful warmup schedule for exactly this reason.)

2. The residual stream is a fixed width (d_model) and is the ONLY thing
   that flows between layers. Attention and MLP both read from it and
   write back into it. This is the "information highway" idea, and it is
   why parameter counts and FLOPs are so predictable: every layer does
   O(d_model^2) work per token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ==========================================================================
# Normalization
# ==========================================================================

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm, without the mean and without bias.

    LayerNorm computes (x - mean(x)) / std(x). RMSNorm just drops the
    mean-subtraction:

        rms(x) = sqrt(mean(x^2) + eps)
        out    = (x / rms(x)) * g

    Why is it better? It is one fewer operation and, empirically, just as
    stable. It is what LLaMA and most modern models use.

    `g` is a learned per-channel gain, initialized to 1.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # Compute in float32 for numerical stability even if x is bf16.
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (x32 * rms).to(dtype)
        return out * self.weight


# ==========================================================================
# Rotary Position Embeddings (RoPE)
# ==========================================================================

class RotaryEmbedding(nn.Module):
    """Position information via rotation instead of addition.

    The old way (learned absolute position embeddings): add a vector to the
    token embedding, and the model has to learn every position separately.

    The RoPE way: take the query and key vectors and *rotate* them by an
    angle that depends on the absolute position. Because rotation is a
    linear operation, the dot product q_i . k_j depends only on the
    relative distance (i - j). That is exactly what attention wants.

    Concretely, we split each head's dimension into pairs and rotate each
    pair by angle = position * inv_freq, where the frequencies decay
    geometrically across the dimension (so low dimensions rotate slowly and
    encode long-range position, high dimensions rotate fast and encode
    fine position).

    The consequence you care about: RoPE extrapolates to longer sequences
    better than learned absolute embeddings, and it has zero parameters.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # inv_freq[i] = 1 / base^(2i/head_dim)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        # (max_seq_len, head_dim/2)
        freqs = torch.outer(pos, inv_freq)
        # Duplicate so we can use the "half-split" rotation convention.
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, start_pos: int, length: int, device, dtype):
        """Return cos/sin for a contiguous range of positions.

        cos/sin shape: (T, head_dim)
        """
        end = start_pos + length
        if end > self.max_seq_len:
            raise ValueError(
                f"position {end} exceeds max_seq_len {self.max_seq_len}"
            )
        cos = self.cos_cached[start_pos:end].to(device=device, dtype=dtype)
        sin = self.sin_cached[start_pos:end].to(device=device, dtype=dtype)
        return cos, sin

    def forward_positions(
        self, positions: torch.Tensor, device, dtype
    ):
        """Return cos/sin for ARBITRARY positions.

        cos/sin shape: (B, T, head_dim)

        Why we need this: when several prompts are batched together they
        have different lengths, so we left-pad them to the same width.
        Each sequence's first REAL token should still be at RoPE position
        0 -- otherwise the model sees a prompt shifted by the padding
        width, and the batched output differs from generating one prompt
        at a time.
        """
        if int(positions.max().item()) >= self.max_seq_len:
            raise ValueError("position exceeds max_seq_len")
        cos = self.cos_cached[positions].to(device=device, dtype=dtype)
        sin = self.sin_cached[positions].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: [a,b] -> [-b,a]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query/key tensors.

    x:  (B, H, T, D)
    cos/sin: either (T, D) -- one shared position range for the whole
             batch -- or (B, T, D) -- per-sequence positions, needed when
             prompts of different lengths are padded into one batch.
    """
    if cos.dim() == 2:
        # (1, 1, T, D) so it broadcasts against (B, H, T, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        # (B, 1, T, D)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


# ==========================================================================
# Attention
# ==========================================================================

class KVCache:
    """Storage for past keys and values during incremental decoding.

    During generation we compute one token at a time. Recomputing the
    whole prefix every step would be O(n^2) total work; caching the
    key/value tensors makes it O(n) total.
    """

    def __init__(self, n_layers: int) -> None:
        self.k: list[Optional[torch.Tensor]] = [None] * n_layers
        self.v: list[Optional[torch.Tensor]] = [None] * n_layers

    def reset(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None


class CausalSelfAttention(nn.Module):
    """Multi-head attention with Grouped-Query Attention (GQA).

    Standard MHA has n_heads query heads AND n_heads key/value heads.
    GQA lets several query heads SHARE one key/value head:

        n_kv_heads=4, n_heads=12  ->  each KV head serves 3 query heads

    Why? The key/value tensors are what you must store for every token
    during generation, and they are read by every layer of every future
    token. Shrinking them shrinks the memory cache and the memory bandwidth
    of generation, with very little quality loss. Llama 2 70B, Mistral, and
    Qwen all use GQA.

    The four projections:
        q_proj: d_model -> n_heads    * head_dim   (what we want to look at)
        k_proj: d_model -> n_kv_heads * head_dim   (what we have to offer)
        v_proj: d_model -> n_kv_heads * head_dim   (what we have to say)
        o_proj: n_heads * head_dim -> d_model      (combine the heads back)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        d = cfg.d_model

        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        attn_impl: str = "sdpa",
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        # (B, T, H*D) -> (B, H, T, D)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # --- append to the cache -------------------------------------
        if cache is not None:
            if cache.k[layer_idx] is not None:
                k = torch.cat([cache.k[layer_idx], k], dim=2)
                v = torch.cat([cache.v[layer_idx], v], dim=2)
            cache.k[layer_idx] = k
            cache.v[layer_idx] = v

        # --- attention ------------------------------------------------
        if attn_impl == "math":
            out = self._attention_math(q, k, v, attn_mask)
        else:
            out = self._attention_sdpa(q, k, v, attn_mask)

        # (B, H, T, D) -> (B, T, H*D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), k, v

    def _causal_mask(
        self,
        T_q: int,
        T_k: int,
        device,
    ) -> torch.Tensor:
        """Which query may attend to which key.

        When using a KV cache, T_q < T_k: query i (at absolute position
        `start + i`) may attend to keys 0 .. start+i. The `offset` is what
        makes that work.
        """
        offset = T_k - T_q
        return torch.ones(T_q, T_k, dtype=torch.bool, device=device).tril(offset)

    def _attention_math(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The textbook version. Slower, but you can read exactly what happens.

        1. scores = Q @ K^T / sqrt(d)      -> "how much does query i attend
                                              to key j?"
        2. mask out j > i (causal)         -> "you cannot see the future"
        3. softmax over j                   -> "turn scores into a
                                              distribution"
        4. out = probs @ V                 -> "average the values you attend
                                              to"
        """
        # GQA: each KV head is shared by n_rep query heads.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        T_q, T_k = q.shape[2], k.shape[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T_q, T_k)

        # Causal mask, plus any caller-supplied key mask (e.g. padding).
        mask = self._causal_mask(T_q, T_k, q.device)
        if attn_mask is not None:
            # attn_mask: (B, 1, T_q, T_k) bool, True = attend.
            mask = mask.unsqueeze(0) & attn_mask
        else:
            mask = mask.unsqueeze(0)

        scores = scores.masked_fill(~mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        return probs @ v

    def _attention_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Same math, fused into a single optimized kernel."""
        T_q, T_k = q.shape[2], k.shape[2]
        is_causal = T_q == T_k and attn_mask is None
        mask = attn_mask
        if not is_causal:
            causal = self._causal_mask(T_q, T_k, q.device)
            if mask is None:
                mask = causal.unsqueeze(0).unsqueeze(0)
            else:
                mask = causal.unsqueeze(0).unsqueeze(0) & mask
        try:
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                is_causal=is_causal,
                enable_gqa=self.n_rep > 1,
            )
        except (RuntimeError, TypeError):
            # Older torch without enable_gqa: expand manually.
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=is_causal
            )


# ==========================================================================
# Feed-forward (MLP)
# ==========================================================================

class SwiGLU(nn.Module):
    """The gated feed-forward block used by LLaMA and friends.

    Standard MLP:            h = act(W1 x);  out = W2 h
    SwiGLU (as in PaLM):     h = act(W_gate x) * (W_up x);  out = W_down h

    The extra `W_up` path gives the model two independent ways to build the
    hidden representation, and the elementwise product acts like a learned
    switch: the gate can suppress any part of the up-projection it does
    not want. The cost is that you need 3 matrices instead of 2, so the
    hidden width is usually set to ~8/3 * d_model to keep the parameter
    count comparable to a 2-matrix MLP.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.mlp_hidden
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ==========================================================================
# Transformer block
# ==========================================================================

class Block(nn.Module):
    """One transformer layer: attention then MLP, both with residuals."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        attn_impl: str = "sdpa",
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- attention sublayer ---
        h = self.attn_norm(x)
        attn_out, _, _ = self.attn(h, cos, sin, cache, layer_idx, attn_impl,
                                   attn_mask)
        x = x + attn_out

        # --- MLP sublayer ---
        h = self.mlp_norm(x)
        x = x + self.mlp(h)

        return x


# ==========================================================================
# The full model
# ==========================================================================

@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class Transformer(nn.Module):
    """A decoder-only transformer language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Weight tying: the output projection IS the input embedding.
            # One fewer matrix to learn, and empirically a small win for
            # small models (the embedding gets a gradient signal from
            # "which token was predicted", not just "which token was here").
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale down the residual-writing projections so that the residual
        # stream does not grow with depth (GPT-2's trick).
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.o_proj.weight.mul_(scale)
                block.mlp.down_proj.weight.mul_(scale)

        # Cache of cos/sin for the full training context (fast path).
        self.register_buffer(
            "cos_full", self.rope.cos_cached, persistent=False
        )
        self.register_buffer(
            "sin_full", self.rope.sin_cached, persistent=False
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        start_pos: int = 0,
        attn_impl: Optional[str] = None,
        attn_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Run the model.

        Args:
            idx: (B, T) token ids.
            targets: (B, T) next-token targets for the loss.
            cache: optional KVCache for incremental decoding.
            start_pos: absolute position of the first token in `idx`.
            attn_impl: override the configured attention implementation.
            attn_mask: optional (B, T_keys) boolean mask. True means "this
                key position is real and may be attended to". Used when
                batching prompts of different lengths: the padded
                positions are masked out so they cannot influence the
                real tokens. Must cover ALL key positions, including any
                that are already in the cache.
            positions: optional (B, T) long tensor of explicit RoPE
                positions per sequence. Needed for batched generation,
                where each padded sequence's real tokens must start at
                position 0 rather than at the padding offset.
        """
        B, T = idx.shape
        impl = attn_impl or self._attn_impl

        # Embed tokens and scale. The sqrt(d_model) factor matters: if you
        # do not scale, the logits start with enormous magnitude and the
        # first softmax saturates, which kills early training.
        x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)

        # Position information (RoPE).
        if positions is None:
            cos, sin = self.rope(start_pos, T, x.device, x.dtype)
        else:
            cos, sin = self.rope.forward_positions(
                positions.to(x.device), x.device, x.dtype
            )

        # Expand the key mask to (B, 1, 1, T_keys) so it broadcasts
        # against the (B, H, T_q, T_k) attention scores.
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :].bool()

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i, impl, attn_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Next-token prediction: position i predicts targets[i].
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return ModelOutput(logits=logits, loss=loss)

    # ------------------------------------------------------------------
    # Config plumbing
    # ------------------------------------------------------------------

    _attn_impl: str = "sdpa"

    def set_attn_impl(self, impl: str) -> None:
        assert impl in ("sdpa", "math"), impl
        self._attn_impl = impl

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg.n_layers)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        torch.save(
            {
                "config": self.cfg.to_dict(),
                "model": self.state_dict(),
                "attn_impl": self._attn_impl,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu") -> tuple["Transformer", ModelConfig]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig(**payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["model"])
        model.set_attn_impl(payload.get("attn_impl", "sdpa"))
        return model, cfg
