"""
Model + training configuration.

Everything that defines "how big the model is" and "how we train it" lives
here so that the model code stays about *math* and the training code stays
about *process*.

Two presets are provided:

    tiny  -- ~13M params. Trains in minutes on a laptop CPU. Used to validate
             the whole pipeline end-to-end before we spend days on a big run.
    small -- ~126M params. The real target. Takes days on a CPU.

A "config" here is just a dataclass holding numbers. Swapping the preset
changes the shape of the model, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = ROOT / "runs"

# Short alias used by the scripts.
PROCESSED = PROCESSED_DIR


@dataclass
class ModelConfig:
    """Shape of the transformer.

    The four numbers that matter most:

    d_model     -- width of the residual stream. Every activation is this wide.
    n_layers    -- depth. Roughly how many times we let the model "think".
    n_heads     -- how many independent attention patterns run in parallel.
    n_kv_heads  -- how many key/value heads (Grouped-Query Attention). Fewer
                   than n_heads means less memory/compute for the same quality.
    vocab_size  -- how many tokens the tokenizer knows.

    head_dim is derived rather than specified, so that d_model == n_heads *
    head_dim always holds (the standard, well-behaved choice).
    """

    name: str = "small"
    vocab_size: int = 32_768
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    n_kv_heads: int = 4
    max_seq_len: int = 512
    rope_base: float = 10_000.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    # MLP inner width. LLaMA's rule of thumb: 8/3 * d_model, rounded to a
    # multiple of 64 (which keeps the GEMM shapes friendly to the CPU).
    mlp_hidden: int | None = None
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) for GQA"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.mlp_hidden is None:
            # 8/3 * 768 = 2048 exactly, and 2048 is a multiple of 64.
            raw = int(8 * self.d_model / 3)
            self.mlp_hidden = ((raw + 63) // 64) * 64

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def n_params(self) -> int:
        """Exact parameter count (without building the model)."""
        d, h, kv, hd = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim
        m = self.mlp_hidden
        emb = self.vocab_size * d
        per_layer = (
            d * h * hd          # q_proj
            + d * kv * hd       # k_proj
            + d * kv * hd       # v_proj
            + h * hd * d        # o_proj
            + d * m             # gate_proj
            + d * m             # up_proj
            + m * d             # down_proj
            + 2 * d             # two RMSNorms
        )
        # When embeddings are tied, lm_head shares tok_emb.weight and adds
        # NO new parameters. Only an untied head costs an extra d_model.
        head = 0 if self.tie_embeddings else self.d_model
        return emb + self.n_layers * per_layer + head + d  # + final norm

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """How we run the optimization loop."""

    # --- batching ---
    batch_size: int = 8           # sequences per micro-step
    grad_accum: int = 4          # micro-steps per optimizer step
    # effective batch = batch_size * grad_accum * seq_len tokens

    # --- optimization ---
    lr: float = 3e-4
    min_lr_ratio: float = 0.1     # cosine decays to this fraction of peak
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # --- budget ---
    max_tokens: int = 120_000_000  # total training tokens to see
    max_steps: int = 10_000_000    # hard cap on optimizer steps

    # --- runtime ---
    seed: int = 1337
    log_every: int = 20
    ckpt_every: int = 250
    eval_every: int = 250
    eval_batches: int = 20
    # 16 threads, not torch's default of 24.
    #
    # This CPU is an i9-13900K: 8 P-cores + 16 E-cores. On hybrid CPUs the
    # E-cores are much slower at FP32 GEMM and they do not hide the
    # P-core latency, so oversubscribing them makes everything SLOWER.
    # Measured on this machine (126M model, batch 16x512):
    #
    #     8 threads -> 363 tok/s
    #    16 threads -> 467 tok/s   <- best
    #    24 threads -> 269 tok/s   <- torch's default!
    #    32 threads -> 256 tok/s
    #
    # Set to 0 to use torch's default instead.
    num_threads: int = 16
    # float32, NOT bfloat16.
    #
    # On a GPU, bf16 is 2-4x faster than fp32. On this CPU it is 93x
    # SLOWER (measured: 5 tok/s vs 467 tok/s), because the i9-13900K has
    # AVX2 but no AVX-512-BF16, so every bf16 matmul is emulated.
    # Reduced precision is only fast where the hardware has native
    # instructions for it.
    dtype: str = "float32"        # float32 | bfloat16
    attn_impl: str = "sdpa"       # sdpa | math (math is the readable one)

    def effective_batch_tokens(self, seq_len: int) -> int:
        return self.batch_size * self.grad_accum * seq_len


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

def tiny() -> tuple[ModelConfig, TrainConfig]:
    """~13M params. The pipeline-validation workhorse.

    Trains on a CPU in minutes so you can change the model and immediately
    see whether the change helped.
    """
    model = ModelConfig(
        name="tiny",
        vocab_size=32_768,
        d_model=256,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
    )
    train = TrainConfig(
        batch_size=16,
        grad_accum=2,
        lr=1e-3,
        warmup_steps=100,
        max_tokens=8_000_000,
        dtype="float32",
    )
    return model, train


def small() -> tuple[ModelConfig, TrainConfig]:
    """~126M params. The real thing.

    Budget: 60M tokens at ~584 tok/s on 16 threads == roughly 28 hours.
    That is about 0.4 epochs of the corpus we built, and well below the
    Chinchilla-optimal ~2.5B tokens for a model this size -- we simply do
    not have the compute on a CPU. Within that constraint, 60M tokens with
    a complete cosine schedule beats 120M tokens with an unfinished one.

    On a 32-core CPU this is a multi-day run. That is fine: we checkpoint
    every few hundred steps, so there is always a usable model on disk and
    the loss curve is visible while it runs.
    """
    model = ModelConfig(
        name="small",
        vocab_size=32_768,
        d_model=768,
        n_layers=16,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=512,
    )
    train = TrainConfig(
        batch_size=16,
        grad_accum=2,
        lr=6e-4,
        warmup_steps=500,
        max_tokens=60_000_000,
        dtype="float32",
    )
    return model, train


PRESETS = {
    "tiny": tiny,
    "small": small,
    "medium": lambda: (
        ModelConfig(
            name="medium",
            vocab_size=32_768,
            d_model=1024,
            n_layers=20,
            n_heads=16,
            n_kv_heads=4,
            max_seq_len=512,
        ),
        TrainConfig(
            batch_size=8,
            grad_accum=4,
            lr=5e-4,
            warmup_steps=1000,
            max_tokens=200_000_000,
        ),
    ),
}


def get_preset(name: str) -> tuple[ModelConfig, TrainConfig]:
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; available: {sorted(PRESETS)}"
        )
    return PRESETS[name]()
