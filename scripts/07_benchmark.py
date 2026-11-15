#!/usr/bin/env python3
"""
Benchmark training throughput so we can size the big run honestly.

On a CPU, tokens/second is the only number that matters, and it is very
easy to be off by 5x in your planning. This script measures it directly:

  - forward + backward + optimizer step, exactly like the real trainer
  - for a given model config and batch size
  - reporting tokens/sec, and therefore "how long will this take?"

It also compares the two attention implementations (the readable `math`
one vs the fused `sdpa` one) and float32 vs bfloat16, so you can see the
real speed/quality tradeoff instead of guessing.

Run:
    python scripts/07_benchmark.py --preset small
    python scripts/07_benchmark.py --preset small --batch-sizes 4 8 16 32
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myllm.config import get_preset  # noqa: E402
from myllm.model import Transformer  # noqa: E402
from myllm.train import build_optimizer  # noqa: E402


def bench(
    model: Transformer,
    batch_size: int,
    seq_len: int,
    steps: int = 6,
    dtype: str = "float32",
    attn_impl: str = "sdpa",
) -> dict:
    """Time a few full training steps."""
    model.set_attn_impl(attn_impl)
    model.train()
    _, train_cfg = get_preset("tiny")
    train_cfg.weight_decay = 0.1
    train_cfg.lr = 1e-4
    opt = build_optimizer(model, train_cfg)
    opt.zero_grad(set_to_none=True)

    ctx = (
        torch.autocast("cpu", dtype=torch.bfloat16)
        if dtype == "bfloat16"
        else contextlib.nullcontext()
    )

    times = []
    for i in range(steps):
        x = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len))
        t0 = time.time()
        with ctx:
            out = model(x, y)
            out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        times.append(time.time() - t0)

    # Discard the first step (lazy allocator / warmup effects).
    times = sorted(times[1:])
    median = times[len(times) // 2]
    tokens = batch_size * seq_len
    return {
        "tokens": tokens,
        "step_time": median,
        "tok_per_s": tokens / median,
        "params": model.num_params(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--dtypes", nargs="+", default=["float32"])
    ap.add_argument("--attn-impls", nargs="+", default=["sdpa", "math"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--target-tokens", type=int, default=120_000_000)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    model_cfg, _ = get_preset(args.preset)
    print(f"preset: {args.preset} | {model_cfg.d_model}d, "
          f"{model_cfg.n_layers}L, ctx {model_cfg.max_seq_len}")
    print(f"threads: {torch.get_num_threads()}\n")

    model = Transformer(model_cfg)
    print(f"params: {model.num_params() / 1e6:.1f}M "
          f"(non-embedding {model.num_params(True) / 1e6:.1f}M)\n")

    print(f"{'dtype':>9} {'attn':>5} {'batch':>6} {'step_s':>8} {'tok/s':>10} "
          f"{'hours_for_target':>17}")
    print("-" * 64)

    best = None
    for dtype in args.dtypes:
        for impl in args.attn_impls:
            for bs in args.batch_sizes:
                try:
                    r = bench(model, bs, model_cfg.max_seq_len, args.steps,
                              dtype, impl)
                except Exception as e:  # noqa: BLE001
                    print(f"{dtype:>9} {impl:>5} {bs:>6}  FAILED: {e}")
                    continue
                hours = args.target_tokens / r["tok_per_s"] / 3600
                print(f"{dtype:>9} {impl:>5} {bs:>6} {r['step_time']:>8.3f} "
                      f"{r['tok_per_s']:>10.0f} {hours:>16.1f}h")
                if best is None or r["tok_per_s"] > best[1]:
                    best = ((dtype, impl, bs), r["tok_per_s"])

    if best:
        (dtype, impl, bs), tps = best
        print(f"\nbest: {dtype} / {impl} / batch {bs} -> {tps:.0f} tok/s")
        print(f"      {args.target_tokens / 1e6:.0f}M tokens would take "
              f"{args.target_tokens / tps / 3600:.1f} hours")
    return 0


if __name__ == "__main__":
    sys.exit(main())
