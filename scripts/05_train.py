#!/usr/bin/env python3
"""
Train the model.

Examples
--------
    # fast pipeline validation (~13M params, minutes)
    python scripts/05_train.py --preset tiny

    # the real run (~126M params, days on a CPU)
    python scripts/05_train.py --preset small

    # continue a previous run
    python scripts/05_train.py --preset small --resume runs/small

    # override individual settings
    python scripts/05_train.py --preset small --lr 3e-4 --batch-size 8 --grad-accum 4

Any flag not given falls back to the preset's value, so you can always
start from the preset and nudge one thing.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myllm.config import PROCESSED, RUNS_DIR, get_preset  # noqa: E402
from myllm.data import TokenCorpus  # noqa: E402
from myllm.model import Transformer  # noqa: E402
from myllm.train import Trainer, find_latest_checkpoint  # noqa: E402
from myllm.utils import configure_threads, set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=["tiny", "small", "medium"])
    ap.add_argument("--out", default=None, help="run directory")
    ap.add_argument("--tokenizer", default=str(PROCESSED / "tokenizer.json"))
    ap.add_argument("--train-bin", default=str(PROCESSED / "train.bin"))
    ap.add_argument("--val-bin", default=str(PROCESSED / "val.bin"))

    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights from a checkpoint (fresh optimizer)")

    # overrides
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--attn-impl", default=None, choices=["sdpa", "math"])
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16"])
    # NOTE: default is None, not 0. If the default were 0 we would
    # overwrite the preset's (measured-optimal) thread count with
    # "leave torch's default", which on a hybrid P/E-core CPU is ~1.8x
    # slower. None means "only override if the user asked".
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    args = ap.parse_args()

    # ---- config -------------------------------------------------------
    model_cfg, train_cfg = get_preset(args.preset)
    out_dir = Path(args.out) if args.out else RUNS_DIR / args.preset

    # Map CLI flag names onto TrainConfig field names.
    overrides = {
        "lr": "lr",
        "batch_size": "batch_size",
        "grad_accum": "grad_accum",
        "max_tokens": "max_tokens",
        "max_steps": "max_steps",
        "warmup_steps": "warmup_steps",
        "attn_impl": "attn_impl",
        "dtype": "dtype",
        "seed": "seed",
        "log_every": "log_every",
        "ckpt_every": "ckpt_every",
        "eval_every": "eval_every",
        "threads": "num_threads",
    }
    for flag, field in overrides.items():
        val = getattr(args, flag, None)
        if val is not None:
            setattr(train_cfg, field, val)

    set_seed(train_cfg.seed)
    n_threads = configure_threads(train_cfg.num_threads)

    # ---- tokenizer ----------------------------------------------------
    # The model's embedding matrix must have exactly one row per token
    # the tokenizer can emit, or the ids will not line up. The tokenizer
    # may have stopped short of its target vocab size (merges stop when
    # the best pair is too rare), so we read the real size from it.
    from myllm.tokenizer import BPETokenizer

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"error: {tokenizer_path} not found -- run 03_train_tokenizer.py")
        return 1
    tokenizer = BPETokenizer.load(tokenizer_path)
    real_vocab = len(tokenizer)
    if real_vocab != model_cfg.vocab_size:
        print(f"[tokenizer] vocab size is {real_vocab:,}, "
              f"but the preset asked for {model_cfg.vocab_size:,} "
              f"-- using the tokenizer's real size.")
        model_cfg.vocab_size = real_vocab

    # ---- data ----------------------------------------------------------
    train_bin = Path(args.train_bin)
    val_bin = Path(args.val_bin)
    if not train_bin.exists():
        print(f"error: {train_bin} not found -- run 04_tokenize_corpus.py")
        return 1

    print("[data] loading corpus (memory-mapped) ...")
    train_corpus = TokenCorpus(train_bin)
    val_corpus = TokenCorpus(val_bin) if val_bin.exists() else None
    print(f"[data] train: {train_corpus.n_tokens:,} tokens")
    if val_corpus:
        print(f"[data] val:   {val_corpus.n_tokens:,} tokens")

    # --- SANITY CHECK: the embedding must cover every id in the corpus --
    #
    # The model's embedding has `vocab_size` rows, so it accepts ids
    # 0..vocab_size-1. If the corpus contains a larger id the model
    # crashes with a bare `IndexError: index out of range in self` deep
    # inside torch.nn.Embedding, some random number of steps into
    # training. Checking it up front turns a mystery crash into a clear
    # message.
    corpus_max = int(train_corpus.tokens.max())
    if corpus_max >= model_cfg.vocab_size:
        print(f"\nERROR: the corpus contains token id {corpus_max}, but the "
              f"model's embedding only has {model_cfg.vocab_size} rows "
              f"(valid ids 0..{model_cfg.vocab_size - 1}).")
        print("       The corpus and the tokenizer disagree. Re-tokenize "
              "the corpus with the tokenizer you are training against:")
        print("         python scripts/04_tokenize_corpus.py")
        return 1
    if val_corpus is not None:
        val_max = int(val_corpus.tokens.max())
        if val_max >= model_cfg.vocab_size:
            print(f"\nERROR: the validation corpus contains token id "
                  f"{val_max}, but the model's embedding only has "
                  f"{model_cfg.vocab_size} rows.")
            return 1
    print(f"[data] max token id {corpus_max} < vocab_size "
          f"{model_cfg.vocab_size} -- ok")

    # ---- model ---------------------------------------------------------
    model = Transformer(model_cfg)
    n_params = model.num_params()
    print(f"[model] {model_cfg.name}: {n_params / 1e6:.1f}M params "
          f"({n_params / 1e6:.1f}M non-embedding)")
    print(f"[model] d_model={model_cfg.d_model} layers={model_cfg.n_layers} "
          f"heads={model_cfg.n_heads}/{model_cfg.n_kv_heads} kv "
          f"mlp={model_cfg.mlp_hidden} ctx={model_cfg.max_seq_len}")

    trainer = Trainer(
        model=model,
        cfg=train_cfg,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        out_dir=out_dir,
    )

    # ---- checkpoints ----------------------------------------------------
    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        print(f"[init] warm-started weights from {args.init_from}")
    elif args.resume:
        ckpt = find_latest_checkpoint(out_dir)
        if ckpt is None:
            print(f"[resume] no checkpoint found in {out_dir}; starting fresh")
        else:
            trainer.load_checkpoint(ckpt)

    # ---- go -------------------------------------------------------------
    print(f"[run] output -> {out_dir}")
    print(f"[run] threads: {n_threads}\n")
    t0 = time.time()
    trainer.train()
    print(f"\n[run] wall time: {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
