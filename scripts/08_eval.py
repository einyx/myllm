#!/usr/bin/env python3
"""
Evaluate a checkpoint properly.

`06_generate.py --eval` gives you one number (validation perplexity).
This gives you the number you actually want to reason about: perplexity
BROKEN DOWN BY DOCUMENT TYPE.

Why that matters:
    A single perplexity number tells you the model is "good" or "bad".
    A breakdown tells you *what* it is good at. If the model scores 12
    perplexity on instruction documents but 45 on raw prose, you know it
    has learned the answer format but not the underlying writing -- and
    you know to change the mix.

It also reports:
    - generation speed (tokens/sec) with and without the KV cache
    - sample completions from held-out prompts
    - a crude hallucination check on generated CVE identifiers

Run:
    python scripts/08_eval.py --ckpt runs/small/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myllm.config import PROCESSED, ModelConfig  # noqa: E402
from myllm.data import TokenCorpus  # noqa: E402
from myllm.generate import generate, perplexity  # noqa: E402
from myllm.model import Transformer  # noqa: E402
from myllm.tokenizer import BPETokenizer  # noqa: E402

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}")


def load(ckpt_path: Path) -> tuple[Transformer, dict]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.no_grad()
def per_document_eval(
    model: Transformer,
    tokenizer: BPETokenizer,
    val_jsonl: Path,
    max_docs_per_kind: int = 400,
) -> dict:
    """Per-document loss, grouped by (kind, source).

    We evaluate document-by-document rather than over a packed stream,
    because a packed stream would let the model use the tail of one
    document as context for the head of the next. That is what the model
    was TRAINED on, so it is the honest number, but grouping by document
    is what lets us attribute the loss to a document type.
    """
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen: dict[tuple[str, str], int] = defaultdict(int)

    with open(val_jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("kind", "?"), rec.get("source", "?"))
            if seen[key] >= max_docs_per_kind:
                continue

            ids = tokenizer.encode(rec["text"])
            # A document can be longer than the model's context window.
            # RoPE only has positions up to max_seq_len, so we must
            # truncate -- otherwise this crashes.
            if len(ids) > model.cfg.max_seq_len:
                ids = ids[: model.cfg.max_seq_len]
            if len(ids) < 32:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor([ids[1:]], dtype=torch.long)
            out = model(x, y)
            buckets[key].append(float(out.loss))
            seen[key] += 1

    return {k: v for k, v in buckets.items()}


@torch.no_grad()
def measure_generation_speed(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str,
    n_tokens: int = 64,
) -> float:
    """Tokens per second during cached generation."""
    t0 = time.time()
    generate(model, tokenizer, prompt, max_new_tokens=n_tokens,
             temperature=0.8, seed=0)
    dt = time.time() - t0
    return n_tokens / max(dt, 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=str(PROCESSED / "tokenizer.json"))
    ap.add_argument("--val-bin", default=str(PROCESSED / "val.bin"))
    ap.add_argument("--val-jsonl", default=str(PROCESSED / "val.jsonl"))
    ap.add_argument("--max-docs-per-kind", type=int, default=400)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"error: {ckpt} not found")
        return 1

    tokenizer = BPETokenizer.load(args.tokenizer)
    model, payload = load(ckpt)
    print(f"model: {model.cfg.name}  {model.num_params() / 1e6:.1f}M params  "
          f"step {payload.get('step', '?')}")

    # ---- overall validation perplexity -------------------------------
    val_bin = Path(args.val_bin)
    if val_bin.exists():
        corpus = TokenCorpus(val_bin)
        ppl = perplexity(model, corpus, max_batches=60)
        print(f"\nvalidation perplexity: {ppl:.2f}   "
              f"(random over {model.cfg.vocab_size:,} tokens = "
              f"{model.cfg.vocab_size:,})")

    # ---- what is the validation set actually made of? ----------------
    # A validation number is only meaningful if the validation set has
    # the same shape as the training set. If your corpus is written in
    # source order and you evaluate the first N windows, you may be
    # measuring a single source and calling it "validation".
    val_jsonl = Path(args.val_jsonl)
    if val_jsonl.exists():
        composition: Counter = Counter()
        with open(val_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                composition[rec.get("source", "?")] += 1
        total = sum(composition.values())
        print("\nvalidation set composition:")
        for src, n in composition.most_common():
            print(f"  {src:<18} {n:>8,}  ({100 * n / total:5.1f}%)")

    # ---- breakdown by document type -----------------------------------
    val_jsonl = Path(args.val_jsonl)
    if val_jsonl.exists():
        print("\nper-document perplexity by type:")
        print(f"  {'kind':<14} {'source':<16} {'docs':>5} {'loss':>8} {'ppl':>10}")
        print("  " + "-" * 58)
        buckets = per_document_eval(
            model, tokenizer, val_jsonl, args.max_docs_per_kind
        )
        for key in sorted(buckets):
            losses = buckets[key]
            avg = sum(losses) / len(losses)
            print(f"  {key[0]:<14} {key[1]:<16} {len(losses):>5} "
                  f"{avg:>8.3f} {math.exp(min(20, avg)):>10.2f}")

    # ---- generation speed ---------------------------------------------
    tok_per_s = measure_generation_speed(
        model, tokenizer,
        "### Instruction:\nExplain SQL injection.\n\n### Response:\n",
    )
    print(f"\ngeneration speed: {tok_per_s:.1f} tok/s (with KV cache)")

    # ---- sample completions -------------------------------------------
    print("\n" + "=" * 72)
    print("SAMPLE COMPLETIONS (held-out prompts)")
    print("=" * 72)
    prompts = [
        "### Instruction:\nWhat is a buffer overflow and how do I prevent it?"
        "\n\n### Response:\n",
        "### Instruction:\nDescribe CVE-2021-44228.\n\n### Response:\n",
        "### Instruction:\nHow should I store database credentials?\n\n"
        "### Response:\n",
        "The CISA Known Exploited Vulnerabilities catalog tracks",
    ]
    for i, p in enumerate(prompts):
        out = generate(model, tokenizer, p, max_new_tokens=120,
                       temperature=0.7, seed=1234 + i)
        print(f"\n>>> {p.strip()[:90]}")
        print("-" * 72)
        print(out.strip()[:900])

    # ---- hallucination check -------------------------------------------
    print("\n" + "=" * 72)
    print("CVE ID CHECK")
    print("=" * 72)
    print("(A small model will invent CVE IDs. This is a sanity check on")
    print(" how many of the IDs it produces actually exist in the corpus.)")
    real_ids = set()
    train_jsonl = ROOT / "data" / "processed" / "train.jsonl"
    if train_jsonl.exists():
        with open(train_jsonl, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 200_000:
                    break
                real_ids.update(CVE_RE.findall(line))
        print(f"  loaded {len(real_ids):,} real CVE ids from the training corpus")

        invented = 0
        checked = 0
        for p in prompts[:2]:
            out = generate(model, tokenizer, p, max_new_tokens=120,
                           temperature=0.7, seed=7)
            for cid in CVE_RE.findall(out):
                checked += 1
                if cid not in real_ids:
                    invented += 1
        if checked:
            print(f"  generated {checked} CVE ids, {invented} not in the "
                  f"corpus ({100 * invented / checked:.0f}% invented)")
            print("  -> expect this to be high. It is a base model, not a")
            print("     retrieval system.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
