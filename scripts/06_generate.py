#!/usr/bin/env python3
"""
Sample from a trained checkpoint and evaluate it.

Examples
--------
    # one-shot: prompt and print the completion
    python scripts/06_generate.py --ckpt runs/small/best.pt \
        --prompt "### Instruction:
What is a buffer overflow?

### Response:
"

    # interactive chat loop
    python scripts/06_generate.py --ckpt runs/small/best.pt --interactive

    # evaluate perplexity on the validation set
    python scripts/06_generate.py --ckpt runs/small/best.pt --eval

    # sample several prompts at once
    python scripts/06_generate.py --ckpt runs/small/best.pt --prompts-file prompts.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myllm.config import PROCESSED  # noqa: E402
from myllm.data import TokenCorpus  # noqa: E402
from myllm.generate import generate, generate_batch, perplexity  # noqa: E402
from myllm.model import Transformer  # noqa: E402
from myllm.tokenizer import BPETokenizer  # noqa: E402


DEFAULT_INSTRUCTION = (
    "### Instruction:\n{q}\n\n### Response:\n"
)


def load(ckpt_path: Path):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from myllm.config import ModelConfig

    cfg = ModelConfig(**payload["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"[load] {ckpt_path}  (step {payload.get('step', '?')}, "
          f"{model.num_params() / 1e6:.1f}M params)")
    if payload.get("val_loss"):
        print(f"[load] checkpoint val loss: {payload['val_loss']:.4f} "
              f"(ppl {math.exp(min(20, payload['val_loss'])):.2f})")
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--tokenizer", default=str(PROCESSED / "tokenizer.json"))
    ap.add_argument("--val-bin", default=str(PROCESSED / "val.bin"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--eval", action="store_true", help="report val perplexity")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"error: {ckpt} not found")
        return 1

    tokenizer = BPETokenizer.load(args.tokenizer)
    model, _ = load(ckpt)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    # ---- evaluation ----
    if args.eval:
        val_bin = Path(args.val_bin)
        if not val_bin.exists():
            print(f"error: {val_bin} not found")
            return 1
        corpus = TokenCorpus(val_bin)
        ppl = perplexity(model, corpus, max_batches=40)
        print(f"\nvalidation perplexity: {ppl:.2f}")
        print(f"  (random over a {model.cfg.vocab_size:,}-token vocab "
              f"would be {model.cfg.vocab_size:,})")

    # ---- one-shot ----
    if args.prompt is not None:
        print("\n" + "=" * 72)
        print(args.prompt)
        print("-" * 72)
        out = generate(model, tokenizer, args.prompt, **gen_kwargs)
        print(out)
        print("=" * 72)

    # ---- batch from file ----
    if args.prompts_file:
        prompts = [
            DEFAULT_INSTRUCTION.format(q=line.strip())
            for line in Path(args.prompts_file).read_text().splitlines()
            if line.strip()
        ]
        outs = generate_batch(model, tokenizer, prompts, **gen_kwargs)
        for q, o in zip(prompts, outs):
            print("\n" + "=" * 72)
            print(q)
            print("-" * 72)
            print(o)
        print("=" * 72)

    # ---- interactive ----
    if args.interactive:
        print("\nInteractive mode. Ctrl-D or 'exit' to quit.\n")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            prompt = DEFAULT_INSTRUCTION.format(q=q)
            out = generate(model, tokenizer, prompt, **gen_kwargs)
            print(f"\nmodel> {out}\n")

    if not (args.eval or args.prompt or args.prompts_file or args.interactive):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
