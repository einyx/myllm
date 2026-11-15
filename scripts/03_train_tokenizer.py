#!/usr/bin/env python3
"""
Train the byte-level BPE tokenizer on our corpus.

WHY WE TRAIN OUR OWN
    You could `pip install tokenizers` and use a pretrained GPT-2 or
    Llama tokenizer. That would be faster and probably slightly better at
    compression. But you would not understand how a tokenizer works, and
    more importantly you would not be able to *see* what your model is
    actually learning. For a learning project, writing it yourself is the
    point. It is also only ~450 lines.

SAMPLING
    We train on a random SUBSAMPLE of the corpus (default 60M chars).
    Why not all of it?

      - BPE quality saturates well before the corpus is exhausted. The
        merges that matter are the frequent ones, and 60MB contains
        hundreds of millions of token occurrences. That is plenty to
        learn "the", " vulnerability", " buffer overflow".
      - Training is O(merges x affected_words). Using 10x more text makes
        the merge loop 10x slower for a barely different vocabulary.

    We sample RANDOMLY rather than taking the first N bytes, because the
    corpus file is ordered by source (all NIST, then all NVD, ...). The
    first N bytes would be all NIST and the tokenizer would learn a
    NIST-only vocabulary.

Run:  python scripts/03_train_tokenizer.py [--vocab-size 32768]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def load_sample(path: Path, target_chars: int, seed: int = 0) -> list[str]:
    """Read the corpus and return a random sample of documents."""
    print(f"[tok] reading {path} ...")
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"[tok] {len(texts):,} documents loaded "
          f"({sum(len(t) for t in texts) / 1e6:.0f}M chars)")

    rng = random.Random(seed)
    rng.shuffle(texts)

    sample: list[str] = []
    total = 0
    for t in texts:
        sample.append(t)
        total += len(t)
        if total >= target_chars:
            break
    print(f"[tok] sampled {len(sample):,} documents, {total / 1e6:.0f}M chars")
    return sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(PROCESSED / "train.jsonl"))
    ap.add_argument("--out", default=str(PROCESSED / "tokenizer.json"))
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--sample-chars", type=int, default=60_000_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from myllm.tokenizer import BPETokenizer

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"error: {corpus} not found -- run 02_build_corpus.py first")
        return 1

    sample = load_sample(corpus, args.sample_chars)

    tok = BPETokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    t0 = time.time()
    tok.train(sample)
    elapsed = time.time() - t0
    print(f"[tok] trained in {elapsed / 60:.1f} min")

    tok.save(args.out)
    print(f"[tok] saved -> {args.out}")

    # --- sanity checks -------------------------------------------------
    print("\n" + "=" * 60)
    print("ROUND TRIP TEST")
    print("=" * 60)
    tests = [
        "Hello world!",
        "Buffer overflow in memcpy allows remote attackers to execute code.",
        "### Instruction:\nWhat is CVE-2021-44228?\n\n### Response:\n"
        "It is a remote code execution vulnerability in Apache Log4j.",
        "def main():\n    print('héllo wörld')\n",
        "",
    ]
    ok = True
    for t in tests:
        ids = tok.encode(t)
        back = tok.decode(ids)
        match = back == t
        ok = ok and match
        print(f"  {len(ids):>4} tokens | {'OK ' if match else 'FAIL'} | {t[:44]!r}")

    # Compression ratio: how many characters per token.
    sample_text = " ".join(sample[:2000])
    n_tok = len(tok.encode(sample_text))
    ratio = len(sample_text) / max(1, n_tok)
    print(f"\n  compression: {ratio:.2f} characters per token")
    print(f"  (a word-level tokenizer would be ~5-7; a char-level one ~1)")

    print("\n" + tok.describe(show=30))

    if not ok:
        print("\nWARNING: round trip failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
