#!/usr/bin/env python3
"""
Tokenize the corpus into flat binary files that the trainer memory-maps.

Output:
    data/processed/train.bin   uint16 token ids, documents separated by EOS
    data/processed/val.bin     same format, held-out

WHY uint16?
    Our vocabulary is 32768 tokens, which fits in 16 bits (0..65535).
    Storing each token in 2 bytes instead of 4 halves the file size and
    halves the RAM needed to hold the corpus. For a 170M-token corpus
    that is the difference between 340MB and 680MB -- not a big deal on
    this machine, but the right habit.

PARALLELISM
    Pure-Python tokenization is the slowest part of the whole pipeline
    (the model trains at ~300 tok/s but we can tokenize at ~200k tok/s
    across 32 cores). We split the JSONL file into byte ranges and give
    one range per worker process.

Run:  python scripts/04_tokenize_corpus.py [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

_TOKENIZER_PATH: str = ""


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER_PATH
    _TOKENIZER_PATH = tokenizer_path


def _tokenize_range(job: tuple[int, int, str, int]) -> tuple[int, str]:
    """Tokenize one byte range of the corpus into its own .bin file."""
    start_byte, n_lines, out_path, chunk_idx = job
    from myllm.tokenizer import BPETokenizer

    tok = BPETokenizer.load(_TOKENIZER_PATH)
    eos_id = tok.eos_id

    ids_out: list[int] = []
    with open(_CORPUS_PATH, "rb") as f:
        f.seek(start_byte)
        for _ in range(n_lines):
            line = f.readline()
            if not line:
                break
            try:
                text = json.loads(line.decode("utf-8"))["text"]
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                continue
            ids = tok.encode(text)
            if ids:
                ids_out.extend(ids)
                ids_out.append(eos_id)

    arr = np.asarray(ids_out, dtype=np.uint16)
    arr.tofile(out_path)
    return len(arr), out_path


_CORPUS_PATH = ""


def line_offsets(path: Path) -> list[int]:
    """Byte offset of the start of each line."""
    offsets = [0]
    with open(path, "rb") as f:
        for line in f:
            offsets.append(offsets[-1] + len(line))
    return offsets


def tokenize_file(
    corpus: Path,
    out: Path,
    tokenizer_path: Path,
    workers: int = 16,
) -> int:
    global _CORPUS_PATH
    _CORPUS_PATH = str(corpus)

    offsets = line_offsets(corpus)
    n_lines = len(offsets) - 1
    if n_lines == 0:
        raise ValueError(f"{corpus} is empty")

    print(f"[tok] {n_lines:,} documents, {corpus.stat().st_size / 1e6:.0f}MB")
    print(f"[tok] splitting across {workers} workers ...")

    # Round-robin the lines across workers so each worker gets a
    # representative slice of the (source-ordered) corpus.
    n_chunks = workers
    per = (n_lines + n_chunks - 1) // n_chunks
    jobs = []
    tmp_paths = []
    for i in range(n_chunks):
        lo = i * per
        hi = min(n_lines, lo + per)
        if lo >= hi:
            break
        tmp = out.parent / f".{out.name}.part{i}"
        jobs.append((offsets[lo], hi - lo, str(tmp), i))
        tmp_paths.append(tmp)

    t0 = time.time()
    ctx = mp.get_context("fork")
    _init_worker(str(tokenizer_path))
    with ctx.Pool(len(jobs)) as pool:
        results = pool.map(_tokenize_range, jobs)

    total = sum(r[0] for r in results)
    print(f"[tok] {total:,} tokens in {time.time() - t0:.0f}s")

    # Concatenate the parts in order.
    with open(out, "wb") as dst:
        for _, part in results:
            with open(part, "rb") as src:
                while chunk := src.read(1 << 22):
                    dst.write(chunk)
            Path(part).unlink()

    print(f"[tok] -> {out} ({out.stat().st_size / 1e6:.0f}MB, "
          f"{total:,} tokens)")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(PROCESSED / "train.jsonl"))
    ap.add_argument("--val", default=str(PROCESSED / "val.jsonl"))
    ap.add_argument("--tokenizer", default=str(PROCESSED / "tokenizer.json"))
    ap.add_argument("--out-dir", default=str(PROCESSED))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"error: {tokenizer_path} not found -- run 03_train_tokenizer.py")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = tokenize_file(
        Path(args.train), out_dir / "train.bin", tokenizer_path, args.workers
    )
    n_val = tokenize_file(
        Path(args.val), out_dir / "val.bin", tokenizer_path,
        min(args.workers, 4),
    )

    print("\n--- tokenized corpus ---")
    print(f"  train : {n_train:,} tokens")
    print(f"  val   : {n_val:,} tokens")
    print(f"  ratio : {n_train / max(1, n_val):.0f}x")
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    sys.exit(main())
