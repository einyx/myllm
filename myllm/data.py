"""
Turning text into tensors.

Two stages:

  1. CORPUS  -> token ids  (see scripts/02_build_corpus.py and
                              scripts/04_tokenize_corpus.py)
  2. token ids -> (input, target) batches for training

The important trick in stage 2 is PACKING. A naive pipeline would pad
every document to a fixed length, wasting most of the compute (most
security documents are not 512 tokens long, and short ones would be
mostly <pad>). Instead we concatenate ALL documents into one long stream
of token ids, separated by an <|endoftext|> marker, and then cut random
fixed-length windows out of that stream.

Because the stream is continuous, the model still learns "documents are
separated by <|endoftext|>" -- but every single token we compute a loss
on is a real token. No padding waste at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from .tokenizer import BPETokenizer


# ==========================================================================
# Stage 1: text -> token stream
# ==========================================================================

def tokenize_text_file(
    tokenizer: BPETokenizer,
    in_path: str | Path,
    out_path: str | Path,
    eos_id: Optional[int] = None,
    dtype: np.dtype = np.uint16,
    append_eos: bool = True,
    progress_every: int = 200_000,
    verbose: bool = True,
) -> int:
    """Tokenize a text file into a flat binary file of token ids.

    Each line of the input file is one document. Documents are joined with
    an <|endoftext|> id so the model learns document boundaries.

    Returns the number of tokens written.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if eos_id is None:
        eos_id = tokenizer.eos_id

    n_tokens = 0
    n_lines = 0
    is_jsonl = in_path.suffix == ".jsonl"

    with open(out_path, "wb") as f:
        buf = np.empty(1 << 16, dtype=dtype)
        buf_pos = 0

        with open(in_path, "r", encoding="utf-8", errors="replace") as src:
            for line in src:
                if is_jsonl:
                    # Documents may contain newlines, so they are stored as
                    # JSONL: one JSON object per line, with the text under
                    # the "text" key.
                    import json as _json
                    line = _json.loads(line).get("text", "")
                ids = tokenizer.encode(line.rstrip("\n"))
                if not ids:
                    continue
                if append_eos:
                    ids.append(eos_id)
                n_lines += 1
                n_tokens += len(ids)

                # Write into the buffer in chunks to avoid one huge array.
                for start in range(0, len(ids), len(buf) - buf_pos):
                    chunk = ids[start:start + len(buf) - buf_pos]
                    buf[buf_pos:buf_pos + len(chunk)] = chunk
                    buf_pos += len(chunk)
                    if buf_pos == len(buf):
                        f.write(buf.tobytes())
                        buf_pos = 0

                if verbose and n_lines % progress_every == 0:
                    print(f"  [tokenize] {n_lines:,} docs, {n_tokens:,} tokens")

        if buf_pos:
            f.write(buf[:buf_pos].tobytes())

    if verbose:
        print(f"  [tokenize] wrote {n_tokens:,} tokens from {n_lines:,} docs -> {out_path}")
    return n_tokens


# ==========================================================================
# Stage 2: token stream -> batches
# ==========================================================================

class TokenCorpus:
    """A memory-mapped array of token ids that can be sampled in windows.

    Memory mapping matters: a 200M-token corpus is 400MB as uint16. If we
    held it in RAM as Python ints that would be gigabytes. `np.memmap`
    lets the OS page it in from disk, and we only ever touch the windows
    we are about to train on.
    """

    def __init__(self, path: str | Path, dtype: np.dtype = np.uint16) -> None:
        self.path = Path(path)
        self.dtype = dtype
        self.tokens = np.memmap(self.path, dtype=dtype, mode="r")
        self.n_tokens = len(self.tokens)

    def __len__(self) -> int:
        return self.n_tokens

    def _window(self, start: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        # Gather a contiguous window with fancy indexing (one memcpy).
        offsets = np.arange(seq_len)
        inp = self.tokens[start + offsets]
        tgt = self.tokens[start + 1 + offsets]
        return inp.astype(np.int64), tgt.astype(np.int64)

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random windows (the training distribution)."""
        max_start = self.n_tokens - seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"corpus too small ({self.n_tokens} tokens) for seq_len {seq_len}"
            )
        starts = rng.integers(0, max_start, size=batch_size)
        xs, ys = [], []
        for s in starts:
            x, y = self._window(int(s), seq_len)
            xs.append(x)
            ys.append(y)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def sequential_batches(
        self,
        batch_size: int,
        seq_len: int,
        max_batches: Optional[int] = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Non-overlapping windows (the evaluation distribution).

        Deterministic, and it walks the whole validation set, so the
        validation loss is a stable number you can compare across runs.
        """
        stride = seq_len + 1
        n_windows = self.n_tokens // stride
        if max_batches is not None:
            n_windows = min(n_windows, max_batches * batch_size)

        for b in range(n_windows // batch_size):
            xs, ys = [], []
            for i in range(batch_size):
                start = (b * batch_size + i) * stride
                x, y = self._window(start, seq_len)
                xs.append(x)
                ys.append(y)
            yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def head_text(self, tokenizer: BPETokenizer, n: int = 400) -> str:
        """A peek at the beginning of the corpus, decoded. Handy for sanity checks."""
        ids = self.tokens[:n].tolist()
        return tokenizer.decode(ids)
