"""
A byte-level BPE tokenizer, written from scratch.

Why a tokenizer at all?
-----------------------
An LLM does not see letters. It sees a sequence of integers ("tokens").
The tokenizer is the translator between text and those integers. Everything
downstream -- the embedding matrix, the context window, the loss -- is
measured in tokens, so the tokenizer is a real architectural decision, not a
detail.

Why byte-level BPE?
-------------------
Two problems with naive tokenizers:

  1. Out-of-vocabulary (UNK) tokens. If you build a vocabulary from a
     fixed word list, any unseen word becomes <UNK> and the model learns
     nothing about it. Fix: start from the 256 possible bytes. Every byte
     sequence is representable, always. No UNK, ever.

  2. Sequences too long. Character-level models need 5x more steps to
     predict a word than a good BPE. Fix: learn merges that group common
     byte sequences into single tokens.

So: start with every byte as its own token, then repeatedly merge the most
frequent adjacent pair of tokens. That is Byte-Pair Encoding.

The vocabulary layout produced here is:

    id 0    .. 255      -> the raw bytes 0x00..0xFF
    id 256  .. 256+S-1  -> special tokens (<|endoftext|>, ...)
    id 256+S ..         -> learned BPE merges, in merge order

Because ids 0..255 are literally the bytes, decoding is exact and lossless:
we concatenate the byte strings and utf-8 decode at the end.

The training algorithm (the interesting part)
---------------------------------------------
Naive BPE training re-scans the whole corpus once per merge. With 32k merges
over a large corpus that is ~10^11 operations -- far too slow in Python.

Instead we maintain, incrementally:
    pair_counts   -- how often each adjacent token pair occurs (weighted by
                      how often the whole word occurs)
    pair_to_words -- which words contain each pair

When we merge pair (A,B), only the words that actually contained (A,B)
change. We update just those. This turns training from O(merges x corpus)
into roughly O(total work touched), and is the same trick the production
Rust implementation uses.
"""

from __future__ import annotations

import json
import collections
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import regex

# GPT-2's pre-tokenizer pattern. This is not arbitrary:
#
#   's|'t|'re|'ve|'m|'ll|'d   English contractions
#   ?\p{L}+                   a word, optionally with a leading space
#   ?\p{N}+                   a number, optionally with a leading space
#   ?[^\s\p{L}\p{N}]+         punctuation/symbols
#   \s+(?!\S)|\s+              whitespace (the lookahead keeps the last
#                              space attached to the following word)
#
# Why bother? Because BPE merges must not cross a "word" boundary. If we
# merged across boundaries, the token " the" and the token "there" could
# interfere, and the tokenizer would learn vocabulary shaped by how the
# corpus happens to be laid out.
#
# The leading-space convention is deliberate: " hello" and "hello" are
# different tokens. That costs a little vocabulary but saves the model from
# having to relearn that a space exists.
PRETOKEN_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)

DEFAULT_SPECIALS = ("<|endoftext|>",)


class BPETokenizer:
    """Byte-level BPE tokenizer.

    Args:
        vocab_size: total vocabulary size including the 256 bytes and
            special tokens.
        specials: strings reserved for control sequences.
        min_frequency: stop training merges once the best pair occurs
            fewer than this many times (avoids learning junk merges).
    """

    def __init__(
        self,
        vocab_size: int = 32_768,
        specials: Sequence[str] = DEFAULT_SPECIALS,
        min_frequency: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.specials = tuple(specials)
        self.min_frequency = min_frequency

        # --- base vocabulary: every byte is a token ---
        self.vocab: dict[bytes, int] = {
            bytes([b]): b for b in range(256)
        }
        # Token id -> its byte string. Used for decoding and for reporting.
        self.inv_vocab: dict[int, bytes] = {
            b: bytes([b]) for b in range(256)
        }

        # Special tokens live above the bytes but below the merges.
        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}
        self.n_base = 256
        for i, s in enumerate(self.specials):
            tid = self.n_base + i
            self.special_to_id[s] = tid
            self.id_to_special[tid] = s

        # --- merge table ---
        # (left_bytes, right_bytes) -> id of the merged token.
        # `ranks` gives the priority: lower rank == merged earlier == higher
        # priority during encoding.
        self.merges: dict[tuple[bytes, bytes], int] = {}
        self.ranks: dict[tuple[bytes, bytes], int] = {}

        # Cache of word -> token ids, so repeated words are cheap.
        self._cache: dict[tuple[bytes, ...], list[int]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_merges(self) -> int:
        return len(self.merges)

    @property
    def eos_id(self) -> int:
        """Id of <|endoftext|>, used to separate documents."""
        return self.special_to_id[self.specials[0]]

    def __len__(self) -> int:
        """The number of distinct token ids the tokenizer can emit.

        This is the size of the ID SPACE, which is not the same as
        `len(self.vocab)`.

        The special tokens live in `special_to_id` / `id_to_special` and
        are deliberately NOT part of `self.vocab` (a merge could in
        principle produce the same bytes as a special token, and we do
        not want that collision). So counting `vocab` alone undercounts
        by `len(specials)` -- and an embedding matrix built one row too
        small will crash on the rarest token, at some random point
        during training.

        That is not hypothetical: it is exactly what happened here. The
        embedding had 32767 rows (valid ids 0..32766) but the tokenizer
        could emit id 32767, so training died with
        `IndexError: index out of range in self` 60 steps in.
        """
        n = len(self.vocab)
        # The highest merge id, +1, is the true end of the id space.
        # (The special token's id can sit *inside* the merge range, so
        # counting specials alone is not enough.)
        if self.inv_vocab:
            n = max(n, max(self.inv_vocab.keys()) + 1)
        for tid in self.id_to_special:
            n = max(n, tid + 1)
        return n

    def max_token_id(self) -> int:
        """The largest id this tokenizer can ever emit."""
        return len(self) - 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _pretokenize(self, text: str) -> Iterator[str]:
        """Split text into word-ish pieces using the GPT-2 pattern."""
        return regex.findall(PRETOKEN_PATTERN, text)

    def train(
        self,
        texts: Iterable[str],
        verbose: bool = True,
    ) -> "BPETokenizer":
        """Learn merges from a corpus.

        Args:
            texts: an iterable of raw strings (e.g. lines of a text file).
        """
        # ---- step 1: count unique pre-tokens -------------------------
        # We count *words* rather than raw bytes because the same word
        # appears thousands of times. Merging decisions only depend on
        # relative frequency, so counting unique words is enough.
        word_freqs: collections.Counter[tuple[bytes, ...]] = collections.Counter()
        for text in texts:
            for piece in self._pretokenize(text):
                word_freqs[tuple(bytes([b]) for b in piece.encode("utf-8"))] += 1

        if not word_freqs:
            raise ValueError("cannot train tokenizer on empty text")

        # Drop hapax legomena (words seen once) -- they cannot drive any
        # useful merge and they massively slow training.
        words = [w for w, c in word_freqs.items() if c >= 2]
        freqs = [word_freqs[w] for w in words]
        if verbose:
            print(
                f"[tokenizer] {len(word_freqs):,} unique pieces "
                f"({len(words):,} kept, {len(word_freqs) - len(words):,} dropped)"
            )

        # ---- step 2: build the initial pair statistics ----------------
        pair_counts: collections.Counter[tuple[bytes, bytes]] = collections.Counter()
        pair_to_words: dict[tuple[bytes, bytes], set[int]] = collections.defaultdict(set)

        for wi, word in enumerate(words):
            f = freqs[wi]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += f
                pair_to_words[pair].add(wi)

        target_merges = self.vocab_size - self.n_base - len(self.specials)

        # ---- step 3: repeatedly merge the best pair --------------------
        for merge_idx in range(target_merges):
            if not pair_counts:
                break

            # Highest count wins. Ties broken by the pair's byte content so
            # training is deterministic regardless of dict ordering.
            best_pair, best_count = max(
                pair_counts.items(), key=lambda kv: (kv[1], kv[0])
            )
            if best_count < self.min_frequency:
                if verbose:
                    print(
                        f"[tokenizer] stopping early: best pair count "
                        f"{best_count} < min_frequency {self.min_frequency}"
                    )
                break

            left, right = best_pair
            new_token = left + right
            new_id = self.n_base + len(self.specials) + merge_idx

            self.vocab[new_token] = new_id
            self.inv_vocab[new_id] = new_token
            self.merges[best_pair] = new_id
            self.ranks[best_pair] = merge_idx

            # Merge every occurrence of this pair in every affected word.
            affected = list(pair_to_words[best_pair])
            for wi in affected:
                word = words[wi]
                f = freqs[wi]

                # remove the word's old pairs from the global counters
                old_pairs = self._pair_counts(word)
                for p, c in old_pairs.items():
                    pair_counts[p] -= c * f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        pair_to_words.pop(p, None)
                    else:
                        pair_to_words[p].discard(wi)

                # actually perform the merge
                new_word = self._merge_word(word, best_pair)

                # add the word's new pairs
                words[wi] = new_word
                for p, c in self._pair_counts(new_word).items():
                    pair_counts[p] += c * f
                    pair_to_words[p].add(wi)

            # the merged pair is gone for good
            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            if verbose and (merge_idx + 1) % 2000 == 0:
                print(
                    f"[tokenizer] merge {merge_idx + 1:,}/{target_merges:,} "
                    f"| {best_count:,} occurrences "
                    f"| {left[:16]!r} + {right[:16]!r}"
                )

        self._cache.clear()
        if verbose:
            print(f"[tokenizer] done: vocab size = {len(self.vocab):,}")
        return self

    @staticmethod
    def _pair_counts(
        word: Sequence[bytes],
    ) -> dict[tuple[bytes, bytes], int]:
        """Count adjacent pairs within a single word."""
        counts: dict[tuple[bytes, bytes], int] = {}
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge_word(
        word: Sequence[bytes],
        target: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        """Replace every occurrence of `target` in `word` with its merge."""
        out: list[bytes] = []
        i = 0
        n = len(word)
        while i < n:
            if i + 1 < n and (word[i], word[i + 1]) == target:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _bpe(self, word: tuple[bytes, ...]) -> list[bytes]:
        """Apply the merge table to a word, highest-priority merge first."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        symbols = list(word)
        while len(symbols) > 1:
            # Find the adjacent pair with the lowest rank (merged earliest).
            best_rank = None
            best_idx = -1
            for i in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break  # no more applicable merges

            target = (symbols[best_idx], symbols[best_idx + 1])
            merged: list[bytes] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i + 1 < n and (symbols[i], symbols[i + 1]) == target:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        self._cache[word] = symbols
        return symbols

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Turn text into token ids.

        Special tokens are only emitted if `allow_special` is True; by
        default they are treated as ordinary text (which is what you want
        when encoding user data).
        """
        ids: list[int] = []

        # Split on special tokens first so they can be honoured as tokens.
        if allow_special and self.specials:
            pattern = "(" + "|".join(regex.escape(s) for s in self.specials) + ")"
            for chunk in regex.split(pattern, text):
                if chunk in self.special_to_id:
                    ids.append(self.special_to_id[chunk])
                else:
                    ids.extend(self._encode_plain(chunk))
        else:
            ids.extend(self._encode_plain(text))

        return ids

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._pretokenize(text):
            word = tuple(bytes([b]) for b in piece.encode("utf-8"))
            for sym in self._bpe(word):
                ids.append(self.vocab[sym])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Turn token ids back into text."""
        pieces: list[bytes] = []
        for tid in ids:
            if tid in self.id_to_special:
                pieces.append(self.id_to_special[tid].encode("utf-8"))
            elif tid in self.inv_vocab:
                pieces.append(self.inv_vocab[tid])
            else:
                raise KeyError(f"unknown token id {tid}")
        raw = b"".join(pieces)
        # errors="replace" because a byte sequence in the middle of the
        # corpus may not be valid utf-8 on its own.
        return raw.decode("utf-8", errors="replace")

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        out = bytearray()
        for tid in ids:
            if tid in self.id_to_special:
                out += self.id_to_special[tid].encode("utf-8")
            else:
                out += self.inv_vocab[tid]
        return bytes(out)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # merges are stored as hex so the JSON stays ASCII-safe and
        # lossless for arbitrary byte content.
        payload = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "specials": list(self.specials),
            "min_frequency": self.min_frequency,
            "n_base": self.n_base,
            "merges": [
                [left.hex(), right.hex(), tid]
                for (left, right), tid in self.merges.items()
            ],
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text())
        tok = cls(
            vocab_size=payload["vocab_size"],
            specials=tuple(payload["specials"]),
            min_frequency=payload.get("min_frequency", 2),
        )
        for left_hex, right_hex, tid in payload["merges"]:
            pair = (bytes.fromhex(left_hex), bytes.fromhex(right_hex))
            tok.merges[pair] = tid
            tok.ranks[pair] = len(tok.ranks)
            new_token = pair[0] + pair[1]
            tok.vocab[new_token] = tid
            tok.inv_vocab[tid] = new_token
        return tok

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self, show: int = 40) -> str:
        """A human-readable look at what the tokenizer learned."""
        lines = []
        lines.append(f"vocab size : {len(self.vocab):,}")
        lines.append(f"merges     : {len(self.merges):,}")
        lines.append(f"specials   : {self.specials}")
        lines.append("\nlongest tokens:")
        by_id = sorted(self.inv_vocab.items(), key=lambda kv: len(kv[1]), reverse=True)
        for tid, tok in by_id[:show]:
            lines.append(f"  {tid:6d}  {tok!r}")
        return "\n".join(lines)
