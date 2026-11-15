#!/usr/bin/env python3
"""
Turn the raw downloaded datasets into one unified, deduplicated corpus.

THE MOST IMPORTANT DESIGN DECISION IN THE WHOLE PROJECT.

A language model learns exactly the distribution you give it. If you train
it on CVE identifiers and nothing else, it becomes a CVE identifier
completer. The mix you choose is what the model becomes.

We use four document "shapes":

  (A) INSTRUCTION  "### Instruction:\\n<q>\\n\\n### Response:\\n<a>"
      Teaches the model to answer questions. Without this, a base model
      just continues text forever and never responds. This is what makes
      the thing usable as an assistant.

  (B) PROSE        Raw technical writing.
      Teaches grammar, long-range structure and vocabulary. Instruction
      documents alone tend to make a model stilted.

  (C) RECORD       Structured facts (a CVE, a score).
      Teaches the model the *shape* of domain knowledge and how to state
      it precisely.

  (D) CONTINUATION Raw text with no instruction wrapper.
      Keeps the model able to just write prose when asked to.

MIX BUDGET
    Each source gets a character budget. This is how you control what the
    model becomes, and it is the knob you will want to turn most often.
    The default mix is:

        nist           40%   long technical prose + standards Q&A
        nvd_qa         22%   deep technical Q&A (the "interview" style)
        circl          16%   CVE records: facts and severity
        pretrain_640k  20%   broad coverage, already clean

    NOTE: nvd_qa contains ~3M rows. Without a budget it would drown
    everything else. We round-robin across its shards so we sample the
    whole set rather than just the first file.

FILTERS
    - drop documents shorter than MIN_CHARS (junk, empty descriptions)
    - drop exact duplicates (CVE lists repeat a LOT)
    - normalise whitespace
    - strip leading ISO timestamps (the pretrain corpus has them)

Output: data/processed/{train,val}.jsonl  -- one JSON object per line:
        {"text": "...", "kind": "instruction", "source": "nist"}

Run:  python scripts/02_build_corpus.py [--total-chars N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

INSTRUCTION_FMT = "### Instruction:\n{q}\n\n### Response:\n{a}"

MIN_CHARS = 120
DEFAULT_TOTAL_CHARS = 600_000_000  # ~600 MB of text, ~170M tokens
VAL_EVERY = 100                     # deterministic 1% validation holdout

# (name, weight) -- weights are fractions of the total character budget.
#
# NOTE ON `pretrain_640k`
#     `11-47/Organized_PreTrain_Cyber_Security_640k` carries NO license
#     tag on the HuggingFace Hub, which means no permission was granted
#     to redistribute or derive from it. We exclude it from the release
#     corpus. The remaining three sources are permissive:
#
#         NIST cybersecurity training ....... CC0-1.0  (public domain)
#         rezaduty/cybersecurity-qa-v2 ...... Apache-2.0
#         CIRCL/vulnerability-scores ........ CC-BY-4.0 (attribution)
#
#     The weights below are renormalised after the exclusion, so the
#     corpus keeps the same total size and the same relative balance
#     between the three remaining sources.
MIX = {
    "nist": 0.40,
    "nvd_qa": 0.22,
    "circl": 0.16,
    "pretrain_640k": 0.00,
}

# Sources excluded from the released corpus, for the record.
EXCLUDED = ["pretrain_640k"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\n")


def normalize(text: str) -> str:
    """Collapse whitespace, strip control chars, drop leading timestamps."""
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = _TS.sub("", text)
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


def dedup_key(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def shorten_cpe(cpe: str) -> str:
    """cpe:2.3:o:vendor:product:1.1:*:*... -> vendor:product

    The version and the trailing wildcards are noise for language
    learning; vendor:product carries the actual information.
    """
    parts = cpe.split(":")
    if len(parts) >= 5:
        return f"{parts[3]}:{parts[4]}"
    return cpe


def severity_label(score: float | None) -> str:
    """CVSS base score -> qualitative rating."""
    if score is None:
        return "unrated"
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def best_cvss(row: dict) -> tuple[float | None, str]:
    """Pick the most recent CVSS score available and label the version."""
    for key, label in (
        ("cvss_v4_0", "CVSS v4.0"),
        ("cvss_v3_1", "CVSS v3.1"),
        ("cvss_v3_0", "CVSS v3.0"),
        ("cvss_v2_0", "CVSS v2.0"),
    ):
        v = row.get(key)
        if v is not None:
            try:
                return float(v), label
            except (TypeError, ValueError):
                continue
    return None, "CVSS"


def chat_to_instruction(messages: list[dict]) -> tuple[str, str]:
    """Pull (user, assistant) out of a chat record.

    The `system` message is DROPPED on purpose: it is the same boilerplate
    in every record, and repeating a constant 60-token string thousands of
    times teaches the model to emit that constant.
    """
    user = assistant = ""
    for m in messages or []:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user" and not user:
            user = content
        elif role == "assistant" and not assistant:
            assistant = content
    return user, assistant


# --------------------------------------------------------------------------
# Source 1: NIST  (chat format with real publication text as the answer)
# --------------------------------------------------------------------------

def iter_nist() -> Iterator[tuple[str, str]]:
    """Yield (text, kind) from the NIST dataset.

    Each record is a chat: a question about a NIST section, and an answer
    containing the actual publication text.

    We emit BOTH shapes:
      - instruction form, so the model learns Q&A over standards
      - the raw answer text, so the model learns to write standards prose
    """
    for split in ("train", "valid"):
        path = RAW / "nist" / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user, assistant = chat_to_instruction(rec.get("messages"))
                assistant = normalize(assistant)
                if len(assistant) < MIN_CHARS:
                    continue
                if user:
                    q = normalize(user)
                    text = INSTRUCTION_FMT.format(q=q, a=assistant)
                    if len(text) >= MIN_CHARS:
                        yield text, "instruction"
                yield assistant, "prose"


# --------------------------------------------------------------------------
# Source 2: CIRCL CVE records
# --------------------------------------------------------------------------

def iter_circl() -> Iterator[tuple[str, str]]:
    """Yield CVE records in a readable, factual form."""
    for split in ("train", "validation", "test"):
        path = RAW / "circl_vuln" / "data" / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                desc = normalize(row.get("description") or "")
                title = normalize(row.get("title") or "")
                if len(desc) < 40 and len(title) < MIN_CHARS:
                    continue

                cid = row.get("id") or ""
                score, score_label = best_cvss(row)

                pieces = []
                if title and title.lower() != desc.lower():
                    pieces.append(f"{cid}: {title}.")
                else:
                    pieces.append(f"{cid}: {desc}")
                if score is not None:
                    pieces.append(
                        f"{score_label} base score {score} "
                        f"({severity_label(score)})."
                    )
                cpes = row.get("cpes") or []
                if cpes:
                    short: list[str] = []
                    for c in cpes[:8]:
                        s = shorten_cpe(c)
                        if s not in short:
                            short.append(s)
                    if short:
                        pieces.append(
                            "Affected products include: " + ", ".join(short) + "."
                        )
                patch = row.get("patch_commit_url")
                if patch:
                    pieces.append(f"Reference patch: {patch}")

                text = " ".join(pieces)
                if len(text) >= MIN_CHARS:
                    yield text, "record"


# --------------------------------------------------------------------------
# Source 3: NVD question/answer pairs (chat format, ~3M rows)
# --------------------------------------------------------------------------

def iter_nvd_qa() -> Iterator[tuple[str, str]]:
    """Yield instruction pairs from the NVD Q&A dataset.

    Round-robins across the parquet shards so we sample the whole dataset
    rather than exhausting the first file.
    """
    data_dir = RAW / "nvd_qa" / "data"
    if not data_dir.exists():
        return

    readers = []
    for path in sorted(data_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        readers.append(iter(pf.iter_batches(batch_size=2048, columns=["messages"])))

    exhausted = 0
    while readers and exhausted < len(readers):
        for i, reader in enumerate(list(readers)):
            try:
                batch = next(reader)
            except StopIteration:
                readers.remove(reader)
                exhausted += 1
                continue
            for row in batch.to_pylist():
                user, assistant = chat_to_instruction(row.get("messages"))
                if len(user) < 20:
                    continue
                assistant = normalize(assistant)
                if len(assistant) < MIN_CHARS:
                    continue
                text = INSTRUCTION_FMT.format(q=normalize(user), a=assistant)
                if len(text) >= MIN_CHARS:
                    yield text, "instruction"


# --------------------------------------------------------------------------
# Source 4: the prebuilt PreTrain-640k corpus
# --------------------------------------------------------------------------

def iter_pretrain_640k() -> Iterator[tuple[str, str]]:
    """Yield documents from the prebuilt pretraining corpus."""
    root = RAW / "pretrain_640k"
    if not root.exists():
        return
    for path in sorted(root.rglob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                text = row.get("text")
                if not isinstance(text, str) or len(text) <= MIN_CHARS:
                    strs = [v for v in row.values() if isinstance(v, str)]
                    if not strs:
                        continue
                    text = max(strs, key=len)
                text = normalize(text)
                if len(text) >= MIN_CHARS:
                    yield text, "continuation"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

SOURCES = {
    "nist": iter_nist,
    "nvd_qa": iter_nvd_qa,
    "circl": iter_circl,
    "pretrain_640k": iter_pretrain_640k,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--total-chars", type=int, default=DEFAULT_TOTAL_CHARS,
        help="total characters of text to build (default 600M)",
    )
    ap.add_argument(
        "--include-pretrain-640k", action="store_true",
        help="also use 11-47/Organized_PreTrain_Cyber_Security_640k, which "
             "has no license grant. Off by default for the released corpus.",
    )
    ap.add_argument(
        "--out-dir", default=str(OUT),
        help="where to write train.jsonl / val.jsonl",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mix = dict(MIX)
    if args.include_pretrain_640k:
        mix["pretrain_640k"] = 0.20
        print("[mix] INCLUDING pretrain_640k (no license grant) "
              "-- this corpus is NOT clean for redistribution.")
    else:
        mix["pretrain_640k"] = 0.0

    # Renormalise so the total budget is still fully allocated.
    total_weight = sum(mix.values())
    if total_weight <= 0:
        raise SystemExit("all mix weights are zero")
    mix = {k: v / total_weight for k, v in mix.items() if v > 0}
    print("[mix] " + "  ".join(f"{k}={v:.1%}" for k, v in mix.items()))

    OUT.mkdir(parents=True, exist_ok=True)

    seen: set[bytes] = set()
    kind_counts: Counter = Counter()
    src_counts: Counter = Counter()
    src_chars: Counter = Counter()

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    val_docs: list[dict] = []
    n_written = 0

    with open(train_path, "w", encoding="utf-8") as ftr:
        for name, gen in SOURCES.items():
            if mix.get(name, 0) <= 0:
                print(f"[{name}] excluded (weight 0) -- skipped")
                continue
            budget = int(args.total_chars * mix[name])
            got = 0
            n_docs = 0
            print(f"[{name}] budget {budget / 1e6:.0f}M chars ...")

            for text, kind in gen():
                if got >= budget:
                    break
                key = dedup_key(text)
                if key in seen:
                    continue
                seen.add(key)

                got += len(text)
                n_docs += 1
                kind_counts[kind] += 1
                src_counts[name] += 1
                src_chars[name] += len(text)

                rec = {"text": text, "kind": kind, "source": name}
                n_written += 1
                # Deterministic 1% holdout for validation.
                if n_written % VAL_EVERY == 0:
                    val_docs.append(rec)
                else:
                    ftr.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"         -> {n_docs:,} docs, {got / 1e6:.1f}M chars")

    # --- write the validation split ----------------------------------
    #
    # IMPORTANT: the validation documents are SHUFFLED before writing.
    #
    # The corpus is written in source order (all NIST, then all NVD, then
    # all CIRCL, ...). If the validation split kept that order, then the
    # first N tokens of val.bin would be 100% NIST -- and since the
    # evaluator reads windows sequentially from the start, your validation
    # loss would silently measure a single source. That is a real bug we
    # hit: val loss 5.95 while the true held-out loss across all sources
    # was ~4.5.
    #
    # Shuffling makes any prefix of val.bin a representative sample.
    random.Random(20260808).shuffle(val_docs)
    with open(val_path, "w", encoding="utf-8") as fva:
        for rec in val_docs:
            fva.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_val = len(val_docs)

    print("\n--- corpus summary ---")
    print("  by kind:")
    for k, c in sorted(kind_counts.items()):
        print(f"    {k:16s} {c:>10,}")
    print("  by source:")
    for s in sorted(src_counts):
        print(f"    {s:16s} {src_counts[s]:>10,} docs  "
              f"{src_chars[s] / 1e6:>8.1f}M chars")
    total_chars = sum(src_chars.values())
    print(f"\n  train docs : {n_written - n_val:,}")
    print(f"  val docs   : {n_val:,}")
    print(f"  total chars: {total_chars / 1e6:.1f}M "
          f"(~{total_chars / 3.6 / 1e6:.0f}M tokens at ~3.6 chars/token)")
    print(f"  -> {train_path}")
    print(f"  -> {val_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
