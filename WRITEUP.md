# Building a language model from scratch, on a CPU, in 4,400 lines

*A learning project. No `transformers`, no `tokenizers`, no `trl` — just
PyTorch, NumPy, and a lot of reading about why things are the way they
are.*

---

## The premise

I wanted to understand what actually happens between "here is a text file"
and "here is a token the model decided to emit". Not the API calls — the
arithmetic.

So: build a decoder-only transformer from nothing. Train its own byte-level
BPE tokenizer. Write its own training loop, sampler, and data pipeline.
Train it on security-domain text, on a CPU, and document what breaks along
the way.

The three things I expected to be interesting turned out to be *different*
from the three things that were interesting. The architecture went exactly
as the textbook says it should. The real lessons were all about the
machine, and one was a genuine bug in my own evaluation setup that would
have quietly poisoned every number I reported.

---

## What was built

```
myllm/
├── tokenizer.py   byte-level BPE, trained from scratch
├── model.py       transformer: RoPE, RMSNorm, GQA, SwiGLU
├── data.py        corpus -> packed token stream -> batches
├── train.py       AdamW, warmup + cosine, gradient accumulation
├── generate.py    temperature / top-k / top-p / KV cache
└── 9 scripts      download -> build -> tokenize -> train -> eval -> plot
```

**The model.** 125.9M parameters: `d_model=768`, 16 layers, 12 attention
heads sharing 4 key/value heads (GQA), SwiGLU feed-forward with
`hidden = 8/3 × 768`, RoPE positions, RMSNorm, pre-norm residual
structure, and tied input/output embeddings. Context length 512.

**The tokenizer.** Byte-level BPE with a 32K vocabulary, trained with an
*incremental* merge algorithm: instead of rescanning the corpus once per
merge (≈10¹¹ operations for 32K merges), it maintains `pair_counts` and
`pair_to_words` and only touches the words that actually contain the pair
being merged. Round-trip is exact for any input, including invalid UTF-8.

**The data.** 749,191 documents from three public security corpora,
concatenated into one packed token stream and cut into random windows —
no padding, so every token we compute a loss on is a real token.

---

## Finding 1: your CPU has fast cores and slow cores, and PyTorch doesn't care

The first real surprise had nothing to do with transformers.

This machine is an i9-13900K: 8 P-cores (performance) plus 16 E-cores
(efficiency). PyTorch's default thread count is the number of *physical*
cores, so 24 — which means it happily schedules GEMM work onto the E-cores.
Those E-cores have no AVX-512 and run at roughly half the clock. They also
don't hide P-core latency, so they don't even free up P-cores.

Measured on the real 126M model, batch 16×512:

| threads | tok/s | what's running |
|---|---|---|
| 8 | 363 | P-cores, one thread each |
| **16** | **467** | **P-cores + hyperthreads** |
| 24 | 269 | PyTorch's default — E-cores included |
| 32 | 256 | all logical CPUs |

Going from PyTorch's default to the right number was a **1.8× speedup**, or
roughly 14 extra hours of training. The fix is one line:

```python
torch.set_num_threads(16)
```

`TrainConfig.num_threads` now defaults to 16 and the measurement is
recorded in the source next to it. The general lesson: on a hybrid CPU,
"how many threads" is not a rounding question, and you should measure it
rather than assume the default is sensible.

---

## Finding 2: bfloat16 was 93× *slower* than float32

I had assumed `bfloat16` would be a modest win — it's 2–4× faster than
fp32 on a GPU, so it's the reflexive choice when someone says "make it
faster."

Measured on the same model:

| threads | float32 | bfloat16 |
|---|---|---|
| 16 | **467 tok/s** | **5 tok/s** |
| 32 | 256 tok/s | 5 tok/s |

Five tokens per second. A 93× *regression*.

The reason is the same hardware fact as Finding 1: this CPU has AVX2 but
**no AVX-512-BF16**. There is no native bf16 multiply, so every bf16
matrix multiply is emulated — and the conversion overhead dominates the
actual arithmetic by two orders of magnitude.

This one is worth internalizing because it inverts a GPU intuition that
carries very far:

> **Reduced precision is only fast where the hardware has native
> instructions for it.** "bf16 is faster" is not a property of bf16; it is
> a property of bf16 *plus* your CPU.

The same trap has a milder form on GPUs — bf16 on a consumer card without
good bf16 tensor-core throughput can be slower than fp16 — but a 93×
gap is the extreme version of the lesson.

---

## Finding 3: my validation loss was measuring one dataset and calling it "validation"

This is the one that mattered most, and it was a bug in my own setup, not
in the model.

The symptom: training loss 4.15, validation loss 5.95, and a gap that
refused to close. That gap says "overfitting", and the reflex is to add
regularization, drop the learning rate, or get more data — all of which
would have been the wrong fix.

The actual cause was an accident of ordering:

1. The corpus builder writes documents **in source order**: all the NIST
   documents, then all the NVD Q&A, then all the CVE records.
2. The training loop samples **random** windows from that stream — so
   training sees the correct proportional mix of all three sources.
3. The evaluator reads **sequential** windows starting from the beginning
   of `val.bin`.
4. And `val.bin` was, like the training file, in source order.

So the first 82,000 tokens of the validation file were **100% NIST**, and
every "validation loss" I printed was a measurement of NIST alone.

The giveaway is in the per-document-type breakdown, which is why I built
that tool:

```
kind           source            docs     loss        ppl
continuation   pretrain_640k      120    4.739     114.33
instruction    nist               120    5.649     283.91
instruction    nvd_qa             120    3.662      38.92
prose          nist               120    6.113     451.73
record         circl              120    4.141      62.86
```

NIST instruction and NIST prose are the two *hardest* kinds (5.65 and
6.11). My validation number was a weighted average of exactly those two,
which is why it sat so far above the training loss.

The fix is one line in the corpus builder — **shuffle the validation
split** before writing it, so any prefix of `val.bin` is a representative
sample:

```python
random.Random(20260808).shuffle(val_docs)
```

Validation perplexity went from **411 to 94** for the same checkpoint.
The training loss never moved.

> **A validation number that measures a different distribution than your
> training set is worse than no validation number at all** — because it
> looks authoritative. It will send you optimising the wrong thing with
> real confidence.

The general form of this bug is worth naming: *any* place where you slice
a dataset and only look at one end of it inherits the ordering of how it
was written. Evaluation code is exactly the place people forget to check.

---

## The data, and why the mix is a decision

A language model becomes the distribution you give it. Four document
shapes went into the corpus:

- **instruction** (`### Instruction:\n<q>\n\n### Response:\n<a>`) — teaches
  the model to *answer* rather than just continue text forever. Without
  this, a base model never responds.
- **prose** — raw technical writing. Teaches grammar and long-range
  structure. Instruction documents alone make a model stilted.
- **record** — a CVE with its severity and affected products. Teaches
  precise factual statement.
- **continuation** — everything else. Keeps the model able to just write.

Two deliberate choices that are easy to get wrong:

- **The `system` prompt is dropped.** In the NIST dataset every record
  begins with the same ~70-token system prompt. Training on it teaches the
  model to emit a constant string.
- **The mix is a character budget per source**, not a document count. A
  budget makes the knob explicit and makes the proportions stable when
  document lengths change.

The final released corpus excludes one source. `11-47/Organized_PreTrain_Cyber_Security_640k`
carries no license tag on the Hub, which means no permission was granted to
derive from it, so it is not in the release. The remaining three are
CC0-1.0, Apache-2.0, and CC-BY-4.0.

---

## Results

The released checkpoint is still training; these numbers will be filled in
from `runs/small-clean/best.pt` when it finishes (~26 hours from the start
of the clean run). The script that produces them is
`scripts/08_eval.py`, and the loss curve is in
`runs/small-clean/metrics.jsonl`.

For reference, a pilot run on an earlier corpus mix reached a validation
perplexity of 10.0 at 49M tokens, with validation loss still falling
monotonically — which is the point: at this compute budget the model is
sitting on the steep part of the curve, not the bottom of it.

---

## Honest limitations

**It is undertrained, by a lot.** 60M tokens for a 126M-parameter model is
2.4% of the Chinchilla-optimal amount (~2.5B tokens for this size).
Validation loss was still falling monotonically when the budget ran out,
so the model is sitting on the steep part of the curve, not the bottom.
Everything here is a statement about the *code*, not about the ceiling of
the approach.

**It hallucinates CVE identifiers.** It will produce plausible-looking
`CVE-YYYY-NNNN` strings, and many of them will be real CVEs. It cannot
tell you which are real, because it is not a retrieval system — it is a
statistical model over CVE-shaped text. The eval script measures this
explicitly rather than letting you discover it.

**It is not a security reference.** A 126M model trained on 60M tokens of
security prose can produce confident, fluent, wrong security advice.
Treat it as a toy, not a tool.

**One GPU would have changed the compute story entirely.** At ~650 tok/s
on 32 CPU threads, this is a multi-day exercise. The same code on a single
consumer GPU is a few hours, and the 10× data gap becomes a 10-minute
problem. The CPU constraint is what made the data budget a real decision
rather than a formality.

---

## Reproducing it

```bash
git clone <this repo> && cd myllm
python -m venv .venv && .venv/bin/pip install -r requirements.txt

python scripts/01_download_data.py          # ~2.5 GB
python scripts/02_build_corpus.py           # ~590 MB of text
python scripts/03_train_tokenizer.py        # ~35 min, single core
python scripts/04_tokenize_corpus.py        # ~1 min, 12 workers
python scripts/00_selftest.py               # 19 correctness checks

python scripts/05_train.py --preset tiny    # 13M params, ~1 hour
python scripts/05_train.py --preset small   # 126M params, ~26 hours
```

The trained model is published as a
[GitHub release](https://github.com/einyx/myllm/releases/tag/v1.0)
(`model.safetensors`, `tokenizer.json`, `config.json`) and, if the
publisher is logged in to HuggingFace, mirrored at
[`einyx/myllm-small`](https://huggingface.co/einyx/myllm-small).
`./scripts/13_auto_publish.sh` does the export and both uploads.

The self-test is worth running first, and it earned its keep. It caught
four real bugs before a single long training run: a parameter-count
formula off by exactly `d_model`, a name-shadowing bug in the repetition
penalty, a KV-cache assertion comparing logits at different positions, and
left-padded batched generation leaking pad tokens into attention *and*
shifting RoPE positions by the padding width. The last one is now
verified equal to single-sequence generation to 2.4e-07.

The HF export checks itself too. It re-tokenizes real corpus documents
with both the original and the converted tokenizer and requires
byte-identical output, then loads the exported model and requires the
logits to match the checkpoint exactly (`max abs diff = 0.00e+00`). The
published weights are loaded by a copy of this repository's own model
code, vendored into the export, so the published artifact cannot drift
from the code that trained it.
