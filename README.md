# myllm

A language model built from scratch — tokenizer, transformer, training loop,
and sampler — trained on cybersecurity data, on a CPU, in this repo.

No `transformers`, no `tokenizers`, no `trl`. Just PyTorch and NumPy.
The point is not to beat Llama. The point is that you can read every line
that goes from raw text to generated text.

**Read the write-up:** https://einyx.github.io/myllm/ (also in
[`WRITEUP.md`](WRITEUP.md)).

**Model:** https://github.com/einyx/myllm/releases/tag/v1.0 (also
[`MODEL_CARD.md`](MODEL_CARD.md))

---

## 1. Quick start

```bash
cd ~/myllm
source .venv/bin/activate

# 1. download the corpora (~2.5 GB)
python scripts/01_download_data.py

# 2. turn them into one mixed corpus (~600 MB of text)
python scripts/02_build_corpus.py

# 3. learn a byte-level BPE vocabulary
python scripts/03_train_tokenizer.py

# 4. tokenize into flat binary files
python scripts/04_tokenize_corpus.py

# 5. check how fast this machine is, then size your run
python scripts/07_benchmark.py --preset small

# 6. train
python scripts/05_train.py --preset tiny      # ~13M params, minutes
python scripts/05_train.py --preset small     # ~126M params, ~28h

# 7. watch it train
python scripts/09_plot.py --run runs/small

# 8. evaluate
python scripts/08_eval.py --ckpt runs/small/best.pt

# 9. talk to it
python scripts/06_generate.py --ckpt runs/small/best.pt --interactive

# 10. export it for HuggingFace (verifies the conversion before writing)
python scripts/10_export_hf.py --ckpt runs/small/best.pt \
    --tokenizer data/processed/tokenizer.json --out dist/myllm-small
```

### Using the exported model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

dist = "dist/myllm-small"          # or a downloaded release
tok = AutoTokenizer.from_pretrained(f"{dist}/tokenizer.json")
model = AutoModelForCausalLM.from_pretrained(dist, trust_remote_code=True)

ids = tok("### Instruction:\nWhat is a buffer overflow?\n\n### Response:\n",
          return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=120, temperature=0.8, top_p=0.95)
print(tok.decode(out[0][ids.shape[1]:]))
```

The export script does not trust itself: it re-tokenizes real corpus
documents with both tokenizers and requires byte-identical output, then
loads the exported model and requires the logits to match the original
checkpoint exactly (max abs difference `0.00e+00`). The published weights
are loaded by a *copy of this repository's own model code*, not a
re-implementation, so they cannot drift.

### Publishing

```bash
# waits for training to finish, then exports and publishes
./scripts/13_auto_publish.sh runs/small-clean/best.pt v1.0
```

Weights go to a **GitHub Release** (no setup required) and to
**HuggingFace** if you have run `hf auth login`.

---

## 2. What the model actually is

A **decoder-only transformer** — the "GPT" architecture. It reads a
sequence of tokens and, at every position, predicts the next one.

```
tokens:  [  t0  t1  t2  t3  ]
                ↓  embed + position
         ┌──────────────────────┐
         │  x = x + Attention(  │   "what should I look at?"
         │           RMSNorm(x)) │
         │  x = x + MLP(         │   "what should I think about it?"
         │           RMSNorm(x)) │
         └──────────────────────┘
              × 16 layers
                ↓  final RMSNorm
         logits → softmax → next token
```

Everything is in `myllm/model.py`, with the shapes and the reasoning in
the comments. The pieces that are worth understanding:

### RMSNorm instead of LayerNorm

```python
rms = sqrt(mean(x^2) + eps)
out = (x / rms) * g
```

LayerNorm subtracts the mean first; RMSNorm does not. One less operation,
same stability. Used by LLaMA and most modern models.

### RoPE (rotary position embeddings)

Transformers have no idea about order. RoPE encodes it by *rotating* the
query and key vectors by an angle proportional to the position.

The magic property: because rotation is linear, the dot product `q_i · k_j`
depends only on the **relative** distance `i - j` — which is exactly what
attention needs. It also extrapolates to longer sequences better than
learned absolute embeddings, and costs zero parameters.

### GQA (grouped-query attention)

`n_heads=12` query heads but only `n_kv_heads=4` key/value heads; each KV
head is shared by 3 query heads. This shrinks the KV cache (the thing you
must store for every token during generation) and the memory bandwidth of
generation, with very little quality loss.

### SwiGLU

```python
h = silu(gate_proj(x)) * up_proj(x)
out = down_proj(h)
```

Three matrices instead of two, so the hidden width is set to `8/3 × d_model`
to keep the parameter count comparable. The gate acts as a learned switch:
it can suppress any part of the up-projection it does not want.

### Pre-norm

The norm goes *before* each sublayer, not after. The residual stream `x`
stays a clean highway from the embedding to the output, which is what makes
deep stacks trainable. The original Transformer used post-norm and needed a
very careful warmup schedule because of it.

### Weight tying

The output projection (`lm_head`) *is* the input embedding (`tok_emb`). One
fewer matrix, and the embedding gets a second gradient signal: "which token
was predicted", not just "which token was here".

---

## 3. The tokenizer (`myllm/tokenizer.py`)

A **byte-level BPE**, written from scratch.

An LLM never sees letters — it sees integers. The tokenizer is the
translator, and it is a real architectural decision: the context window is
measured in tokens, the loss is per-token, and a bad tokenizer wastes
compute.

**Why byte-level?** Start the vocabulary at all 256 possible bytes. Then
every byte sequence is representable and there is never an UNK token. This
is why our tokenizer has no `[UNK]` and why it never fails on weird input.

**Why BPE on top of bytes?** Character-level models need ~5× more steps to
predict a word. BPE learns merges that group common byte sequences:
`" vulner"` + `"ability"` → one token.

**The training algorithm** is the interesting part. Naive BPE rescans the
whole corpus once per merge — ~10¹¹ operations for 32k merges. Instead we
maintain `pair_counts` and `pair_to_words` incrementally and only touch the
words that actually contain the pair being merged.

```
$ python scripts/03_train_tokenizer.py --sample-chars 60000000
[tokenizer] 1,204,331 unique pieces (612,882 kept, 591,449 dropped)
[tokenizer] merge 32,000/32,512 | 118,204 occurrences | b' vulner' + b'ability'
[tokenizer] done: vocab size = 32,767
```

Round-trip is exact (`encode(decode(x)) == x`) for any input, including
non-UTF-8 bytes. Compression on our corpus: **4.07 characters per token**.

---

## 4. The data

This is where most of the "what will the model become" decision lives.

| Source | Dataset | License | Shape |
|---|---|---|---|
| NIST publications | `ethanolivertroy/nist-cybersecurity-training` | CC0-1.0 | instruction + long prose |
| NVD Q&A | `rezaduty/cybersecurity-qa-v2` | Apache-2.0 | deep technical Q&A |
| CVE records | `CIRCL/vulnerability-scores` | CC-BY-4.0 | structured facts |

We emit four *document shapes*:

- **instruction** — `### Instruction:\n<q>\n\n### Response:\n<a>`
  Teaches the model to answer instead of just continuing.
- **prose** — raw NIST publication text. Teaches grammar and structure.
- **record** — a CVE with its score and affected products. Teaches precise
  factual statement.
- **continuation** — everything else. Keeps the model able to just write.

> **One source is deliberately excluded.** `11-47/Organized_PreTrain_Cyber_Security_640k`
> carries no license tag on the Hub, meaning no permission was granted to
> redistribute or derive from it. It is not used in the released corpus.
> Pass `--include-pretrain-640k` to `scripts/02_build_corpus.py` if you
> want it for your own experiments.

The mix is a **character budget per source** (`MIX` in
`scripts/02_build_corpus.py`). That is the knob to turn when you want a
different model. Right now:

```
nist 51.3% | nvd_qa 28.2% | circl 20.5%
```

Resulting corpus:

```
749,191 train documents, 7,567 validation documents, 594M characters
  by kind:   record 369,083 | instruction 274,791 | prose 105,317
  by source: circl 369,083 | nist 220,580 | nvd_qa 159,528
-> 137,627,881 training tokens (4.33 chars/token)
->   1,422,239 validation tokens
```

Two things we deliberately do:

- **Drop the `system` prompt.** It is the same boilerplate in every record.
  Training on it teaches the model to emit a constant.
- **Strip leading ISO timestamps** from the pretrain corpus.

### The validation set must match the training distribution

This one bit us, so it is worth spelling out.

Our corpus is written in **source order**: all the NIST documents, then all
the NVD Q&A, then all the CVE records, then the rest. The training loop
samples **random** windows from that stream, so training sees the correct
proportional mix of all four sources.

But the evaluator reads **sequential** windows from the start of
`val.bin`. And `val.bin` was, until we fixed it, also in source order. So
the first 82,000 tokens of the validation file were **100% NIST** — and
the "validation loss" was silently measuring one source out of four.

The symptom: train loss 4.15, val loss 5.95, and a gap that would not go
away. The per-document breakdown in `scripts/08_eval.py` showed why:

```
  kind           source            docs     loss        ppl
  continuation   pretrain_640k      120    4.739     114.33
  instruction    nist               120    5.649     283.91
  instruction    nvd_qa             120    3.662      38.92
  prose          nist               120    6.113     451.73
  record         circl              120    4.141      62.86
```

The fix is one line in the corpus builder: **shuffle the validation
documents before writing them**, so any prefix of `val.bin` is a
representative sample. Always check that your validation set has the same
shape as your training set — a validation number that measures a different
distribution is worse than no validation number, because it looks
authoritative.

### Packing

All documents are concatenated into one long stream of token ids, separated
by `<|endoftext|>`, and random fixed-length windows are cut out of that
stream. No padding: **every token we compute a loss on is a real token.**
That is roughly a 3-4× compute saving versus padding to fixed length.

---

## 5. Training (`myllm/train.py`)

Per optimizer step:

1. Sample `batch_size × grad_accum` windows.
2. Forward → logits → cross-entropy.
3. Backward, accumulating gradients over the micro-batches.
4. Clip gradient norm to 1.0.
5. AdamW step with a warmup + cosine learning rate.
6. Log, checkpoint, evaluate.

**Gradient accumulation** is why we can use a big effective batch on a
machine with no GPU: 8 micro-batches of 2×512 tokens accumulate into one
optimizer step of 8192 tokens, and it is mathematically identical to one
big batch.

**Weight decay** applies to matrices only, never to norm gains. Decaying a
norm gain toward zero would slow the model down for no reason.

**AdamW** rather than Adam + L2: AdamW applies decay directly to the
weights instead of adding it to the gradient, so the effective decay does
not get rescaled by the adaptive learning rate.

### Reading the loss

```
step   1,250/22,900 | loss 3.8421 | ppl  46.54 | lr 6.00e-04 | gnorm  1.87 |  312.4 tok/s
```

- **loss** is cross-entropy in nats. `ppl = exp(loss)`.
- **gnorm** is the gradient norm. If it pins at the clip value (1.0) for a
  long time, the learning rate is probably too high.
- **tok/s** is your actual budget. Multiply by seconds to plan the run.

---

## 6. Layout

```
myllm/
├── myllm/
│   ├── config.py      model + training configs, presets
│   ├── tokenizer.py   byte-level BPE, from scratch
│   ├── model.py       transformer: RoPE, RMSNorm, GQA, SwiGLU
│   ├── data.py        corpus → token stream → batches
│   ├── train.py       the training loop
│   ├── generate.py    sampling: temperature, top-k, top-p, KV cache
│   └── utils.py       seeding, logging, timing
├── scripts/
│   ├── 00_selftest.py       correctness checks (run these first!)
│   ├── 01_download_data.py
│   ├── 02_build_corpus.py
│   ├── 03_train_tokenizer.py
│   ├── 04_tokenize_corpus.py
│   ├── 05_train.py
│   ├── 06_generate.py       sample / chat
│   ├── 07_benchmark.py      throughput measurement
│   ├── 08_eval.py           perplexity broken down by document type
│   ├── 09_plot.py           ASCII loss curve + health check
│   ├── 10_export_hf.py      export to HuggingFace format (+ verifies it)
│   └── run_all.sh           the whole pipeline end to end
├── data/
│   ├── raw/           downloaded datasets
│   └── processed/     corpus.jsonl, tokenizer.json, train.bin, val.bin
└── runs/
    └── <preset>/      checkpoints + metrics.jsonl + config.json
```

---

## 7. Experiments worth trying

Each of these changes one thing and is a real lesson:

1. **Ablate the data mix.** Set `MIX` to 100% NIST and retrain. Does it
   still answer questions? Does it still know CVE IDs?
2. **Change the vocabulary size** to 8k / 16k / 32k / 65k and watch the
   compression ratio and the token/s tradeoff.
3. **Swap RoPE for learned absolute positions.** Which extrapolates better
   past the training context length?
4. **Remove GQA** (set `n_kv_heads = n_heads`) and measure the generation
   speed difference.
5. **Read the attention maps.** The `math` attention implementation exists
   so you can print the attention matrix and literally look at which
   tokens the model attends to.
6. **Sweep the learning rate.** 1e-4 vs 3e-4 vs 1e-3. The instability at
   high LR is where the warmup schedule earns its keep.
7. **Change the model depth, not the width.** 8 layers of 768 vs 16 layers
   of 768 vs 32 layers of 384. Same parameter count, different model.

---

## 8. Hardware notes

The machine: Intel i9-13900K (8 P-cores + 16 E-cores), 32 threads,
125 GB RAM, **no GPU**.

### Thread count matters more than anything else

This CPU is hybrid: the 16 E-cores are much slower at FP32 GEMM and do
not hide P-core latency. Measured on the real 126M model (batch 16×512):

| threads | tok/s | |
|---|---|---|
| 8 | 363 | P-cores only, no HT |
| **16** | **467** | **P-cores + hyperthreads — the sweet spot** |
| 24 | 269 | torch's default |
| 32 | 256 | all logical CPUs |

Oversubscribing the E-cores makes training **~1.8× slower**. `TrainConfig`
therefore defaults to `num_threads=16`, not torch's default of 24.

### `bfloat16` is 93x SLOWER here — do not use it

This is not a guess; it is measured on the real 126M model (batch 16×512):

| threads | fp32 tok/s | bf16 tok/s |
|---|---|---|
| 16 | **467** | **5** |
| 32 | 256 | 5 |

The reason: this CPU (i9-13900K) has AVX2 but **no AVX-512-BF16**. Every
bfloat16 matrix multiply has to be emulated, and the conversion overhead
dominates the actual math by two orders of magnitude.

The general lesson, which does not apply to GPUs: reduced precision is
only fast if the hardware has native instructions for it. On a GPU, bf16
is 2-4x faster than fp32. On this CPU it is 90x slower. `TrainConfig.dtype`
therefore defaults to `"float32"`, and `scripts/07_benchmark.py --dtypes
bfloat16` lets you re-measure it on whatever machine you move to.

### Consequences

- **~584 tok/s** for the 126M model (measured on the real run). A 60M-token
  run is ~28 hours; 120M tokens is ~3 days. Plan around that.
- `float32` beats `bfloat16` here by **93x** (see the table above).
- Checkpoints every 250 steps mean there is always a usable model on disk
  and you can watch the loss curve while the run continues.
- Memory: ~15 GB resident for the 126M model (weights + gradients +
  AdamW states + activations). No problem with 125 GB.

### The compute reality

A 126M model is Chinchilla-optimal at ~2.5B tokens. We can afford ~60M.
That is 2.4% of the compute-optimal amount, so the model will be
undertrained by the standards of the literature — it will produce
grammatical, domain-fluent text that is often subtly wrong.

Two honest options if you want to spend the same wall-clock time better:

```
# what we do: bigger model, less data
python scripts/05_train.py --preset small        # 126M, 60M tokens, ~28h

# alternative: smaller model, more data (often better at fixed FLOPs)
python scripts/05_train.py --preset medium       # 209M, 200M tokens, ~5d
# or edit the 'small' preset in config.py to d_model=640, n_layers=12
#     -> 62M params, and double max_tokens for the same wall time
```

### Why 126M and not 62M (given we are so far below the optimum)

At a fixed FLOP budget, the loss keeps dropping as you move from
"model-heavy" to "data-heavy" allocations -- but only up to the
Chinchilla optimum of ~20 tokens per parameter. We are at **0.48** tokens
per parameter (60M tokens for 126M params), and the alternative
allocation (62M params, 120M tokens) would be 1.94.

Both are far below 20, so on pure loss grounds the smaller model would
win slightly. We keep 126M anyway because:

- a 62M model trained on the same wall-clock budget is still
  undertrained, and the extra depth of the 126M model is what makes the
  outputs read like prose rather than like a template;
- the model you can point at and say "this is mine, and it is 126 million
  parameters" is the one that teaches you the most.

If you want to try the other allocation, the `small` preset is three
numbers: `d_model`, `n_layers`, and `max_tokens`.

---

## 9. Results so far

### `tiny` preset — 12.8M params, 2.5M tokens, 1.06 hours

```
step     20/305 | loss 8.5763 | ppl  5304.44 | lr 6.67e-04 | gnorm 1.55 |  695 tok/s
step    100/305 | loss 5.8061 | ppl   332.31 | lr 8.67e-04 | gnorm 0.56 |  676 tok/s
step    200/305 | loss 5.0394 | ppl   154.38 | lr 3.92e-04 | gnorm 0.57 |  655 tok/s
step    300/305 | loss 4.1457 | ppl    63.16 | lr 1.01e-04 | gnorm 0.52 |  656 tok/s
Done. final val loss 5.9539
```

Loss fell from `ln(32768) = 10.397` (a perfectly uniform distribution over
the vocabulary — i.e. a model that has learned nothing) down to **4.15**
(train) / **4.5** (held-out, averaged over all four sources).

Per-document-type perplexity of that checkpoint:

```
kind           source            docs     loss        ppl
continuation   pretrain_640k      120    4.739     114.33
instruction    nist               120    5.649     283.91
instruction    nvd_qa             120    3.662      38.92
prose          nist               120    6.113     451.73
record         circl              120    4.141      62.86
```

Which is a nice summary of the whole project in one table: after 2.5M
tokens the model is best at imitating Q&A answers (the most templated
shape), worst at free-form prose, and middling on structured records.

### `small` preset — 125.9M params, 60M tokens, ~28 hours

Running. Watch it with:

```
python scripts/09_plot.py --run runs/small
```

---

## 10. Honest expectations

A 126M model trained for ~120M tokens on a domain-specific corpus will:

- produce grammatically coherent security prose
- know the *shape* of CVE descriptions and QA pairs
- hallucinate CVE IDs (it will invent plausible-looking ones — they will
  often be real, but it cannot tell you which are real)
- be noticeably worse than a general-purpose model at non-security topics

That is the expected outcome of ~10¹⁷ FLOPs on one CPU. The interesting
part is that every number in that sentence is now something you can point
at in the code and change.
