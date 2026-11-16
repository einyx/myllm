# Building a language model from scratch, on a CPU, in 4,400 lines

*A learning project. No `transformers`, no `tokenizers`, no `trl`. Just
PyTorch, NumPy, and a lot of reading about why things are the way they
are.*

---

## The premise

I wanted to understand what actually happens between "here is a text file" and "here is a token the model decided to emit". Not the API calls. The arithmetic.

So I built a decoder-only transformer from nothing. Trained its own byte level BPE tokenizer. Wrote its own training loop, sampler, and data pipeline. Trained it on security domain text, on a CPU, and wrote down what broke along the way.

The architecture went exactly as the textbook says it should, which was reassuring and completely uninteresting. The real lessons were not about transformers at all.

Three of the four things worth writing down are about the machine. A CPU whose fast cores and slow cores are not interchangeable. An instruction set that does not exist on it. A validation split that was quietly measuring the wrong dataset. The fourth is an off by one in my own code that killed the run sixty steps in and would have shipped a broken model if the corpus had been slightly different.

They share a shape, and the shape is the actual finding:

> Every one of them was a mismatch between two components that were each individually correct. Not one was a bug inside a component. All of them were bugs in the agreement between components, which is exactly where nobody looks.

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

**The model.** 125.9M parameters. `d_model=768`, 16 layers, 12 attention heads sharing 4 key/value heads (GQA), SwiGLU feed forward with `hidden = 8/3 * 768`, RoPE positions, RMSNorm, pre-norm residual blocks, tied input and output embeddings. Context length 512.

**The tokenizer.** Byte level BPE with a 32K vocabulary, trained with an incremental merge algorithm. Instead of rescanning the corpus once per merge, which is about 10^11 operations for 32K merges, it keeps `pair_counts` and `pair_to_words` in memory and only touches the words that actually contain the pair being merged. Round trip is exact for any input, including invalid UTF-8.

**The data.** 749,191 documents from three public security corpora, concatenated into one packed token stream and cut into random windows. No padding, so every token we compute a loss on is a real token.

---

## Finding 1: your CPU has fast cores and slow cores, and PyTorch doesn't care

The first real surprise had nothing to do with transformers.

This machine is an i9-13900K. Eight P cores for performance, sixteen E cores for efficiency. PyTorch's default thread count is the number of physical cores, so 24. That means it happily schedules matrix multiply work onto the E cores. Those cores have no AVX-512 and run at roughly half the clock. They also do not hide P core latency, so they do not even free up the P cores.

Measured on the real 126M model, batch 16 by 512:

| threads | tok/s | what's running |
|---|---|---|
| 8 | 363 | P cores, one thread each |
| **16** | **467** | P cores plus hyperthreads |
| 24 | 269 | PyTorch's default, E cores included |
| 32 | 256 | all logical CPUs |

Going from the default to the right number was a 1.8x speedup. That is roughly fourteen extra hours of training. The fix is one line:

```python
torch.set_num_threads(16)
```

The general lesson: on a hybrid CPU, thread count is not a rounding question. Measure it instead of assuming the default is sensible.

---

## Finding 2: bfloat16 was 93x slower than float32

I assumed `bfloat16` would be a modest win. It is 2 to 4 times faster than fp32 on a GPU, so it is the reflexive answer when someone says "make it faster".

Measured on the same model:

| threads | float32 | bfloat16 |
|---|---|---|
| 16 | **467 tok/s** | **5 tok/s** |
| 32 | 256 tok/s | 5 tok/s |

Five tokens per second. A 93x regression.

The reason is the same hardware fact as finding 1. This CPU has AVX2 but no AVX-512-BF16. There is no native bf16 multiply, so every bf16 matrix multiply gets emulated, and the conversion overhead dominates the actual arithmetic by two orders of magnitude.

This one is worth keeping in mind because it inverts a GPU intuition that carries very far:

> Reduced precision is only fast where the hardware has native instructions for it. "bf16 is faster" is not a property of bf16. It is a property of bf16 plus your CPU.

The same trap has a milder form on GPUs. bf16 on a consumer card without good bf16 tensor core throughput can be slower than fp16. But 93x is the extreme version of the lesson.

---

## Finding 3: my validation loss was measuring one dataset and calling it "validation"

This is the one that mattered most, and it was a bug in my own setup rather than in the model.

The symptom: training loss 4.15, validation loss 5.95, and a gap that refused to close. That gap says overfitting. The reflex is to add regularization, drop the learning rate, or get more data. All of those would have been the wrong fix.

The actual cause was an accident of ordering.

1. The corpus builder writes documents in source order. All the NIST documents, then all the NVD Q&A, then all the CVE records.
2. The training loop samples random windows from that stream, so training sees the correct proportional mix of all three sources.
3. The evaluator reads sequential windows starting from the beginning of `val.bin`.
4. And `val.bin` was, like the training file, in source order.

The first 82,000 tokens of the validation file were 100 percent NIST. Every validation number I printed was a measurement of NIST alone.

The giveaway is in the per document type breakdown:

```
kind           source            docs     loss        ppl
continuation   pretrain_640k      120    4.739     114.33
instruction    nist               120    5.649     283.91
instruction    nvd_qa             120    3.662      38.92
prose          nist               120    6.113     451.73
record         circl              120    4.141      62.86
```

NIST instruction and NIST prose are the two hardest kinds, at 5.65 and 6.11. My validation number was an average of exactly those two, which is why it sat so far above the training loss.

The fix is one line in the corpus builder. Shuffle the validation split before writing it, so any prefix of `val.bin` is a representative sample.

```python
random.Random(20260808).shuffle(val_docs)
```

Validation perplexity went from 411 to 94 for the same checkpoint. The training loss never moved.

> A validation number that measures a different distribution than your training set is worse than no validation number at all, because it looks authoritative. It will send you optimising the wrong thing with real confidence.

The general form of this bug is worth naming. Anywhere you slice a dataset and only look at one end of it, you inherit the order in which it was written. Evaluation code is exactly the place people forget to check.

---

## Finding 4: the vocabulary was one too small, and it crashed 60 steps in

This one is a plain off by one in my own code, and it is worth writing down because of how it failed.

The tokenizer's special token `<|endoftext|>` is deliberately not stored in the token vocabulary. A learned merge could in principle produce the same bytes and collide with it. So the vocabulary has two parts, the byte and merge tokens and the special tokens, kept separately.

`__len__` returned the size of the first part only. With 256 byte tokens and 32,511 merges that is 32,767. But the highest id the tokenizer can actually emit is 32,767, so the real id space is 0..32,767, which is 32,768 ids.

The training script sized the embedding matrix from `len(tokenizer)`. The embedding had 32,767 rows and accepted ids 0..32,766, while the corpus contained id 32,767.

Training died 60 steps in:

```
File "myllm/model.py", line 530, in forward
    x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)
IndexError: index out of range in self
```

Three things about that failure are worth dwelling on.

The message tells you nothing. "index out of range in self" means the id was bigger than the weight matrix. It never mentions ids, vocabularies, or tokenizers. The actual cause was four files away, in a class whose job is to count things.

It was data dependent. The rarest token in the vocabulary occurs once every few million tokens. A smaller corpus, a luckier sample, or a slightly different vocabulary would have trained to completion and published a model that crashed on the first rare input. I only caught it because the crash happened to land inside this run.

The adaptive code was the vector. I had deliberately added `vocab_size = len(tokenizer)` to the training script so the model would automatically match whatever tokenizer it was given. That line, written to make things robust, carried the wrong number into the model. Being clever about coupling is only safe when the thing you couple to is itself correct.

Three changes, in increasing order of how much I like them:

1. `__len__` returns the true id space size, which is the maximum emittable id plus one.
2. The trainer asserts `max_token_id_in_corpus < vocab_size` before training starts, and fails with a message that says what to do.
3. The self test checks that the id space is contiguous and that an embedding sized `len(tokenizer)` accepts every id the tokenizer can emit.

---

## The data, and why the mix is a decision

A language model becomes the distribution you give it. Four document shapes went into the corpus.

Instruction documents, in the form `### Instruction:` then a question then `### Response:` then an answer. These teach the model to answer rather than just continue text forever. Without them a base model never responds.

Prose. Raw technical writing. This teaches grammar and long range structure. Instruction documents alone make a model stilted.

Records. A CVE with its severity and affected products. This teaches precise factual statement.

Continuation. Everything else. Keeps the model able to just write.

Two deliberate choices that are easy to get wrong.

The system prompt is dropped. In the NIST dataset every record begins with the same 70 token system prompt. Training on it teaches the model to emit a constant string.

The mix is a character budget per source, not a document count. A budget makes the knob explicit and keeps the proportions stable when document lengths change.

The final released corpus excludes one source. `11-47/Organized_PreTrain_Cyber_Security_640k` carries no license tag on the Hub, which means no permission was granted to derive from it, so it is not in the release. The remaining three are CC0-1.0, Apache-2.0, and CC-BY-4.0.

---

## Results

Final numbers, from the released checkpoint at step 3,662 of 3,662, fully annealed.

Held out perplexity, broken down by what the model was asked to do:

| document kind | source | loss | perplexity |
|---|---|---|---|
| instruction | nvd_qa | 0.888 | **2.43** |
| record | circl | 1.991 | 7.33 |
| instruction | nist | 2.554 | 12.86 |
| prose | nist | 2.928 | 18.69 |

Overall validation perplexity 9.90, from a starting point of 32,768 for a uniformly random model. The validation curve fell monotonically for the whole run, 4.44 at step 250 down to 2.25 at the end, with no sign of overfitting at any point.

The model is extremely good at the most templated shape in the corpus. Answer a question in the instruction and response form, and it reaches 2.4 perplexity. It is weakest at free prose, which is both the hardest thing to model and the first thing to run out of data.

And here is the number I would put on the front of the model card, because it decides whether this model is safe to hand to someone:

| | |
|---|---|
| CVE identifiers generated (60 samples) | 27 |
| present in the training corpus | 23 (85%) |
| **fabricated** | **4 (15%)** |

Roughly one in seven CVE identifiers this model produces is invented. And the part that matters more than the rate: at 7 percent of training the model produced obviously malformed identifiers. Six digits, impossible years. At full training it produced `CVE-2021-41774`, which is one digit away from the real `CVE-2021-41773` that appeared in the prompt.

The failure mode got subtler as the model got better. A malformed identifier is caught by anyone who glances at it. A nearly correct one passes casual review, because the only thing wrong with it is that it does not exist. For a model whose job is producing security reference material, that is the most dangerous direction the quality curve could move in.

Asked about a buffer overflow, the released model says:

> The following vulnerability has been resolved: Assessment: A.0x A.0:•19, 2018-1a through 2.0.1.2.1, and earlier allows remote attackers to execute arbitrary code execution... (CVSS 4.0 to 8.9) severity with a CVSS score of 4.1. Reference: https://github.com

The formatting is right. The CVSS vocabulary is right. The confidence is right. The version ranges are invented, the CVSS pairing is nonsense, and the reference URL is truncated mid string. That is what a 126M model trained on 60M tokens looks like. Fluent, confident, and wrong in the details, in a way that reads as authoritative.

---

## Honest limitations

**It is undertrained, by a lot.** 60M tokens for a 126M parameter model is 2.4 percent of the Chinchilla optimal amount for this size. Validation loss was still falling when the budget ran out, so the model sits on the steep part of the curve, not the bottom. Everything here is a statement about the code, not about the ceiling of the approach.

**It hallucinates CVE identifiers.** It will produce plausible looking `CVE-YYYY-NNNN` strings, many of them real, and it cannot tell you which. It is a statistical model over CVE shaped text, not a database.

**It is not a security reference.** A 126M model trained on 60M tokens of security prose can produce confident, fluent, wrong security advice. Treat it as a toy, not a tool.

**One GPU would have changed the compute story entirely.** At around 650 tok/s on 32 CPU threads this is a multi day exercise. The same code on a single consumer GPU is a few hours, and the data gap becomes a ten minute problem. The CPU constraint is what made the data budget a real decision instead of a formality.

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

The self test is worth running first, and it earned its keep. It caught four real bugs before a single long training run. A parameter count formula off by exactly `d_model`. A name shadowing bug in the repetition penalty. A KV cache assertion comparing logits at different positions. And left padded batched generation leaking pad tokens into attention while also shifting RoPE positions by the padding width. The last one is now verified equal to single sequence generation to 2.4e-07.

The HF export checks itself too. It re-tokenizes real corpus documents with both the original and the converted tokenizer and requires byte identical output, then loads the exported model and requires the logits to match the checkpoint exactly. The published weights are loaded by a copy of this repository's own model code, vendored into the export, so the published artifact cannot drift from the code that trained it.

The trained model is published as a
[GitHub release](https://github.com/einyx/myllm/releases/tag/v1.0)
with `model.safetensors`, `tokenizer.json`, and `config.json`. `./scripts/13_auto_publish.sh` does the export and the upload.
