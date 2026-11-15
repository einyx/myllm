# Model Card: myllm-small

A decoder-only transformer language model, trained from scratch on
security-domain text. Specialized for cybersecurity Q&A and CVE records.

**This is a learning artifact, not a security tool.** See
[Limitations](#limitations) — it will invent CVE identifiers, and it is
not a reference for anything.

---

## Model Details

| | |
|---|---|
| Architecture | Decoder-only transformer (GPT-style) |
| Parameters | 125.9M |
| `d_model` | 768 |
| Layers | 16 |
| Attention heads | 12 query / 4 key-value (grouped-query attention) |
| Feed-forward | SwiGLU, hidden 2048 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (base 10000) |
| Embeddings | Tied input/output |
| Context length | 512 tokens |
| Tokenizer | Byte-level BPE, 32,767 tokens, 4.33 chars/token |
| Precision | float32 |
| Framework | PyTorch only — no `transformers` at training time |

Implemented in this repository from scratch: the tokenizer, the model, the
training loop, and the sampler. See
[github.com/einyx/myllm](https://github.com/einyx/myllm).

---

## Training Data

Three public cybersecurity corpora, mixed by character budget and packed
into a single token stream.

| Source | Dataset | License | Characters | Share |
|---|---|---|---|---|
| NIST publications | `ethanolivertroy/nist-cybersecurity-training` | CC0-1.0 | 308M | 51.3% |
| CVE Q&A | `rezaduty/cybersecurity-qa-v2` | Apache-2.0 | 169M | 28.2% |
| CVE records | `CIRCL/vulnerability-scores` | CC-BY-4.0 | 123M | 20.5% |

**749,191 training documents → 137,627,881 training tokens**
(1,422,239 held-out validation tokens).

### Attribution

The CC-BY-4.0 source (`CIRCL/vulnerability-scores`) requires attribution:

> CIRCL — Vulnerability Scores. Dataset available through Hugging Face
> (`CIRCL/vulnerability-scores`), licensed CC-BY-4.0. Contains vulnerability
> data derived from the CVE Program's cvelistV5 repository.

The NIST corpus is US Government work released under CC0-1.0. The Q&A
corpus is Apache-2.0 and is itself derived from the NIST National
Vulnerability Database.

### A corpus that is deliberately excluded

`11-47/Organized_PreTrain_Cyber_Security_640k` carries **no license tag**
on the HuggingFace Hub, meaning no permission was granted to redistribute
or derive from it. It was used during development (20% of the corpus) but
is **not** in this model. The tokenizer was also retrained without it, so
no part of the released artifact derives from an unlicensed source.

### Document shapes

The corpus mixes four shapes, because a model trained on one narrow kind
of text becomes a model that only does that:

- **instruction** — `### Instruction:\n<q>\n\n### Response:\n<a>`. Teaches
  the model to answer rather than continue text indefinitely.
- **prose** — raw NIST publication text. Teaches grammar and structure.
- **record** — a CVE with its CVSS score and affected products. Teaches
  precise factual statement.
- **continuation** — everything else.

---

## Training Procedure

- Optimizer: AdamW (β = 0.9/0.95, ε = 1e-8), weight decay 0.1 on
  matrices only
- Learning rate: 6e-4 peak, 500-step linear warmup, cosine decay to 0.1×
- Gradient clipping: 1.0
- Batch: 16 sequences × 512 tokens × 2 gradient accumulation = 16,384
  tokens per optimizer step
- Total: 3,662 steps = 60M tokens (0.44 epochs over the corpus)
- Hardware: Intel i9-13900K, 32 threads, no GPU, float32, ~645 tok/s
- Wall clock: ~26 hours

---

## Evaluation

Measured with `scripts/08_eval.py` on the **step-250 checkpoint** (7% of
training, 4.1M tokens). These will be refreshed from the final
checkpoint when training completes.

### Held-out perplexity

| document kind | source | loss | perplexity |
|---|---|---|---|
| instruction | nvd_qa | 2.477 | **11.9** |
| record | circl | 3.813 | 45.3 |
| instruction | nist | 5.201 | 181.5 |
| prose | nist | 5.625 | 277.3 |

Overall validation perplexity: **90.9** (a uniformly random model over
this vocabulary would be 32,768).

The shape of that table is the honest summary of the model: it is
already very good at the most templated shape in the corpus — answering
a question in the `### Instruction / ### Response` form — and weakest at
free-form prose, which is the hardest thing to model and the first thing
that runs out of data.

### CVE identifier hallucination

This is the number that matters most, so it was measured rather than
asserted. Sampling 32 completions from held-out prompts and extracting
every CVE-shaped string:

| | |
|---|---|
| CVE identifiers generated | 10 |
| present in the training corpus | 7 (**70%**) |
| **fabricated** | **3 (30%)** |

Examples of the fabricated ones: `CVE-2014-601013` (six digits — a
malformed identifier), `CVE-2025-4335`.

So: **roughly one in three CVE identifiers this model produces is
invented**, and the invented ones are *more* likely to be wrong in
subtle ways (impossible years, malformed digits) than the real ones are
to be right. The model has learned the statistical shape of a CVE
identifier perfectly and the underlying facts not at all.

**Never use its CVE output as a reference.** Verify against NVD, MITRE
CVE, or your vendor's advisory.

### Generation speed

32.6 tokens/second on a CPU with the KV cache, batch size 1, at
`d_model=768` / 16 layers. On a GPU this would be orders of magnitude
faster.

### Sample output

From the step-250 checkpoint, prompt `### Instruction:\nDescribe
CVE-2021-44228.\n\n### Response:\n`:

> Detecting exploitation of CVE-2026-11509 is classified as CWE-63 — a
> weakness weakness weakness. This classification helps defenders
> understand the root cause and apply appropriate countermeasures
> targeting CWE-310 patterns are particularly effective.

The formatting, the CWE references, the confidence, and the CVE id are
all *structurally* right. The id is invented, the CWE pairing is wrong,
and the sentence is not English. That is exactly what a 126M model at 7%
of training looks like — and it is why the warning above is the first
thing in this card rather than the last.

---

## Intended Use

- Educational: understanding how decoder-only transformers are built and
  trained, end to end
- Demonstrating a domain-specialised small language model
- A base for experimentation with data mixes, architectures, and
  training schedules

---

## Limitations

**It hallucinates CVE identifiers.** The model produces
`CVE-YYYY-NNNN`-shaped strings, and many will correspond to real
vulnerabilities — but it has no way to tell you which ones are real. It
is a statistical model over CVE-shaped text, not a database. Measured:
**30% of the CVE identifiers it generates are fabricated.** **Do not use
its CVE output for anything.** Verify any CVE identifier against
NVD, MITRE, or your vendor's advisory.

**It is undertrained.** 60M tokens for a 126M-parameter model is roughly
2.4% of the compute-optimal amount for that size. Validation loss was
still falling monotonically when the budget ran out. The model sits on
the steep part of the learning curve, not the bottom of it.

**It can produce fluent, confident, wrong security advice.** This is the
failure mode that matters most for a security-domain model. Treat every
output as a generated artefact, not as guidance.

**Its knowledge is narrow by construction.** It knows security text and
little else. It is not a general-purpose model and will be poor at
anything outside its domain.

**No safety alignment.** This is a base model. It will follow the
instruction format because that format is in its training data, not
because it has been trained to be helpful or harmless.

---

## Ethical Considerations

The training data is entirely defensive-security material: public CVE
records, NIST standards publications, and technical Q&A. No credential
dumps, no malware samples, no exploit code.

That said, a model that can generate plausible-looking vulnerability
descriptions could be used to make synthetic advisory text more
convincing. The realistic mitigation is not to publish it, but to publish
it *with this warning attached* — which is why the limitations section
above leads with the CVE hallucination behaviour rather than burying it.

The model is small enough (126M) and undertrained enough that it is not
useful for convincing social engineering or scaled content generation.

---

## How to Use

### Download

**GitHub Release** (primary):
https://github.com/einyx/myllm/releases/tag/v1.0

| file | what |
|---|---|
| `model.safetensors` | the weights (503 MB) |
| `tokenizer.json` | the tokenizer, in HuggingFace `tokenizers` format |
| `config.json` | architecture + training metadata |

**HuggingFace** (mirror, if published):
https://huggingface.co/einyx/myllm-small

### Load it

```python
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

dist = Path("path/to/release")

tok = AutoTokenizer.from_pretrained(str(dist / "tokenizer.json"))
model = AutoModelForCausalLM.from_pretrained(
    str(dist), trust_remote_code=True
)

prompt = ("### Instruction:\nWhat is a buffer overflow and how do I "
          "prevent it?\n\n### Response:\n")
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=120, temperature=0.8, top_p=0.95)
print(tok.decode(out[0][ids.shape[1]:]))
```

The release contains `config.json`, `model.safetensors` and
`tokenizer.json`; `from_pretrained` needs the directory (it reads the
config from `config.json` and the weights from `model.safetensors`).

On HuggingFace the same directory layout works via
`AutoModelForCausalLM.from_pretrained("einyx/myllm-small",
trust_remote_code=True)`.

Sampling notes: temperature 0.8 and top-p 0.95 work well. A repetition
penalty around 1.15 helps small models. The context window is 512 tokens
(total prompt + completion).

The published weights are loaded by a **copy of the training
repository's own model code** (vendored into the export as
`myllm_vendor_*.py`), not by a re-implementation, so the published
artifact cannot drift from the code that trained it.

---

## Citation

```bibtex
@software{myllm2026,
  author  = {Alessio},
  title   = {myllm: a language model built from scratch},
  year    = {2026},
  url     = {https://github.com/einyx/myllm},
  note    = {Byte-level BPE tokenizer, transformer and training loop
             implemented from scratch in PyTorch.}
}
```
