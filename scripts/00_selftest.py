#!/usr/bin/env python3
"""
Self-tests. Run these before you trust a multi-day training run.

A transformer has a lot of ways to be subtly wrong, and most of them
produce a model that trains to a plausible-looking loss curve while
being broken. These tests catch the classic ones:

  1. The two attention implementations must agree. The `math` version is
     the readable reference; `sdpa` is the fast path. If they disagree,
     one of them is wrong.
  2. Causality: changing token t must not change the logits at
     positions < t. If it does, the model can "see the future" and the
     loss numbers are meaningless.
  3. Shapes and dtypes line up end to end.
  4. The gradient actually flows to every parameter.
  5. The KV cache produces the same output as a full forward pass.
  6. The tokenizer round-trips exactly.

Run:  python scripts/00_selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myllm.config import ModelConfig, get_preset  # noqa: E402
from myllm.model import Transformer  # noqa: E402
from myllm.tokenizer import BPETokenizer  # noqa: E402

PASS, FAIL = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL}  {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    print("\n[1] attention implementations agree")
    # ------------------------------------------------------------------
    cfg = ModelConfig(
        name="test", vocab_size=512, d_model=128, n_layers=2,
        n_heads=4, n_kv_heads=2, max_seq_len=64,
    )
    model = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))

    model.set_attn_impl("math")
    with torch.no_grad():
        out_math = model(x).logits
    model.set_attn_impl("sdpa")
    with torch.no_grad():
        out_sdpa = model(x).logits

    diff = (out_math - out_sdpa).abs().max().item()
    check("math vs sdpa logits match", diff < 1e-4, f"max abs diff = {diff:.2e}")

    # ------------------------------------------------------------------
    print("\n[2] causality: no information flows backwards")
    # ------------------------------------------------------------------
    x2 = x.clone()
    x2[0, 10] = (x2[0, 10] + 137) % cfg.vocab_size  # change a LATE token
    with torch.no_grad():
        out2 = model(x2).logits

    early = (out_math[0, :10] - out2[0, :10]).abs().max().item()
    late = (out_math[0, 10:] - out2[0, 10:]).abs().max().item()
    check("logits before the changed token are identical", early < 1e-6,
          f"max diff = {early:.2e}")
    check("logits after the changed token DID change", late > 1e-4,
          f"max diff = {late:.2e}")

    # ------------------------------------------------------------------
    print("\n[3] shapes and parameter count")
    # ------------------------------------------------------------------
    for preset in ("tiny", "small"):
        mcfg, _ = get_preset(preset)
        m = Transformer(mcfg)
        n_expected = mcfg.n_params()
        n_actual = m.num_params()
        check(f"{preset}: parameter count matches the formula",
              n_expected == n_actual,
              f"{n_actual:,} vs {n_expected:,}")

        xx = torch.randint(0, mcfg.vocab_size, (2, 8))
        yy = torch.randint(0, mcfg.vocab_size, (2, 8))
        o = m(xx, yy)
        check(f"{preset}: logits shape",
              tuple(o.logits.shape) == (2, 8, mcfg.vocab_size),
              str(tuple(o.logits.shape)))
        check(f"{preset}: loss is a scalar", o.loss.dim() == 0,
              f"loss={float(o.loss.detach()):.4f}")

    # ------------------------------------------------------------------
    print("\n[4] gradients reach every parameter")
    # ------------------------------------------------------------------
    mcfg, _ = get_preset("tiny")
    m = Transformer(mcfg)
    xx = torch.randint(0, mcfg.vocab_size, (2, 16))
    yy = torch.randint(0, mcfg.vocab_size, (2, 16))
    m(xx, yy).loss.backward()

    missing = []
    for name, p in m.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif p.grad.abs().sum().item() == 0:
            missing.append(f"{name} (all zero)")
    check("every parameter received a nonzero gradient", not missing,
          ", ".join(missing[:4]) if missing else f"{len(list(m.parameters()))} params")

    tied = m.tok_emb.weight is m.lm_head.weight
    check("embeddings are tied to the LM head", tied)

    # ------------------------------------------------------------------
    print("\n[5] KV cache matches a full forward pass")
    # ------------------------------------------------------------------
    mcfg = ModelConfig(name="test", vocab_size=256, d_model=64, n_layers=2,
                        n_heads=4, n_kv_heads=2, max_seq_len=64)
    m = Transformer(mcfg).eval()
    with torch.no_grad():
        prompt = torch.randint(0, 256, (1, 6))
        nxt = torch.randint(0, 256, (1, 1))

        # (a) a plain forward over the prompt
        plain = m(prompt).logits[:, -1]

        # (b) the same prompt, but routed through a cache
        cache_a = m.new_cache()
        cached = m(prompt, cache=cache_a, start_pos=0).logits[:, -1]

        # (c) full sequence [prompt, nxt] in one forward
        full_seq = torch.cat([prompt, nxt], dim=1)
        ref = m(full_seq).logits[:, -1]

        # (d) [prompt, nxt] fed through the cache one step at a time
        cache_b = m.new_cache()
        m(prompt, cache=cache_b, start_pos=0)
        stepped = m(nxt, cache=cache_b, start_pos=prompt.shape[1]).logits[:, -1]

    diff = (plain - cached).abs().max().item()
    check("cached prompt == plain forward", diff < 1e-4,
          f"max abs diff = {diff:.2e}")

    diff = (ref - stepped).abs().max().item()
    check("step-by-step cache == full forward", diff < 1e-4,
          f"max abs diff = {diff:.2e}")

    # ------------------------------------------------------------------
    print("\n[5b] batched generation with padding == single generation")
    # ------------------------------------------------------------------
    # Left-padding a batch of different-length prompts introduces two
    # ways to go wrong: the padded tokens leak into attention, and RoPE
    # positions get shifted by the padding width. This checks both.
    mcfg = ModelConfig(name="test", vocab_size=256, d_model=64, n_layers=2,
                        n_heads=4, n_kv_heads=2, max_seq_len=64)
    m = Transformer(mcfg).eval()

    lens = [7, 3, 11]
    toks = [torch.randint(0, 256, (1, L)) for L in lens]
    maxlen = max(lens)

    with torch.no_grad():
        # reference: each sequence on its own
        refs = [m(t).logits[:, -1] for t in toks]

        # batched with left padding + mask + per-sequence positions
        batch = torch.full((3, maxlen), 0, dtype=torch.long)
        key_mask = torch.zeros((3, maxlen), dtype=torch.bool)
        positions = torch.zeros((3, maxlen), dtype=torch.long)
        for i, t in enumerate(toks):
            pad = maxlen - t.shape[1]
            batch[i, pad:] = t[0]
            key_mask[i, pad:] = True
            positions[i, pad:] = torch.arange(t.shape[1])
        batched = m(batch, attn_mask=key_mask, positions=positions).logits[:, -1]

    worst = 0.0
    for i, ref in enumerate(refs):
        worst = max(worst, (ref - batched[i:i + 1]).abs().max().item())
    check("padding + mask + positions reproduce single-sequence logits",
          worst < 1e-4, f"max abs diff = {worst:.2e}")

    # ------------------------------------------------------------------
    print("\n[6] tokenizer")
    # ------------------------------------------------------------------
    tok_path = ROOT / "data" / "processed" / "tokenizer.json"
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        check("tokenizer loaded", len(tok) > 256, f"vocab {len(tok):,}")

        # The id space must be contiguous and cover every id we can emit.
        # If len(tok) undercounts, an embedding built from it is too
        # small and training dies later with an opaque IndexError.
        max_id = max(max(tok.inv_vocab.keys()), max(tok.id_to_special.keys()))
        check("len(tokenizer) covers the whole id space",
              len(tok) > max_id,
              f"len={len(tok):,} max_id={max_id:,} "
              f"-> embedding rows {len(tok):,} accept ids 0..{len(tok) - 1:,}")
        check("id space has no gaps",
              len(tok) == len(set(tok.inv_vocab.keys()) |
                             set(tok.id_to_special.keys())),
              f"{len(tok):,} ids")

        # And the embedding built from it must accept every id the
        # tokenizer can produce.
        emb = torch.nn.Embedding(len(tok), 8)
        probe = torch.tensor(sorted(set(tok.inv_vocab.keys()) |
                                    set(tok.id_to_special.keys())))
        try:
            emb(probe)
            ok = True
        except IndexError:
            ok = False
        check("an embedding sized len(tokenizer) accepts every id", ok)

        samples = [
            "Hello world!",
            "### Instruction:\nWhat is CVE-2021-44228?\n\n### Response:\n"
            "A remote code execution vulnerability.",
            "def f(x):\n    return x ** 2  # unicode: café naïve",
        ]
        all_ok = all(tok.decode(tok.encode(s)) == s for s in samples)
        check("round trip is exact", all_ok)
        n = len(tok.encode(" ".join(samples)))
        check("tokenizer produces sane lengths", 0 < n < len(" ".join(samples)),
              f"{n} tokens")
    else:
        print("  (skipped: tokenizer.json not built yet)")

    # ------------------------------------------------------------------
    print("\n[7] generation smoke test")
    # ------------------------------------------------------------------
    from myllm.generate import generate, generate_batch

    # The model and the tokenizer must agree on the vocabulary size.
    if tok_path.exists():
        gen_model = Transformer(ModelConfig(
            name="gen", vocab_size=len(tok), d_model=64, n_layers=2,
            n_heads=4, n_kv_heads=2, max_seq_len=64,
        )).eval()
    else:
        gen_model = m
        tok = _dummy_tok()

    out = generate(gen_model, tok, prompt="Hello", max_new_tokens=5, seed=0)
    check("generate() returns a string", isinstance(out, str), repr(out[:40]))

    out2 = generate_batch(
        gen_model, tok, ["Hello", "A much longer prompt here"],
        max_new_tokens=4, seed=0,
    )
    check("generate_batch() returns one string per prompt",
          isinstance(out2, list) and len(out2) == 2,
          f"{len(out2)} outputs")

    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        print("\nFAILED:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 64)
    return 1 if n_fail else 0


def _dummy_tok():
    """A minimal tokenizer stand-in for when the real one is not built yet."""
    tok = BPETokenizer(vocab_size=256)
    return tok


if __name__ == "__main__":
    sys.exit(main())
