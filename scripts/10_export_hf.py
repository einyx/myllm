#!/usr/bin/env python3
"""
Export a checkpoint to HuggingFace format.

    python scripts/10_export_hf.py \
        --ckpt runs/small-clean/best.pt \
        --tokenizer data/processed/clean/tokenizer.json \
        --out dist/myllm-small

Produces a directory that `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` can load.

THE IMPORTANT DESIGN CHOICE
    We do NOT re-implement the model for HuggingFace. Instead we copy
    this repository's own `myllm/` package into the export directory as
    `myllm_vendor/`, and the generated `modeling_myllm.py` imports it.

    That guarantees the published weights are loaded by the exact code
    that trained them. A hand-written HF re-implementation would be a
    second implementation, and the two would silently drift.

THE TOKENIZER
    Our tokenizer is a byte-level BPE whose vocabulary is literally
    "ids 0..255 are the raw bytes, then the merges in merge order".
    HuggingFace's `tokenizers` library models byte-level BPE as GPT-2
    does: the 256 base symbols are the bytes mapped through the GPT-2
    byte-to-unicode table, and merges are pairs of those.

    So the conversion is mechanical:
        our token (a byte string)  ->  HF token (the byte-to-unicode
                                         rendering of those bytes)
        our merge (left, right)    ->  HF merge "left_hf right_hf"

    Because the conversion can go wrong in subtle ways (wrong merge
    order, wrong alphabet, a different pre-tokenizer regex), we VERIFY
    it: tokenize a large sample of the real corpus with both
    implementations and require byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myllm.config import ModelConfig  # noqa: E402
from myllm.model import Transformer  # noqa: E402
from myllm.tokenizer import BPETokenizer  # noqa: E402


# ==========================================================================
# GPT-2 byte <-> unicode
# ==========================================================================

def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte-to-unicode table.

    Maps each of the 256 byte values to a printable unicode character so
    that a byte-level BPE vocabulary can be written as plain text. This
    is the same table HuggingFace's `ByteLevel` uses.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


B2U = bytes_to_unicode()


def to_hf_str(token_bytes: bytes) -> str:
    """Our token (raw bytes) -> HuggingFace token (byte-level string)."""
    return "".join(B2U[b] for b in token_bytes)


# ==========================================================================
# Tokenizer conversion
# ==========================================================================

def build_hf_tokenizer(tok: BPETokenizer):
    """Build a `tokenizers` BPE equivalent to our tokenizer."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    # --- vocabulary -------------------------------------------------
    vocab: dict[str, int] = {}
    for tid, token_bytes in tok.inv_vocab.items():
        vocab[to_hf_str(token_bytes)] = tid
    for special, tid in tok.special_to_id.items():
        vocab[special] = tid

    # --- merges, in priority order -----------------------------------
    # `tok.merges` is a dict; we need the original merge ORDER, which
    # is the order the merges were learned (== ascending id).
    merges = []
    for (left, right), tid in sorted(tok.merges.items(), key=lambda kv: kv[1]):
        merges.append((to_hf_str(left), to_hf_str(right)))

    backend = BPE(
        vocab=vocab,
        merges=merges,
        unk_token=None,
        fuse_unk=False,
        byte_fallback=False,
    )

    hf = Tokenizer(backend)
    hf.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    hf.decoder = ByteLevelDecoder()

    # Register the special token so it survives as a single id.
    from tokenizers import AddedToken
    for special, tid in tok.special_to_id.items():
        hf.add_special_tokens([AddedToken(special, single_word=False,
                                         normalized=False, special=True)])

    return hf, len(merges)


def verify_tokenizers(
    ours: BPETokenizer,
    hf,
    corpus_path: Path,
    n_samples: int = 2000,
) -> tuple[int, int]:
    """Tokenize real corpus text with both implementations and compare.

    Returns (n_checked, n_mismatched).
    """
    import json as _json

    checked = mismatched = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if checked >= n_samples:
                break
            try:
                text = _json.loads(line)["text"]
            except Exception:
                continue
            ours_ids = ours.encode(text)
            hf_ids = hf.encode(text).ids
            checked += 1
            if ours_ids != hf_ids:
                mismatched += 1
                if mismatched == 1:
                    print("\n  first mismatch:")
                    print(f"    ours: {ours_ids[:40]}")
                    print(f"    hf  : {hf_ids[:40]}")
                    for i, (a, b) in enumerate(zip(ours_ids, hf_ids)):
                        if a != b:
                            print(f"    diverges at token {i}: "
                                  f"{a} vs {b}")
                            print(f"    context: {text[max(0,i-20):i+20]!r}")
                            break
    return checked, mismatched


# ==========================================================================
# Model export
# ==========================================================================

def export_weights(ckpt_path: Path, out_dir: Path) -> dict:
    """Write model.safetensors + config.json."""
    from safetensors.torch import save_file

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["model_config"])
    state = payload["model"]

    # We wrap our Transformer in an attribute named `myllm`, so the
    # checkpoint keys get that prefix.
    prefixed = {f"myllm.{k}": v.contiguous() for k, v in state.items()}

    # safetensors refuses to save shared tensors. Our lm_head.weight IS
    # tok_emb.weight (weight tying). HF handles tying itself via
    # config.tie_word_embeddings, so we drop the duplicate key.
    tied = prefixed.pop("myllm.lm_head.weight", None)
    if tied is not None:
        assert torch.equal(tied, prefixed["myllm.tok_emb.weight"]), \
            "lm_head is not tied to tok_emb; refusing to drop the key"

    save_file(prefixed, str(out_dir / "model.safetensors"))

    hf_config = {
        "architectures": ["MyllmForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_myllm.MyllmConfig",
            "AutoModelForCausalLM": "modeling_myllm.MyllmForCausalLM",
        },
        "model_type": "myllm",
        "tie_word_embeddings": True,
        # --- our architecture fields, mirrored exactly ---
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "max_seq_len": cfg.max_seq_len,
        "rope_base": cfg.rope_base,
        "norm_eps": cfg.norm_eps,
        "mlp_hidden": cfg.mlp_hidden,
        "init_std": cfg.init_std,
        # --- training metadata, for provenance ---
        "myllm_training": {
            "step": payload.get("step"),
            "tokens_seen": payload.get("tokens_seen"),
            "val_loss": payload.get("val_loss"),
            "train_config": payload.get("train_config"),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(hf_config, indent=2))
    return hf_config


# ==========================================================================
# The HF wrapper files
# ==========================================================================

CONFIGURATION_TEMPLATE = '''"""HuggingFace config for the myllm architecture."""

from transformers import PretrainedConfig


class MyllmConfig(PretrainedConfig):
    """Mirrors `myllm.config.ModelConfig` exactly.

    The model is a decoder-only transformer: RoPE positions, RMSNorm,
    grouped-query attention, SwiGLU feed-forward, pre-norm residual
    blocks, and tied input/output embeddings.
    """

    model_type = "myllm"

    def __init__(
        self,
        vocab_size=32768,
        d_model=768,
        n_layers=16,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=512,
        rope_base=10000.0,
        norm_eps=1e-5,
        mlp_hidden=None,
        init_std=0.02,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.mlp_hidden = mlp_hidden
        self.init_std = init_std
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=None,
            eos_token_id=256,   # <|endoftext|>
            pad_token_id=None,
            **kwargs,
        )
'''

MODELING_TEMPLATE = '''"""HuggingFace wrapper around the myllm architecture.

The actual model lives in `myllm_vendor_*.py` -- verbatim copies of this
repository's `myllm/` modules (with only their relative imports rewritten
to point at the sibling copies). Importing the original implementation
rather than re-writing it guarantees the published weights are loaded by
the same code that trained them.

NOTE ON IMPORTS: these are RELATIVE imports on purpose. HuggingFace's
dynamic-module loader treats an absolute import as a third-party package
and refuses to load the model if it is not installed; a relative import
is understood as "a sibling file in this repo" and is copied alongside
the modeling code automatically.
"""

from typing import Optional

import torch
from torch import nn

from .configuration_myllm import MyllmConfig
from .myllm_vendor_config import ModelConfig
from .myllm_vendor_model import Transformer

try:
    # transformers >= 4.44
    from transformers.modeling_outputs import CausalLMOutput
except ImportError:  # pragma: no cover - very old transformers
    from transformers.utils import ModelOutput as CausalLMOutput


class MyllmForCausalLM(nn.Module):
    """Causal language model with a `.generate()` interface.

    Implemented as a plain nn.Module rather than a PreTrainedModel so it
    has no hard dependency on a specific transformers version. The
    checkpoint is loaded from safetensors by `from_pretrained` below.
    """

    config_class = MyllmConfig

    @classmethod
    def register_for_auto_class(cls, auto_class, *args, **kwargs):
        """Hook that transformers calls when it registers this class.

        It exists so that `save_pretrained` can round-trip through the
        auto classes. We implement our own export script instead, so
        there is nothing to register.
        """
        return None

    def __init__(self, config: MyllmConfig):
        super().__init__()
        self.config = config
        inner = ModelConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            norm_eps=config.norm_eps,
            mlp_hidden=config.mlp_hidden,
            init_std=config.init_std,
            tie_embeddings=getattr(config, "tie_word_embeddings", True),
        )
        self.myllm = Transformer(inner)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        out = self.myllm(input_ids, targets=labels)
        return CausalLMOutput(logits=out.logits, loss=out.loss)

    # -- convenience ------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=64,
        temperature=0.8,
        top_k=0,
        top_p=0.95,
        repetition_penalty=1.15,
        eos_token_id: Optional[int] = None,
        generator=None,
    ):
        """Sample continuations. Returns a LongTensor of new token ids."""
        from .myllm_vendor_generate import (
            apply_temperature,
            apply_repetition_penalty,
            top_k_filter,
            top_p_filter,
        )
        import torch.nn.functional as F

        self.eval()
        device = next(self.parameters()).device
        x = input_ids.to(device)
        past = x
        cache = self.myllm.new_cache()
        start = 0
        generated = []

        for _ in range(max_new_tokens):
            out = self.myllm(x, cache=cache, start_pos=start)
            start += x.shape[1]
            logits = out.logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(
                    logits, past, repetition_penalty)
            logits = apply_temperature(logits, temperature)
            if top_k:
                logits = top_k_filter(logits, top_k)
            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(
                probs, num_samples=1, generator=generator
            ).squeeze(-1)
            if eos_token_id is not None:
                done = (nxt == eos_token_id)
                if bool(done.all()):
                    break
            generated.append(nxt)
            past = torch.cat([past, nxt[:, None]], dim=1)
            x = nxt[:, None]

        if not generated:
            return torch.zeros((input_ids.shape[0], 0), dtype=torch.long)
        return torch.stack(generated, dim=1)

    # -- loading ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, path, config=None, **kwargs):
        """Load from a directory containing config.json + model.safetensors.

        `config` is accepted (and preferred) because the transformers
        auto-loader builds it via `AutoConfig` and passes it in.
        """
        import json
        import os

        from safetensors.torch import load_file

        path = str(path)
        if os.path.isdir(path):
            cfg_path = os.path.join(path, "config.json")
            weights_path = os.path.join(path, "model.safetensors")
        else:
            cfg_path = path
            weights_path = path.replace(".json", ".safetensors")

        if config is None:
            with open(cfg_path) as f:
                config = MyllmConfig(**json.load(f))

        model = cls(config)
        state = load_file(weights_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # `lm_head.weight` is intentionally missing: it is tied to
        # `tok_emb.weight`.
        missing = [k for k in missing if k != "myllm.lm_head.weight"]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
            )
        return model
'''


def write_hf_wrapper_files(out_dir: Path) -> None:
    (out_dir / "configuration_myllm.py").write_text(CONFIGURATION_TEMPLATE)
    (out_dir / "modeling_myllm.py").write_text(MODELING_TEMPLATE)

    # Vendor our package so the published model uses the published code.
    #
    # HuggingFace's dynamic-module loader resolves relative imports as
    # SIBLING FILES (`./foo.py`), not as packages -- it copies
    # `from .foo import bar` by looking for `foo.py` next to the modeling
    # file. A subpackage directory would not be found. So we flatten the
    # package into `myllm_vendor_<module>.py` files and rewrite the one
    # kind of import that changes.
    #
    # The rewrite is mechanical and total: every `from .x import y`
    # becomes `from .myllm_vendor_x import y`. Nothing else is touched,
    # so the published model code is the training code.
    src = ROOT / "myllm"
    for name in ("config", "model", "tokenizer", "data", "generate",
                 "utils", "train"):
        text = (src / f"{name}.py").read_text()
        text = re.sub(
            r"from\s+\.(\w+)\s+import",
            r"from .myllm_vendor_\1 import",
            text,
        )
        (out_dir / f"myllm_vendor_{name}.py").write_text(text)


def verify_model(out_dir: Path, ckpt_path: Path, n_tokens: int = 512) -> float:
    """Check that the exported model reproduces the original exactly.

    Loads the original checkpoint with our own code and the exported one
    through the HuggingFace wrapper, then compares logits on the same
    input. Returns the max absolute difference (should be exactly 0).
    """
    import importlib
    import shutil
    import tempfile

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["model_config"])
    original = Transformer(cfg)
    original.load_state_dict(payload["model"])
    original.eval()

    # The exported modeling code uses RELATIVE imports, so it has to be
    # loaded as a package. Copy it into a temp package and import it.
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "hfcheck"
        shutil.copytree(out_dir, pkg)
        (pkg / "__init__.py").write_text("")
        sys.path.insert(0, tmp)
        try:
            for name in list(sys.modules):
                if name.startswith("hfcheck"):
                    del sys.modules[name]
            modeling = importlib.import_module("hfcheck.modeling_myllm")
            exported = modeling.MyllmForCausalLM.from_pretrained(str(pkg))
            exported.eval()

            torch.manual_seed(0)
            n_tokens = min(n_tokens, cfg.max_seq_len)
            x = torch.randint(0, cfg.vocab_size, (2, n_tokens))
            with torch.no_grad():
                a = original(x).logits
                b = exported(x)["logits"]
            diff = (a - b).abs().max().item()

            # Also check the tokenizer round-trips through the HF path.
            from tokenizers import Tokenizer
            hf_tok = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
            sample = "### Instruction:\nWhat is CVE-2021-44228?\n\n### Response:\n"
            ids = hf_tok.encode(sample).ids
            decoded = hf_tok.decode(ids)
            if decoded != sample:
                raise RuntimeError(
                    "HF tokenizer does not round-trip: "
                    f"{decoded!r} != {sample!r}"
                )
            print(f"[export]   model logits match, max abs diff = {diff:.2e}")
            print(f"[export]   HF tokenizer round-trips exactly")
            return diff
        finally:
            sys.path.remove(tmp)


# ==========================================================================
# Driver
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "data" / "processed" / "clean" / "train.jsonl"),
                    help="corpus sample used to verify the tokenizer")
    ap.add_argument("--verify-samples", type=int, default=2000)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer ----------------------------------------------------
    print("[export] loading tokenizer ...")
    ours = BPETokenizer.load(args.tokenizer)
    hf_tok, n_merges = build_hf_tokenizer(ours)
    print(f"[export] {len(ours):,} tokens, {n_merges:,} merges")

    # ---- verify -------------------------------------------------------
    if not args.skip_verify:
        corpus = Path(args.corpus)
        if corpus.exists():
            print(f"[export] verifying tokenization on up to "
                  f"{args.verify_samples:,} real documents ...")
            checked, mismatched = verify_tokenizers(
                ours, hf_tok, corpus, args.verify_samples
            )
            rate = 100 * mismatched / max(1, checked)
            print(f"[export]   checked {checked:,}, mismatched {mismatched:,} "
                  f"({rate:.3f}%)")
            if mismatched:
                print("[export] FAILED: the converted tokenizer does not "
                      "match. Refusing to publish a broken tokenizer.")
                return 1
            print("[export]   tokenizer conversion is exact.")
        else:
            print(f"[export] WARNING: {corpus} not found, skipping "
                  "the tokenization check.")

    # ---- write tokenizer files ------------------------------------------
    print("[export] writing tokenizer files ...")
    hf_tok.save(str(out_dir / "tokenizer.json"))

    eos = ours.eos_id
    (out_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": None,
        "eos_token": ours.specials[0],
        "pad_token": None,
        "unk_token": None,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 4096,
    }, indent=2))
    (out_dir / "special_tokens_map.json").write_text(json.dumps({
        "eos_token": {"content": ours.specials[0], "lstrip": False,
                      "normalized": False, "rstrip": False,
                      "single_word": False},
        "bos_token": None,
        "pad_token": None,
        "unk_token": None,
    }, indent=2))

    # ---- model ---------------------------------------------------------
    print("[export] exporting weights ...")
    cfg = export_weights(Path(args.ckpt), out_dir)
    print(f"[export]   {cfg['n_layers']} layers, d_model {cfg['d_model']}, "
          f"vocab {cfg['vocab_size']:,}")

    # ---- wrapper code ---------------------------------------------------
    print("[export] writing modeling code ...")
    write_hf_wrapper_files(out_dir)

    # ---- verify the exported artifact ------------------------------------
    print("[export] verifying the exported model ...")
    try:
        diff = verify_model(out_dir, Path(args.ckpt))
        if diff > 1e-5:
            print(f"[export] FAILED: logits differ by {diff:.2e}")
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"[export] FAILED to verify: {type(e).__name__}: {e}")
        return 1

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\n[export] wrote {out_dir}  ({total / 1e6:.1f} MB)")
    for f in sorted(out_dir.iterdir()):
        if f.is_dir():
            print(f"  {f.name}/")
        else:
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
