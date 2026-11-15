#!/usr/bin/env bash
# Export a checkpoint and publish it.
#
#   ./scripts/11_publish.sh <ckpt> [tag] [tokenizer] [corpus]
#
# Example:
#   ./scripts/11_publish.sh runs/small-clean/best.pt v1.0
#
# Two destinations:
#
#   1. A GitHub RELEASE in einyx/myllm. This always works -- `gh` is
#      already authenticated, and GitHub accepts release assets up to
#      2GB (our safetensors file is ~503MB). This is the primary
#      distribution channel.
#
#   2. HuggingFace, if `hf auth login` has been run. Skipped silently if
#      not authenticated, so the script never hard-fails on it.
#
# Step 1 (the export) verifies itself before anything is uploaded: it
# re-tokenizes real corpus documents with both tokenizers and requires
# byte-identical output, then loads the exported model and requires the
# logits to match the checkpoint exactly.
set -euo pipefail

CKPT="${1:-runs/small-clean/best.pt}"
TAG="${2:-v1.0}"
TOKENIZER="${3:-data/processed/clean/tokenizer.json}"
CORPUS="${4:-data/processed/clean/train.jsonl}"
GITHUB_REPO="${GITHUB_REPO:-einyx/myllm}"
HF_REPO="${HF_REPO:-einyx/myllm-small}"

cd "$(dirname "$0")/.."
DIST="dist/myllm-small"

if [[ ! -f "$CKPT" ]]; then
  echo "error: checkpoint $CKPT not found"
  exit 1
fi

echo "=== 1/4 export ==="
.venv/bin/python scripts/10_export_hf.py \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --out "$DIST" \
  --corpus "$CORPUS" \
  --verify-samples 1000

echo
echo "=== 2/4 model card ==="
cp MODEL_CARD.md "$DIST/README.md"
cat > "$DIST/.gitattributes" <<'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
tokenizer.json diff=lfs merge=lfs -text
EOF
echo "  -> $DIST/README.md"

echo
echo "=== 3/4 GitHub release ($GITHUB_REPO $TAG) ==="
# Ship the weights and the tokenizer as release assets. The whole
# directory is also archived, so a user can download one file and get
# everything.
ASSETS=("$DIST/model.safetensors" "$DIST/tokenizer.json" "$DIST/config.json")

if gh release view "$TAG" --repo "$GITHUB_REPO" >/dev/null 2>&1; then
  echo "  release $TAG exists -- replacing assets"
  gh release upload "$TAG" "${ASSETS[@]}" --repo "$GITHUB_REPO" --clobber
else
  gh release create "$TAG" \
    "${ASSETS[@]}" \
    --repo "$GITHUB_REPO" \
    --title "myllm-small $TAG -- 126M decoder-only transformer, trained from scratch" \
    --notes "$(cat <<'NOTES'
A decoder-only transformer language model trained from scratch in PyTorch on
security-domain text -- byte-level BPE tokenizer, model, training loop and
sampler all implemented in this repository. No `transformers`, no
`tokenizers`, no `trl`.

**125.9M parameters** | 16 layers | d_model 768 | 12 heads / 4 KV heads
(GQA) | SwiGLU | RoPE | RMSNorm | context 512 | 32,767-token byte-level BPE

Trained on 60M tokens of NIST security standards publications (CC0-1.0),
NIST NVD-derived Q&A (Apache-2.0), and CIRCL vulnerability records
(CC-BY-4.0) on a CPU in ~26 hours.

### Download

| file | what |
|---|---|
| `model.safetensors` | the weights |
| `tokenizer.json` | the tokenizer (HuggingFace `tokenizers` format) |
| `config.json` | architecture + training metadata |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("path/to/tokenizer.json")
model = AutoModelForCausalLM.from_pretrained(
    "path/to/dir-with-config.json-and-model.safetensors",
    trust_remote_code=True,
)
```

The modeling code ships with the repository and is vendored into the
export so the published weights are loaded by the same code that trained
them.

### Please read the model card

This model **invents CVE identifiers**. Many of them will be real
vulnerabilities; it cannot tell you which. Verify any CVE against NVD
before acting on it. It is also undertrained (60M tokens, ~2.4% of
compute-optimal) and can produce confident, wrong security advice.

Full model card: MODEL_CARD.md in the repository.
Write-up: https://einyx.github.io/myllm/
NOTES
)"
fi
echo "  -> https://github.com/$GITHUB_REPO/releases/tag/$TAG"

echo
echo "=== 4/4 HuggingFace ($HF_REPO) ==="
if .venv/bin/hf auth whoami >/dev/null 2>&1; then
  .venv/bin/hf upload "$HF_REPO" "$DIST" \
    --repo-type model \
    --commit-message "Add myllm-small $TAG: 126M decoder-only transformer trained from scratch on security data"
  echo "  -> https://huggingface.co/$HF_REPO"
else
  echo "  skipped: not logged in to HuggingFace."
  echo "  run '.venv/bin/hf auth login' then re-run this script to publish there too."
fi

echo
echo "published."
