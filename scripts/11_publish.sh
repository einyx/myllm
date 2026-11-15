#!/usr/bin/env bash
# Export a checkpoint to HuggingFace and publish it.
#
#   ./scripts/11_publish.sh runs/small-clean/best.pt einyx/myllm-small
#
# Steps:
#   1. export to HuggingFace format (this VERIFIES the tokenizer
#      conversion and the model logits before anything is uploaded)
#   2. copy MODEL_CARD.md in as the model card
#   3. upload
#
# Requires:  hf auth login
set -euo pipefail

CKPT="${1:-runs/small-clean/best.pt}"
REPO="${2:-einyx/myllm-small}"
TOKENIZER="${3:-data/processed/clean/tokenizer.json}"
CORPUS="${4:-data/processed/clean/train.jsonl}"
DIST="dist/$(basename "$REPO")"

cd "$(dirname "$0")/.."

if [[ ! -f "$CKPT" ]]; then
  echo "error: checkpoint $CKPT not found"
  exit 1
fi

echo "=== 1/3 export ==="
.venv/bin/python scripts/10_export_hf.py \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --out "$DIST" \
  --corpus "$CORPUS" \
  --verify-samples 1000

echo
echo "=== 2/3 model card ==="
cp MODEL_CARD.md "$DIST/README.md"

cat > "$DIST/.gitattributes" <<'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
tokenizer.json diff=lfs merge=lfs -text
EOF

echo "  -> $DIST/README.md"

echo
echo "=== 3/3 upload ==="
.venv/bin/hf upload "$REPO" "$DIST" \
  --repo-type model \
  --commit-message "Add $REPO: 126M decoder-only transformer trained from scratch on security data"

echo
echo "published: https://huggingface.co/$REPO"
