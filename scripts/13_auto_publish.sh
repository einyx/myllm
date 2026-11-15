#!/usr/bin/env bash
# Wait for a training run to finish, then export and publish the result.
#
#   ./scripts/13_auto_publish.sh [ckpt] [tag] [tokenizer] [corpus]
#
# Example:
#   ./scripts/13_auto_publish.sh runs/small-clean/best.pt v1.0
#
# This exists because the full training run takes ~26 hours. Rather than
# have a human remember to come back and publish, this waits for the
# trainer to exit and then runs the verified export + upload.
#
# Publishing to GitHub Releases needs no extra setup. Publishing to
# HuggingFace additionally requires `hf auth login`; if that has not been
# done, the HuggingFace step is skipped and the GitHub release still goes
# out.
set -euo pipefail

CKPT="${1:-runs/small-clean/best.pt}"
TAG="${2:-v1.0}"
TOKENIZER="${3:-data/processed/clean/tokenizer.json}"
CORPUS="${4:-data/processed/clean/train.jsonl}"
LOG="runs/auto_publish.log"

cd "$(dirname "$0")/.."
mkdir -p runs

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

log "waiting for the training run to finish ..."
while pgrep -f "scripts/05_train.py" > /dev/null; do
    sleep 300
done
log "trainer has exited."

# The trainer writes last.pt / best.pt as its final action; give the
# filesystem a moment to flush.
sleep 30

if [[ ! -f "$CKPT" ]]; then
    log "ERROR: $CKPT not found. Nothing to publish."
    exit 1
fi

log "publishing $CKPT as $TAG ..."
if ./scripts/11_publish.sh "$CKPT" "$TAG" "$TOKENIZER" "$CORPUS" >> "$LOG" 2>&1; then
    log "done: https://github.com/einyx/myllm/releases/tag/$TAG"
else
    log "ERROR: publish failed -- see $LOG"
    exit 1
fi
