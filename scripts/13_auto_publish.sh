#!/usr/bin/env bash
# Wait for a training run to finish, then export and publish the result.
#
#   ./scripts/13_auto_publish.sh runs/small-clean/best.pt einyx/myllm-small
#
# Requires `hf auth login` to have been run already.
#
# This exists because the full training run takes ~26 hours. Rather than
# have a human remember to come back and publish, this waits for the
# trainer to exit and then runs the verified export + upload.
set -euo pipefail

CKPT="${1:-runs/small-clean/best.pt}"
REPO="${2:-einyx/myllm-small}"
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

log "publishing $CKPT to $REPO ..."
if ./scripts/11_publish.sh "$CKPT" "$REPO" "$TOKENIZER" "$CORPUS" >> "$LOG" 2>&1; then
    log "done: https://huggingface.co/$REPO"
else
    log "ERROR: publish failed -- see $LOG"
    exit 1
fi
