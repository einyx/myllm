#!/usr/bin/env bash
# Run the whole pipeline end to end.
#
#   ./scripts/run_all.sh              # full pipeline
#   ./scripts/run_all.sh --skip-data  # reuse the data already downloaded
#
# Each stage is idempotent: rerunning it skips work that is already done.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
SKIP_DATA=0
[[ "${1:-}" == "--skip-data" ]] && SKIP_DATA=1

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }

step "0. self-test"
$PY scripts/00_selftest.py

if [[ $SKIP_DATA -eq 0 ]]; then
  step "1. download data"
  $PY scripts/01_download_data.py

  step "2. build corpus"
  $PY scripts/02_build_corpus.py --total-chars 600000000
fi

step "3. train tokenizer"
$PY scripts/03_train_tokenizer.py --sample-chars 60000000 --vocab-size 32768

step "4. tokenize corpus"
$PY scripts/04_tokenize_corpus.py --workers 16

step "5. benchmark"
$PY scripts/07_benchmark.py --preset small --batch-sizes 8 16 32 --steps 4

step "6. train (tiny -- validates the pipeline)"
$PY scripts/05_train.py --preset tiny

step "7. evaluate"
$PY scripts/08_eval.py --ckpt runs/tiny/best.pt

step "8. build the write-up site"
$PY scripts/12_build_site.py

cat <<'EOF'

Pipeline complete. To start the real (multi-day) run:

    .venv/bin/python scripts/05_train.py --preset small

To talk to a checkpoint:

    .venv/bin/python scripts/06_generate.py --ckpt runs/small/best.pt --interactive

To publish the trained model to HuggingFace (needs `hf auth login`):

    ./scripts/13_auto_publish.sh runs/small/best.pt einyx/myllm-small
EOF
